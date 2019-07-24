from typing import Dict, Optional

import torch

from allennlp.models import Model
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

try:
    from allennlp.nn.util import last_dim_softmax
except ImportError:
    print('cant last_dim_softmax')
from allennlp.nn.util import weighted_sum, get_text_field_mask, device_mapping

from .utils.utils import *

class ESIM(torch.nn.Module):
    """
    Parameters
    ----------
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).

    """
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,                 
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 premise_composer: Optional[Seq2SeqEncoder] = None,
                 hypothesis_composer: Optional[Seq2SeqEncoder] = None,
                 combine_feedforward: Optional[FeedForward] = None,
                 aggregate_feedforward: Optional[FeedForward] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),                 
                 num_wrapping_dims=0,
                 vocab=None) -> None:
        
        super(ESIM, self).__init__()

        self.vocab = vocab
        
        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = MatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder
        self._premise_composer = premise_composer
        self._hypothesis_composer = hypothesis_composer or premise_composer

        self._combine_feedforward = combine_feedforward
        self._aggregate_feedforward = aggregate_feedforward

        self._num_wrapping_dims = num_wrapping_dims

        #initializer(self)
        
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                premise_mask=None,
                hypothesis_mask=None,
                wrap_output=False):

        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``

        """

        #print(self._num_wrapping_dims)
        #print(premise['tokens'].shape)
        #print(premise['elmo'].shape)

        #print(self._text_field_embedder.token_embedder_tokens.weight.data[:10,0])
        #print(self._text_field_embedder.token_embedder_tokens._projection.weight.data[0,0])
        embedded_premise = self._text_field_embedder(premise,
                                                     num_wrapping_dims=self._num_wrapping_dims)
        embedded_hypothesis = self._text_field_embedder(hypothesis)

        #print(embedded_premise.shape)
        
        if premise_mask is None:
            premise_mask = get_text_field_mask(premise,
                                               num_wrapping_dims=self._num_wrapping_dims).float()
        if hypothesis_mask is None:
            hypothesis_mask = get_text_field_mask(hypothesis).float()

        '''
        key = 'tokens'
        for q in range(1): #(premise[key].size(0)):
            print([int(i.data[0]) for i in hypothesis[key][q]])
            print([self.vocab._index_to_token[key][i.data[0]] for i in hypothesis[key][q]])
            print([int(i.data[0]) for i in premise[key][q,0]])
            print([self.vocab._index_to_token[key][i.data[0]] for i in premise[key][q,0]])
            print(embedded_premise[q,0,0,0])
            print([int(i.data[0]) for i in premise_mask[q,0]])
        '''
        #print(premise_mask.shape)
            
        premise_dim = embedded_premise.size(-1)
        max_premise_length = embedded_premise.size(-2)
        batch_size = embedded_premise.size(0)
        
        if len(premise_mask.shape) > 2:
            num_evidence = premise_mask.size(1)
        else:
            num_evidence = 1

        #print(batch_size, num_evidence, max_premise_length, premise_dim)
            
        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise.view(batch_size*num_evidence,
                                                                           -1,premise_dim),
                                                     premise_mask.view(batch_size*num_evidence, -1))
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        #for now, treat the premise as a B x E*T input instead of B x E x T
        premise_mask = premise_mask.view(batch_size, -1)
        embedded_premise = embedded_premise.view(batch_size,
                                                 num_evidence*max_premise_length, -1)

        #print(premise_mask)
        #print(premise_mask.sum(dim=-1))
        #print(embedded_premise.shape)
        #print(premise_mask.shape)
        
        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        
        # Shape: (batch_size, num_evidence*premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise,
                                                   projected_hypothesis)

        #print(similarity_matrix.shape)
        #print(similarity_matrix[0])
        
        if self._num_wrapping_dims and (self._combine_feedforward is not None or wrap_output):
            embedded_hypothesis = embedded_hypothesis.unsqueeze(1).expand(batch_size, num_evidence,
                                                                          embedded_hypothesis.size(1),
                                                                          -1).contiguous().view(batch_size*num_evidence,
                                                                                                embedded_hypothesis.size(1), -1)
            hypothesis_mask = hypothesis_mask.unsqueeze(1).expand(batch_size,
                                                                  num_evidence,
                                                                  -1).contiguous().view(batch_size*num_evidence, -1)
            embedded_premise = embedded_premise.view(batch_size*num_evidence, -1, premise_dim)
            premise_mask = premise_mask.view(batch_size*num_evidence, -1)
        
            similarity_matrix = similarity_matrix.view(batch_size*num_evidence,
                                                       max_premise_length, -1)
            
        # Shape: (batch_size*num_evidence, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size*num_evidence, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size*num_evidence, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(),
                                         premise_mask)
        # Shape: (batch_size*num_evidence, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis,
                                           #UNDO
                                           embedded_premise-attended_hypothesis,
                                           embedded_premise*attended_hypothesis
        ], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise,
                                              #UNDO
                                              embedded_hypothesis-attended_premise,
                                              embedded_hypothesis*attended_premise
        ], dim=-1)

        #print(premise_compare_input.shape)
        #print(hypothesis_compare_input.shape)
        
        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size*num_evidence, premise_length, compare_dim)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size*num_evidence, hypothesis_length, compare_dim)

        if self._premise_composer is not None:
            compared_premise = self._premise_composer(compared_premise, premise_mask)
        if self._hypothesis_composer is not None:
            compared_hypothesis = self._hypothesis_composer(compared_hypothesis, hypothesis_mask)

        #compared_premise = embedded_premise
        #compared_hypothesis = embedded_hypothesis
            
        '''
        #UNDO
        compared_premise = compared_premise.sum(dim=1)
        compared_hypothesis = compared_hypothesis.sum(dim=1)
        output = torch.cat([compared_premise, compared_hypothesis], dim=-1)

        '''
        
        #get sequence max and mean
        premise_len = premise_mask.sum(dim=-1)
        #print(premise_len)
        hypothesis_len = hypothesis_mask.sum(dim=-1)
        #print(hypothesis_len)
        #print(compared_premise.shape)
        #print(compared_hypothesis.shape)

        pooled_premise = torch.cat([sequence_max(compared_premise, premise_mask),
                                    sequence_mean(compared_premise, premise_len)],
                                   dim=-1)
        pooled_hypothesis = torch.cat([sequence_max(compared_hypothesis, hypothesis_mask),
                                       sequence_mean(compared_hypothesis, hypothesis_len)],
                                      dim=-1)            

        #print(pooled_premise.shape)
        #print(pooled_hypothesis.shape)
        
        output = torch.cat([pooled_premise, pooled_hypothesis], dim=-1)

        if self._num_wrapping_dims and self._combine_feedforward is not None:
            output = self._combine_feedforward(output).view(batch_size,
                                                            num_evidence, -1)
                
            #get seq max and mean
            evidence_mask = premise_len.gt(0).view(batch_size, num_evidence)
            evidence_len = evidence_mask.sum(dim=-1).float()
            output = torch.cat([sequence_max(output, evidence_mask),
                                     sequence_mean(output, evidence_len)],
                                    dim=-1)
            
        if self._aggregate_feedforward:
            output = self._aggregate_feedforward(output)

        #print(output)
            
        return output
        
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, vocab_weight=None) -> 'ESIM':

        embedder_params = params.pop("text_field_embedder")
        if vocab_weight is None:
            text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        else:
            token_embedders = {}
            keys = list(embedder_params.keys())
            for key in keys:
                e_params = embedder_params.pop(key)
                if e_params['type'] == 'embedding':
                    token_embedders[key] = Embedding(vocab_weight.size(0),
                                                     vocab_weight.size(1),
                                                     e_params.pop('projection_dim'),
                                                     vocab_weight,
                                                     trainable=False)
                else:                    
                    token_embedders[key] = TokenEmbedder.from_params(vocab, e_params)
            text_field_embedder = BasicTextFieldEmbedder(token_embedders)
            
        premise_encoder_params = params.pop("premise_encoder", None)
        if premise_encoder_params is not None:
            premise_encoder = Seq2SeqEncoder.from_params(premise_encoder_params)
        else:
            premise_encoder = None

        hypothesis_encoder_params = params.pop("hypothesis_encoder", None)
        if hypothesis_encoder_params is not None:
            hypothesis_encoder = Seq2SeqEncoder.from_params(hypothesis_encoder_params)
        else:
            hypothesis_encoder = None

        attend_feedforward = FeedForward.from_params(params.pop('attend_feedforward'))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        compare_feedforward = FeedForward.from_params(params.pop('compare_feedforward'))
            
        premise_composer_params = params.pop("premise_composer", None)
        if premise_composer_params is not None:
            premise_composer = Seq2SeqEncoder.from_params(premise_composer_params)
        else:
            premise_composer = None

        hypothesis_composer_params = params.pop("hypothesis_composer", None)
        if hypothesis_composer_params is not None:
            hypothesis_composer = Seq2SeqEncoder.from_params(hypothesis_composer_params)
        else:
            hypothesis_composer = None

        combine_feedforward_params = params.pop("combine_feedforward", None)
        combine_feedforward = None
        if combine_feedforward_params is not None:
            combine_feedforward = FeedForward.from_params(combine_feedforward_params)

        aggregate_feedforward_params = params.pop("aggregate_feedforward", None)
        aggregate_feedforward = None
        if aggregate_feedforward_params is not None:
            aggregate_feedforward = FeedForward.from_params(aggregate_feedforward_params)

        num_wrapping_dims = params.pop("num_wrapping_dims", 0)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        
        params.assert_empty(cls.__name__)
        
        return cls(text_field_embedder=text_field_embedder,
                   attend_feedforward=attend_feedforward,
                   similarity_function=similarity_function,
                   compare_feedforward=compare_feedforward,
                   premise_encoder=premise_encoder,
                   hypothesis_encoder=hypothesis_encoder,
                   premise_composer=premise_composer,
                   hypothesis_composer=hypothesis_composer,
                   combine_feedforward=combine_feedforward,
                   aggregate_feedforward=aggregate_feedforward,
                   initializer=initializer,
                   num_wrapping_dims=num_wrapping_dims,
                   vocab=vocab)
