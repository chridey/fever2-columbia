import itertools
import random

from typing import Dict, Optional

import torch
import numpy as np

from allennlp.common import Params
#from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model, load_archive
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
try:
    from allennlp.nn.util import last_dim_softmax
except ImportError:
    from allennlp.nn.util import masked_softmax as last_dim_softmax
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from .metrics import FeverScore
from .utils.utils import *

from .rl_ptr_extractor import ActorCritic
from .esim import ESIM
from .feature_model import FeatureModel
#from my_decomposable_attention import MyDecomposableAttention

from torch.nn.utils import clip_grad_norm

def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            try:
                grad_log['grad_norm'+n] = tot_grad.data[0]
            except IndexError:
                grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm(
	    [p for p in params if p.requires_grad], clip_grad)

        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

@Model.register("esim_rl_ptr_extractor")
class ESIMRLPtrExtractor(Model):

    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
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
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 sentence_selection_esim: ESIM,
                 entailment_esim: ESIM,
                 ptr_extract_summ: ActorCritic,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ei_reward_weight=1,
                 fix_entailment_params=False,
                 fix_sentence_extraction_params=False,
                 nei_label=0,
                 train_gold_evidence=False,
                 use_decoder_states=False,
                 beam_size=5) -> None:

        super(ESIMRLPtrExtractor, self).__init__(vocab, regularizer)

        self._sentence_selection_esim = sentence_selection_esim
        self._entailment_esim = entailment_esim
        self._ptr_extract_summ = ptr_extract_summ
        
        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._evidence_f1 = F1Measure(1)
        self._fever = FeverScore(nei_label)
        self._fever_evidence_only = FeverScore(nei_label, True)
        
        self._loss = torch.nn.CrossEntropyLoss()
        self._evidence_loss = torch.nn.functional.cross_entropy
        initializer(self)

        self.lambda_weight = 1
        self._ei_reward_weight = ei_reward_weight
        #print(self._ei_reward_weight)
        
        self.grad_fn = get_grad_fn(self._ptr_extract_summ, 5)

        self._fix_entailment_params = fix_entailment_params
        self._fix_sentence_extraction_params = fix_sentence_extraction_params

        self._nei_label = nei_label
        #print(self._fix_entailment_params, self._fix_sentence_extraction_params)
        self._train_gold_evidence = train_gold_evidence
        self._use_decoder_states = use_decoder_states
        self._beam_size = beam_size
        
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                evidence: torch.IntTensor = None,
                pad_idx=-1,
                max_select=5,
                gamma=0.95,
                teacher_forcing_ratio=1,
                features=None,
                metadata=None) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        evidence : torch.IntTensor, optional (default = None)
            From a ``ListField``
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        #print([int(i.data[0]) for i in premise['tokens'][0,0]])
            
        premise_mask = get_text_field_mask(premise,
                                           num_wrapping_dims=1).float()

        hypothesis_mask = get_text_field_mask(hypothesis).float()
        
        aggregated_input = self._sentence_selection_esim(premise, hypothesis,
                                                        premise_mask,
                                                         hypothesis_mask,
                                                         wrap_output=True,
                                                         features=features)

        batch_size, num_evidence, max_premise_length = premise_mask.shape
        #print(premise_mask.shape)
        aggregated_input = aggregated_input.view(batch_size, num_evidence, -1)
        evidence_mask = premise_mask.sum(dim=-1).gt(0)
        evidence_len = evidence_mask.view(batch_size, -1).sum(dim=-1)
        #print(aggregated_input.shape)
        #print(evidence_len)
        
        #for each element in the batch
        valid_indices = []
        indices = []
        probs = []
        baselines = []
        states = []
        selected_evidence_lengths = []
        for i in range(evidence.size(0)):
            #print(label[i].data[0], evidence[i])
            
            gold_evidence=None
            #teacher forcing, give a list of indices and get the probabilities
            #print(label[i])
            try:
                curr_label = label[i].data[0]
            except IndexError:
                curr_label = label[i].item()
                
            if random.random() > teacher_forcing_ratio and curr_label != self._nei_label and float(evidence[i].ne(pad_idx).sum()) > 0:
                gold_evidence = evidence[i]
            #print(gold_evidence)

            output = self._ptr_extract_summ(aggregated_input[i],
                                                            max_select,
                                                            evidence_mask[i],
                                                            gold_evidence,
                                                            beam_size=self._beam_size)
            #print(output['states'].shape)
            #print(idxs)
            states.append(output['states'])
            
            valid_idx = []
            try:
                curr_evidence_len = evidence_len[i].data[0]
            except IndexError:
                curr_evidence_len = evidence_len[i].item()
            for idx in output['idxs'][:min(max_select, curr_evidence_len)]:
                try:
                    curr_idx = idx.view(-1).data[0]
                except IndexError:
                    curr_idx = idx.view(-1).item()

                if curr_idx == num_evidence:
                    break
                valid_idx.append(curr_idx)
                
                if valid_idx[-1] >= curr_evidence_len:
                    valid_idx[-1] = 0

            #TODO: if it selects none, use the first one?
                
            selected_evidence_lengths.append(len(valid_idx))
            #print(selected_evidence_lengths[-1])
            indices.append(valid_idx)
            if 'scores' in output:
                baselines.append(output['scores'][:len(valid_idx)])
            if 'probs' in output:
                probs.append(output['probs'][:len(valid_idx)])

            valid_indices.append(torch.LongTensor(valid_idx + \
                                             [-1]*(max_select-len(valid_idx))))

        '''
        for q in range(label.size(0)):
            if selected_evidence_lengths[q] >= 5:
                continue
            print(label[q])
            print(evidence[q])
            print(valid_indices[q])
            if len(baselines):
                print(probs[q][0].probs)            
                print(baselines[q])
        '''

        output_dict = {'predicted_sentences': torch.stack(valid_indices)}
        
        predictions = torch.autograd.Variable(torch.stack(valid_indices))

        selected_premise = {}
        index=predictions.unsqueeze(2).expand(batch_size,
                                              max_select,
                                              max_premise_length)
        #B x num_selected
        l = torch.autograd.Variable(len_mask(selected_evidence_lengths,
                                             max_len=max_select,
                                             dtype=torch.FloatTensor))
        
        index = index * l.long().unsqueeze(-1)
        if torch.cuda.is_available() and premise_mask.is_cuda:
            idx = premise_mask.get_device()
            index = index.cuda(idx)                
            l = l.cuda(idx)
            predictions = predictions.cuda(idx)
            
        if self._use_decoder_states:
            states = torch.cat(states, dim=0)
            label_sequence = make_label_sequence(predictions, evidence, label,
                                                 pad_idx=pad_idx,
                                                 nei_label=self._nei_label)
            #print(states.shape)
            batch_size, max_length, _ = states.shape
            label_logits = self._entailment_esim(features=states.view(batch_size*max_length,1,-1))
            if 'loss' not in output_dict:
                output_dict['loss'] = 0
            output_dict['loss'] += sequence_loss(label_logits.view(batch_size,
                                                                   max_length, -1),
                                                 label_sequence, self._evidence_loss,
                                                 pad_idx=pad_idx)
            output_dict['label_sequence_logits'] = label_logits.view(batch_size,max_length,-1)
            label_logits = output_dict['label_sequence_logits'][:,-1,:]
        else:
            for key in premise:
                selected_premise[key] = torch.gather(premise[key], dim=1, index=index)

            selected_mask = torch.gather(premise_mask, dim=1,
                                         index=index)

            selected_mask = selected_mask * l.unsqueeze(-1)
            
            selected_features = None
            if features is not None:
                index=predictions.unsqueeze(2).expand(batch_size,
                                                      max_select,
                                                      features.size(-1))
                index = index * l.long().unsqueeze(-1)
                selected_features = torch.gather(features, dim=1,
                                                 index=index)
            
            label_logits = self._entailment_esim(selected_premise, hypothesis,
                                                premise_mask=selected_mask,
                                                features=selected_features)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        #print(label_probs[0])
        
        '''
        key = 'tokens'
        for q in range(premise[key].size(0)):
            print(index[q,:,0])
            print([int(i.data[0]) for i in hypothesis[key][q]])
            print([self.vocab._index_to_token[key][i.data[0]] for i in hypothesis[key][q]])
            print([int(i.data[0]) for i in premise[key][q,0]])
            print([self.vocab._index_to_token[key][i.data[0]] for i in premise[key][q,0]])
            print([self.vocab._index_to_token[key][i.data[0]] for i in premise[key][q,index[q,0,0].data[0]]])            
            print([self.vocab._index_to_token[key][i.data[0]] for i in selected_premise[key][q,0]])
        
            print([int(i.data[0]) for i in premise_mask[q,0]])
            print(l[q])
            print([int(i.data[0]) for i in premise_mask[q,index[q,0,0].data[0]]])            
            for z in range(5):
                print([int(i.data[0]) for i in selected_mask[q,z]])

            print(label[q], label_probs[q])
        '''
        
        output_dict.update({"label_logits": label_logits,
                            "label_probs": label_probs})
        
        #get fever score, recall, and accuracy

        if len(label.shape) > 1:
            self._accuracy(label_logits, label.squeeze(-1))
        else:
            self._accuracy(label_logits, label)

        fever_reward = self._fever(label_logits, label.squeeze(-1), predictions, evidence,
                                   indices=True, pad_idx=pad_idx)

        if not self._fix_sentence_extraction_params:
            #multiply the reward for the support/refute labels by a constant so that the model selects the correct evidence instead of just trying to predict the not enough info labels
            fever_reward = fever_reward * label.squeeze(-1).ne(self._nei_label) * self._ei_reward_weight + fever_reward * label.squeeze(-1).eq(self._nei_label)

            #compute discounted reward
            rewards = []
            #print(fever_reward[0])
            avg_reward = 0
            for i in range(evidence.size(0)):
                avg_reward += float(fever_reward[i])
                #rewards.append(gamma ** torch.range(selected_evidence_lengths[i]-1,0,-1) * float(fever_reward[i]))
                rewards.append(gamma ** torch.arange(selected_evidence_lengths[i]).float() * fever_reward[i].float())
            #print(fever_reward[0])
            #print(rewards[0])

            reward = torch.autograd.Variable(torch.cat(rewards), requires_grad=False)
            if torch.cuda.is_available() and fever_reward.is_cuda:
                idx = fever_reward.get_device()
                reward = reward.cuda(idx)                        

            #print(reward)
            if len(baselines):
                indices = list(itertools.chain(*indices))
                probs = list(itertools.chain(*probs))
                baselines = list(itertools.chain(*baselines))

                #print(baselines)

                # standardize rewards
                reward = (reward - reward.mean()) / (
                    reward.std() + float(np.finfo(np.float32).eps))

                #print(reward)
                baseline = torch.cat(baselines).squeeze()
                avg_advantage = 0
                losses = []
                for action, p, r, b in zip(indices, probs, reward, baseline):
                    #print(action, p, r, b)
                    action = torch.autograd.Variable(torch.LongTensor([action]))
                    if torch.cuda.is_available() and r.is_cuda:
                        idx = r.get_device()
                        action = action.cuda(idx)

                    advantage = r - b
                    #print(r, b, advantage)
                    avg_advantage += advantage
                    losses.append(-p.log_prob(action)
                                  * (advantage/len(indices))) # divide by T*B
                    #print(losses[-1])

                critic_loss = F.mse_loss(baseline, reward)

                output_dict['loss'] = critic_loss + sum(losses)

                #output_dict['loss'].backward(retain_graph=True)
                #grad_log = self.grad_fn()
                #print(grad_log)

                try:
                    output_dict['advantage'] = avg_advantage.data[0]/len(indices)
                    output_dict['mse'] = critic_loss.data[0]
                except IndexError:
                    output_dict['advantage'] = avg_advantage.item()/len(indices)
                    output_dict['mse'] = critic_loss.item()
                    
            #output_dict['reward'] = avg_reward / evidence.size(0)

        if self.training and self._train_gold_evidence:
            
            if 'loss' not in output_dict:
                output_dict['loss'] = 0
            if evidence.sum() != -1*torch.numel(evidence):                
                if len(evidence.shape) > 2:
                    evidence = evidence.squeeze(-1)
                #print(evidence_len.long().data.cpu().numpy().tolist())
                #print(evidence.shape, evidence_len.shape)
                #print(evidence, evidence_len)
                output = self._ptr_extract_summ(aggregated_input,
                                                   None,
                                                   None,
                                                   evidence,
                                                   evidence_len.long().data.cpu().numpy().tolist())
                #print(output['states'].shape)

                loss = sequence_loss(output['scores'][:,:-1,:],
                                     evidence, self._evidence_loss, pad_idx=pad_idx)
                
                output_dict['loss'] += self.lambda_weight * loss
                
        if not self._fix_entailment_params:
            if self._use_decoder_states:
                if self.training:
                    label_sequence = make_label_sequence(evidence, evidence, label,
                                                         pad_idx=pad_idx, nei_label=self._nei_label)
                    batch_size, max_length, _ = output['states'].shape
                    label_logits = self._entailment_esim(features=output['states'][:,1:,:].contiguous().view(batch_size*(max_length-1),1,-1))
                    if 'loss' not in output_dict:
                        output_dict['loss'] = 0
                    #print(label_logits.shape, label_sequence.shape)
                    output_dict['loss'] += sequence_loss(label_logits.view(batch_size,
                                                                           max_length-1, -1),
                                                         label_sequence, self._evidence_loss,
                                                         pad_idx=pad_idx)
            else:
                #TODO: only update classifier if we have correct evidence            
                evidence_reward = self._fever_evidence_only(label_logits, label.squeeze(-1),
                                                            predictions, evidence,
                                                            indices=True, pad_idx=pad_idx)
                ###print(evidence_reward)
                ###print(label)            
                #mask = evidence_reward > 0
                #target = mask * label.byte() + mask.eq(0) * self._nei_label

                mask = fever_reward != 2**7
                target = label.view(-1).masked_select(mask)

                ###print(target)

                mask = fever_reward != 2**7
                logit = label_logits.masked_select(
                    mask.unsqueeze(1).expand_as(label_logits)
                ).contiguous().view(-1, label_logits.size(-1))

                loss = self._loss(logit, target.long()) #label_logits, label.long().view(-1))
                if 'loss' in output_dict:
                    output_dict["loss"] += self.lambda_weight * loss
                else:
                    output_dict["loss"] = self.lambda_weight * loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        fever, recall = self._fever.get_metric(reset)
        return {
            'accuracy': self._accuracy.get_metric(reset),
            'evidence_recall': recall,
            'FEVER': fever
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ESIMPtrExtractor':

        entailment_params = params.pop("entailment_esim")
        fix_entailment_params = params.pop('fix_entailment_params', False)
        if 'archive_file' in entailment_params:
            model = load_archive(entailment_params.pop('archive_file')).model
            if model._combine_feedforward is not None:
                model._entailment_esim._combine_feedforward = model._combine_feedforward
            if model._aggregate_feedforward is not None:
                model._entailment_esim._aggregate_feedforward = model._aggregate_feedforward
            entailment_esim = model._entailment_esim

            fix_entailment_params = entailment_params.pop('fix_entailment_params', True)
            if fix_entailment_params:
                for parameter in entailment_esim.parameters():
                    parameter.requires_grad = False
        elif entailment_params.pop('model', None) == 'feature_model':
            weights_file = entailment_params.pop('weights_file', None)
            entailment_esim = FeatureModel(**entailment_params)
            if weights_file is not None:
                entailment_esim.load_state_dict(torch.load(weights_file))
        else:
            entailment_esim = ESIM.from_params(vocab, entailment_params)
                    
        sentence_selection_params = params.pop("sentence_esim")
        pretrained_ptr_extractor = None
        fix_sentence_selection_esim_params = False
        if 'archive_file' in sentence_selection_params:
            archive_file = sentence_selection_params.pop('archive_file')
            pretrained_ptr_extractor = load_archive(archive_file).model
            sentence_selection_esim = pretrained_ptr_extractor._entailment_esim

            fix_sentence_selection_esim_params = sentence_selection_params.pop('fix_sentence_selection_esim_params', False)
            if fix_sentence_selection_esim_params:
                for parameter in sentence_selection_esim.parameters():
                    parameter.requires_grad = False
        elif sentence_selection_params.pop('model', None) == 'feature_model':            
            sentence_selection_esim = FeatureModel(**sentence_selection_params)
        else:
            sentence_selection_esim = ESIM.from_params(vocab, sentence_selection_params,
                                                       vocab_weight=entailment_esim._text_field_embedder.token_embedder_tokens.weight.data)

        ptr_extract_summ_params = params.pop('ptr_extract_summ')
        fix_ptr_extract_summ_params = False
        if 'archive_file' in ptr_extract_summ_params:
            archive_file = ptr_extract_summ_params.pop('archive_file')            
            if pretrained_ptr_extractor is None:
                pretrained_ptr_extractor = load_archive(archive_file).model
            ptr_extract_summ_params['pretrained'] = pretrained_ptr_extractor._ptr_extract_summ

            fix_ptr_extract_summ_params = ptr_extract_summ_params.pop('fix_ptr_extract_summ_params',
                                                                      False)
            if fix_ptr_extract_summ_params:
                for parameter in ptr_extract_summ_params['pretrained'].parameters():
                    parameter.requires_grad = False
                    
        ptr_extract_summ = ActorCritic(**ptr_extract_summ_params)
            
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        ei_reward_weight = params.pop("ei_reward_weight", 1)
        nei_label = params.pop("nei_label", 0)
        train_gold_evidence = params.pop("train_gold_evidence", False)
        use_decoder_states = params.pop("use_decoder_states", False)
        beam_size = params.pop("beam_size", 5)
        
        fix_sentence_extraction_params = params.pop("fix_sentence_extraction_params", False)
        
        params.assert_empty(cls.__name__)
        
        return cls(vocab=vocab,
                   sentence_selection_esim=sentence_selection_esim,
                   entailment_esim=entailment_esim,                   
                   ptr_extract_summ=ptr_extract_summ,
                   initializer=initializer,
                   regularizer=regularizer,
                   ei_reward_weight=ei_reward_weight,
                   fix_entailment_params=fix_entailment_params,
                   fix_sentence_extraction_params=fix_sentence_extraction_params or fix_ptr_extract_summ_params and fix_sentence_selection_esim_params,
                   nei_label=nei_label,
                   train_gold_evidence=train_gold_evidence,
                   use_decoder_states=use_decoder_states,
                   beam_size=beam_size)

