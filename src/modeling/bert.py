from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import logging
import argparse
import random
import json
import tempfile
import tarfile

import numpy as np
import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.file_utils import cached_path
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel, BertConfig, CONFIG_NAME, WEIGHTS_NAME
CLASSIFIER_CONFIG_NAME = "classifier_config.json"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class BertForMultipleSequenceClassification(PreTrainedBertModel):
    def __init__(self, config, model=None, num_sequences=5, num_labels=3,
                 pooling=('concat',), return_reps=True):
        super(BertForMultipleSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.num_sequences = num_sequences
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        num_inputs = len(set(pooling) - {'concat'})
        if 'concat' in pooling:
            num_inputs += num_sequences
        
        self.hidden_layer = nn.Linear(num_inputs*config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        #self.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        #self.classifier.bias.data.zero_()
        #self.apply(self.init_bert_weights)
        if model is not None:
            self.bert = model.bert
        else:
            self.bert = BertModel(config)
            
        assert(set(pooling).issubset({'concat', 'mean', 'max', 'min'})) 
        self.pooling = pooling

        self.return_reps = return_reps

    def load_weights(self, serialization_dir, cuda_device='cpu'):
        weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
        if cuda_device == "cpu":
            state_dict = torch.load(weights_path, map_location=cuda_device)
        else:
            state_dict = torch.load(weights_path)
            
        weight_groups = {i.split('.')[0] for i in state_dict}
        for group in weight_groups:
            if getattr(self, group) is not None:
                try:
                    getattr(self, group).load_state_dict(collections.OrderedDict((i.replace(group + '.', ''),
                                                                             state_dict[i]) for i in state_dict if i.startswith(group + '.')))
                except Exception:
                    print('WARNING: could not load weight group {} from {}'.format(group,
                                                                                   serialization_dir))
                getattr(self, group).to(cuda_device)                                     

        #self.model.load_state_dict(state_dict)        
        #self.model = self.model.to(cuda_device)        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        batch_size, num_sequences, max_length = input_ids.shape
        if not self.return_reps:
            assert(num_sequences == self.num_sequences)

        input_ids = input_ids.view(batch_size*num_sequences, -1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(batch_size*num_sequences, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size*num_sequences, -1)

        #TODO: try this with and without backpropagating all the way through BERT                    
        with torch.no_grad():
            encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)        
        #pooled_output = self.dropout(pooled_output)

        #get the sum of the top 4 layers
        pooled_output = torch.stack([i[:,0].unsqueeze(1) for i in encoded_layers[-4:]], dim=1).sum(dim=1)
        #print(pooled_output.shape)
        
        combined_output = []
        pooled_output = pooled_output.view(batch_size, num_sequences, -1)
        if self.return_reps:
            return pooled_output
            
        if 'concat' in self.pooling:
            combined_output.append(pooled_output.view(batch_size, -1))
        if 'mean'  in self.pooling:
            combined_output.append(pooled_output.mean(dim=1))
        if 'max' in self.pooling:
            combined_output.append(pooled_output.max(dim=1)[0])
        if 'min' in self.pooling:
            combined_output.append(pooled_output.min(dim=1)[0])
        pooled_output = torch.cat(combined_output, dim=-1)
        
        logits = self.classifier(torch.nn.functional.relu(self.hidden_layer(pooled_output)))

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits            


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_list=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_list: (Optional) list of strings
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.text_list = text_list
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class BertFeatureExtractor:
    '''
    adapted from https://github.com/huggingface/pytorch-pretrained-BERT run_classifier.py
    '''
    def __init__(self,
                 bert_model_name,
                 pretrained_file_name,
                 label_map,
                 do_lower_case=True,
                 max_seq_length=128,
                 batch_size=32,
                 cuda_device=-1,
                 reorder=False):
        
        #initialize tokenizer and models here
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)

        pretrained_file_name = cached_path(pretrained_file_name)
        tempdir = tempfile.mkdtemp()
        with tarfile.open(pretrained_file_name, 'r:gz') as archive:
            archive.extractall(tempdir)
        serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        if cuda_device == -1:
            cuda_device = "cpu"
        self.cuda_device = cuda_device

        with open(os.path.join(serialization_dir, CLASSIFIER_CONFIG_NAME)) as f:
            classifier_config = json.load(f)
        # Instantiate model.
        self.model = BertForMultipleSequenceClassification(config, **classifier_config)
        self.model.load_weights(serialization_dir, cuda_device)
        #self.model = BertForMultipleSequenceClassification.from_pretrained(pretrained_file_name)
        
        self.label_map = label_map
        self.max_seq_length = max_seq_length

        self.batch_size = batch_size
        self.verbose = False
        self.reorder = reorder
        
    def convert_examples_to_features(self, examples, single_sentence=False):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)

            tokens_list = None
            if example.text_list:
                tokens_list = [self.tokenizer.tokenize(i) for i in example.text_list]

            if tokens_list:
                if not single_sentence:
                    tokens_a_list = []
                    tokens_b_list = []
                    for tokens_b in tokens_list:
                        tokens_a_list.append(list(tokens_a))
                        _truncate_seq_pair(tokens_a_list[-1], tokens_b, self.max_seq_length - 3)
                        tokens_b_list.append(tokens_b)
                else:
                    '''
                    tokens_a_list = []
                    for tokens_b in [tokens_a] + tokens_list:
                        if len(tokens_b) > max_seq_length - 2:
                            tokens_b = tokens_b[0:(max_seq_length - 2)]
                        tokens_a_list.append(tokens_b)

                    tokens_b_list = [None] * (len(tokens_list)+1)
                    '''

                    #a,None, b,a, a,c
                    tokens_a_list = [list(tokens_a)]
                    tokens_b_list = [None]

                    ba = list(tokens_a)
                    _truncate_seq_pair(tokens_list[0], ba, self.max_seq_length - 3)
                    tokens_a_list.append(tokens_list[0])
                    tokens_b_list.append(ba)

                    ac = list(tokens_a)
                    _truncate_seq_pair(ac, tokens_list[1], self.max_seq_length - 3)
                    tokens_a_list.append(ac)
                    tokens_b_list.append(tokens_list[1])

            elif tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            if not tokens_list:
                tokens_a_list = [tokens_a]
                tokens_b_list = [tokens_b]

            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []

            for tokens_a, tokens_b in zip(tokens_a_list, tokens_b_list):

                if self.reorder:
                    tokens_a, tokens_b = tokens_b, tokens_a
                    
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                if tokens_a:
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)

                if tokens_b:
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)

                if example.label is None:
                    label_id = 0
                else:
                    label_id = self.label_map[example.label]
                if ex_index < 5 and self.verbose:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    logger.info("tokens: %s" % " ".join(
                            [str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    logger.info("label: %s (id = %d)" % (example.label, label_id))

            if not tokens_list:
                input_ids = input_ids_list[0]
                input_mask = input_mask_list[0]
                segment_ids = segment_ids_list[0]
            else:
                input_ids = input_ids_list
                input_mask = input_mask_list
                segment_ids = segment_ids_list

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))


        return features

    def convert_examples_to_reps(self, examples):
        features = self.convert_examples_to_features(examples)
        for reps in self.convert_features_to_reps(features):
            yield reps
    
    def convert_features_to_reps(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        self.model.eval()        
        for batch_num,(input_ids, input_mask, segment_ids, label_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.cuda_device)
            input_mask = input_mask.to(self.cuda_device)
            segment_ids = segment_ids.to(self.cuda_device)
            label_ids = label_ids.to(self.cuda_device)

            yield self.model(input_ids, segment_ids, input_mask, label_ids)

    def forward_on_single(self, id, claim, sentences, label=None):
        input_example = InputExample(id, claim, label=label, text_list=sentences)
        return list(self.convert_examples_to_reps([input_example]))[0][0]

    def forward(self, examples):
        input_examples = []
        for example in examples:
            input_examples.append(InputExample(*example))
            
        ret = []
        for reps in self.convert_examples_to_reps(input_examples):
            ret.append(reps)
        return torch.cat(ret, dim=0)
                    
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
        
