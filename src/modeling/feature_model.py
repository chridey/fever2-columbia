from typing import Dict, Optional

import torch

from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import FeedForward

from .esim import ESIM


class FeatureModel(ESIM):                 

    def __init__(self,
                 num_sequences,
                 num_labels=2,
                 hidden_size=None,
                 pooling=('concat',)) -> None:

        torch.nn.Module.__init__(self)

        if type(pooling) == str:
            pooling = pooling.split(',')
        
        num_inputs = len(set(pooling) - {'concat'})
        if 'concat' in pooling:
            num_inputs += num_sequences

        self._num_sequences = num_sequences
        self._num_labels = num_labels
        self._hidden_size = hidden_size

        self._hidden_layer = None
        self._classifier = None
        if hidden_size is not None:
            self._hidden_layer = torch.nn.Linear(num_inputs*hidden_size, hidden_size)
            self._classifier = torch.nn.Linear(hidden_size, num_labels)
        
        self._pooling = pooling
        
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor] = None,
                hypothesis: Dict[str, torch.LongTensor] = None,
                premise_mask=None,
                hypothesis_mask=None,
                wrap_output=False,
                features=None):

        features = features[:, :self._num_sequences]
        if self._hidden_size is None:
            return features
            
        #features is batch_size x num_evidence x dim
        batch_size, num_evidence, _ = features.shape
        
        combined_output = []
        if 'concat' in self._pooling:
            combined_output.append(features.view(batch_size, -1))
        if 'mean'  in self._pooling:
            combined_output.append(features.mean(dim=1))
        if 'max' in self._pooling:
            combined_output.append(features.max(dim=1)[0])
        if 'min' in self._pooling:
            combined_output.append(features.min(dim=1)[0])
        pooled_output = torch.cat(combined_output, dim=-1)

        return self._classifier(torch.nn.functional.relu(self._hidden_layer(pooled_output)))

    def set_linear_layers(self,
                          hidden_layer,
                          classifier):
        self._hidden_layer = hidden_layer
        self._classifier = classifier
        
