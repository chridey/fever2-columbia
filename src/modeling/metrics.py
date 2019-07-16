from typing import Optional

from overrides import overrides

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

class FeverScore(Metric):
    def __init__(self, nei_label=0, max_select=5) -> None:
        self.correct_count = 0.
        self.total_count = 0.
        self.correct_evidence_count = 0.
        self.total_evidence_count = 0.
        self.nei_label = nei_label
        self.max_select = max_select
        
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 evidence_predictions: torch.Tensor,
                 evidence: torch.Tensor,
                 indices=False,
                 metadata=None,
                 pad_idx=-1):

        top_k = predictions.max(-1)[1].unsqueeze(-1)
        correct = top_k.eq(gold_labels.long().unsqueeze(-1)).view(-1)
        evidence_predictions = evidence_predictions.data.cpu().numpy()

        total_evidence_count = 0
        correct_evidence_count = 0
        total_correct = 0
        fever_recall = []
        for idx,(is_correct, evidence_prediction) in enumerate(zip(correct,
                                                                evidence_predictions)):
            #print(predictions[idx], gold_labels[idx].item(), is_correct.item(),
            #      evidence_prediction, metadata[idx])
            if gold_labels[idx] != self.nei_label:
                total_evidence_count += 1
                #TODO: subset evidence
                evidence_metadata = {tuple(metadata[idx]['evidence'][i]) for i in evidence_prediction[:self.max_select] if i < len(metadata[idx]['evidence'])}
                found_evidence = False
                for evidence_set in metadata[idx]['gold']:
                    #print(evidence_set, evidence_set.issubset(evidence_metadata))
                    if evidence_set.issubset(evidence_metadata):
                        found_evidence = True
                        break                
                correct_evidence_count += found_evidence
                
            if is_correct and (gold_labels[idx] == self.nei_label or found_evidence):
                total_correct += 1

                fever_recall.append(1)
            else:
                fever_recall.append(0)

            #print(total_evidence_count, correct_evidence_count, total_correct)
                
        self.total_evidence_count += total_evidence_count
        self.correct_evidence_count += correct_evidence_count
        
        self.total_count += int(gold_labels.shape[0])
        self.correct_count += total_correct

        fever_recall = torch.autograd.Variable(torch.FloatTensor(fever_recall))
        if torch.cuda.is_available() and predictions.is_cuda:
            idx = predictions.get_device()            
            fever_recall = fever_recall.cuda(idx)                        
        return fever_recall
        
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = 0.0
        if float(self.total_count) > 0:
            accuracy = float(self.correct_count) / float(self.total_count)

        recall = 0.0
        if float(self.total_evidence_count) > 0:
            recall = float(self.correct_evidence_count) / float(self.total_evidence_count)
            
        if reset:
            self.reset()
        return accuracy, recall

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.correct_evidence_count = 0.
        self.total_evidence_count = 0.
