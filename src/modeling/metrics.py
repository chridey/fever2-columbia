from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

class FeverScore(Metric):
    def __init__(self, nei_label=0, correct_evidence_only=False) -> None:
        self.correct_count = 0.
        self.total_count = 0.
        self.correct_evidence_count = 0.
        self.total_evidence_count = 0.
        self.nei_label = nei_label
        self.correct_evidence_only = correct_evidence_only
        
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 evidence_predictions: torch.Tensor,
                 evidence_labels: torch.Tensor,
                 indices=False,
                 pad_idx=-1):

        if self.correct_evidence_only:
            correct = 1
        else:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
            correct = top_k.eq(gold_labels.long().unsqueeze(-1)).view(-1)

        #print(predictions, gold_labels, top_k, correct)
        
        if indices:
            correct_evidence = []
            for i in range(evidence_predictions.size(0)):
                if int(evidence_labels[i].sum()) != pad_idx * 5:
                    #print(evidence_labels[i])
                    #print(evidence_labels[i].ne(pad_idx))
                    #print(evidence_labels[i].masked_select(evidence_labels[i].ne(pad_idx)))
                    e = evidence_labels[i].masked_select(evidence_labels[i].ne(pad_idx)).data.cpu().numpy()
                    p = evidence_predictions[i].data.cpu().numpy()
                    #print(e)
                    #print(p)
                    correct_evidence.append([set(e).issubset(p)])
                else:
                    correct_evidence.append([False])
            correct_evidence = torch.autograd.Variable(torch.FloatTensor(correct_evidence))
            if torch.cuda.is_available() and evidence_predictions.is_cuda:
                idx = evidence_predictions.get_device()
                correct_evidence = correct_evidence.cuda(idx)                
        else:
            correct_evidence = evidence_labels.float() * (evidence_predictions[:,:,1] > evidence_predictions[:,:,0]).float()
            #TODO: this is overcounting, includes the ones where evidence is 0
            correct_evidence = ((correct_evidence.sum(dim=1) >= evidence_labels.sum(dim=1).float()) & (evidence_labels.sum(dim=1) > 0)).view(-1,1)
        #print(evidence_predictions, evidence_labels, correct_evidence)
        #TODO: if more than one evidence sentence is required, this is an overestimate for non-inidices and an under for indices

        correct_evidence_count = float(((correct_evidence.sum(dim=1) > 0) & (gold_labels != self.nei_label)).sum())
        total_evidence_count = float((gold_labels != self.nei_label).sum())
        self.total_evidence_count += total_evidence_count
        self.correct_evidence_count += correct_evidence_count
        
        #print(correct_evidence_count, total_evidence_count)
        
        fever_recall = ((correct_evidence.sum(dim=1) > 0) | (gold_labels == self.nei_label)) & correct
        total_count = float(predictions.size(0))
        correct_count = float(fever_recall.sum())
        self.total_count += total_count
        self.correct_count += correct_count

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
