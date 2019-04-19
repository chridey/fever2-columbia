import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

import math

def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score


#################### general sequence helper #########################
def len_mask(lens, max_len=None, dtype=torch.ByteTensor):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    
    if max_len is None:
        max_len = max(lens)
    batch_size = len(lens)
    mask = dtype(batch_size, int(max_len))
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

def sequence_max(sequence, seq_lens):
    dim=1
    if type(seq_lens) == list:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        mx = torch.stack([s[:int(l)].max(dim=0)[0] if l else torch.zeros_like(s[0]) for s, l in zip(sequence, seq_lens)], dim=0)
    elif seq_lens is not None:
        #this is a mask of batch_size * len
        seq_lens = seq_lens.eq(0).float() * -1e8
        mx = torch.max(sequence + seq_lens.unsqueeze(-1), dim=dim)[0]
    else:
        mx = torch.max(sequence, dim=dim)[0]
    return mx

def sequence_mean(sequence, seq_lens, dim=1):
    if type(seq_lens) == list:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim)
        mean = torch.stack([s/l if l else torch.zeros_like(s) for s, l in zip(sum_, seq_lens)], dim=0)
    elif seq_lens is not None:
        seq_lens = seq_lens.eq(0).float() + seq_lens
        sum_ = torch.sum(sequence, dim=dim)
        mean = sum_/seq_lens.unsqueeze(-1)
    else:
        mean = torch.mean(sequence, dim=dim)
    return mean

def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    #assert (not math.isnan(loss.mean().item())
    #        and not math.isinf(loss.mean().item()))
    return loss

class StackedLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, cells, dropout=0.0):
        super().__init__()
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout

    def forward(self, input_, state):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return self._cells[0].bidirectional

class MultiLayerLSTMCells(StackedLSTMCells):
    """
    This class is a one-step version of the cudnn LSTM
    , or multi-layer version of LSTMCell
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.0):
        """ same as nn.LSTM but without (bidirectional)"""
        cells = []
        cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers-1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        super().__init__(cells, dropout)

    @property
    def bidirectional(self):
        return False

    def reset_parameters(self):
        for cell in self._cells:
            # xavier initilization
            gate_size = self.hidden_size / 4
            for weight in [cell.weight_ih, cell.weight_hh]:
                for w in torch.chunk(weight, 4, dim=0):
                    init.xavier_normal_(w)
            #forget bias = 1
            for bias in [cell.bias_ih, cell.bias_hh]:
                torch.chunk(bias, 4, dim=0)[1].data.fill_(1)

    @staticmethod
    def convert(lstm):
        """ convert from a cudnn LSTM"""
        lstm_cell = MultiLayerLSTMCells(
            lstm.input_size, lstm.hidden_size,
            lstm.num_layers, dropout=lstm.dropout)
        for i, cell in enumerate(lstm_cell._cells):
            cell.weight_ih.data.copy_(getattr(lstm, 'weight_ih_l{}'.format(i)).data)
            cell.weight_hh.data.copy_(getattr(lstm, 'weight_hh_l{}'.format(i)).data)
            cell.bias_ih.data.copy_(getattr(lstm, 'bias_ih_l{}'.format(i)).data)
            cell.bias_hh.data.copy_(getattr(lstm, 'bias_hh_l{}'.format(i)).data)
        return lstm_cell

