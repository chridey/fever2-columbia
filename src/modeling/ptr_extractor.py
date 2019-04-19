import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .utils import *

INI = 1e-2

class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        #print('HOPS', n_hop)
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform(self._init_h, -INI, INI)
        init.uniform(self._init_c, -INI, INI)
        init.uniform(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal(self._attn_wm)
        init.xavier_normal(self._attn_wq)
        init.uniform(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal(self._hop_wm)
        init.xavier_normal(self._hop_wq)
        init.uniform(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)

        #print(init_i.shape)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        #print(lstm_in.shape)
        
        query, final_states = self._lstm(lstm_in, lstm_states)
        
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k, mask=None):

        beam_size = 5
        if attn_mem.size(0) > 1:
            beam_size = 1
        elif mem_sizes is not None and mem_sizes[0] < beam_size:
            beam_size = mem_sizes[0]
        elif mask is not None and int(mask[0].sum()) < beam_size:
            beam_size = int(mask[0].sum())
        
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)

        lstm_in = lstm_in.transpose(0,1)
        
        '''
        lstm_in = lstm_in.squeeze(1)        
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm)

            if False:
                self._lstm_cell = self._lstm_cell.cuda()
        '''

        batch_size, max_mem_size, mem_dim = attn_mem.shape
        
        extracts = []
        beam_states = []
        if beam_size > 1:
            #print(mask.shape)
            mask = mask.unsqueeze(0).expand(beam_size, batch_size, mask.size(-1)).contiguous().view(beam_size*batch_size, -1)
            lstm_in = lstm_in.expand(beam_size, batch_size, lstm_in.size(-1)).contiguous().view(1, beam_size*batch_size, -1)
            lstm_states = [i.expand(beam_size, batch_size, i.size(-1)).contiguous().view(1, beam_size*batch_size, -1) for i in lstm_states]
                                     
        for state in range(k):
            #print(lstm_in.shape)
            output, lstm_states = self._lstm(lstm_in, lstm_states)
            query = output.transpose(0, 1)
            #h, c = self._lstm_cell(lstm_in, lstm_states)
            #query = h[-1]
            #print(query.shape)
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes, mask)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            #print(score.shape)
            
            #score = score.squeeze() * (mask.eq(0).float() * 1e-6 + mask.eq(1).float())
            score = score.squeeze() + mask.eq(0).float() * -1e8
            
            #print(score.shape)
            for e in extracts:
                for i,p in enumerate(e):
                    #TODO: index fill: score.index_fill_
                    score[i,int(p)] = -1e6

            if beam_size > 1: #only works if batch_size == 1
                score = score
                if state == 0:
                    scores, ptrs = score.view(beam_size, -1)[0,:].topk(beam_size)
                elif state == k-1:
                    scores, ptrs = score.view(-1).topk(1)
                else:
                    scores, ptrs = score.view(-1).topk(beam_size)
                    
                #print(scores, ptrs)
                back_ptrs = ptrs / max_mem_size
                ext = ptrs % max_mem_size
                #print(ptrs, ext, back_ptrs)
                                     
                extracts = []
                for i in range(beam_size if state < k-1 else 1):
                    if len(beam_states):
                        extracts.append(beam_states[back_ptrs[i].view(-1).data[0]] + [ext[i].view(-1).data[0]])
                    else:
                        extracts.append([ext[i].view(-1).data[0]])
                beam_states = extracts
                #print(extracts)
                extracts = torch.autograd.Variable(torch.LongTensor(extracts).transpose(0,1))
                #print(beam_states, extracts)
            else:
                ext = score.max(dim=1)[1] #.item()
                extracts.append(ext)

            ##print([i.shape for i in c])
            #lstm_states = (h, c)

            #print(ext)
            #print(attn_mem.shape)
            #lstm_in = attn_mem[:, ext, :]

            lstm_in = torch.gather(
                attn_mem.expand(attn_mem.size(0)*beam_size, max_mem_size, -1),
                dim=1,
                index=ext.view(-1,1,1).expand(attn_mem.size(0)*beam_size,
                                              1,
                                              attn_mem.size(-1))
            ).transpose(0,1)
            #print(lstm_in.shape)
            #print(type(lstm_states), [i.shape for i in lstm_states])

        if type(extracts) == list:
            extracts = torch.stack(extracts, dim=0).transpose(0,1)
        else:
            extracts = extracts.transpose(0,1)
            
        if torch.cuda.is_available() and attn_mem.is_cuda:
            idx = attn_mem.get_device()
            extracts = extracts.cuda(idx)

        #print(extracts)
            
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes, mask=None):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None and mask is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            if mask is None:
                mask = torch.autograd.Variable(len_mask(mem_sizes))
                if torch.cuda.is_available() and score.is_cuda:
                    idx = score.get_device()
                    mask = mask.cuda(idx)
            mask = mask.unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output

class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, input_dim, 
                 lstm_hidden, lstm_layer,
                 n_hop=1, dropout=0.0):
        super().__init__()

        self._extractor = LSTMPointerNet(
            input_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

    def forward(self, enc_out, sent_nums, target):
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.clamp(min=0).unsqueeze(2).expand(bs, nt, d)
        )
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, enc_out, sent_nums=None, k=5, mask=None):
        output = self._extractor.extract(enc_out, sent_nums, k, mask)
        return output
    
