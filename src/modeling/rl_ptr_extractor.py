import copy

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .ptr_extractor import LSTMPointerNet

INI = 1e-2

class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""    
    def __init__(self, input_dim, lstm_hidden, lstm_layer,
                 dropout=0.0, n_hop=1, pretrained=None):
        super().__init__()
        
        self._ptr_net = LSTMPointerNet(input_dim, lstm_hidden, lstm_layer,
                                       dropout, n_hop)
        if pretrained is not None:
            self._ptr_net.load_state_dict(pretrained)
            
        # regression layer
        self._score_linear = nn.Linear(input_dim, 1)

    def forward(self, attn_mem, n_step, memory_mask=None):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._ptr_net._attn_wm)
        hop_feat = torch.mm(attn_mem, self._ptr_net._hop_wm)
        scores = []
        lstm_in = self._ptr_net._init_i.view(1,1,-1)
        lstm_states = (self._ptr_net._init_h.unsqueeze(1), self._ptr_net._init_c.unsqueeze(1))
        
        for _ in range(n_step):
            output, lstm_states = self._ptr_net._lstm(lstm_in, lstm_states)
            query = output[:, -1, :]

            for _ in range(self._ptr_net._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._ptr_net._hop_v, self._ptr_net._hop_wq, mask=memory_mask)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._ptr_net._attn_v, self._ptr_net._attn_wq, mask=memory_mask)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output.view(1,1,-1)
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w, mask=None):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        if mask is not None:
            score += mask.eq(0).float() * -1e8
        score = F.softmax(score, dim=-1)
        output = torch.mm(score, attention)
        return output


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    """ works only on single sample in RL setting"""    
    def __init__(self, input_dim, lstm_hidden, lstm_layer,
                 dropout=0.0, n_hop=1, pretrained=None, use_stop=False):
        super().__init__()
        
        self._ext = LSTMPointerNet(input_dim, lstm_hidden, lstm_layer, dropout, n_hop)
        if pretrained is not None:
            pretrained = pretrained._extractor.state_dict()            
            self._ext.load_state_dict(pretrained)
            
        self._scr = PtrScorer(input_dim, lstm_hidden, lstm_layer, dropout, n_hop, pretrained)

        self._use_stop = use_stop
        if use_stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform(self._stop, -INI, INI)
        
    def _extract(self, attn_mem, n_step, memory_mask=None, gold_evidence=None, k=1):
        """atten_mem: Tensor of size [num_sents, input_dim]"""

        max_step = attn_mem.size(0)
        if self._use_stop:
            attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
            o = torch.autograd.Variable(torch.ones(1).byte())
            if torch.cuda.is_available() and memory_mask.is_cuda:
                idx = memory_mask.get_device()
                o = o.cuda(idx)
            memory_mask = torch.cat([memory_mask, o],
                                    dim=0)
            
        attn_feat = torch.mm(attn_mem, self._ext._attn_wm)
        hop_feat = torch.mm(attn_mem, self._ext._hop_wm)
        
        outputs = []
        dists = []
        states = []
        
        lstm_in = self._ext._init_i.view(1,1,-1)
        lstm_states = (self._ext._init_h.unsqueeze(1), self._ext._init_c.unsqueeze(1))
        
        for step in range(n_step):
            output, lstm_states = self._ext._lstm(lstm_in, lstm_states)
            query = output[:, -1, :]
            for hop in range(self._ext._n_hop):
                query = ActorCritic.attention(hop_feat, query,
                                              self._ext._hop_v, self._ext._hop_wq, mask=memory_mask)
            states.append(query)
            
            score = ActorCritic.attention_score(
                attn_feat, query, self._ext._attn_v, self._ext._attn_wq, mask=memory_mask)
            #print(score.shape)

            if step == 0 and self._use_stop:
                fill = torch.autograd.Variable(torch.LongTensor([max_step]))
                if torch.cuda.is_available() and score.is_cuda:
                    idx = score.get_device()
                    fill = fill.cuda(idx)
                score.index_fill_(1, fill, -1e18)
                
            if self.training:
                prob = F.softmax(score, dim=-1)
                #print(prob)
                m = torch.distributions.Categorical(prob)
                if gold_evidence is None:
                    out = m.sample()
                else:
                    out = gold_evidence[step]
                    if out.data[0] < 0:
                        break
                                    
                dists.append(m)
            else:
                #print(score.shape)
                if len(outputs):
                    score.index_fill_(1, torch.cat(outputs).view(-1), -1e18)
                out = score.max(dim=1, keepdim=True)[1]
            #print(out, out.shape)
            outputs.append(out)

            #print(out.view(-1).data[0])
            if out.view(-1).data[0] == max_step:
                break
            
            lstm_in = attn_mem[out.data[0]].view(1,1,-1)

        return dict(idxs=outputs, probs=dists, states=torch.cat(states, dim=0).unsqueeze(0))

    @staticmethod
    def attention_score(attention, query, v, w, mask=None):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        
        if mask is None:
            return score
        
        return score + mask.eq(0).float() * -1e8
    
    @staticmethod
    def attention(attention, query, v, w, mask=None):
        """ attention context vector"""
        score = F.softmax(
            ActorCritic.attention_score(attention, query, v, w, mask=mask), dim=-1)
        output = torch.mm(score, attention)
        return output
        
    def forward(self, enc_out, n_abs, memory_mask=None, gold_evidence=None, evidence_len=None,
                beam_size = 5):

        if not self.training and beam_size >= 1:
            enc_out = enc_out.unsqueeze(0)
            memory_mask = memory_mask.unsqueeze(0)
            outputs = self._ext.extract(enc_out, None, n_abs, mask=memory_mask,
                                        beam_size=beam_size)
            outputs['idxs'] = outputs['idxs'].view(-1)
            #print(outputs)
        elif self.training and evidence_len is not None and gold_evidence is not None:
            bs, nt = gold_evidence.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=gold_evidence.clamp(min=0).unsqueeze(2).expand(bs, nt, d)
            )
            return self._ext(enc_out, evidence_len, ptr_in)
        elif n_abs is None:            
            outputs = self._extract(enc_out, memory_mask=memory_mask, gold_evidence=gold_evidence)
        else:
            outputs = self._extract(enc_out, n_abs, memory_mask=memory_mask,
                                gold_evidence=gold_evidence)
            
        if self.training:
            outputs['scores'] = self._scr(enc_out, n_abs, memory_mask=memory_mask)
        return outputs
