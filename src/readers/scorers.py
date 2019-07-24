import gzip
import json
import collections

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
            
class Scorer(object):
    def __init__(self):
        pass
    
    def _embed(self, tokens):
        raise NotImplementedError

    def _batch_embed(self, token_list):
        ret = []
        for tokens in token_list:
            ret.append(self._embed(tokens))
        return ret
    
    def embed(self, tokens):
        if type(tokens) == dict:
            if 'embedding' in tokens:
                if type(tokens['embedding']) == dict and self.name in tokens['embedding']:
                    return tokens['embedding'][self.name]

                return tokens['embedding']
            elif 'sentence' in tokens:
                tokens = tokens['sentence']

        return self._embed(tokens)

    def batch_embed(self, token_list):
        sentences = []
        ret_flag = False
        for tokens in token_list:
            if type(tokens) == dict:
                if 'embedding' in tokens:
                    ret_flag = True
                    if type(tokens['embedding']) == dict and self.name in tokens['embedding']:
                        tokens = tokens['embedding'][self.name]
                    else:
                        tokens = tokens['embedding']
                elif 'sentence' in tokens:
                    tokens = tokens['sentence']
            sentences.append(tokens)

        if ret_flag:
            return sentences
            
        return self._batch_embed(sentences)
    
    def score(self, *args):
        raise NotImplementedError

    def batch_score(self, token_list1, token_list2):
        scores = np.zeros((len(token_list1), len(token_list2)))
        for i1,tokens1 in enumerate(token_list1):
            for i2,tokens2 in enumerate(token_list2):
                scores[i1,i2] = self.score(tokens1, tokens2)
        return scores
        
class GloveScorer(Scorer):
    name = 'Glove'
    
    def __init__(self, filename='data/index/glove.6B.300d.txt.gz', lower=True,
                 idf=None, pooling=np.sum):

        self.filename = filename
        self.lower = lower
        self._embeddings = {}

        self.pool = pooling
        
    def _load_embeddings(self):
        print('loading {} ...'.format(self.filename))
        with gzip.open(self.filename, 'rb') as f:
            for line in f:
                token, *embedding = line.decode('utf-8').split()
                self._embeddings[token] = np.array(list(map(float, embedding)))

    @property
    def embeddings(self):
        if not len(self._embeddings):
            self._load_embeddings()
        return self._embeddings
    
    def _embed(self, tokens):

        tf = collections.Counter(x.lower() if self.lower else x for x in tokens)
        weights = tf
            
        ret = []
        total_weight = 0
        for token in tf:
            weight = weights[token]
                
            #print(token, token in self.embeddings)
            if token in self.embeddings:
                ret.append(weight * self.embeddings[token])
                total_weight += weight
                
        #print(list(self.embeddings.keys())[:10])

        if self.pool is None:
            return ret
        
        if not total_weight or not len(ret):
            #TODO
            return self.embeddings['the']

        return self.pool(ret, axis=0)/total_weight #np.mean(ret, axis=0)
                
    def score(self, tokens1, tokens2):
        sentence1 = self.embed(tokens1)
        sentence2 = self.embed(tokens2)
            
        #print(sentence1, sentence2)
        cos_sim = 1 - cosine(sentence1, sentence2)

        return cos_sim

    def batch_score(self, token_list1, token_list2):
        embedding_list1 = []
        for tokens in token_list1:
            embedding_list1.append(self.embed(tokens))
        embedding_list2 = []
        for tokens in token_list2:
            embedding_list2.append(self.embed(tokens))

        return cosine_similarity(np.array(embedding_list1), np.array(embedding_list2), True)
        
class JaccardScorer(Scorer):
    name = 'Jaccard'
    
    def __init__(self, lower=False, ngrams=(1,)):
        self.lower = lower
        self.ngrams = ngrams
        
    def _embed(self, tokens):
        if type(tokens) == set:
            return tokens

        if self.lower:
            tokens = list(map(lambda x:str(x).lower(), tokens))

        #final_tokens = tokens
        final_tokens = []
        for i in self.ngrams:
            final_tokens.extend(list(zip(*(tokens[j:] for j in range(i)))))
        #print(set(final_tokens))
        
        return set(final_tokens)
    
    def score(self, tokens1, tokens2):
        set1 = self.embed(tokens1)
        set2 = self.embed(tokens2)
        score = 1.*len(set1 & set2) / len(set1 | set2)
        return score
        
class WeightedScorer(Scorer):
    def __init__(self, scorers, weights):
        self.scorers = scorers
        self.names = [scorer.name for scorer in scorers]
        self._weights = np.array(weights)

    def _weighted_average(self, scores, weight):
        if not weight:
            return scores

        a_len = []
        b_len = []
        for score in scores.values():
            try:
                iter(score)
            except TypeError:
                a_len.append(1)
                b_len.append(1)
            else:
                a_len.append(len(list(score)))
                try:
                    iter(score[0])
                except TypeError:
                    b_len.append(1)
                else:
                    b_len.append(len(list(score[0])))

        assert(sum(a_len) // len(a_len) == a_len[0])
        assert(sum(b_len) // len(b_len) == b_len[0])        
        a_len = sum(a_len) // len(a_len)
        b_len = sum(b_len) // len(b_len)
        
        scores = np.array([np.array(scores[i]).reshape(a_len,b_len) for i in self.names]).reshape(len(self.scorers), a_len, b_len)
        
        return np.sum(scores * self._weights[:,None,None], axis=0)

    def set_weights(self, weights):
        self._weights = np.array(weights)
    
    def _embed(self, tokens):
        embeddings = {}
        for scorer in self.scorers:
            embeddings[scorer.name] = scorer.embed(tokens)
        return embeddings
    
    def score(self, tokens1, tokens2, precalculated=None, weight=False):
        if precalculated is not None:
            return self._weighted_average(precalculated, weight)
        
        scores = {}
        for scorer in self.scorers:
            scores[scorer.name] = scorer.score(tokens1, tokens2)

        return self._weighted_average(scores, weight)

    def batch_score(self, token_list1, token_list2, precalculated=None, weight=False):
        if precalculated is not None:
            return self._weighted_average(precalculated, weight)
        
        scores = {}
        for scorer in self.scorers:
            scores[scorer.name] = scorer.batch_score(token_list1, token_list2)

        return self._weighted_average(scores, weight)

    
class SimpleSentenceRanker:
    scorers = {JaccardScorer.name: JaccardScorer,
               GloveScorer.name: GloveScorer}
        
    def __init__(self,
                 scorer_names,
                 weights,
                 kwargs=None):
        assert(len(scorer_names)==len(weights))
        self.scorer = WeightedScorer([self.scorers[scorer_names[i]](**(kwargs[i] if kwargs is not None else {})) for i in range(len(scorer_names))],
                                     weights)

    def score(self,
             query,
             sentences):

        if not len(sentences):
            return np.array([])
        
        scores = self.scorer.batch_score([query], sentences, weight=True)
        return scores[0]

    def rank(self,
             query,
             sentences):
        scores = self.score(query, sentences)
        indices = scores.argsort()[::-1]
        return indices
