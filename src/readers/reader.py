import os
import functools
import json
import collections
import logging

from typing import List, Union, Dict, Any, Iterable
from overrides import overrides

import numpy as np
import spacy
from spacy.lang.en import English

import tqdm

try:
    from allennlp.data.dataset import Batch as Dataset
except ImportError:
    from allennlp.data import Dataset

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField, IndexField, MetadataField, ArrayField

from allennlp.data.instance import Instance

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from .batched_db import BatchedDB, normalize, process_markup
from .base_reader import BaseReader
from .scorers import SimpleSentenceRanker
from .my_text_field import MyTextField
from modeling.bert import BertFeatureExtractor
    
@DatasetReader.register("fever")
class FEVERReader(BaseReader):
    def __init__(self,
                 db: str,
                 sentence_level = False,                 
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 include_evidence = False,
                 evidence_indices = False,
                 list_field = False,
                 split_evidence_groups = False,
                 include_features = False,
                 include_metadata = False,
                 label_lookup = None,
                 choose_min_evidence=False,
                 lazy: bool = True,
                 batch_size: int = 100,
                 bert_extractor_settings=None,
                 evidence_memory_size=50,
                 max_selected_evidence=5,
                 sentence_ranker_settings=None,
                 prepend_title=True,
                 bert_batch_mode=False,
                 cached_features_size=0,
                 titles_only=False,
                 cuda_device=-1) -> None:

        assert(cached_features_size == 0 or cached_features_size % batch_size == 0)
        
        super().__init__(lazy)

        self._sentence_level = sentence_level
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.include_evidence = include_evidence
        self.evidence_indices = evidence_indices        
        self.list_field = list_field
        self.split_evidence_groups = split_evidence_groups

        self.include_features = include_features
        self.include_metadata = include_metadata        
        
        self.label_lookup = label_lookup
        if label_lookup is None:
            self.label_lookup = {'NOT ENOUGH INFO': 0,
                                 'REFUTES': 1,
                                 'SUPPORTS': 2}

        self._choose_min_evidence = choose_min_evidence
        self.db = BatchedDB(db)

        self.sentence_ranker = None
        if sentence_ranker_settings is not None:
            nlp = spacy.load('en')
            self.tokenizer = English().Defaults.create_tokenizer(nlp)
            self.sentence_ranker = SimpleSentenceRanker(**sentence_ranker_settings)
        
        self.bert_feature_extractor = None
        self.bert_batch_mode = False
        if bert_extractor_settings is not None:
            bert_extractor_settings['cuda_device'] = cuda_device
            self.bert_feature_extractor = BertFeatureExtractor(**bert_extractor_settings,
                                                               label_map=self.label_lookup)
            self.bert_batch_mode = bert_batch_mode
            
        self.batch_size = batch_size
        self.evidence_memory_size = evidence_memory_size
        self.max_selected_evidence = max_selected_evidence
        self._prepend_title = prepend_title

        self._read = None
        self._features_cache = collections.defaultdict(dict)
        self._cached_features_size = cached_features_size
        
        self._titles_only = titles_only
        
    def prepend_title(self, doc, line):
        if not self._prepend_title:
            return line
        return ' '.join(doc.split('_')) + ' : ' + line

        
    def get_doc_line(self,doc,line):
        if self._titles_only:
            return normalize(process_markup(doc.split('_')))
                
        if self.db is None:
            return '{}_{}'.format(doc,line) #TODO
        
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            try:
                if lines is not None:
                    return self.prepend_title(doc, lines.split("\n")[line].split("\t")[1])
                doc = doc.replace(' ', '_').replace('__', '_')
                lines = self.db.get_doc_lines(doc)
                return self.prepend_title(doc, lines.split("\n")[line].split("\t")[1])
            except IndexError:
                print('problem with ', doc, line)
            except AttributeError:
                print('problem with ', doc, line)
                return ''
            
        non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
        return self.prepend_title(doc, non_empty_lines[np.random.choice(len(non_empty_lines), 1)])


    def get_doc_line_for_date_claim(self,doc,line):
        if self.db is None:
            return '{}_{}'.format(doc,line) #TODO
        
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            try:
                if lines is not None:
                    return lines.split("\n")[line].split("\t")[1]
                doc = doc.replace(' ', '_').replace('__', '_')
                lines = self.db.get_doc_lines(doc)
                return lines.split("\n")[line].split("\t")[1]
            except IndexError:
                print('problem with ', doc, line)
            except AttributeError:
                print('problem with ', doc, line)
                return ''
            
        non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
        return non_empty_lines[np.random.choice(len(non_empty_lines), 1)]


    def get_top_sentences_from_pages(self, instance):
        claim = list(map(str, self.tokenizer(instance['claim'])))
        sentences = []
        sentence_metadata = []
        for page in instance['predicted_pages']:
            lines = self.db.get_doc_lines(page[0])
            if lines is None:
                continue
            for index, line in enumerate(lines.split("\n")):
                try:
                    line = line.split("\t")[1]
                except IndexError:
                    continue
                sentence_metadata.append([page[0], index])
                sentences.append(line.split(' '))
        indices = self.sentence_ranker.rank(claim, sentences)
        instance['evidence'] = [sentence_metadata[i] for i in indices[:self.evidence_memory_size]]
        return instance

    def read(self, file_path: str, data=None,
             replace_with_gold=False, pad_with_nearest=0,
             include_metadata=False, dedupe=False, start=0) -> Iterable[Instance]:

        def iter_data(file_path):
            for i in self._iter_data(file_path, data, replace_with_gold, pad_with_nearest,
                                     include_metadata, dedupe, start):
                yield i
        
        self._read = iter_data
        
        return super().read(file_path)
                
    def _iter_data(self, file_path: str, data=None,
              replace_with_gold=False, pad_with_nearest=0,
             include_metadata=False, dedupe=False, start=0) -> Iterable[Instance]:
     
        counter = collections.Counter()
        
        duplicate_claims = set()
        if dedupe:
            claim_count = collections.Counter(i['claim'] for i in self.iter_data(file_path,data))
            duplicate_claims = {i for i in claim_count if claim_count[i] > 1}
            print('found {} duplicate claims'.format(len(duplicate_claims)))

        if self.include_features:
            if os.path.exists(file_path + '.npy'):
                print('loading features...')
                features = np.load(file_path + '.npy')
                indices = range(features.shape[0])
                self._features_cache[file_path] = dict(zip(indices, features))

        examples = []
        batch_evidence = []
        for line_index,instance in enumerate(self.iter_data(file_path,data)):
            if instance is None or instance['claim'] in duplicate_claims or line_index < start:
                self._features_cache[line_index] = []
                continue

            if 'gold_evidence' not in instance and 'evidence' in instance:
                instance['gold_evidence'] = instance['evidence']
            if self._titles_only:
                instance['gold_evidence'] = [{(None,None,j[-2],0) for j in i} for i in instance['gold_evidence']]

            gold = [{(e[-2],e[-1]) for e in evidence_set} for evidence_set in instance['gold_evidence']]
                
            if 'predicted_pages' in instance:
                if self._titles_only:
                    instance['evidence'] = [[i[0],0] for i in instance['predicted_pages']]
                elif self.sentence_ranker is not None:
                    instance = self.get_top_sentences_from_pages(instance)
                else:
                    instance['evidence'] = instance['predicted_pages']

            if 'gold_evidence' in instance and instance['gold_evidence'] is not None and len(instance['gold_evidence']):
                if self._choose_min_evidence:
                    choice = min(zip(range(len(instance['gold_evidence'])),
                                    instance['gold_evidence']),
                                key=lambda x:len(x[1]))[0]
                    instance['gold_evidence'] = [(ev[-2],ev[-1]) for ev in instance['gold_evidence'][choice]]
                else:
                    gold_evidence = []
                    for evidence_set in instance['gold_evidence']:
                        gold_evidence.extend([(ev[-2],ev[-1]) for ev in evidence_set])
                    instance['gold_evidence'] = gold_evidence
                    
            instance_evidence_list = instance['evidence']

            if not self.split_evidence_groups:
                instance_evidence_list = [instance_evidence_list]

            for instance_evidence in instance_evidence_list:
                instance_evidence = instance_evidence[:self.evidence_memory_size]
                #print(instance_evidence)
                
                evidence = None
                if self.include_evidence:
                    if 'gold_evidence' not in instance or instance['gold_evidence'] is None:
                        #print('assuming evidence is gold evidence')
                        gold_evidence = set((i[0], i[1]) for i in instance_evidence)
                    else:
                        gold_evidence = set((i[0], i[1]) for i in instance['gold_evidence'])
                else:
                    gold_evidence = set()

                if not self._sentence_level:
                    pages = set(ev[0] for ev in instance_evidence)
                    premise = " ".join([self.db.get_doc_text(p) for p in pages])
                else:
                    if self.include_evidence:
                        evidence = instance_evidence[:self.evidence_memory_size]
                        if replace_with_gold and instance["label_text"] != 'NOT ENOUGH INFO':
                            #randomly replace some of the extracted evidence with the gold evidence
                            gold_indices = {i for i in range(len(evidence)) if tuple(evidence[i][:2]) in gold_evidence}
                            while len(gold_indices) < len(gold_evidence) and len(gold_indices) < len(evidence):
                                index = random.randint(0, len(evidence)-1)
                                if index in gold_indices:
                                    continue
                                gold_indices.add(index)
                            for index, gold in zip(gold_indices, gold_evidence):
                                evidence[index] = gold

                        lines = [(self.get_doc_line(d[0],d[1]),
                                  ((d[0], d[1]) in gold_evidence))  for d in evidence]

                        if not len(lines):
                            lines = [('',False)]
                            instance_evidence = [None]
                            
                        evidence_map = {i[0]:j for i,j in zip(lines, instance_evidence)}
                        #print(evidence_map)
                        if not self.include_features:
                            lines = set(lines)
                        lines, evidence = zip(*list(lines))

                        evidence = list(evidence)
                        evidence_count = sum(evidence)
                        if evidence_count < pad_with_nearest:
                            i = 0
                            while evidence_count < pad_with_nearest and i < len(evidence):
                                if not evidence[i]:
                                    evidence[i] = True                               
                                    evidence_count += 1
                                i += 1

                        if self.evidence_indices:
                            evidence = [i for i,j in enumerate(evidence) if int(j) != 0]
                            if len(evidence) < self.max_selected_evidence:
                                evidence += [-1]*(self.max_selected_evidence-len(evidence))
                        else:
                            evidence = list(map(str, evidence))
                        #print(evidence)
                    else:
                        lines = [self.get_doc_line(d[0],d[1]) for d in instance_evidence]
                        evidence_map = dict(zip(lines, instance_evidence))
                        if not self.include_features:
                            lines = set(lines)

                    premise = lines 

                batch_evidence.append((evidence, evidence_map, gold, line_index))
                    
                if evidence is not None:
                    counter.update(evidence)

                hypothesis = instance["claim"]
                label = "NOT ENOUGH INFO"
                if "label" in instance and instance['label'] is not None:
                    label = instance["label"].upper()

                def cache_filename(file_path, line_index):
                    #get the nearest multiple of cache size
                    base_line_index = line_index // (self._cached_features_size if self._cached_features_size else 1) * self._cached_features_size
                    return '.'.join([file_path, str(base_line_index), 'npy'])
                                        
                #read in filename + line_index if it exists
                if line_index not in self._features_cache[file_path] and os.path.exists(cache_filename(file_path, line_index)):
                    print('loading features from {}...'.format(line_index))
                    features = np.load(cache_filename(file_path, line_index))
                    indices = range(line_index, line_index+features.shape[0])
                    self._features_cache[file_path] = dict(zip(indices, features))

                if self.include_features and not self.bert_batch_mode and line_index not in self._features_cache[file_path]:
                    instance_features = self.bert_feature_extractor.forward_on_single(instance['id'],
                                                                                      hypothesis,
                                                                                      premise,
                                                                                      label).cpu().numpy().tolist()
                    self._features_cache[file_path][line_index] = instance_features
                elif self.bert_batch_mode and len(premise) < self.evidence_memory_size:
                    premise = list(premise) + ['']*(self.evidence_memory_size-len(premise))
                
                examples.append([instance['id'], hypothesis, None, label, premise])

                if len(examples) >= self.batch_size:
                    batch_features = None
                    if self.include_features and batch_evidence[0][-1] not in self._features_cache[file_path]:
                        batch_features = self.bert_feature_extractor.forward(examples).cpu().numpy().tolist()
                        indices = range(line_index-len(examples)+1, line_index+1)
                        self._features_cache[file_path] = dict(zip(indices, batch_features))

                    for idx,(_,hypothesis,_,label,premise) in enumerate(examples):
                        evidence, evidence_map, gold_evidence, line_index = batch_evidence[idx]
                        
                        evidence_metadata = None
                        if include_metadata:
                            evidence_metadata = {'evidence': [evidence_map[i] for i in premise if len(i)],
                                                 'gold': gold_evidence}

                        instance_features = None
                        if self.include_features:
                            instance_features = self._features_cache[file_path][line_index]
                                
                        yield self.text_to_instance(premise, hypothesis, label, evidence,
                                                instance_features, evidence_metadata)

                    if self._cached_features_size == 0:
                        self._features_cache[file_path] = {}                                            
                    examples = []
                    batch_evidence = []

                #write cache to disk if it is full and filename + line_index does not exist
                if self._cached_features_size and len(self._features_cache[file_path]) >= self._cached_features_size and not os.path.exists(cache_filename(file_path, line_index)):
                    _,features = zip(*sorted(self._features_cache[file_path].items(),
                                             key=lambda x:x[0]))
                    print(line_index, len(self._features_cache[file_path]))
                    np.save(cache_filename(file_path, line_index),
                            np.array(features))
                #clear cache if we are at the end of the batch
                if self._cached_features_size != 0 and ((line_index + 1) % self._cached_features_size == 0):
                    self._features_cache[file_path] = {}                                            

        #handle the leftovers

        if len(examples):
            batch_features = None
            if self.include_features and batch_evidence[0][-1] not in self._features_cache[file_path]:
                batch_features = self.bert_feature_extractor.forward(examples).cpu().numpy().tolist()
                indices = range(line_index, line_index+self.batch_size)
                self._features_cache[file_path] = dict(zip(indices, batch_features))

            for idx,(_,hypothesis,_,label,premise) in enumerate(examples):
                evidence, evidence_map, gold_evidence, line_index = batch_evidence[idx]

                evidence_metadata = None
                if include_metadata:
                    evidence_metadata = {'evidence': [evidence_map[i] for i in premise if len(i)],
                                         'gold': gold_evidence}

                instance_features = None
                if self.include_features:
                    instance_features = self._features_cache[file_path][line_index]

                yield self.text_to_instance(premise, hypothesis, label, evidence,
                                        instance_features, evidence_metadata)

        if self._cached_features_size and len(self._features_cache[file_path]) and not os.path.exists(cache_filename(file_path, line_index)):
            _,features = zip(*sorted(self._features_cache[file_path].items(),
                                     key=lambda x:x[0]))
            print(line_index, len(self._features_cache[file_path]))
            np.save(cache_filename(file_path, line_index),
                    np.array(features))
        self._features_cache[file_path] = {}                                            
                                            
    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None,
                         evidence: Union[str,int] = None,
                         features = None,
                         evidence_metadata=None) -> Instance:
        
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['premise'] = None
        if premise is not None:            
            if self.list_field:
                if len(premise) < self.max_selected_evidence:
                    evidence_metadata['evidence'].extend([['N/A',0]]*(self.max_selected_evidence-len(premise)))
                    premise = list(premise) + ['']*(self.max_selected_evidence-len(premise))
                #if evidence_metadata is None:
                premise_tokens = [TextField(self._wiki_tokenizer.tokenize(p)[:self.evidence_memory_size],
                                                self._token_indexers) for p in premise]
                #else:
                #    premise_tokens = [MyTextField(self._wiki_tokenizer.tokenize(p)[:self.evidence_memory_size],
                #                                self._token_indexers,
                #                                m) for p,m in zip(premise,
                #                                                  evidence_metadata)]
                fields['premise'] = ListField(premise_tokens)
            else:
                fields['premise'] = TextField(self._wiki_tokenizer.tokenize(' '.join(premise)),
                                              self._token_indexers)

        if evidence_metadata is not None:
            fields['metadata'] = MetadataField(evidence_metadata)
                
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        
        if label is not None:
            fields['label'] = LabelField(self.label_lookup[label], skip_indexing=True)

        if evidence is not None:
            if self.evidence_indices:
                evidence_indices = [IndexField(e, fields['premise']) for e in evidence]
                fields['evidence'] = ListField(evidence_indices)
            else:
                fields['evidence'] = SequenceLabelField(evidence, fields['premise'], 'evidence_labels')

        if features is not None:
            if len(features) < self.max_selected_evidence:
                features += [[0] * len(features[0])] * (self.max_selected_evidence - len(features))
            fields['features'] = ArrayField(np.array(features[:self.evidence_memory_size]))
            
        return Instance(fields)
                
            
        
