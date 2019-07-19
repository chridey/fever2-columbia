import sqlite3
import unicodedata
import functools
import itertools

import tqdm

from drqa.retriever import DocDB, utils

def normalize(page):
    return unicodedata.normalize('NFD',page)

def process_markup(s):
    s = ' '.join(s)
    s = s.replace('-LRB-', '(')
    s = s.replace('-RRB-', ')')
    s = s.replace('-COLON-', ':')
    return s

def get_nonempty_doc_lines(lines):
    if lines is None:
        return []
    return [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]

class BatchedDB(DocDB):
    '''
    enhanced fever db class with preprocessing for large queries
    '''
    def __init__(self,db, data_to_preprocess=None, batch_size=100000):
        self.db = db
        super().__init__(db)
        
        self.cache = {}
        self.batch_size = batch_size
        if data_to_preprocess is not None:
            self.preprocess(data_to_preprocess)

    def clear_cache(self):
        self.cache = {}

    @functools.lru_cache(maxsize=128)        
    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        if doc_id in self.cache:
            return self.cache[doc_id]
        
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]        

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results                                            
    
    def get_first_two_lines(self, doc_id):
        if doc_id in self.cache:
            return self.cache[doc_id]

        return get_nonempty_doc_lines(super().get_doc_lines(doc_id))[:2]
    
    def get_batch_doc_lines(self, data, keep_lines=None,
                            keys=('predicted_pages',), ignore=None):
        #first read the data and get all the relevant wikipedia articles
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        to_lookup = set()
        
        for js in tqdm.tqdm(data):
            for key in keys:
                evidence = js[key]
                if key == 'predicted_pages':
                    index = 0
                elif key == 'evidence':
                    index = 2
                    evidence = itertools.chain(*evidence)
                else:
                    raise NotImplementedError
                
                for j in evidence:
                    if j[index] is None:
                        continue
                    page = normalize(j[index]) 
                    if page not in self.cache and (ignore is None or page not in ignore):
                        to_lookup.add(page)

        print(len(to_lookup))
        self.cache = {}
        to_lookup = list(to_lookup)
        
        for batch_num in range(len(to_lookup) // self.batch_size + 1):
            lookup = to_lookup[batch_num*self.batch_size : (batch_num+1)*self.batch_size]
            print(batch_num, len(lookup))
            c.execute('SELECT id,lines from documents where id in (%s)' % ','.join('?'*len(lookup)), lookup)
            for row in tqdm.tqdm(c):
                id,lines = row
                if keep_lines is not None:
                    lines = get_nonempty_doc_lines(lines)
                    lines = [lines[i] for i in keep_lines if i<len(lines)]
                        
                yield normalize(id), lines
        
    def preprocess(self, data, keep_lines=None,
                    keys=('predicted_pages',), ignore=None):
        for id,lines in self.get_batch_doc_lines(data, keep_lines, keys, ignore):
            self.cache[id] = lines
