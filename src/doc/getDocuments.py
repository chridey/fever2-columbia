import sys
import os
import json
import ast
from time import sleep
from googleapiclient.discovery import build #google-api-python-client
import pprint
import nltk
import wikipedia
import ast
import unicodedata
from allennlp.service.predictors import Predictor
import time
import logging

logger = logging.getLogger(__name__)

class GoogleConfig:
    def __init__(self, api_key=None, cse_id=None, site='wikipedia', num=5, max_docs=2):
        if 'API_KEY' in os.environ:
            api_key = os.environ['API_KEY']
        if 'SEARCH_ID' in os.environ:
            cse_id = os.environ['SEARCH_ID']
        assert(api_key is not None and cse_id is not None)
        
        self.api_key = api_key
        self.cse_id = cse_id
        self.site = site
        self.num = num
        self.max_docs = max_docs

    def __str__(self):
        return 'GoogleConfig(api_key={}, cse_id={}, site={}, num={}, max_docs={}'.format(self.api_key,
                                                                                         self.cse_id,
                                                                                         self.site,
                                                                                         self.num,
                                                                                         self.max_docs)
        
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    if 'items' in res:
        return res['items']
    else:
        return []

def getDocumentsForClaimFromGoogle(claim, google_config):    
    results = google_search(google_config.site+' '+claim, google_config.api_key,
                            google_config.cse_id, num=google_config.num)
    logger.info(google_config)
    res = []
    c = 0
    for result in results:
        logger.info(result['formattedUrl'])
        #if 'https://en.wikipedia.org/wiki/' in result['formattedUrl'] and c<google_config.max_docs:
        if 'en.wikipedia.org' in result['formattedUrl'] and c<google_config.max_docs:
            b = result['formattedUrl'].split('/')[-1].replace(' ', '')#replace('https://en.wikipedia.org/wiki/','')
            c = c+1
            b = b.replace('(','-LRB-')
            b = b.replace(')','-RRB-')
            b = b.replace(':','-COLON-')
            b = b.replace('%26','&')
            b = b.replace('%27',"'")
            b = b.replace('%22','"')
            b = b.replace('%3F','?')
            res.append(b)
    return res

def getDocumentsFromConstituentParse(claim, predictor):
    results = predictor.predict(sentence=claim)

    nodes = [results['hierplane_tree']['root']]
    candidates = []
    while len(nodes):
        node = nodes.pop(0)
        if node['nodeType'] == 'NP':
            #print(node)
            candidates.append(node['word'])
        if 'children' in node:
            nodes.extend(node['children'])    

    entities = []
    for candidate in candidates:
        entity = wikipedia.search(candidate,1)
        if len(entity)>0:
            x = entity[0]
            x = x.replace(' ','_')
            x = x.replace('(','-LRB-')
            x = x.replace(')','-RRB-')
            x = x.replace(':','-COLON-')
            entities.append(x)
    return entities
            
def getDocumentsFromDepParse(claim):
    claim = nltk.word_tokenize(claim)
    pos = nltk.pos_tag(claim)
    s = ''
    for w,p in zip(claim,pos):
        if p[1][0]=='V' and w.islower():
            s = s[:-1]
            break
        s = s+ w + ' '
    if s:
        entity = wikipedia.search(s,1)
    else:
        entity = []
    if len(entity)>0:
        x = entity[0]
        x = x.replace(' ','_')
        x = x.replace('(','-LRB-')
        x = x.replace(')','-RRB-')
        x = x.replace(':','-COLON-')
        return [x]
    else:
        return []


def addHeuristics(l,claim):
    if 'film' in claim.split() or '(film)' in claim.split() or 'film)' in claim.split():
        if 'is' in claim.split():
            l.append(claim.split('is')[0]+' '+'film')
        if 'was' in claim.split():
            l.append(claim.split('was')[0]+' '+'film')
        if 'in the film' in claim:
            l.append(claim.split('in the film')[1]+' '+'film')
        elif '(film)' in claim:
            l.append(claim.split('(film)')[0]+' '+'film')
        elif 'film)' in claim:
            l.append(claim.split(')')[0]+') ')
        elif 'called' in claim:
            l.append(claim.split('called')[1]+' film')
    if 'directed by' in claim:
        if 'is directed by' in claim:
            l.append(claim.split('is directed by')[0]+' '+'film')
        elif 'was directed by' in claim:
            l.append(claim.split('was directed by')[0]+' '+'film')
    if 'stars' in claim:
        l.append(claim.split('stars')[0]+' '+'film')
    if 'is a' in claim and 'film' not in claim:
        l.append(claim.split('is a')[0])
    if 'is the' in claim:
        l.append(claim.split('is the')[0])
    if 'premiered in' in claim:
        l.append(claim.split('premiered in')[0])
    if 'based on' in claim:
        if 'is ' in claim:
            l.append(claim.split('is ')[0])
        elif 'are ' in claim:
            l.append(claim.split('are ')[0])
    if 'was' in claim and 'film' not in claim:
        l.append(claim.split('was')[0])
    if 'has' in claim:
        l.append(claim.split('has')[0])
    if 'is' in claim and 'actor' in claim:
        l.append(claim.split('is')[0]+' '+'actor')
    if 'is' in claim and 'actress' in claim:
        l.append(claim.split('is')[0]+' '+'actress')
    if 'features' in claim:
        l.append(claim.split('features')[0])

    return l

def getDocumentsForNer(claim,predictor):
    results = predictor.predict(sentence=claim)
    f = open('NER.txt','w')
    for word, tag in zip(results["words"], results["tags"]):
        f.write(word+'\t'+tag+'\n')
    f.write('\n')
    f.close()
    l =[]
    s = ''
    a = []
    for line in open('NER.txt','r'):   #change files names based on dev /test
        if 'DATE' in line.strip().split('\t')[-1]:
            continue
        if len(line.strip())==0:
            l.append(a)
            a = []
            continue
        if line.strip().split('\t')[-1]=='O':
            if s!='':
                a.append(s)
            s =''
            continue
        if s!='':
            s = s+' '+line.strip().split('\t')[0]
        else:
            s = line.strip().split('\t')[0]
    

    entities = l[0]
    entities = addHeuristics(entities,claim)
    rec = []
    a = list(set(entities))
    for ner in a:
        try:
            y = wikipedia.search(ner,1)
        except Exception:
            y = []
        for x in y:
            x = x.replace(' ','_')
            x = x.replace('(','-LRB-')
            x = x.replace(')','-RRB-')
            x = x.replace(':','-COLON-')
            rec.append(x)
    rec = list(set(rec))
    return rec

def getDocsFromTFIDF(claim, ranker, k=10):
    doc_names, doc_scores = ranker.closest_docs(claim, k)

    pages = list(zip(doc_names, doc_scores))
    return pages

def getDocsForClaim(claim,google_config,predictor,ranker):
    try:
        docs_google = getDocumentsForClaimFromGoogle(claim,google_config)
    except Exception:
        docs_google = []
    try:
        docs_ner = getDocumentsForNer(claim,predictor)
    except Exception:
        docs_google = []
    try:
        docs_dep_parse = getDocumentsFromDepParse(claim)
    except Exception:
        docs_dep_parse = []
    try:
        docs_tfidf = getDocsFromTFIDF(claim, ranker)
    except Exception:
        docs_tfidf = []
    docs = []
    for elem in docs_google:
            if 'disambiguation' not in elem or 'List_of_' not in elem:
                docs.append(elem)
    for elem in docs_ner:
            if 'disambiguation' not in elem or 'List_of_' not in elem:
                    if elem not in docs:
                        docs.append(elem)
    for elem in docs_dep_parse:
            if 'disambiguation' not in elem or 'List_of_' not in elem:
                    if elem not in docs:
                        docs.append(elem)
    for elem,_ in docs_tfidf:
            if 'disambiguation' not in elem or 'List_of_' not in elem:
                    if elem not in docs:
                        docs.append(elem)
                        
    docs = [[d] for d in docs ]
    return dict(predicted_pages=docs, predicted_google=docs_google,
                predicted_ner=docs_ner, predicted_dep_parse=docs_dep_parse,
                predicted_tfidf=docs_tfidf)

def getDocsSingle(data,google_config,predictor,ranker):
    if str(type(data))=="<class 'dict'>":
        return getDocsForClaim(data['claim'],google_config,predictor,ranker)
    if str(type(data))=="<class 'list'>":
        print('here')
        a = []
        for d in data:
            claim = d['claim']
            docs = getDocsForClaim(claim,google_config,predictor,ranker)
            d.update(docs)
            a.append(d)
        return a


def getDocsBatch(file,google_config,predictor,ranker):
    for line in open(file):
        line = json.loads(line.strip())
        line.update(getDocsForClaim(line['claim'],google_config,predictor,ranker))
        yield line

if __name__ == '__main__':
    filename = sys.argv[1]
    api_key = sys.argv[2]
    cse_id = sys.argv[3]
    outfilename = sys.argv[4]

    google_config = GoogleConfig(api_key, cse_id)
    with open(outfilename, 'w') as outfile:
        for docs in getDocsBatch(filename, google_config, predictor):
            print(json.dumps(docs), file=outfile)
