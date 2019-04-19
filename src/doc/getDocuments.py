import sys

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


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    if 'items' in res:
    	return res['items']
    else:
    	return []




def getDocumentsForClaimFromGoogle(claim,api_key, cse_id):
	results = google_search('wikipedia '+claim, api_key, cse_id, num=5)
	res = []
	c = 0
	for result in results:
		if 'https://en.wikipedia.org/wiki/' in result['formattedUrl'] and c<2:
			b = result['formattedUrl'].replace('https://en.wikipedia.org/wiki/','')
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


def getDocumentsFromDepParse(claim):
	claim = nltk.word_tokenize(claim)
	pos = nltk.pos_tag(claim)
	s = ''
	for w,p in zip(claim,pos):
		if p[1][0]=='V' and w.islower():
			s = s[:-1]
			break
		s = s+ w + ' '
	entity = wikipedia.search(s,1)
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

def getDocumentsForNer(claim):
	predictor = Predictor.from_path("fine-grained-ner-model-elmo-2018.12.21.tar.gz")
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
		y = wikipedia.search(ner,1)
		for x in y:
			x = x.replace(' ','_')
			x = x.replace('(','-LRB-')
			x = x.replace(')','-RRB-')
			x = x.replace(':','-COLON-')
			rec.append(x)
	rec = list(set(rec))
	return rec


def getDocsForClaim(claim,api_key,cse_id):
	docs_google = getDocumentsForClaimFromGoogle(claim,api_key,cse_id)
	docs_ner = getDocumentsForNer(claim)
	docs_dep_parse = getDocumentsFromDepParse(claim)
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

	return docs



def getDocsSingle(data,api_key,cse_id):
	if str(type(data))=="<class 'dict'>":
		return getDocsForClaim(data['claim'],api_key,cse_id)
	if str(type(data))=="<class 'list'>":
		print('here')
		a = []
		for d in data:
			claim = d['claim']
			docs = getDocsForClaim(claim,api_key,cse_id)
			d['predicted_pages'] = docs
			a.append(d)
		return a


def getDocsBatch(file,api_key,cse_id):
	for line in open(file):
		line = json.loads(line.strip())
		line['predicted_pages'] = getDocsForClaim(line['claim'],api_key,cse_id)
		yield line

# getDocsSingle({'id':0,'claim':'The Dark Tower is a fantasy film.'})
#print(getDocsSingle([{'id':0,'claim':'The Dark Tower is a fantasy film.'},{'id':1,'claim':'"Down With Love is a 2003 comedy film.'}]))

if __name__ == '__main__':
    filename = sys.argv[1]
    api_key = sys.argv[2]
    cse_id = sys.argv[3]
    outfilename = sys.argv[4]

    with open(outfilename, 'w') as outfile:
        for docs in getDocsBatch(filename, api_key, cse_id):
            print(json.dumps(docs, file=outfile))
