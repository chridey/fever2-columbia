import json
import re
from string import Template
from allennlp.predictors.predictor import Predictor
import json


def reformulate_query(claim):
	m = {'first-decade': '00 and 09','second-decade': '10 and 19','third-decade': '20 and 29',
			'fourth-decade': '30 and 39','fifth-decade': '40 and 49','sixth-decade': '50 and 59',
			'seventh-decade': '60 and 69','eighth-decade': '70 and 79','ninth-decade': '80 and 89',
			'tenth-decade': '90 and 99','final-decade': '90 and 91'}
	m1 = {'twenty-first': '20', 'twentieth': '19','eighteenth': '17',
			'seventeenth': '16','sixteenth': '15','fifteenth': '14',
			'fourteenth': '13','thirteenth': '12','twelveth': '11'}
	m2 = {'twenties': '20 and 29', 'thirties': '30 and 39','forties':'40 and 49',
			'fifties':'50 and 59','sixties':'60 and 69','seventies':'70 and 79','eighties':'80 and 89',
			'nineties':'90 and 99','ninties':'90 and 99'}

	match = re.match(r'.*([1-3][0-9]{3})s', claim)
	if match is not None:
		c = claim.strip().split()
		for elem in c:
			if elem[-1] == 's' and elem[0]>='1' and elem[0]<='9' and len(elem)==5:
				x = elem
				elem = elem.replace('s','')
				elem = int(elem)
				elem_next = elem+9
				if 'in the' in claim:
					claim = claim.replace('in the','between')
					claim = claim.replace(x, str(elem)+' and '+str(elem_next))
					if '--' in claim:
						claim = claim.replace('--','and')
	else:
		if '-decade' in claim and 'century' in claim:
			s = Template('in the $decade of the $centu century')
			x = claim.strip().split()
			decadeval = ''
			centuryval = ''
			for elem in x:
				if elem in m:
					decadeval = elem
				if elem in m1:
					centuryval = elem
			s = s.safe_substitute(decade=decadeval, centu=centuryval)
			centval = m1[centuryval]
			decval = m[decadeval].split()
			new_s = 'between '
			for elem in decval:
				if elem!='and':
					new_s = new_s+centval+elem
				else:
					new_s = new_s+' and '
			claim = claim.replace(s,new_s)
			if '--' in claim:
				claim = claim.replace('--','and')
		elif (re.match(r'.*ties', claim) is not None) and ('century' in claim):
			s = Template('in the $decade of the $centu century')
			x = claim.strip().split()
			decadeval = ''
			centuryval = ''
			for elem in x:
				if elem in m2:
					decadeval = elem
				if elem in m1:
					centuryval = elem
			if decadeval!='':
				s = s.safe_substitute(decade=decadeval, centu=centuryval)
				centval = m1[centuryval]
				decval = m2[decadeval].split()
				new_s = 'between '
				for elem in decval:
					if elem!='and':
						new_s = new_s+centval+elem
					else:
						new_s = new_s+' and '
				claim = claim.replace(s,new_s)
				if '--' in claim:
					claim = claim.replace('--','and')
		elif ('before' in claim or 'after' in claim) and ('--' in claim):
			claim = claim.replace('--','and')
			x = claim.split()
			for i in range(len(x)):
				if x[i]=='before' and (re.match(r'.*([1-3][0-9]{3})', x[i+1]) is not None):
					date = int(x[i+1])-1
					claim = claim.replace('before','between')
					claim = claim.replace(x[i+1],str(date))
				elif  x[i]=='after' and (re.match(r'.*([1-3][0-9]{3})', x[i+1]) is not None):
					date = int(x[i+1])+1
					claim = claim.replace('after','between')
					claim = claim.replace(x[i+1],str(date))



	if ('between' in claim) and (re.match(r'.*([1-3][0-9]{3})', claim) is not None):
		x = claim.split()
		c = 0
		for elem in x:
			if (re.match(r'.*([1-3][0-9]{3})', elem) is not None):
				c = c+1
		if c==3:
			s = ''
			c1 = 0
			for elem in x:
				if (re.match(r'.*([1-3][0-9]{3})', elem) is not None):
					c1 = c1+1
					if c1==1:
						s = s+elem+' and '
					if c1==3:
						s = s+elem
				else:
					if elem!='and':
						s = s+elem+' '
			claim = s
	if '--' in claim:
		claim = claim.replace('--','')
	return claim



def getOpenIEdateArguments(argument,predictor):
	response = predictor.predict(sentence=argument)
	verbs = response['verbs']
	events = []
	for x in verbs:
		description = x['description']
		for c in range(len(description)):
			event = ''
			if description[c]=='[':
				event = '['
				for d in range(c+1,len(description)):
					event = event+description[d]
					if description[d]==']':
						break
			else:
				continue
			c = d+1
			if event!='':
				events.append(event)
	e = []
	prev = '[]'
	for ev in events:
		match = re.match(r'.*([1-3][0-9]{3})', ev)
		if match is not None:
			e.append(prev)
			e.append(ev)
		prev = ev
	return e

def getVerdict(claim,evidence,predictor):
	claim_date_tuples = getOpenIEdateArguments(claim,predictor)
	evidence_date_tuples = getOpenIEdateArguments(evidence,predictor)


	date1 = None
	less = False
	greater = False
	prev = None
	d1 = None
	d2 = None
	label = ""
	for elem in claim_date_tuples:
		if '1000' in elem:
			continue
		match = re.match(r'.*([1-3][0-9]{3})', elem)
		if match is not None and 'between' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='between' and x[i+2]=='and':
					if x[i+1].endswith(']'):
						x[i+1] = x[i+1][:-1]
					d1 = int(x[i+1])
					if x[i+3].endswith(']'):
						x[i+3] = x[i+3][:-1]
					d2 = int(x[i+3])
		elif match is not None and 'years before' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='years' and x[i+1]=='before':
					if x[i-1].endswith(']'):
						x[i-1] = x[i-1][:-1]
					if x[i+2].endswith(']'):
						x[i+2] = x[i+2][:-1]
					y = int(x[i-1])
					z = int(x[i+2])
					date1 = z-y
		elif match is not None and 'years after' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='years' and x[i+1]=='after':
					if x[i-1].endswith(']'):
						x[i-1] = x[i-1][:-1]
					if x[i+2].endswith(']'):
						x[i+2] = x[i+2][:-1]
					y = int(x[i-1])
					z = int(x[i+2])
					date1 = y+z
		elif match is not None and 'before' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='before' and isDate(x[i+1]):
					if x[i+1].endswith(']'):
						x[i+1] = x[i+1][:-1]
					date1 = int(x[i+1])
					less = True
		elif match is not None and 'after' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='after' and isDate(x[i+1]):
					if x[i+1].endswith(']'):
						x[i+1] = x[i+1][:-1]
					date1 = int(x[i+1])
					greater = True
		elif match is not None and 'in' in elem:
			x = elem.split()
			for i in range(len(x)):
				if x[i]=='in' and x[i+1]!='the' and isDate(x[i+1]):
					if x[i+1].endswith(']'):
						x[i+1] = x[i+1][:-1]
					date1 = int(x[i+1])
		prev = elem


	prev2 = None
	evdate = None
	for elem in evidence_date_tuples:
		match = re.match(r'.*([1-3][0-9]{3})', elem)
		if (match is not None) and (prev is not None) and ((prev2 is not None and len(set(prev.lower().split()).intersection(set(prev2.lower().split())))>=1) or len(set(elem.lower().split()).intersection(set(prev.lower().split())))>=1):
			if 'between' in elem:
				x = elem.split()
				for i in range(len(x)):
					if x[i]=='between' and x[i+2]=='and':
						if x[i+1].endswith(']'):
							x[i+1] = x[i+1][:-1]
						if d1 is not None and int(x[i+1])!=d1:
							label = "REFUTES"
						if x[i+3].endswith(']'):
							x[i+3] = x[i+3][:-1]
						if d2 is not None and int(x[i+3])!=d2:
							label = "REFUTES"
			else:
				x = elem.split()
				for i in range(len(x)):
					if len(x[i])==4 and x[i][0]>='0' and x[i][0]<='9' and x[i][3]>='0' and x[i][3]<='9':
						if x[i].endswith(']'):
							x[i] = x[i][:-1]
						evdate = int(x[i])
	if evdate == None:
		label = "REFUTES"
	if label=="":
		if less == False and greater==False:
			if date1 == evdate:
				label = "SUPPORTS"
			else:
				label = "REFUTES"
		if less == True:
			if evdate < date1:
				label = "SUPPORTS"
			else:
				label = "REFUTES"
		if greater == True:
			if evdate > date1:
				label = "SUPPORTS"
			else:
				label = "REFUTES"
		if d1 is not None and d2 is not None:
			if evdate>=d1 and evdate<=d2:
				label = "SUPPORTS"
			else:
				label = "REFUTES"

	return label


def isDate(str):
	match = re.match(r'.*([1-3][0-9]{3})', str)
	if match is not None:
		return True
	else:
		return False

def isDateRange(str):
	match = re.match(r'.*([1-3][0-9]{3})s', str)
	if match is not None:
		return True
	else:
		return False


def isClaimEligibleForDateCalculation(claim):
	claim = reformulate_query(claim)
	words = claim.split()
	for i in range(len(words)):
		if (words[i]=='before' or words[i]=='after') and (i+1<len(words) and (isDate(words[i+1]) or isDateRange(words[i+1]))):
			return claim,True

		if (i-1>=0 and (words[i-1]=='years' or words[i-1]=='year')) and (i+1<len(words) and (isDate(words[i+1]) or isDateRange(words[i+1]))):
			return claim ,True

		if (i-2>=0 and words[i-2]=='in') and (i-1>=0 and words[i-1]=='the') and (isDateRange(words[i])):
			return claim ,True

		if (words[i]=='between') and (i+1<len(words) and isDate(words[i+1])):
			return claim ,True

	return claim , False



def getDateClaimLabel(claim,evidence,reader,predictor):
	s = ''
	for ev in evidence:
		doc = ev[0]
		lineno = ev[1]
		evi = reader.get_doc_line_for_date_claim(doc,lineno)
		if re.match(r'.*([1-3][0-9]{3})', evi) is not None:
			s = s+evi

	return getVerdict(claim,s,predictor)




