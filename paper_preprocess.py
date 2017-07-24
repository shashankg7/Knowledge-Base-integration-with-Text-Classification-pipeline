#D1 : Evaluate with papers dataset
import sys
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import utils
#from read_wiki import stream
from representation import lda, lsi, doc2vec
from gensim.parsing.preprocessing import STOPWORDS
import pdb
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score as acc
from compiler.ast import flatten
from collections import Counter
#read in papers dataset
# Arg1 is path to CS_Citation_Network Arg2 is type of representation to be used

file="CS_Citation_Network"
rep = "doc2vec"
max_vocab = 50000
embed_dim = 25
#Read the keyword and abstract
uniqueFields = {}
keyAbs=[]
with open(file) as infile:
	field = None;
	abstract = None;
	for line in infile:
		#if len(line)==1 :
		if line in ['\n', '\r\n']:
			if abstract != None:
				keyAbs.append([field,abstract])
				#print abstract
				#print field
			field = None
			abstract = None
		else:
			#if line[1]=='*' : # field is title
				#field = line.replace("#*","")
			if line[1]=='f' : #field is keyword
				field = line.replace("#f","")
				field = field.replace("\n","")
				field = field.replace("\r","")
				field = field.replace("_"," ")
				field = field.capitalize()
				#print field
				uniqueFields[field]=0
			if line[1]=='!' :
				abstract = line.replace("#!","")
print 'Citations loaded = '+str(len(keyAbs))
doc2vec_model = Doc2Vec.load('./wiki_model2.doc2vec')


def tokenize(text):
	try:
		return [token for token in utils.simple_preprocess(text) if token not in STOPWORDS]
	except Exception as e:
		print(str(e))


train_abs = keyAbs[:int(0.8 * len(keyAbs))]
test_abs = keyAbs[int(0.8 * len(keyAbs)):]
	
abstracts1 = map(lambda x:x[1], train_abs)
Abstracts1 = map(lambda x:x.rstrip().lower(), abstracts1)
Abs1 = map(lambda x:tokenize(x), Abstracts1)
title1 = map(lambda x:x[0], train_abs)

abstracts2 = map(lambda x:x[1], test_abs)
Abstracts2 = map(lambda x:x.rstrip().lower(), abstracts2)
Abs2 = map(lambda x:tokenize(x), Abstracts2)
title2 = map(lambda x:x[0], test_abs)

tokens1 = flatten(Abs1)
freq = Counter(tokens1)
freq_tokens = freq.most_common(max_vocab)
tokens = map(lambda x:x[0], freq_tokens)
vocab = dict(zip(tokens, range(len(tokens))))
vocab["UNK"] = len(vocab)

#train_tokens = 


#pdb.set_trace()
i = 0
m = 0
embedding = np.random.uniform(low=-0.25, high=0.25, size=(len(vocab), embed_dim))
for token, idx in vocab.items():
	try:
		embedding[idx, :] = doc2vec_model[token]
		i += 1
	except Exception as e:
		m += 1

print("no of missed tokens %d out of %d"%(m, len(tokens)))
#pdb.set_trace()
