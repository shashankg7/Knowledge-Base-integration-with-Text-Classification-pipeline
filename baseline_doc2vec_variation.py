#D1 : Evaluate with papers dataset
import sys
import os
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import utils
#from read_wiki import stream
from representation1 import lda, lsi, doc2vec, word2vec, doc2vec300
import pdb
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import LinearSVC
import time
from collections import defaultdict

#read in papers dataset
# Arg1 is path to CS_Citation_Network Arg2 is type of representation to be used

file=sys.argv[1];
rep = sys.argv[2]
wiki_path = './enwiki-latest-pages-articles.xml.bz2'

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

# avg length of each article's abstract
avg_len = map(lambda x:len(x[1].split()), keyAbs)
avg_len1 = sum(avg_len)/float(len(avg_len))

#print 'Unique fields = '+str(uniqueFields.keys())


#wiki_file = open('wikiIdTitleMap.json','r')
#wikiTitleTextMap = json.load(wiki_file)

#map Citation field name to Wiki article name for string inconsistencies
mapFields={}
mapFields['Programming languages'] = ['Programming language']; mapFields['Real time and embedded systems'] = ['Modeling and Analysis of Real Time and Embedded systems'];
mapFields['Scientific computing'] = ['Computational science']; mapFields['Natural language and speech'] = ['Natural language processing','Speech recognition'];
mapFields['Machine learning and pattern recognition'] = ['Machine learning','Pattern recognition']; mapFields['Operating systems'] = ['Operating system'];
mapFields['World wide web'] = ['World Wide Web'];  mapFields['Bioinformatics and computational biology'] = ['Bioinformatics','Computational biology'];
mapFields['Security and privacy']=['Information security', 'Internet privacy']; mapFields['Distributed and parallel computing'] = ['Distributed computing','Parallel computing'];
mapFields['Databases'] = ['Database'];  mapFields['Simulation'] = ['Computer simulation'];
mapFields['Algorithms and theory'] = ['Algorithm', 'Theoretical computer science']; mapFields['Computer education']=['Computer literacy'];
mapFields['Human-computer interaction']= []; #REVISIT ['Human-computer interaction' is missing in wiki_model2.doc2vec
mapFields['Hardware and architecture']=['Hardware architecture'];
mapFields['Networks and communications'] = ['Computer network', 'Telecommunications engineering']
mapFields['Artificial intelligence'] = ['Artificial intelligence']; mapFields['Data mining'] = ['Data mining'];
mapFields['Computer vision'] = ['Computer vision']; mapFields['Simulation'] = ['Simulation'] ; mapFields['Software engineering'] = ['Software engineering'];
mapFields['Information retrieval']=['Information retrieval']; mapFields['Multimedia'] = ['Multimedia']; mapFields['Graphics'] = ['Graphics']
#print 'Mappings exist for '+str(len(mapFields.keys()))

print("Loading wikipedia page content")

# Get wiki page corresponding to categories
#CategoriesWikiPage = {}
#for field in uniqueFields.keys():
#    keyFields = mapFields[field]
#    for keyField in keyFields:
#            keyField = keyField.encode('utf-8')
#            print(keyField)

#titles1 = sum(mapFields.values(), [])
#titles2 = mapFields.keys()
#titles1.extend(titles2)

#for field in uniqueFields.keys():
#    keyFields = mapFields[field]
#    print(field, keyFields)

# Generate training and testing data
n = 0
for article in keyAbs:
    title = article[0]
    if title == 'Human-computer interaction':
        n+= 1
print(n)

pdb.set_trace()

f_wiki = open('WikiTitleTextMap24.json', 'r')
wiki_data = json.load(f_wiki)
keys = wiki_data.keys()

training_samples = []
test = []
np.random.seed(42)
ind = np.random.permutation(range(len(keyAbs)))
train_ind = ind[:int(0.8 * len(ind))]
test_ind = ind[int(0.8 * len(ind)):]
#keyAbs = np.array(keyAbs)

#train_abstracts = keyAbs[train_ind]
#test_abstracts = keyAbs[test_ind]

train_abstracts = [keyAbs[index] for index in train_ind]
test_abstracts = [keyAbs[index] for index in test_ind]
pdb.set_trace()
# TRAIN-TEST SPLITTING AND DATA GENERATION WHEN ALL CATEGORIES ARE PRESENT
#for article in train_abstracts:
#    titles = article[0]
#    titles.extend(sum(mapFields[titles], []))
#    # Add positive samples to training data
#    for title in titles:
#        x = []
#        wiki = wikiTitleTextMap[title]
#        wiki = wiki.lower()
#        article = article.lower()
#        if rep == 'lda':
#            x.append(lda(article))
#            x.append(lda(wiki))
#        elif rep == 'doc2vec':
#            x.append(doc2vec(article))
#            x.append(doc2vec(wiki))

#        pos_samples.append(x)
#        # Remove the current title (and its matching titles as well)
#        titles3 = set(titles1) - set(titles)
#        rand_title = random.sample(titles3)
#        y = []
#        wiki = wikiTitleTextMap[rand_title]
#        wiki = wiki.lower()
#        article = article.lower()
#        if rep == 'lda':
#            y.append(lda(article))
#            y.append(lda(wiki))
#        elif rep == 'doc2vec':
#            y.append(doc2vec(article))
#            y.append(doc2vec(wiki))
#        neg_samples.append(y)

#    # Sample negative examples from wiki


#for article in test_abstracts:
#    titles = article[0]
#    titles.extend(sum(mapFields[titles], []))
#    for title in titles:
#        x = []
#        wiki = wikiTitleTextMap[title]
#        wiki = wiki.lower()
#        article = article.lower()
#        if rep == 'lda':
#            x.append(lda(article))
#            x.append(lda(wiki))
#        elif rep == 'doc2vec':
#            x.append(doc2vec(article))
#            x.append(doc2vec(wiki))
#
#        test.append(x)


K = int(sys.argv[3])
print("Generating training samples")
# Pre-computing wiki vecs
wiki_vecs = {}
#wiki_lens = {}
if rep == 'doc2vec':
	for k, v in wiki_data.items():
                #wiki_lens[k] = len(v.split())
		wiki_vecs[k] = doc2vec300(v.lower(), k)
elif rep == 'lda':
	for k, v in wiki_data.items():
		wiki_vecs[k] = lda(v.lower())
elif rep == 'word2vec':
	print("Word")
	for k, v in wiki_data.items():
                #wiki_lens[k] = len(v.split())
		wiki_vecs[k] = word2vec(v.lower(), K)
print("wiki articles found")
#f1 = open('docvecs_exp1.json', 'w')
#json.dump(wiki_vecs, f1)


if not os.path.isfile('X_train.npy'):
    for article in train_abstracts:
        title = article[0]
        if title in keys:
            x = []
            y = []
            abstract = article[1].lower()
            y.extend(wiki_vecs[title])
            if rep == 'lda':
                x.extend(lda(abstract))
             
            elif rep == 'doc2vec':
                x.extend(doc2vec300(abstract), k)

            elif rep == 'word2vec':
                x.extend(word2vec(abstract, K))

            training_samples.append((x, y, 1, title))

            neg_title = set(keys) - set(title)
            #random.seed(42)
            title_neg = random.choice(list(neg_title))
            if title_neg in keys:
                y = []
                y.extend(wiki_vecs[title_neg])

            training_samples.append((x, y,0, title))

#dictinary to book-keep performance per class
#per_class_perf_gt = defaultdict(int)
#per_class_perf = defaultdict(int)
#wiki_len  = {}


print("Generating test samples\n")
for article in test_abstracts:
    title = article[0]
    if title in keys:
        #per_class_perf_gt[title] += 1
        x = []
	#y = []
        abstract = article[1].lower()
        if rep == 'lda':
            x.extend(lda(abstract))
            #y.extend(wiki_data[title])
        elif rep == 'doc2vec':
            x.extend(doc2vec300(abstract))
            #y.extend(wiki_data[title])
	elif rep == 'word2vec':
	    x.extend(word2vec(abstract, K))

        test.append((x, title))

print("Data generated\n")
#random shuffle training data
X_train = []
y_train = []

X_test = []
y_test = []


if not os.path.isfile('X_train.npy'):
    for x,y,label,_ in training_samples:
            X = []
            X.extend(x)
            X.extend(y)
            X_train.append(X)
            y_train.append(label)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    X_train = np.nan_to_num(X_train).astype(np.float32)
    np.save('X_train', X_train)
    np.save('y_train', y_train)
   
    del training_samples

if os.path.isfile('X_train.npy'):
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')


print("Training a classifier\n")
# Train a logistic regression
model = lr()
#model = LinearSVC()
# Time variables for benchmarking
#time1 = time.clock()
model.fit(X_train, y_train)
#time2 = time.clock()
#time_elapsed = time2 - time1
#print("Time elapsed in training the system is %f"%time_elapsed)
print("Training done, testing on test data\n")
#pdb.set_trace()
# test
i = 0
w = 0
for x, title in test:
	for k, v in wiki_vecs.items():
		X = []
		X.extend(x)
		X.extend(v)
		n = len(X)
		X = np.array(X)
		X = X.reshape(1, n)
		if len(np.where(np.isnan(X) == True)) > 1:
			#print(X)
			X = np.nan_to_num(X)
			w += 1
		if model.predict(X) == 1:
			if k == title:
                                #per_class_perf[title] += 1
				i += 1
				#print("Right prediction")
				break

print("Total nan cases %d"%w)
print("No of right predictions %d out of %d with accuracy %f"%(i, len(test), float(i)/len(test)))
pdb.set_trace()
