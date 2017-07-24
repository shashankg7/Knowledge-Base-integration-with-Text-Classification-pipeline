#D1 : Evaluate with papers dataset
import sys
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import utils
#from read_wiki import stream
from gensim.models import Word2Vec
from representation import lda, lsi, doc2vec, word2vec, doc2vec300
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


# mapping category names to corresponding freebase entries
mapFields={}
mapFields['Programming languages'] = '/en/programming_language'
mapFields['Real time and embedded systems'] = '/en/embedded_system'
mapFields['Scientific computing'] = '/en/scientific_computing'
mapFields['Natural language and speech'] = '/en/natural_language_processing'
mapFields['Machine learning and pattern recognition'] = '/en/machine_learning'
mapFields['Operating systems'] = '/en/operating_system'
mapFields['World wide web'] = '/en/world_wide_web'
mapFields['Bioinformatics and computational biology'] = '/en/bioinformatics'
mapFields['Security and privacy']= '/en/internet_security'
mapFields['Distributed and parallel computing'] = '/en/parallel_computing'
mapFields['Databases'] = '/en/database'
mapFields['Simulation'] = '/en/simulation'
mapFields['Algorithms and theory'] = '/en/algorithm'
mapFields['Computer education']= '/en/computer_literacy'
mapFields['Human-computer interaction']= '/en/human_computer_interaction'
mapFields['Hardware and architecture']= '/en/hardware_architecture'
mapFields['Networks and communications'] = '/en/computer_network'
mapFields['Artificial intelligence'] = '/en/artificial_intelligence'
mapFields['Data mining'] = '/en/data_mining'
mapFields['Computer vision'] = '/en/computer_vision'
mapFields['Simulation'] = '/en/simulation'
mapFields['Software engineering'] = '/en/software_engineering'
mapFields['Information retrieval']= '/en/information_retrieval'
mapFields['Multimedia'] = '/en/multimedia'
mapFields['Graphics'] ='/en/graphics'
#print 'Mappings exist for '+str(len(mapFields.keys()))

f_wiki = open('WikiTitleTextMap24.json', 'r')
wiki_data1 = json.load(f_wiki)
keys1 = wiki_data1.keys()


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


print("Generating training samples")
# Pre-computing wiki vecs
wiki_vecs = {}
freebase_vecs = Word2Vec.load_word2vec_format('./freebase_vectors.bin', binary=True)
#wiki_lens = {}
for k, v in mapFields.iteritems():
    wiki_vecs[k] = freebase_vecs[v]
keys = wiki_vecs.keys()

for k, v in wiki_data1.items():
    wiki_vecs[k] = np.hstack((wiki_vecs[k], doc2vec300(v.lower())))


pdb.set_trace()
print("wiki articles found")
#f1 = open('docvecs_exp1.json', 'w')
#json.dump(wiki_vecs, f1)

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
            print("doc")
            x.extend(doc2vec300(abstract))

	elif rep == 'word2vec':
	    x.extend(word2vec(abstract))

        training_samples.append((x, y, 1, title))

        neg_title = set(keys) - set(title)
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
	    x.extend(word2vec(abstract))

        test.append((x, title))

print("Data generated\n")
#random shuffle training data
X_train = []
y_train = []

X_test = []
y_test = []

for x,y,label,_ in training_samples:
	X = []
	X.extend(x)
	X.extend(y)
	X_train.append(X)
	y_train.append(label)

del training_samples
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_train = np.nan_to_num(X_train).astype(np.float32)

print("Training a classifier\n")
# Train a logistic regression
model = lr()
#model = LinearSVC()
# Time variables for benchmarking
time1 = time.clock()
model.fit(X_train, y_train)
time2 = time.clock()
time_elapsed = time2 - time1
print("Time elapsed in training the system is %f"%time_elapsed)
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
