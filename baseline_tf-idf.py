#D1 : Evaluate with papers dataset
import sys
#from read_wiki import stream
import pdb
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 
import random
from sklearn.feature_extraction.text import TfidfTransformer


file=sys.argv[1];

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

keys = map(lambda x:x[0], keyAbs)
from collections import Counter
keys_freq = Counter(keys)
keys = map(lambda x:x[0], keys_freq.most_common(10))
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

training_samples = []
test = []
np.random.seed(42)
ind = np.random.permutation(range(len(keyAbs)))
train_ind = ind[:int(0.8 * len(ind))]
test_ind = ind[int(0.8*len(ind)):]

pdb.set_trace()
keyAbs = np.array(keyAbs)

train_abstracts = keyAbs[train_ind]
test_abstracts = keyAbs[test_ind]




f_wiki = open('WikiTitleTextMap24.json','r')
wiki_cats = json.load(f_wiki)

Keys = wiki_cats.keys()
key2int = dict(zip(Keys, range(len(Keys))))




X_train = []
y_train = []

X_test = []
y_test = []


print("Generating training samples")

for article in train_abstracts:
    title = article[0]
    if title in Keys:
   	X_train.append(article[1])
	y_train.append(key2int[article[0]])


print("Generating test samples\n")
for article in test_abstracts:
    title = article[0]
    if title in Keys:
  	X_test.append(article[1])
	y_test.append(key2int[title])

count_vec = CV()
X_train_counts = count_vec.fit_transform(X_train)
X_test_counts = count_vec.transform(X_test)
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tf_transformer.transform(X_test_counts)
pdb.set_trace()

#clf = LinearSVC()
clf = lr()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

Acc = accuracy_score(y_test, y_pred)
print("accuracy %f"%(Acc))
