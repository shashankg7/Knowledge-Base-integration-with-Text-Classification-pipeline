# Collection of models for inferring different representations (lda, lsi etc.) for a given document

import gensim
import numpy as np
from gensim import corpora, utils, similarities
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import Doc2Vec, Word2Vec
#from read_wiki import stream
import pdb
import sys

dir_path = './'
#doc2vec_path = dir_path + 'wiki_model2.doc2vec'
doc2vec300_path = dir_path + 'wiki_model5.doc2vec'
#word2vec_path = dir_path + 'pretrained_word2vec.bin'

print("Loading word2vec and doc2vec models")
#doc2vec_model = Doc2Vec.load(doc2vec_path)
#word2vec_model = Word2Vec.load_word2vec_format(word2vec_path, binary=True)
doc2vec300_model = Doc2Vec.load(doc2vec300_path)
print("Models loaded, proceeding...")
np.random.seed(42)
random_word = np.random.uniform(low=-0.25, high=0.25, size=(300,))


def tokenize(text):
	return [token for token in utils.simple_preprocess(text) if token not in STOPWORDS]


def lda(text):
	dict_path = dir_path + 'cs_lda6.dict'
	lda_path = dir_path + 'wiki_model6.ldamodel'
	dictionary = corpora.Dictionary.load(dict_path)
	model = gensim.models.ldamodel.LdaModel.load(lda_path)
	print(model)
	nvect = dictionary.doc2bow(tokenize(text))
	print(nvect)
	#print(model[nvect])
	return model[nvect]


def lsi(text):
	pass


def doc2vec(text):
	return doc2vec_model.infer_vector(text.split(), alpha=0.1, min_alpha=0.0001, steps=5)

def doc2vec300(text):
	return doc2vec300_model.infer_vector(tokenize(text), alpha=0.025, min_alpha=0.025, steps=5)


def word2vec(text):
	tokens = tokenize(text)
	#if len(tokens) == 0:
	#	print(text)
	#	print(tokens)
	#	pdb.set_trace()
	vec = np.zeros((300,))
	missed = 0
	for token in tokens:
		try:
			vec += word2vec_model[token]
		except:
			missed += 1
			vec += random_word
	#print("Missed tokens %d out of total %d tokens"%(missed, len(tokens)))
	return vec/float(len(tokens))


if __name__ == "__main__":
	test = "Social media mining is the process of representing, analyzing , and extracting actionable patterns from social media data. Social media mining introduces basic concepts and principal algorithms suitable for investing massive social media data; it discusses theories and methodologies from different disciplines such as computer science, data mining, machine learning, social network analysis, optimization and mathematics. It encompasses the tools to formaly represent, measure, model and mine meaningful patterns from large-scale data"
        f = open('hci.txt', 'r')
        hci_text = f.readlines()
	print(doc2vec(test))
	#vec = word2vec(test)
	pdb.set_trace()
