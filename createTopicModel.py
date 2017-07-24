import logging, sys
import itertools

import numpy as np
import gensim

from gensim.utils import smart_open, simple_preprocess, ClippedCorpus
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import LdaModel

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

if len(sys.argv) !=2:
	print('Missed arguments. G3.py <l/r> <ACMhierarchyDump> <CSdataset>');
	sys.exit();

if sys.argv[1] == 'r':
	fileLocation = '/home/priya/wikiVector/'
else:
	fileLocation = '/media/New Volume/Datasets/'


def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens

# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
stream = iter_wiki(fileLocation+'enwiki-latest-pages-articles.xml.bz2')
#for title, tokens in itertools.islice(iter_wiki('/media/New Volume/Datasets/enwiki-latest-pages-articles.xml.bz2'), 8):
#    print title, tokens[:10]  # print the article title and its first ten tokens
doc_stream = (tokens for _, tokens in iter_wiki(fileLocation+'enwiki-latest-pages-articles.xml.bz2'))
id2word_wiki = gensim.corpora.Dictionary(doc_stream)
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
print(id2word_wiki)
'''
doc1 = "Computer science is the scientific and practical approach to computation and its applications. It is the systematic study of the feasibility, structure, expression, and mechanization of the methodical procedures (or algorithms) that underlie the acquisition, representation, processing, storage, communication of, and access to information. An alternate, more succinct definition of computer science is the study of automating algorithmic processes that scale. A computer scientist specializes in the theory of computation and the design of computational systems"
#doc_stream = [token.encode('utf-8') for token in doc.split()]
doc2 = "Its fields can be divided into a variety of theoretical and practical disciplines. Some fields, such as computational complexity theory (which explores the fundamental properties of computational and intractable problems), are highly abstract, while fields such as computer graphics emphasize real-world visual applications. Still other fields focus on challenges in implementing computation."
doc_stream = []
doc_stream.append([token.encode('utf-8') for token in doc1.split()])
doc_stream.append([token.encode('utf-8') for token in doc2.split()])
#doc_stream = (tokens for _, tokens in iter_wiki('/media/New Volume/Datasets/xaaaa.bz2'))
id2word_wiki = Dictionary(doc_stream)
print(id2word_wiki)

#id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
#print(id2word_wiki)

doc = " For example, programming language theory considers various approaches to the description of computation, while the study of computer programming itself investigates various aspects of the use of programming language and complex systems."
bow = id2word_wiki.doc2bow(tokenize(doc))
print(bow)
print(id2word_wiki[21])
'''

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus(fileLocation+'enwiki-latest-pages-articles.xml.bz2', id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream

MmCorpus.serialize(fileLocation+'wikiModels/wiki_bow.mm', wiki_corpus)

mm_corpus = MmCorpus(fileLocation+'wikiModels/wiki_bow.mm')
print(mm_corpus)

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)

# store all trained models to disk
lda_model.save(fileLocation+'wikiModels/lda_wiki.model')
#lsi_model.save('./data/lsi_wiki.model')
#tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save(fileLocation+'wikiModels/wiki.dictionary')

loaded_lda_model = gensim.models.LdaModel.load(fileLocation+'wikiModels/lda_wiki.model')

# select top 50 words for each of the 20 LDA topics
top_words = [[word for _, word in loaded_lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
print(top_words)

