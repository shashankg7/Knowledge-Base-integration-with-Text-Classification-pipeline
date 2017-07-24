import re
import os
from gensim import corpora, models, similarities, utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from gensim.corpora.mmcorpus import MmCorpus
#from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import WikiCorpus
from lxml import etree
from datetime import datetime
import pdb
#abstract_file = '/home/shashank/wikiVector/enwiki-latest-abstract.xml'
dump_file = '/home/priya/wikiVector/enwiki-latest-pages-articles.xml'; #'Wikipedia-20130419041326.xml';#


wikiTitleKeyMap = {}
start=datetime.now()
wikiTitleKeyMap = {}
#with utils.smart_open("/media/New Volume/Datasets/wikiKeyTitleMap.txt") as fin:
with utils.smart_open("/home/priya/wikiVector/wikiTitleKeyMapFull.txt") as fin:
    for line in fin:
        if line.endswith('\n'):
            line = line.replace("\n","")
        key = line.split()[-1].strip()
        title = line.replace(key,'').strip()
        #print title+'-'+key
        wikiTitleKeyMap[title] = key
print 'No of titles in map = '+str(len(wikiTitleKeyMap))

keyset = []

class wikiContext(object):
    global dump_file
    def __iter__(self):
        pages = corpora.wikicorpus.extract_pages(dump_file)
	pdb.set_trace()
        for page in pages:
            if not page[0].startswith("File:"):
                pageId = utils.to_unicode(page[0])
                pageTitle = utils.to_unicode(page[2])
                #if pageTitle in CCCoutputMap.keys():
                #    context_text = page[1].extend(page[1]).extend(page[1]).extend(page[1])
                #else :
                #    context_text = page[1]
                yield TaggedDocument(page[1], [pageId])
        del pages


start=datetime.now()
print 'Building model ...'
documents = wikiContext()

#model = Doc2Vec( documents, size=25, window=8, min_count=5, workers=0)
model = Doc2Vec( alpha=0.025, min_alpha=0.025, size=300, window=10, min_count=5, workers=0) # default size = 25
model.build_vocab(documents) #this step takes 2h             
for epoch in range(10):
    print('TRAINING EPOCH ',epoch)
    model.train(documents)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

print 'words in created model = '+str(len(model.vocab))
print 'Entities in created model = '+str(len(model.docvecs))

# store the model to mmap-able files
model.save('/home/priya/wikiVector/wiki_model7.doc2vec')#model.save_word2vec_format('/tmp/my_model.doc2vec')
print('Model made in '+str(((datetime.now()-start).total_seconds())/60)+' minutes.')

# load the model back
model_loaded = Doc2Vec.load('/home/priya/wikiVector/wiki_model5.doc2vec')#model_loaded = Doc2Vec.load_word2vec_format('/tmp/my_model.doc2vec')
print 'words in loaded model = '+str(len(model_loaded.vocab))
print 'Entities in loaded model = '+str(len(model_loaded.docvecs))

tweet = 'Its a nice cool day'
twVector = model_loaded.infer_vector(tweet.split(),alpha=0.1, min_alpha=0.0001, steps=5)
pdb.set_trace()
print(twVector)
