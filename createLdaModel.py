#create a computer science documents corpus from Wikipedia and train a lda model on it
import re, sys
import os.path
from gensim import corpora, models, similarities, utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
#from gensim.corpora.mmcorpus import MmCorpus
#from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import WikiCorpus
from lxml import etree
from datetime import datetime

def tokenize(text):
    return [token for token in utils.simple_preprocess(text) if token not in STOPWORDS]

if len(sys.argv) !=2:
    print('Missed arguments. createLdaModel.py <l/r> ');
    sys.exit();

if sys.argv[1] == 'r':
    fileLocation = '/home/priya/wikiVector/'
else:
    fileLocation = '/media/New Volume/Datasets/'

#abstract_file = '/home/shashank/wikiVector/enwiki-latest-abstract.xml'
dump_file = fileLocation+'enwiki-latest-pages-articles.xml'; #'Wikipedia-20130419041326.xml';#

iLinks = {}
CSpages = []
with utils.smart_open(fileLocation+"D_link.txt") as fin:
    for line in fin:
        chunks = utils.to_unicode(line).split('['); # print(len(chunks))
        page = chunks[0].split()[0].strip()
        linksList = chunks[1].replace("]","")
        inlinks = linksList.split(',');  #print page+' - '+str(len(inlinks));
        inlinksCleaned = [ x.strip() for x in inlinks]
        CSpages.extend(inlinksCleaned)
        iLinks[page] = inlinks
print 'ilinks loaded = '+str(len(iLinks.keys()))+' CS pages loaded = '+str(len(CSpages)) ; 

domains =['Computer_vision', 'Information_retrieval', 'Artificial_intelligence', 'Data_mining']
domainIlinks = []
for domain in domains:
    domainIlinks.extend(iLinks.get(domain))
print 'Total ilinks from four domains = '+str(len(domainIlinks))
'''
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
'''

class wikiDocs(object):
    global dump_file
    def __iter__(self):
        pages = corpora.wikicorpus.extract_pages(dump_file)
        for page in pages:
            if not page[0].startswith("File:"):
                pageTitle = utils.to_unicode(page[0])
                pageId = utils.to_unicode(page[2])
                pageTitle = pageTitle.replace(" ", "_").strip()
                #print pageTitle
                if pageTitle in CSpages:
                    #print 'Extracting '+pageTitle
                    yield tokenize(page[1])
        del pages

class wikiDocBow(object):
    global dump_file
    def __iter__(self):
        pages = corpora.wikicorpus.extract_pages(dump_file)
        for page in pages:
            if not page[0].startswith("File:"):
                pageTitle = utils.to_unicode(page[0])
                pageId = utils.to_unicode(page[2])
                pageTitle = pageTitle.replace(" ", "_").strip()
                #print pageTitle
                if pageTitle in CSpages:
                    #print 'Extracting '+pageTitle
                    yield dictionary.doc2bow(tokenize(page[1]))
        del pages

MissingFileCount = 0

class getDocs(object):
    global domainIlinks
    def __iter__(self):
        for domainIlink in domainIlinks:
            pageContents = []
            domainIlink = domainIlink.replace("_"," ").strip()
            if os.path.isfile(fileLocation+'CCCwikipage/'+domainIlink) :
                with utils.smart_open(fileLocation+'CCCwikipage/'+domainIlink) as fin:
                    for line in fin:
                        pageContents.extend(tokenize(line))                        
            yield pageContents

class getDocToBow(object):
    global dictionary
    global domainIlinks
    def __iter__(self):
        global MissingFileCount
        for domainIlink in domainIlinks:
            pageContents = []
            domainIlink = domainIlink.replace("_"," ").strip()
            if os.path.isfile(fileLocation+'CCCwikipage/'+domainIlink) :
                with utils.smart_open(fileLocation+'CCCwikipage/'+domainIlink) as fin:
                    for line in fin:
                        pageContents.extend(tokenize(line))       
            else :
                MissingFileCount += 1
                print 'file not found '+domainIlink                 
            yield dictionary.doc2bow(pageContents)


start=datetime.now()
print 'Building model ...'
documents = wikiDocs()

#build a dictionary which maps between words and index numbers:
dictionary = corpora.Dictionary(documents)
dictionary.save(fileLocation+'cs_lda6.dict')
corpus = wikiDocBow()

#model = Doc2Vec( documents, size=25, window=8, min_count=5, workers=0)
ldaModel = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100) # default size = 25
print 'Out of '+str(len(domainIlinks))+ ' domain pages '+str(MissingFileCount)+' were missing.'
print 'Topics = '+str(len(ldaModel.print_topics(num_topics=10, num_words=10)))
print 'Showing topics = '+str(len(ldaModel.show_topics()))

# store the model to mmap-able files
ldaModel.save(fileLocation+'wiki_model6.ldamodel')#model.save_word2vec_format('/tmp/my_model.doc2vec')
print('Model made in '+str(((datetime.now()-start).total_seconds())/60)+' minutes.')

# load the model back
dictionary_loaded = corpora.Dictionary.load(fileLocation+'cs_lda6.dict')
model_loaded = LdaModel.load(fileLocation+'wiki_model6.ldamodel')#model_loaded = Doc2Vec.load_word2vec_format('/tmp/my_model.doc2vec')
print 'Topics in loaded model = '
print model_loaded.print_topics(num_topics=5, num_words=5)

testDoc = "Social media mining is the process of representing, analyzing, and extracting actionable patterns from social media data. Social media mining introduces basic concepts and principal algorithms suitable for investigating massive social media data; it discusses theories and methodologies from different disciplines such as computer science, data mining, machine learning, social network analysis, network science, sociology, ethnography, statistics, optimization, and mathematics. It encompasses the tools to formally represent, measure, model, and mine meaningful patterns from large-scale social media data"
nvect = dictionary_loaded.doc2bow(tokenize(testDoc))
print nvect
print model_loaded[nvect]
print model_loaded.get_document_topics(nvect, minimum_probability=None)
a = list(sorted(model_loaded[nvect], key=lambda x:x[1]))
print 'last topic '+str(dictionary_loaded[a[0][0]])+' has value '+str(a[0][1])
print 'first topic '+str(dictionary_loaded[a[-1][0]])+' has value '+str(a[-1][1])

'''
tweet = 'Its a nice cool day'
twVector = model_loaded.infer_vector(tweet.split(),alpha=0.1, min_alpha=0.0001, steps=5)
print(twVector)
'''