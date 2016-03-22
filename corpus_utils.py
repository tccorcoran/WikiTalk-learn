"""
Utils to create ML-ready corpora from files generated by munge.py
"""
from __future__ import unicode_literals
from multiprocessing import Manager, Pool
from pdb import set_trace
from glob import glob
from collections import Counter
import numpy as np
import codecs
import os
import json

import regex as re

ROOT_DIR = os.path.split(os.path.realpath(__file__))[0]
# list of json files to work with
# needs to be generated by `./munge.py`
TALK_FILES_EXTRACTED_DIR = os.path.join(ROOT_DIR,'data/talk_pages_structured_json')
g = glob(os.path.join(TALK_FILES_EXTRACTED_DIR,'*.json')) 


class MyCorpus(object):
    """
    Class for reading corpus files generated by createCleanCorpus
    corpus is chosen by n_authors
    """
    def __init__(self,num_authors,start=0,stop=float('inf'),vector=''):
        if vector:
            vector = '.vector'
        self.ROOT_DIR = os.path.split(os.path.realpath(__file__))[0]
        self.corpus_file = os.path.join(self.ROOT_DIR,'data/{}_authors.corpus{}'.format(num_authors,vector))
        self.start = start # which line to start reading from
        self.stop = stop
        self.length = None
    def __len__(self):
        """ Number of posts in corpus """
        if self.length is None:
            self.length = 0
            for line in self:
                self.length += 1
        return self.length
    def __iter__(self):
        "output: tokenize post where line[0] is author followed by tokens"
        i = 0
        with codecs.open(self.corpus_file,'r','utf8') as fi:
            for line in fi:
                if i >= self.start and i < self.stop and not line.startswith('\n'):
                    yield line.split()
                    i += 1

def authorStats(g):
    """
    Collect the most prolific authors by number of chars. Use this to create corpora
    based on number of authors
    """
    authors = Counter()
    for p in g:
        with open(p) as fi:
            page = json.load(fi)
        if len(page[0]) > 0:
            for topic in page[0]:
                if len(topic) > 0:
                    for post in topic['posts']:
                        author = post['author']
                        text = post['post']
                        authors[author] += len(text)
    authors = sorted([(k,v) for k, v in authors.iteritems()],key=lambda x:x[1],reverse=True) # sort most prolific authors decending
    
    # TODO: actually check if author is a bot against this list: https://en.wikipedia.org/wiki/Wikipedia:Bots/Status
    no_bots = [x for x in authors if 'bot' not in x[0].lower()]    # NOTE: Not all bots have 'bot' in the username
    with open(os.path.join(ROOT_DIR,'data/authors.json'),'wb') as fo:
        json.dump(no_bots,fo)
        
# Some of the below regex were taken from wikifil.pl by Matt Mahoney (http://mattmahoney.net/dc/textdata.html)
title = re.compile(r'^===?[^=]*===?',flags=re.M) # Topic titles
and_ = re.compile(r'&amp;') # decode URL encoded chars
lt = re.compile(r'&lt;')
gt = re.compile(r'&gt;')
ref = re.compile(r'<ref[^<]*<\/ref>') # remove references <ref...> ... </ref>
curly = re.compile(r'\{\{[^\{]*?\}\}') # curly markup brackets
reply = re.compile(r':',flags=re.M)  # ::: reply structure
user = re.compile(r'\[\[User.*?\]\]') # usernames
special = re.compile(r'\[\[Special\:Contributions\/.*?\]\]') # more usernames 
html_tags =  re.compile(r'<[^>]*>') # XHMTL tags
wiki_links = re.compile(r'\[\[[^\|\]]*\|')  # remove wiki url, preserve visible text
brackets = re.compile(r'(\[\[?|\]\]?)') # remove brackets
link = re.compile(r'https?:?//.*?\s') # links (sometimes people forget the :)
url_encoded = re.compile(r'&[^;]*;') # remove URL encoded chars
def cleanText(text):
    """ clean the text to be more parsable """
    text = text.strip()
    text = text.replace('\n', ' ')
    text= title.sub('',text)
    text = and_.sub('&',text)
    text = lt.sub('<',text)
    text = gt.sub('>',text)
    text = curly.sub('',text) 
    text = curly.sub('',text) # double nested curly brackets
    text = ref.sub('',text)
    text= reply.sub('',text)
    text= special.sub('',text)
    text= user.sub('*USER*',text)
    text= html_tags.sub('',text)
    text= wiki_links.sub('[[',text)
    text= brackets.sub('',text)
    text = link.sub('*LINK*  ',text)
    text = url_encoded.sub(' ',text)
    text = text.strip()
    return text
        
spaces = re.compile(r'\s')
def createCleanCorpus(g,q,num_top_authors):
    """
    output corpus file with cleaned post text with only num_top_authors
    author's name immediately precedes text with whitespace removed. One post per line
    
    Corpus format (utf-8):
    author tokenize text separated by whitespace . newlines removed
    
    e.g.
    
    Moonriddengirl thanks , * user*. your review is certainly sufficient to resolve my concerns . ) --*user * * user * 1330 , 10 november 2014 ( utc )
    
    SlimVirgin when we say he is a " doctor , " what do we mean exactly ? is he employed as a physician , and if so , where ? * user * * user * 2053 , 12 february 2007 ( utc )
    
    """
    with open(os.path.join(ROOT_DIR,'data/authors.json')) as fi:
        authors = json.load(fi)
    authors = authors[:num_top_authors]
    author_list = set(x[0] for x in authors)
    for p in g:
        with open(p) as fi:
            page = json.load(fi)
        if len(page[0]) > 0:
            for topic in page[0]:
                if len(topic) > 0:
                    for post in topic['posts']:
                        author = post['author']
                        if author in author_list:
                            author = spaces.sub('', post['author'])
                            text = post['post']
                            text = tokenize(cleanText(text))
                            text = ' '.join(tok.lower_ for s in text.sents for tok in s).replace('\n', ' ') # seperate toks by whitespace, remove newlines
                            line = '{} {}\n\n'.format(author,text)
                            q.put(line)


def poolCleanCorpus(g, n_authors, n_jobs):
    """
    Process corpus in parallel, sends completed work to queue to be written to single corpus file
    
    g: list of munged corpus jsons
    n_auhtors: number of most prolific authors to limit corpus to
    n_jobs: split into how many processes
    
    Output file name: 'data/{}_authors.corpus'.format(n_authors)
    """
    manager = Manager()
    q = manager.Queue()
    pool = Pool(n_jobs)
    
    cleaned_corpus_file = os.path.join(ROOT_DIR,'data/{}_authors.corpus'.format(n_authors))
    watcher = pool.apply_async(listener, (q,cleaned_corpus_file,))
    
    jobs = []
    #set_trace()
    for chunk in chunks(g,len(g)/n_jobs):
        #set_trace()
        job = pool.apply_async(createCleanCorpus, (chunk, q, n_authors))
        jobs.append(job)
    #collect results from the workers through the pool result queue
    for job in jobs: 
        job.wait()

    #now we are done, kill the listener
    q.put('**EOF**')
    pool.close()
    pool.join()
    
def chunks(l, n):
    """
    split a list into even chunks: http://stackoverflow.com/a/1751478
    """
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def listener(q,fp):
    """
    Listens for input into a queue,q, and writes to a file, fp
    Use for writing to single corpus file being generated on mutiple threads
    i.e. poolCleanCorpus & 
    """
    fo = codecs.open(fp,'wb','utf-8')
    while True:
        m = q.get()
        if m == '**EOF**':
            break
        fo.write(m)
        fo.flush()
    fo.close()

def addToDict(d, start, stop, n_author):
    """
    put each token into a dict with a unique id
    """
    corpus = MyCorpus(n_authors,start=start,stop=stop)
    for post in corpus:
        tokens = set(post)
        for token in tokens:
            if token not in d:
                d[token] = len(d)+1 # give each token a unique ID,
                                    # save 0 for padding ID
                
def poolCreateVocabDict(n_authors,n_jobs):
    """
    Create a vocabulary for the corpus
    
    n_authors: corpus file to read. Looks for 'data/{}_authors.corpus'.format(num_top_authors)
    n_jobs:  number of jobs to run. Splits corpus into n_jobs to run tokenizer in parallel.
           
    output: vocab -- dict where each word in the corpus lexicon has a unique id
            written to 'data/vocab_{}_authors.dict'.format(n_authors)
    """
    len_corpus = len(MyCorpus(n_authors))# read corpus and figure this out
    m=Manager() # shared between processes
    vocab = m.dict()
    pool = Pool(n_jobs)
    
    jobs = []
    for start in range(0,len_corpus,len_corpus/n_jobs):
        if (len_corpus-start) < (len_corpus/n_jobs):
            stop = len_corpus + 1 # make sure the last thread reads the end of the corpus 
        else:
            stop = start+len_corpus/n_jobs
            
        job = pool.apply_async(addToDict, (vocab, start, stop, n_authors))
        jobs.append(job)

    for job in jobs: 
        job.wait()
        
    vocab = dict(vocab)
    
    with open(os.path.join(ROOT_DIR,'data/vocab_{}_authors.dict'.format(n_authors)),'wb') as fo:
        json.dump(vocab,fo)
    return vocab, len(vocab)
    pool.close()
    pool.join()

def vectorize(vocab, q, start, stop, n_authors):
    """
    Turn row of tokens into a vector, each token matched to a corpus-unique id
    """
    corpus = MyCorpus(n_authors,start=start,stop=stop)
    for post in corpus:
        line = []
        for token in post:
            line.append(str(vocab[token]))
        data = ' '.join(line)
        q.put('{}\n'.format(data))

        

def createVectorCorpus(n_authors,n_jobs):
    """
    write turn a .corpus file into .corpus.vector
    each line is a document, tokens are mapped to unique IDs
    author is first id on line
    """
    vocab =  loadVocab(n_authors)
    vector_corpus_file = os.path.join(ROOT_DIR,'data/{}_authors.corpus.vector'.format(n_authors))
    len_corpus = len(MyCorpus(n_authors))# read corpus and figure this out
    m=Manager() # shared between processes
    q = m.Queue()
    pool = Pool(n_jobs)
    
    watcher = pool.apply_async(listener, (q,vector_corpus_file,))

    jobs = []
    for start in range(0,len_corpus,len_corpus/n_jobs):
        if (len_corpus-start) < (len_corpus/n_jobs):
            stop = len_corpus + 1 # make sure the last thread reads the end of the corpus 
        else:
            stop = start+len_corpus/n_jobs
            
        job = pool.apply_async(vectorize, (vocab, q, start, stop, n_authors))
        jobs.append(job)

    for job in jobs: 
        job.wait()
        
    q.put('**EOF**')
    pool.close()
    pool.join()

def loadVocab(n_authors):
    """
    Load vocab dict for the given corpus specified by n_authors
    """
    with open(os.path.join(ROOT_DIR,'data/vocab_{}_authors.dict'.format(n_authors)),'rb') as fo:
        vocab =  json.load(fo)
    return vocab

def getLongestDoc(n_authors):
    """
    Returns the max number of tokens ( in a doc) over the corpus specified by n_authors
    """
    corpus = MyCorpus(n_authors,vector=True)
    i = 0
    for line in corpus:
        if len(line[1:]) > i:
            i = len(line[1:])
    return i

def loadData(n_authors,return_onehot=True):
    """
    Load data into np arrays
    """
    X = [] # features
    y = [] # labels
    authors = {}
    padding = getLongestDoc(n_authors)
    corpus = MyCorpus(n_authors,vector=True)
    
    for line in corpus:
        vec = [int(value) for value in line[1:]]
        while len(vec) < padding: # add padding to each doc
            vec.append(0)
        X.append(np.array(vec)) # add feature vector to data
        
        # Create one hot vector for label
        one_hot = [0]*n_authors
        author = int(line[0])
        if not author in authors:
            authors[author] = len(authors)
        one_hot[authors[author]] = 1
        if return_onehot:
            y.append(np.array(one_hot))
        else:
            y.append(authors[author])
    return np.array(X),np.array(y)

def loadDataNB(n_authors):
    X = [] # features
    y = [] # labels
    authors = {}

    corpus = MyCorpus(n_authors,vector=True)
    for line in corpus:
        vec = [int(value) for value in line[1:]]
        X.append(vec)
        author = int(line[0])
        if not author in authors:
            authors[author] = len(authors)
        y.append(authors[author])
    return X,y

def traindevtestSplit(X,y):
    """
    Split data into three sets: training, validation (for testing during traning)
    and test (for testing at the very end of training)
    """
    train_percent = 0.8
    dev_percent = 0.1
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    train_end = int(train_percent*len(y))
    dev_part = int(dev_percent*len(y))
    
    x_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    x_train, x_dev,x_test = x_shuffled[:train_end], x_shuffled[train_end:train_end+dev_part],x_shuffled[train_end+dev_part:]
    y_train, y_dev, y_test = y_shuffled[:train_end], y_shuffled[train_end:train_end+dev_part],y_shuffled[train_end+dev_part:]
    
    return ((x_train, x_dev, x_test),(y_train, y_dev, y_test))

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    # This code is modified from the original found here: github.com/dennybritz/cnn-text-classification-tf
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
    

if __name__ == '__main__':
#    authorStats(g)
    from spacy.en import English
    tokenize = English(tagger=None,parser=None,entity=None)
    author_sizes = (10,25,50,100)
    print "Creating parsed corpus..."
    for n_authors in author_sizes:
        print "\n\tworking on {} authors\n".format(n_authors)
        poolCleanCorpus(g,n_authors,8)
    print "Done!\n\nCreating vocabulary..."
    for n_authors in author_sizes:
        print "\n\tworking on {} authors\n".format(n_authors)
        poolCreateVocabDict(n_authors,8)
    print "Done!\n\nCreating vectorized corpus..."
    for n_authors in author_sizes:
        print "\n\tworking on {} authors\n".format(n_authors)
        createVectorCorpus(n_authors,8)
    print "Done"
