import os
import json
from glob import glob
from collections import Counter

from gensim import corpora

ROOT_DIR = os.path.split(os.path.realpath(__file__))[0]
TALK_FILES_EXTRACTED_DIR = os.path.join(ROOT_DIR,'data/talk_pages_structured_json')
g = glob(os.path.join(TALK_FILES_EXTRACTED_DIR,'*.json'))

def authorStats(g):
    """ Collect the most prolific authors by number of chars"""
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
    authors = sorted([(k,v) for k, v in authors.iteritems()],key=lambda x:x[1],reverse=True)
    # TODO: actually check if author is a bot against this list: https://en.wikipedia.org/wiki/Wikipedia:Bots/Status
    # NOTE: Not all bots have 'bot' in the username
    no_bots = [x for x in authors if 'bot' not in x[0].lower()]
    with open(os.path.join(ROOT_DIR,'data/authors.json'),'wb') as fo:
        json.dump(no_bots,fo)
        

def createCleanCorpus(g,num_top_authors):
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
                            text = post['post']
                            cleanText(text)


def cleanText(text):
    title = re.compile(r'^===?[\w\s\.\:]*===?') # Topic titles
    reply = re.compile(r'^\:*') 
    user = re.compile(r'\[\[User.*?\]\]')
    special = re.compile(r'\[\[Special\:Contributions\/.*?\]\]')
    html_tags =  re.compile(r'<[^>]*>') # XHMTL tags
    wiki_links = re.compile(r'\[\[(\w+\:.*?\|)') # first capuring group
    brackets = re.compile(r'(\[\[|\]\])')
    
    

def createVocabDict(g,num_top_authors):
    print "Loading Tokenizer..."
    from spacy.en import English
    nlp = English(tagger=None,parser=None,entity=None)
    print "Done!"
    with open(os.path.join(ROOT_DIR,'data/authors.json')) as fi:
        authors = json.load(fi)
    authors = authors[:num_top_authors]
    author_list = set(x[0] for x in authors)
    texts = []
    print "Loading objects into dictionary..."
    for p in g:
        print "\tProcessing {}".format(p)
        with open(p) as fi:
            page = json.load(fi)
        if len(page[0]) > 0:
            for topic in page[0]:
                if len(topic) > 0:
                    for post in topic['posts']:
                        text = post['post']
                        author = post['author']
                        if author in author_list:
                            doc = nlp(text)
                            texts.append([tok.orth_ for sentence in doc.sents for tok in sentence])
    print "Done!\nOrganizing and writing dictionary out..."
    dictionary = corpora.Dictionary(texts)
    dictionary.save(os.path.join(ROOT_DIR,'data/vocab_{}_authors.dict'.format(num_top_authors)))
    print "Done!"
