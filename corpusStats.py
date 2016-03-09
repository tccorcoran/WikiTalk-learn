# TODO: be able to select authors by 
#   select all posts by those authors
import os
import json
from glob import glob
from collections import Counter

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
    with open(os.path.join(ROOT_DIR,'data/authors.json'),'wb') as fo:
        json.dump(authors,fo)
