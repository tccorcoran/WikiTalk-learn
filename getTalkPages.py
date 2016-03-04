from bs4 import BeautifulSoup
from pdb import set_trace
import sys
import os

ROOT_DIR = os.path.split(os.path.realpath(__file__))[0]
TALK_FILES_DIR = os.path.join(ROOT_DIR,'data/talk_pages')

def processBuffer(buf):
    """
    Write each talk page as an XML file to talk_pages dir
    """
    tnode = BeautifulSoup(buf,'xml')
    empty = True # lots of talk pages have no dilogue/ are made by bots, don't include these
    for line in tnode.find('text').text.splitlines():
        if line.startswith('=='):
            empty = False
            break
    if not empty:
        with open(os.path.join(TALK_FILES_DIR,'{0}.xml'.format(tnode.find('id').text)), 'w') as fo:
            fo.write(buf)

def selectTalkPage(wiki_file):
    
    # Ensure we can write out our talk page xml files to their own dir
    if not os.path.exists(TALK_FILES_DIR):
        os.makedirs(TALK_FILES_DIR)
    i=0
    inputbuffer = ''
    with open(wiki_file,'rb') as inputfile:
        append = False
        talk = False
        for line in inputfile:
            if '<page>' in line:
                inputbuffer = line
                append = True
            elif '<ns>1</ns>' in line: # namespace 1 is a talk page
                talk = True
            elif '</page>' in line:
                inputbuffer += line
                append = False
                if talk:
                    print "Processing talk page:", i
                    i+=1
                    processBuffer(inputbuffer)
                talk = False
                inputbuffer = None
                del inputbuffer 
            elif append:
                inputbuffer += line

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print "Trying default wiki meta file: enwiki-20160204-pages-meta-current.xml"
        selectTalkPage(os.path.join(ROOT_DIR,"data/enwiki-20160204-pages-meta-current.xml"))
    else:
        print "Using wiki meta file: ", sys.argv[1]
        selectTalkPage(os.path.join(ROOT_DIR, sys.argv[1]))
