import regex as re
import os
import json
from uuid import uuid1
from bs4 import BeautifulSoup
from pdb import set_trace
from glob import glob
from multiprocessing import Pool


# TODO: import dir structure from other file,
ROOT_DIR = os.path.split(os.path.realpath(__file__))[0]
TALK_FILES_RAW_DIR = os.path.join(ROOT_DIR,'data/talk_pages_raw_xml')
TALK_FILES_EXTRACTED_DIR = os.path.join(ROOT_DIR,'data/talk_pages_structured_json')

# Signature patterns (wiki links to users talk pages)
user = re.compile(r'\[\[User\:(\w+.*?)[\|\/]')
ipaddress = re.compile(r'\[\[Special\:Contributions\/((?:[0-9]{1,3}\.){3}[0-9]{1,3})')
ipaddress_user = re.compile(r'\[\[User\:((?:[0-9]{1,3}\.){3}[0-9]{1,3})\|')

def extractTopic(line):
    """
    Return the topic of a discusion, marked by == Discussion Topic ==
    """
    return line.strip('=').strip()

def parseLine(line):
    """
    Take in a single line of a talk page, return how deep in the discussion thread it is, and if it is signed
    """
    depth = 0 
    if not line.startswith(':'):
        pass
    else:
        for char in line:
            if char == ':':
                depth += 1
            else:
                break
    # signatures
    # [[User:username|
    # [[Special:Contributions/192.168.1.1|
    # NOTE/TODO: This will fail in cases where a user is mentioned in the post, but the post is signed by an IP address
    author = user.findall(line, overlapped=True)
    if author:     
        return (depth, author[-1])
    author = ipaddress.findall(line, overlapped=True)
    if author:
        return (depth, author[-1])
    author = ipaddress_user.findall(line, overlapped=True)
    if author:
        return (depth, author[-1])
    else:
        return (depth, None)
    
def parsePost(post,author,postID,replyToID):
    return {'post':'\n'.join(post),'author':author,'postID':postID,'replyToId':replyToID}

def parseTopic(topic):
    """
    Seperate replies within a single dicussion topic
    """
    conversation = {}
    conversation['topic'] = extractTopic(topic[0])
    posts = []
    post = []
    depth = {}
    top_level = None
    for line in topic:
        parsed = parseLine(line)
        post.append(line)
        if parsed[1]:
            # end of post / found signature on line
            postID = str(uuid1())
            depth[parsed[0]] = postID
            # Here's how we figure out the reply structure:
            # As we walk down the tree we keep track of the depth of the post
            # and store only the most recent post at each depth
            # Consider that each post is a reply to the most immediate post above at depth - 1
            if parsed[0]-1 in depth:
                replyToId = depth[parsed[0]-1] # find parent post: the most recent post at depth - 1
            else:
                if top_level:
                    replyToId = top_level # must be depth = 0 post, proably replying to the inital post
                else:
                    top_level = postID # intial post under topic
                    replyToId = top_level # on intial posts, replyToID is the same as postID
            posts.append(parsePost(post,parsed[1],postID,replyToId))
            post = []
    if not posts:
        return []
    conversation['posts'] = posts
    return conversation
    
    
def parsePage(xml_file):
    """
    Generate a list of discussions held on a wikipedia talk page
    """
    soup = BeautifulSoup(xml_file, 'xml')
    txt = soup.find('text').text
    pageID = soup.find('id').text
    page_title = soup.find('title').text
    first_topic = True
    topic = []
    page = []
    for line in txt.splitlines():
        if line.startswith('=='):
            # new post heading
            # If this this the fist line of the fist topic, trash all the metadata-looking stuff
            # or unorganized chatter we've collected so far in topic[]
            if not first_topic:
                page.append(parseTopic(topic))
            topic = []
            first_topic = False
        topic.append(line)
    if topic:
        page.append(parseTopic(topic)) # parse the last topic
    return (page,pageID,page_title)


def extract_and_dump(talk_page):
    """
    Read in a raw xml file, output a json with extracted posts
    """
    with open(talk_page,'rb') as fi:
        parsed_page = parsePage(fi.read())
    with open(os.path.join(TALK_FILES_EXTRACTED_DIR,parsed_page[1]+'.json'),'wb') as fo:
        json.dump(parsed_page,fo)
            

if __name__ == '__main__':
    if not os.path.exists(TALK_FILES_EXTRACTED_DIR):
        os.makedirs(TALK_FILES_EXTRACTED_DIR)
    g = glob(os.path.join(TALK_FILES_RAW_DIR,'*.xml'))
    pool = Pool(processes=8)
    pool.map(extract_and_dump,g)
    
    
    
            