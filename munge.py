import re
from uuid import uuid1
from bs4 import BeautifulSoup
from pdb import set_trace
user = re.compile(r'\[\[User\:(\w+.*?)\|')
ipaddress = re.compile(r'\[\[Special\:Contributions\/((?:[0-9]{1,3}\.){3}[0-9]{1,3})')
ipaddress_user = re.compile(r'\[\[User\:((?:[0-9]{1,3}\.){3}[0-9]{1,3})\|')

def extractTopic(line):
    return line.strip('=').strip()

def parseLine(line):
    #set_trace()
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
    if user.findall(line):
        return (depth, user.findall(line)[-1])
    elif ipaddress.findall(line):
        return (depth, ipaddress.findall(line)[-1])
    elif ipaddress_user.findall(line):
        return (depth, ipaddress_user.findall(line)[-1])
    else:
        return (depth, None)
    
def parsePost(post,author,postID,replyToID):
    return {'post':'\n'.join(post),'author':author,'postID':postID,'replyToId':replyToID}

def parseTopic(topic):
    # TODO: People commenting at depth=0 are probably replying to the highest depth=0 comment
    #set_trace()
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
            postID = uuid1()
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
    #set_trace()
    soup = BeautifulSoup(xml_file, 'xml')
    txt = soup.find('text').text
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
    return page

