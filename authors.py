# TODO: be able to select authors by 
#   select all posts by those authors

authors = Counter()
for p in g:
    with open(p) as fi:
        page = json.load(fi)
    for topic in page[0]:
        for post in topic['posts']:
            author = post['author']
            text = post['post']
            authors[author] += len(text)
            print authors[author]

