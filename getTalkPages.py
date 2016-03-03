import xml.etree.cElementTree as ET
from pdb import set_trace
i=0
def process_buffer(buf):
    tnode = ET.fromstring(buf)
    #print buf
    with open('talk_pages/{0}.xml'.format(tnode.find('id').text), 'w') as fo:
        fo.write(buf)
    #pull it apart and stick it in the database

inputbuffer = ''
with open('enwiki-20160204-pages-meta-current.xml','rb') as inputfile:
    append = False
    talk = False
    for line in inputfile:
        if '<page>' in line:
            inputbuffer = line
            append = True
        elif '<ns>1</ns>' in line:
            talk = True
        elif '</page>' in line:
            inputbuffer += line
            append = False
            if talk:
                print i
                i+=1
                process_buffer(inputbuffer)
            talk = False
            inputbuffer = None
            del inputbuffer #probably redundant...
        elif append:
            inputbuffer += line