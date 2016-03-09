# WikiTalk-learn
Utilities to create a corpus from the Wikipedia talk pages, and run some interesting machine learning / NLP tasks on the corpus.

Dependencies
------------
* Python 2.7.x
* sklearn >= 0.17
* beautifulsoup4 == 4.4.1


To generate the corpus yourself you need a copy of a Wikipedia metadata database dump, available [here](https://dumps.wikimedia.org/enwiki/).The metadata file with the talk pages looks like: `enwiki-*-pages-meta-current.xml.bz2`. This project is using [enwiki-20160204-pages-meta-current.xml.bz2](https://dumps.wikimedia.org/enwiki/20160204/).

TODOs
------------
* preprocess text
  * remove/standardize wiki markup (links,signatures)
* collect corpus stats
  * vocab size
  * author info
