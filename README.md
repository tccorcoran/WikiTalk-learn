# WikiTalk-learn
Utilities to create a corpus from the Wikipedia talk pages, and run some interesting machine learning / NLP tasks on the corpus. The only ML experiment here now is an author classifier--the classifier can be trained to determine the author of talk page posts with 94% accuracy. The classifier uses 2 layer convolutional neural net build in TensorFlow.
Dependencies
------------
* Python 2.7.x
* sklearn==0.17
* beautifulsoup4==4.4.1
* regex==2016.3.2
* tensorflow==0.7.0

To generate the corpus yourself you need a copy of a Wikipedia metadata database dump, available [here](https://dumps.wikimedia.org/enwiki/).The metadata file with the talk pages looks like: `enwiki-*-pages-meta-current.xml.bz2`. This project is using [enwiki-20160204-pages-meta-current.xml.bz2](https://dumps.wikimedia.org/enwiki/20160204/).

Procedure
---------

* To generate the corpus yourself you need a copy of a Wikipedia metadata database dump, [enwiki-20160204-pages-meta-current.xml.bz2](https://dumps.wikimedia.org/enwiki/20160204/). Download this to the `data/` dir and extract the bzip2 xml file.
* run `python extratFromWikiDump.py` to generate a dir of extracted talk xml files (very large > 1 million files)
* run `python munge.py` to generate json files with some metadata generated from some heuristics (reply structure, author, post topic)
* run `python corpus_utils.py` to generate vectorized corpus files
* run `python main.py` to start the tensorflow learning experiment


**Note:** This repo contains all the corpus files needed to train the learner on a corpus of 10 authors. The only command needed to run the learning experiments is `./main.py`. The parameters can be changed with the appropriate flags. See `./main.py --help' for help on the available flags. 