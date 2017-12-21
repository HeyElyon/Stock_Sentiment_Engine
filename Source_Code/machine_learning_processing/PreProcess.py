import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def read_stopword():
    stopset_dir = '../Resources/lemur-stopwords.txt'
    with open(stopset_dir) as myFile:
        stopset = myFile.read().split('\n') 
    return stopset

def word_preprocess(file):
    return [paragraph.split() for paragraph in file]
    
    
def word_feats(words, stopset=read_stopword()):
    return dict([(word, True) for word in words if word not in stopset])
    
def bigram_word_feats(words, stopset=read_stopword(), score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)  
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])