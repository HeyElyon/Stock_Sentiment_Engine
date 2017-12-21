import sys
import nltk
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from sklearn.svm import LinearSVC
from PreProcess import word_preprocess, word_feats, bigram_word_feats


class MachineLearningNLP:

    def __init__(self,classifier_type='NaiveBayes',feats=word_feats):
        # "Thumbs up? Sentiment Classification using Machine Learning Techniques
        classifier_list = ['NaiveBayes','MaximumEntropy','SVM']
        if classifier_type in classifier_list:
            self.classifier_type=classifier_type
        else:
            print("Classifier Type is not implemented: "+ classifier_type) 
        if self.classifier_type == 'MaximumEntropy':
            self.classifier = MaxentClassifier
        elif self.classifier_type == 'SVM':
            self.classifier = SklearnClassifier(LinearSVC(), sparse=False)
        elif self.classifier_type == 'NaiveBayes':
            self.classifier = NaiveBayesClassifier
        self.feats = feats

    def convert_txt(self, file_neg, file_pos):
        negfeats = list(map(self.feats, word_preprocess(file_neg)))
        posfeats = list(map(self.feats, word_preprocess(file_pos)))
        
        negfeats = list(zip(negfeats,['neg']*len(negfeats)))
        posfeats = list(zip(posfeats,['pos']*len(posfeats)))
#        negfeats = [(self.feats(f), 'neg') for f in word_preprocess(file_neg)]
#        posfeats = [(self.feats(f), 'pos') for f in word_preprocess(file_pos)]
        return (negfeats,posfeats)

    def train(self, train_data, **kwargs):
        self.classifier = self.classifier.train(train_data,**kwargs)
        
    def predict(self, test_data):
        return [self.classifier.classify(feats) for feats, label in test_data]
    
    def annotate(self, text):
        assert isinstance(text, str)
        text_Encoded = self.feats(text.split())
        return self.classifier.classify(text_Encoded)
    
    def performance(self, test_data):
        prediction = self.predict(test_data)
        pos_loc = set([i for i in range(len(prediction)) if prediction[i]=='pos'])
        neg_loc = set(range(len(prediction))) - pos_loc
        pos_ref = set([i for i in range(len(prediction)) if test_data[i][1]=='pos'])
        neg_ref = set(range(len(prediction))) - pos_ref
        print ('===============================\n')
        print ('Model Summary:\n')
        print (self.classifier_type+' with features '+self.feats.__name__+'\n')
        print ('Overall Accuracy: %.3f\n'%(nltk.classify.util.accuracy(self.classifier,test_data)))
        print ('Positive Precision: %.3f\n'%(nltk.precision(pos_ref, pos_loc)))
        print ('Positive Recall: %.3f\n'%(nltk.recall(pos_ref, pos_loc)))
        print ('Negative Precision: %.3f\n'%(nltk.precision(neg_ref, neg_loc)))
        print ('Negative Recall: %.3f\n'%(nltk.recall(neg_ref, neg_loc)))
        