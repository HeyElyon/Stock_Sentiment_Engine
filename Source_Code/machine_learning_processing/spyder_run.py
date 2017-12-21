import pandas as pd
from MachineLearningNLP import MachineLearningNLP
from PreProcess import word_feats, bigram_word_feats

file_path = '../../Resources/amazon_review_polarity_csv/'
try:
    training_csv
except NameError:
    training_csv = pd.read_csv(file_path+'train.csv',header=None)
    label_dict = {1:'neg',2:'pos'}  
    training_csv.columns = ['label','headline','text']
    training_csv['label'] = training_csv['label'].map(label_dict)

text_in_class = training_csv.groupby('label')['text'].apply(list)
file_neg = text_in_class['neg']
file_pos = text_in_class['pos']

size_train = 30000
file_neg = file_neg[:size_train]
file_pos = file_pos[:size_train]

find_model = False
train_on_stocktwits = False

if find_model:
    for classifier_type in ['NaiveBayes','MaximumEntropy','SVM']:
        for feats in [word_feats, bigram_word_feats]:            
            classifier = MachineLearningNLP(classifier_type,feats)
            
            negfeats, posfeats = classifier.convert_txt(file_neg, file_pos)
            negcutoff = int(len(negfeats)*3/4)
            poscutoff = int(len(posfeats)*3/4)
             
            trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
            testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
            
            if classifier_type=='MaximumEntropy':
                classifier.train(trainfeats,max_iter=2)
            else:
                classifier.train(trainfeats)
            classifier.performance(testfeats)
            
classifier_type = 'NaiveBayes'
feats = bigram_word_feats
classifier = MachineLearningNLP(classifier_type,feats)
negfeats, posfeats = classifier.convert_txt(file_neg, file_pos)
trainfeats = negfeats + posfeats
classifier.train(trainfeats)
stocktwits = pd.read_csv('../../Resources/StockTwists_Data_Sentiment.csv')
twit_in_class = stocktwits.groupby('sentiment')['body_clean'].apply(list)
twit_neg = twit_in_class['neg']
twit_pos = twit_in_class['pos']

if train_on_stocktwits:
    classifier_type = 'NaiveBayes'
    feats = bigram_word_feats
    classifier = MachineLearningNLP(classifier_type,feats)
    negtwits, postwits = classifier.convert_txt(twit_neg, twit_pos)
    negcutoff = int(len(negtwits)*3/4)
    poscutoff = int(len(postwits)*3/4)
    trainfeats = negtwits[:negcutoff] + postwits[:poscutoff]
    testfeats = negtwits[negcutoff:] + postwits[poscutoff:]
    classifier.train(trainfeats)
else:
    negtwits, postwits = classifier.convert_txt(twit_neg, twit_pos)
    testfeats = negtwits + postwits
classifier.performance(testfeats)
