import nltk
import numpy as np
import pandas as pd
from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:9000')
stocktwits = pd.read_csv('../../Resources/StockTwists_Data_Sentiment.csv')
prediction = []
for curr_msg in stocktwits.body_clean:
    res = nlp.annotate(curr_msg,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 1000,
                       })
    prediction.append(np.mean([float(s["sentimentValue"]) for s in res["sentences"]]))

    
threshold = 1.5
pos_loc = set([i for i in range(len(prediction)) if prediction[i]>threshold])
neg_loc = set([i for i in range(len(prediction)) if prediction[i]<threshold])
pos_ref = set(stocktwits[stocktwits['sentiment']=='pos'].index)
neg_ref = set(range(len(prediction))) - pos_ref
print ('===============================\n')
print ('Model Summary:\n')
print ('Overall Accuracy: %.3f\n'%(sum((np.array(prediction)>threshold).astype(int)-(np.array(prediction)<threshold).astype(int)==stocktwits['sentiment'].map({'pos':1,'neg':-1}))/len(prediction)))
print ('Positive Precision: %.3f\n'%(nltk.precision(pos_ref, pos_loc)))
print ('Positive Recall: %.3f\n'%(nltk.recall(pos_ref, pos_loc)))
print ('Negative Precision: %.3f\n'%(nltk.precision(neg_ref, neg_loc)))
print ('Negative Recall: %.3f\n'%(nltk.recall(neg_ref, neg_loc)))