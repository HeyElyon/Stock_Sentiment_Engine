import sys
import argparse
import pandas as pd
from MachineLearningNLP import MachineLearningNLP
from PreProcess import word_feats, bigram_word_feats

def get_command_arguments():
    '''
    Read command line input and set values to arguments.
    :return: a list of arguments
    '''
    parser = argparse.ArgumentParser(description='Machine Learning Sentiment Calculator')
    parser.add_argument('--trainingpath', '-tp', type=str, dest='trainingpath', action='store',
                        default='../Resources/amazon_review_polarity_csv/',
                        help="The folder contains the training dataset.")

    parser.add_argument('--traininglength', '-tl', type=int, dest='traininglength', action='store',
                        default=1000,
                        help="The length (number of data points) to use in training set.")

    parser.add_argument('--trainingsource', '-ts', type=str, dest='trainingsource', action='store',
                        default='Amazon',
                        help="The data source for training set. Choose from Amazon or Stocktwits.")

    parser.add_argument('--testingdata', '-td', type=str, dest='testingdata', action='store',
                        default='Stocktwits',
                        help="The testing data trained model. Choose from Amazon or Stocktwits.")

    
    parser.add_argument('--classifiertype', '-ct', type=str, dest='classifiertype', action='store',
                        default='NaiveBayes',
                        help="The Classifier Type. Choose from [NaiveBayes,MaximumEntropy,SVM]")

    parser.add_argument('--feats', '-f', type=str, dest='feats', action='store',
                        default='word_feats',
                        help="The feats type. Choose from [word_feats, bigram_word_feats]")


    args = parser.parse_args()
    return args

def main():
    args = get_command_arguments()
    file_path = args.trainingpath
    size_train = args.traininglength
    data_source = args.trainingsource
    testing_set_name = args.testingdata
    classifier_type = args.classifiertype
    feats = eval(args.feats)
    
    if classifier_type=='SVM' and args.feats=='bigram_word_feats':
        print('SVM using bigram is not implemented! Please change to word_feats. Exiting program...')
        sys.exit(0)

    if data_source=='Amazon':    
        training_csv = pd.read_csv(file_path+'train.csv',header=None)
        label_dict = {1:'neg',2:'pos'}  
        training_csv.columns = ['label','headline','text']
        training_csv['label'] = training_csv['label'].map(label_dict)
    
        text_in_class = training_csv.groupby('label')['text'].apply(list)
        file_neg = text_in_class['neg']
        file_pos = text_in_class['pos']

        cutoff_size = int(size_train*2/3)
        
        file_neg = file_neg[:cutoff_size]
        file_pos = file_pos[:cutoff_size]    
          
        classifier = MachineLearningNLP(classifier_type,feats)
        
        negfeats, posfeats = classifier.convert_txt(file_neg, file_pos)
        negcutoff = int(len(negfeats)*3/4)
        poscutoff = int(len(posfeats)*3/4)
         
        trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
        if classifier_type=='MaximumEntropy':
            classifier.train(trainfeats,max_iter=2)
        else:
            classifier.train(trainfeats)
            
        if testing_set_name=='Amazon':
            testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
        elif testing_set_name=='Stocktwits':
            stocktwits = pd.read_csv('../Resources/StockTwists_Data_Sentiment.csv')
            twit_in_class = stocktwits.groupby('sentiment')['body_clean'].apply(list)
            twit_neg = twit_in_class['neg']
            twit_pos = twit_in_class['pos']
            negtwits, postwits = classifier.convert_txt(twit_neg, twit_pos)
            testfeats = negtwits + postwits
        else:
            print('Error! Testing set can only choose from Amazon or Stocktwits.')
        classifier.performance(testfeats)
            
        
        
    elif data_source=='Stocktwits':
        if testing_set_name=='Stocktwits':
            stocktwits = pd.read_csv('../Resources/StockTwists_Data_Sentiment.csv')
            twit_in_class = stocktwits.groupby('sentiment')['body_clean'].apply(list)
            twit_neg = twit_in_class['neg']
            twit_pos = twit_in_class['pos']
            
            classifier = MachineLearningNLP(classifier_type,feats)
            
            negtwits, postwits = classifier.convert_txt(twit_neg, twit_pos)
            negcutoff = int(len(negtwits)*3/4)
            poscutoff = int(len(postwits)*3/4)
            trainfeats = negtwits[:negcutoff] + postwits[:poscutoff]
            testfeats = negtwits[negcutoff:] + postwits[poscutoff:]
            if classifier_type=='MaximumEntropy':
                classifier.train(trainfeats,max_iter=2)
            else:
                classifier.train(trainfeats)
            classifier.performance(testfeats)
        else:
            print('Error! Testing set can only be Stocktwits if use Stocktwits data to train.')

if __name__ == "__main__":
    main()