# SO-CAL

# Sentiment Engine for Stock Market

The code is to build a sentiment engine fore stock market based on two approaches: machine learning and lexicon-based. The project is highly consistent with the technology review.

### Code Structure
- Data Crawler
- Data Preprocessing
- Sentiment Engine
- Machine Learning Approach
- Lexicon Approach

*****************************************************************************

### Installtion

1. Install Stanford CoreNLP
1).  Download [Newest Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
2). `unzip stanford-corenlp-full-2016-10-31.zip`
3). `cd stanford-corenlp-full-2016-10-31`
4). `java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000`
5). `pip3 install pycorenlp`
2. Install nltk `pip install nltk`
3. Install sklearn `pip install sklearn`
*****************************************************************************

### Instruction to Run Sentiment Engine
In Terminal, set directory to `Source_Code`
For machine learning sentiment engine:
```
sh run_ml_model.sh
```
For lexicon-based sentiment engine
```
sh run_text_preprocessing.sh
sh run_sentiment_calculator.sh
```
*****************************************************************************
#### Data Crawler
The stock sentiment data used for this project is [StockTwits](https://api.stocktwits.com/developers/docs). 
- register on StockTwits.com and generate a token. `'access_token'`
- set the fixed time interval (in seconds) to crawl the live data stream `time_interval = 240`
- run DataCrawler.py and data will be stored into json files.

### Data Preprocessing
First, conduct a sanity check on all the available files. Then, merge files together. Since the elements in one message varies dramatically, filter out the tweet with the “message” element. To use “message”, “symbol”, “sentiment” as headers, construct a pandas dataframe. Remove the duplicates. Polish and clean the “message” content by removing tags. Reformat the data to the desired input I design for the sentiment engine built.

The techniques I learned in MP1 and the lectures can well apply to this step. Remove the stop words. Do the stemming and tokenization. There are multiple methods of tokenization. With a manual check of sample text, I choose one method with most desired clean message.
It is the script in `Preprocess.py`.

### Stock Sentiment Engine
I. Sentiment Engine under Machine Learning Approach
- Naive Bayes
- Support Vector Machine (SVM)
- Maximum Entropy

Instruction: 
In Terminal: under folder `Source_Code`
```
sh run_ml_model.sh
```
To change arguments in `run_ml_model.sh`
-for `-ct`, set to `NaiveBayes` `SVM` `MaximumEntropy`

```
echo "Running Machine Learning Model: "
python3.5 machine_learning_processing/ML_Run.py –ct NaiveBayes –ts StockTwits –td StockTwits 
echo "Done! ML model completed.”
```

Inplementation:
`nltk` package is sourced. The member functions for `class MachineLearningNLP` are in `main_MachineLearningEngine.py`. The main script will call both `Preprocess.py` and `MachineLearningNLP.py` to train the data. 

II. Sentiment Engine under Lexicon Approach
There are many popular methods used in research and industry. After reading many relevant documents and papers, I regard Semantic Orientation CALculator (SO-CAL) as the lexicon-based approach for my project. The details reasonings can be referred to my Technology Review.  
I build my SO-CAL based on the source code by the paper author Taboada. I modify the code and also apply TF-IDF to the SO correction for stock market. Then map the value to [-5,5] range. Manual check and adjust the stock dictionary.

Instruction: 
In Terminal: under folder `Source_Code`
```
sh run_text_preprocessing.sh
sh run_sentiment_calculator.sh
```
*****************************************************************************

### Future Work

- Get a larger StockTwits data set for Training classifier 
- Leverage network information in StockTwits messages to adjust weight

### Reference
[1] Taboada, Maite, Julian Brooke, Milan Tofiloski, Kimberly Voll and Manfred Stede (2011) [Lexicon-Based Methods for Sentiment Analysis](https://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00049). Computational Linguistics 37 (2): 267-307. Source Code by Taboada
[2] Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan, Proceedings of the 14th International Conference on Intelligent Data Engineering and Automated Learning, [Thumbs up? Sentiment Classification using Machine Learning Techniques](https://arxiv.org/abs/cs/0205070), 2002.
[3]Thorsten Joachims, Making large-scale SVM learning practical, Advances in Kernel Methods - Support Vector Learning, 1999.
[4] Mukesh Chapagain, Machine Learning & Sentiment Analysis: Text Classification, 2016
[5]Councill, Issac G, Ryan McDonanld, and Leonid Velikovich, Proceedubgs of the Wirkshop on NEgation and Speculation in Natural Language Processing, [What’s great and what’s not: Learning to classify the scope of negation for improved sentiment analysis](https://dl.acm.org/citation.cfm?id=1858969), 2010.
