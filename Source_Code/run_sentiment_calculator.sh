#!/bin/sh


echo "Start Sentiment Calculator: "
Python3.5 sentiment_calculator/SO_Run.py -i "../MyData/output/Preprocessed_Output/Twits" -o "../MyData/output/SO_CAL_Output/Twits"
echo "Done! Sentiment Calculation"
