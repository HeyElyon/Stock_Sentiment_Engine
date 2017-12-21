#!/bin/sh

echo "Preprocess Raw Text Input: "
python3.5 text_preprocessing/preprocess.py  -i '../MyData/input/Raw_Text/Twits/' -o '../MyData/output/Preprocessed_Output/Twits/' -a 'tokenize,ssplit,pos'
echo "Done! Generated Preprocessed Data"


