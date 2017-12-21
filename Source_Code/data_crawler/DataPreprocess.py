import json
import pandas as pd
import numpy as np
from glob import glob

all_txts = glob('*.txt')
test_txts = all_txts[:-3]

### Merge and Data Cleansing
all_dict = {}
for curr_txt in test_txts:
    try:
        message_list = json.load(open(curr_txt))['messages']
    except Exception as e:
        print (curr_txt)
        continue
    for curr_msg in message_list:
        txt_dict = {}
        curr_keys = curr_msg.keys()
        if not ('body' in curr_keys or 'symbols' in curr_keys or 'entities' in curr_keys):
            continue
        if 'body' in curr_keys:
            txt_dict['body'] = curr_msg['body']
        else:
            txt_dict['body'] = np.nan
        if 'symbols' in curr_keys:
            txt_dict['symbol'] = ', '.join([curr_symbol['symbol'] for curr_symbol in curr_msg['symbols']])
        else:
            txt_dict['symbol'] = np.nan
        
        if 'entities' in curr_keys:
            if curr_msg['entities']['sentiment']:
                txt_dict['sentiment'] = curr_msg['entities']['sentiment']['basic']
        else:
            txt_dict['sentiment'] = np.nan
        
        txt_dict['timestamp'] = curr_msg['created_at']
        all_dict[curr_msg['id']] = txt_dict

data = pd.DataFrame(all_dict).transpose().drop_duplicates()

### For Stats
data_filter=data[data['symbol'].notnull()]
#data_filter
data_sentiment=data[data['sentiment'].notnull()]
#data_sentiment
print("Data with Stock Ticker: ",len(data_filter)/len(data))
print("Data with Sentiment Label: ",len(data_sentiment)/len(data))

### Reformat data
data_out=data[['body_clean','sentiment']]
data_out["sentiment"].replace('Bullish','pos',inplace=True)
data_out["sentiment"].replace('Bearish','neg',inplace=True)
with open('StockTwists_Data_Sentiment.csv','w') as myFile:
    data_out[data_out["sentiment"].notnull()].to_csv(myFile,index=False)#,sep='\t')
tmp = pd.read_csv('StockTwists_Data_Sentiment.csv')

### Output Positive and Negative messages
for i, text in enumerate(tmp[tmp['sentiment']=='pos']['body_clean']):
    with open('yes'+str(i+1),'w') as myFile:
        myFile.write(text)
for i, text in enumerate(tmp[tmp['sentiment']=='neg']['body_clean']):
    with open('no'+str(i+1),'w') as myFile:
        myFile.write(text)
