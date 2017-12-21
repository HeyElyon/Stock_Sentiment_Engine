from datetime import datetime
import requests
import time
import json


# Access Live Data Frequency: Fixed Time Interval to crawl StockTwits Data
time_interval = 240



params = {
    'access_token': 'YOUR TOKEN'
}

url = "https://api.stocktwits.com/api/2/streams/home.json"



while(True):
    try:
        r = requests.get(url, params=params)
        with open(str(datetime.now())+'.txt','w') as myFile:
            json.dump(r.json(), myFile)
        print ('File is written.')
    except Exception as e:
        print (e)
        break
    time.sleep(time_interval)
