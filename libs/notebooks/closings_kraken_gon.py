"""
#------------ Get OHLC top 15 cryto - exchange Kraken 
Cryto
bitcoin (btc), ethereum (eth), litecoin (ltc), polkadot (dot), monero (xmr), dogecoin (xdg), stellar lumens (xlm),
ripple (xrp), zcash (zec), nano (nano), tron (trx), bitcoin cash (bch), tezos (xtz), cardano (ada), orchid (oxt)
   

path imports.   
add the following bit of code to the beginning of your imports 
(as long as your notebook is inside a folder inside libs) it will tell Python to add the libs folder 
to the list of folders it looks for imports from 
(to see a list of folders Python is searching print sys.path after import.sys):

import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

Which then will allow you to import from our signals, data, or trading packages, i.e.:
from libs.signals import signals.ewma_crossover
from libs.trading import 'some_trading_function/library'

to see a list of folders Python is searching print sys.path after import.sys):


Note: you can remove a path by doing sys.path.remove('path_you_want_to_remove'), 
but be really careful with that and don't remove any of the default paths!
If y'all would rather just move the files into the libs folder, 





"""

# ---------- imports 
import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path

for p in (sys.path):
    print(p)

os.getcwd()

import pandas as pd 
import numpy as np
import datetime
import time, json, requests, sys
from time import time, ctime
from datetime import datetime
import pathlib
from requests.exceptions import HTTPError
#from libs.signals import signals 



#------------ functions 

#--- Get OHLC from Kraken
def Get_OHLC_Kraken(pair, interval='1440', since=''):
    response_json = {}
    url = 'https://api.kraken.com/0/public/OHLC'  #only last 720 datapoints 
    #pair = cryto
    #interval = '1440'                           # 1440 = 1 day 
    #since = ''                                  # return las 720 datapoints
           
    for i in range(3):
        try:
            response_kraken = requests.post(url, 
            params=
                    {'pair':pair,
                    'interval':interval,     
                    'since':since}, 
                    headers={"content-type":"application/json"})
                
            # If the response was successful, no Exception will be raised
            response_kraken.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  
        except Exception as err:
            print(f'Other error occurred: {err}')  
        else:
             print('Success, URL found!')
        break
        print('--------------------  try #');print(i)
        time.sleep(5)

    if i == 2: sys.exit('URL error GGC')     

    #print(response_kraken.json())
    #g = input("control 2 : "); print (g)
    return response_kraken.json()


# --- Process OHLC json into a list
def Process_Kraken_Jason(kraken_json, pair):
    i = 0
    l_price = []
    #pair = 'XXBTZUSD'
    
    while True:
        try:
            new_row = [ctime(kraken_json['result'][pair][i][0]),    # Date 
                        kraken_json['result'][pair][i][1],           # Open                      
                        kraken_json['result'][pair][i][2],           # High                       
                        kraken_json['result'][pair][i][3],           # Low                       
                        kraken_json['result'][pair][i][4],           # Close
                        kraken_json['result'][pair][i][6]            # Volume 
                       ] 
            #print (new_row)
            l_price += [new_row]
            i += 1           

        except Exception as err:
            print(f'error Process_Kraken_Jason: {err}')  
            break 
        
    return l_price


# ---  tranform OHLC list into a DataFrame '
def l_price_into_df_price(l_price, pair):
    l_labels = ['Date', 'Open_'+pair[0:5], 'High_'+pair[0:5], 'Low_'+pair[0:5], 'Close_'+pair[0:5], 'Vol_'+pair[0:5]]

    df_price = pd.DataFrame.from_records(l_price, columns=l_labels)   # l --> list, l_labels --> column names 
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price.set_index('Date', inplace=True) 

    return(df_price)


# ---- concatenate pair dataframe with the global dataframe 
def Concact_Prices(df_price, ft, df_all_prices):
    if ft == True:
        df_all_prices = pd.concat([df_price], axis="columns", join="outer")
        ft = False
    else:
        df_all_prices = pd.concat([df_all_prices, df_price], axis="columns", join="outer")

    return df_all_prices, ft


# --- Export df to CSV 
def export_df_csv(df_all_prices):
    path_csv = 'Crypto_Bandyts/libs/data/'
    name_csv = 'crypto_prices1.csv'
    OHLC_csv = path_csv + name_csv

    path = pathlib.Path('/Users/gonzalogarciacontreras/rice15/Crypto_Bandyts/libs/data/crypto_prices.csv')

    df_all_prices.to_csv(path)

    return 


# --------- Main 

def main():
    #bitcoin (btc), ethereum (eth), litecoin (ltc), polkadot (dot), monero (xmr), dogecoin (xdg), stellar lumens (xlm),
    #ripple (xrp), zcash (zec), nano (nano), tron (trx), bitcoin cash (bch), tezos (xtz), cardano (ada), orchid (oxt)
    cryto_list = ['XXBTZUSD', 'XETHZUSD', 'XLTCZUSD', 'DOTUSD', 'XXMRZUSD', 'XDGUSD', 'XXLMZUSD', 'XXRPZUSD', 
                'XZECZUSD', 'NANOUSD', 'TRXUSD', 'BCHUSD', 'XTZUSD', 'ADAUSD', 'OXTUSD']

    interval = '1440'
    since = ''
    ft = True 
    df_all_prices = pd.DataFrame()

    for pair in (cryto_list):
        kraken_json = Get_OHLC_Kraken(pair, interval, since)
        #print('\n\n')
        #print(kraken_json)

        l_price = Process_Kraken_Jason(kraken_json, pair)
        #print('\n\n')
        #print(l_price)

        df_price = l_price_into_df_price(l_price, pair)
        #print(df_price.head())

        df_all_prices, ft = Concact_Prices(df_price, ft, df_all_prices)
    
    print(df_all_prices.head())
    print(df_all_prices.tail())

    export_df_csv(df_all_prices)



if __name__== "__main__":
  main()
  g = input("End Program .... Press any key : "); print (g)

  
