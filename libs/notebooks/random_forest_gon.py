
# ------------- Initial imports
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
%matplotlib inline

# Needed for decision tree visualization
import pydotplus
from IPython.display import Image

import sys
import os
os.chdir(r"/Users/gonzalogarciacontreras/rice15/Crypto_Bandyts/notebooks/) 
os.getcwd()


import os 
# change the current working directory  
# to specified path 
#os.chdir('c:\\gfg_dir') 
os.chdir('/Users/gonzalogarciacontreras/rice15/Crypto_Bandyts/notebooks/') 
# varify the path using getcwd() 
cwd = os.getcwd()  
print("Current working directory is:", cwd) 


import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from libs.signals import signals
btc_df = signals.technical_indicator_signal('btc')
#btc_df.tail()


# Loading data
what_columns=['Date', 'Close_XXBTZ', 'Vol_XXBTZ']
cryptos_path = Path("../data/crypto_pricesx.csv")
df_cryptos = pd.read_csv(cryptos_path, parse_dates=True, infer_datetime_format=True, delimiter=',')
df_cryptos['Date'] = pd.to_datetime(df_cryptos.Date, infer_datetime_format=True)
df_cryptos.set_index('Date', inplace=True)
df_cryptos.head()


#----------------------------

from pathlib import Path
import pandas as pd

# Crypto name-pair dictionary to easily look up the different pairs
# the dictionary keys are the 'common' representation of the crypto abbreviation, while the values are the Kraken representation

crypto_pairs = {
    'btc': 'XBT',
    'eth': 'ETH',
    'ltc': 'LTC',
    'dot': 'DOT',
    'xmr': 'XMR',
    'xdg': 'XDG',
    'xlm': 'XLM',
    'xrp': 'XRP',
    'zec': 'ZEC',
    'nano': 'NANO',
    'trx': 'TRX',
    'bch': 'BCH',
    'xtz': 'XTZ',
    'ada': 'ADA',
    'oxt': 'OXT'
}

# store all the data returned into one single 'data' dataframe - will parse out next into individual crypto dataframes

data_file = Path('../data/crypto_prices_cleaned.csv')
data = pd.read_csv(data_file, parse_dates=True, infer_datetime_format=True, delimiter=',')
data['Date'] = pd.to_datetime(data.Date, infer_datetime_format=True)
data.set_index('Date', inplace=True)

# parse out 'data' df into individual crypto dataframes (i.e.- btc_df)

# create a function that creates a list of the columns for the individual cryptos from the 'data' dataframe
def all_columns(data):
    all_columns_list = []
    for column in data.columns:
        all_columns_list.append(column)
    return all_columns_list

# create a function that creates a dictionary of list of columns names that will be used to create the crypto dataframes
def crypto_columns(data):

    # initiate the columns dictionary
    columns_dict = {}
    # create a list for each asset
    for crypto in crypto_pairs:
        columns_dict[crypto_pairs[crypto]] = []
    
    # populate each list with column names from data
    all_columns_list = all_columns(data)
    for crypto in crypto_pairs:
        for column in all_columns_list:
            if crypto_pairs[crypto] in column:
                columns_dict[crypto_pairs[crypto]].append(column)
    
    return columns_dict

# create a function that now creates individual dataframes for each crypto
def dataframe(data, asset):
    # create a dictionary of empty dataframes for each crypto
    dataframes = {}
    for crypto in crypto_pairs:
        dataframes[crypto] = pd.DataFrame({})
    # populate the dataframes from above with their respective columns from 'data'
    for crypto in crypto_pairs:
        dataframes[crypto] = data[crypto_columns(data)[crypto_pairs[crypto]]]

    return dataframes[asset]


def kraken_data(asset):
    '''
    Takes in asset abbreviation:  'btc', 'eth', 'ltc', 'dot', 'xmr', 'xdg', 'xlm', 'xrp', 'zec', 'nano', 'trx', 'bch', 'xtz', 'ada','oxt'
    and returns a OHLCV dataframe
    '''

    df = dataframe(data, asset)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(inplace=True)
    return df

    