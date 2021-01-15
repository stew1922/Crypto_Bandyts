"""
By feeding the trading signals (as our features) and the binary daily return (positive or negative, as the target) in a RFM, we will get  a model that predicts whether there will be a positive return or not.  And that should be our trading signal.  
As we get live data it will be sent through the trained RFM which will then return a prediction for tomorrow's return (assuming daily data).  The prediction would determine if we go long or short for the day.  
Really good example in the class repo:  class/15-Algorithmic-Trading/3/Activities/02-Ins_Random_Forest_Trading 

"""



# ------- Initial imports
# Initial imports
import pandas as pd
import numpy as np 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#%matplotlib inline

# Needed for decision tree visualization
import pydotplus
from IPython.display import Image

import warnings
warnings.filterwarnings('ignore')


import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from libs.signals import signals


#----------  get signals from libs (in this version BTC)
trading_signals_df  = pd.DataFrame()
trading_signals_df = signals.technical_indicator_signal('btc')
trading_signals_df.drop(['close','signal'], axis=1, inplace=True)

trading_signals_df.reset_index(inplace=True)
trading_signals_df['Date'] = pd.to_datetime(trading_signals_df['Date']).dt.date
trading_signals_df.set_index('Date', inplace=True)

print(trading_signals_df.head())


#----------  Import cryto sentiment signals (csv)
#Date,Ticker,Compound,Positive,Negative,Neutral,Sentiment Signal

#what_columns=['Date', 'Open_XXBTZ', 'High_XXBTZ', 'Low_XXBTZ', 'Close_XXBTZ', 'Vol_XXBTZ']
sentiment_path = Path("../data/BitcoinSentimentSignals.csv")
#crypto_price = pd.read_csv(cryptos_path, usecols=what_columns, parse_dates=True, infer_datetime_format=True, delimiter=',')
crypto_sentiment = pd.read_csv(sentiment_path, parse_dates=True, infer_datetime_format=True, delimiter=',')
crypto_sentiment['Date'] = pd.to_datetime(crypto_sentiment.Date, infer_datetime_format=True)
crypto_sentiment.set_index('Date', inplace=True)

# Rename columns 
crypto_sentiment.rename(columns={'Sentiment Signal':'Sentiment_Signal'}, inplace=True)

crypto_sentiment.drop('Ticker', axis=1, inplace=True)
#crypto_price.drop(btc_df.index[0], inplace=True)         #delete 1st row nan 

crypto_sentiment = crypto_sentiment.loc[::-1]               #reverse rows order

print(crypto_sentiment.head())


#------ # Import cryto prices (in this version BTC)
  
what_columns=['Date', 'Open_XXBTZ', 'High_XXBTZ', 'Low_XXBTZ', 'Close_XXBTZ', 'Vol_XXBTZ']
cryptos_path = Path("../data/crypto_pricesx.csv")
crypto_price = pd.read_csv(cryptos_path, usecols=what_columns, parse_dates=True, infer_datetime_format=True, delimiter=',')
crypto_price['Date'] = pd.to_datetime(crypto_price.Date, infer_datetime_format=True)
crypto_price['Date'] = pd.to_datetime(crypto_price['Date']).dt.date
crypto_price.set_index('Date', inplace=True)

# Rename columns 
crypto_price.rename(columns={'Open_XXBTZ':'open', 'High_XXBTZ':'high', 'Low_XXBTZ':'low', 'Close_XXBTZ':'close', 'Vol_XXBTZ': 'volume'}, inplace=True)

# Compute daily_return
crypto_price['daily_return'] = (crypto_price[['close']].pct_change(fill_method='ffill'))
crypto_price['daily_return'] = crypto_price['daily_return'].replace(-np.inf, np.nan).dropna()
#crypto_price.drop('close', axis=1, inplace=True)

# Compute daily change in volume 
crypto_price['vol_change'] = (crypto_price[['volume']].pct_change())
crypto_price['vol_change'] = crypto_price['vol_change'].replace(-np.inf, np.nan).dropna()
#crypto_price.drop('volume', axis=1, inplace=True)

#crypto_price.drop(btc_df.index[0], inplace=True)         #delete 1st row nan 

print(crypto_price.head())


#------------ dataframes shape sanity check 
print(f'trading_signals_df -> {trading_signals_df.shape}')
print(f'crypto_price -> {crypto_price.shape}')
print(f'crypto_sentiment -> {crypto_sentiment.shape}')


#------------ # Merge signal and prices in a single dataframe 
trading_signals_df = trading_signals_df.join(crypto_price, how="inner")
trading_signals_df = trading_signals_df.join(crypto_sentiment, how="inner")

print(f'trading_signals_df -> {trading_signals_df.shape}')
print(trading_signals_df.head())


#----------------# Construct the dependent variable y where if daily return is greater than 0, then 1, else, 0.
trading_signals_df['positive_return'] = np.where(trading_signals_df['daily_return'] > 0, 1.0, 0.0)
trading_signals_df['positive_volume'] = np.where(trading_signals_df['vol_change'] > 0, 1.0, 0.0)
print(trading_signals_df.head())


#---------------# Set X variable list of features
x_var_list = ['ewma_x', 'macd', 'bollinger', 'rsi', 'psar', 'positive_volume', 'Sentiment_Signal']

# Filter by x-variable list
print(trading_signals_df[x_var_list].tail())


#---------------- # Shift DataFrame values by 1
trading_signals_df[x_var_list] = trading_signals_df[x_var_list].shift(1)
print(trading_signals_df[x_var_list].tail())


#----------------- # Drop NAs and replace positive/negative infinity values
trading_signals_df.dropna(subset=x_var_list, inplace=True)
trading_signals_df.dropna(subset=['daily_return', 'vol_change'], inplace=True)
trading_signals_df = trading_signals_df.replace([np.inf, -np.inf], np.nan)
print(trading_signals_df.head())
print(trading_signals_df.shape)


#---------------- # Construct training start and end dates
training_start = trading_signals_df.index.min().strftime(format= '%Y-%m-%d')
training_end = '2020-06-30'

# Construct testing start and end dates
testing_start =  '2020-07-01'
testing_end = trading_signals_df.index.max().strftime(format= '%Y-%m-%d')

# Print training and testing start/end dates
print(f"Training Start: {training_start}")
print(f"Training End: {training_end}")
print(f"Testing Start: {testing_start}")
print(f"Testing End: {testing_end}")


#-------------- # Construct the X_train and y_train datasets
X_train = trading_signals_df[x_var_list][training_start:training_end]
y_train = trading_signals_df['positive_return'][training_start:training_end]

print(X_train.tail())
print(y_train.tail())


#---------------- # Separate X test and y test datasets
X_test = trading_signals_df[x_var_list][testing_start:testing_end]
y_test = trading_signals_df['positive_return'][testing_start:testing_end]

print(X_test.tail())
print(y_test.tail())


#---------------- # Creating StandardScaler instance
scaler = StandardScaler()

# Fitting Standard Scaller
X_scaler = scaler.fit(X_train)

# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


#----------------- Import SKLearn Library and Classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#---------------- Train the RFM 
# Fit a SKLearn linear regression using just the training set (X_train, Y_train):
# The parameters for the RandomTress seems to be very weak, only 100 trees and 3 levels
# Might it be better n_estimators=2000, max_depth=10

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
model.fit(X_train_scaled, y_train)

# Make a prediction of "y" values from the X_test dataset
predictions = model.predict(X_test_scaled)

# Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
Results = y_test.to_frame()                             #Column positive return 
Results["predicted_value"] = predictions                #prediction 
print(Results)











