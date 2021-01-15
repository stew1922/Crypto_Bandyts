#-----NOTICE: This script is only valid for BTC right now, will update later----#

# import the required packages
import pandas as pd
import numpy as np
import hvplot.pandas

import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from libs.notebooks import rfm_crypto1
from libs.signals import signals

# From the machine learning library pull in your model data
# first, get the different dataframes with techinical trading signals and the sentiment analysis
trading_signals = rfm_crypto1.get_trading_signals()
sentiment_signals = rfm_crypto1.get_sentiment_signals()
crypto_price = rfm_crypto1.get_crypto_prices()
crypto_price.drop(columns=['volume'], inplace=True)
# next, merge the dataframes into one, asset_df
asset_df = rfm_crypto1.get_merged_dataframes(trading_signals, sentiment_signals, crypto_price)

# then, gather your X_train, X_test, y_train, and y_test data
trading_signals, X_train, y_train, X_test, y_test = rfm_crypto1.construct_X_y(asset_df)

# then, gather you testing and training start/end dates
training_start = X_train.index.min().strftime(format= '%Y-%m-%d')
training_end = X_train.index.max().strftime(format= '%Y-%m-%d')

testing_start = X_test.index.min().strftime(format= '%Y-%m-%d')
testing_end = X_test.index.max().strftime(format= '%Y-%m-%d')

# finally, run the model and gather results and predictions
model = rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)[0]
predictions = rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)[1]
results = rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)[2]

# create a dataframe with the results of our model predictions
results['close'] = trading_signals.close
results['actual_returns'] = trading_signals.daily_return
results.positive_return.replace(0, -1, inplace=True)
results['predicted_return'] = abs(results.actual_returns) * results.positive_return * results.predicted_value

def model_stats():
    '''
    -Takes the results dataframe and returns an output of model statistics
        *Optionally, takes in the X_test and y_test of the model input to return the model score
    '''

    # capture the inital price of the asset and set the initial capital to the same.  
    asset_initial_price = results.close[0]
    initial_capital = results.close[0]

    # calculate the cumulative return of the model
    cum_return = initial_capital * (1 + results['predicted_return']).cumprod()

    # capture the final price of the asset and the end price of the capital
    final_capital = cum_return[-1]
    asset_final_price = results.close[-1]

    # set a risk free rate for the Sharpe Ratio calculation
    # in this case we will use the 10-year treasury bond which is currently (1/11/2021) = 0.96%
    risk_free_rate = 0.0096

    #print out the stats
    print(f'Starting Capital = ${initial_capital}')
    print(f'Ending Capital = ${format(float(final_capital), "0.2f")}')
    print(f'Percent Return = {format(float(((final_capital - initial_capital) / initial_capital) * 100), "0.0f")}%')
    print(f'Beat Asset by {format(float(((final_capital - asset_final_price) / asset_final_price) * 100), "0.0f")}%')
    print(f'Model Score = {format(float(score), "0.2f")}')
    print()
    print(f'Standard Dev Asset = {format(float(results.close.std()), "0.2f")}')
    print(f'Sharpe Ratio Asset= {format(float(((results.positive_return.mean() - risk_free_rate) * 365)/(results.positive_return.std() * np.sqrt(365))), "0.2f")}')
    print()
    print(f'Standard Dev Prediction = {format(float(cum_return.std()), "0.2f")}')
    print(f'Sharpe Ratio Prediction = {format(float(((results.predicted_signal.mean() - risk_free_rate) * 365)/(results.predicted_signal.std() * np.sqrt(365))), "0.2f")}')

    return cum_return

def model_plot(data, start_date="", end_date=""):
    '''
    -Plots the model output versus the asset price to see how the model performs versus a buy and hold
        *Optionally, takes a starting and ending data for the plot
    '''
    cum_return_plot = model_stats(data)[start_date:end_date].hvplot()
    asset_price = data.close[start_date:end_date].hvplot()
    return (cum_return_plot * asset_price).opts(show_legend=False, ylabel="Asset/Portfolio Value, $s")

    