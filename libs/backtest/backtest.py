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

from tensorflow.keras import metrics
from sklearn.metrics import classification_report

# From the machine learning library pull in your model data
def asset_df():
    # first, get the different dataframes with techinical trading signals and the sentiment analysis
    trading_signals = rfm_crypto1.get_trading_signals()
    sentiment_signals = rfm_crypto1.get_sentiment_signals()
    crypto_price = rfm_crypto1.get_crypto_prices()
    # crypto_price.drop(columns=['volume'], inplace=True)
    # next, merge the dataframes into one, asset_df
    asset_df = rfm_crypto1.get_merged_dataframes(trading_signals, sentiment_signals, crypto_price)
    return asset_df

# then, gather you testing and training start/end dates
def train_start(train):
    return train.index.min().strftime(format= '%Y-%m-%d')

def train_end(train):
    return train.index.max().strftime(format= '%Y-%m-%d')
    
def test_start(test):
    return test.index.min().strftime(format= '%Y-%m-%d')

def test_end(test):
    return test.index.max().strftime(format= '%Y-%m-%d')

# finally, run the model and gather results and predictions
def model_output(X_train, X_test, y_train, y_test):
    model, predictions, results = rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)
    return model, predictions, results

def model(asset_df, X_train, X_test, y_train, y_test):
    return rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)[0]

def predictions(X_train, X_test, y_train, y_test):
    return rfm_crypto1.model_train_predict(X_train, X_test, y_train, y_test)[1]

def results_df(asset_df, X_train, X_test, y_train, y_test):
    results = model_output(X_train, X_test, y_train, y_test)[2]
    results['close'] = asset_df.close
    results['actual_returns'] = asset_df.daily_return
    results['positive_return'].replace(0, -1, inplace=True)
    results['predicted_return'] = abs(results.actual_returns) * results.positive_return * results.predicted_value
    results = results.replace([np.inf, -np.inf], np.nan)
    results.dropna(inplace=True)
    return results

# find the model score
def model_score(X_train, X_test, y_train, y_test):
    return model_output(X_train, X_test, y_train, y_test)[0].score(X_test, y_test)

# find the mean squared error of the predictions and actuals
def mse(y_actual, y_predict):
    mse = metrics.MeanSquaredError()
    mse.update_state(y_actual, y_predict)
    return mse.result().numpy()



#----This is the main function --> takes an asset dataframe and optionally if the chart should be run on the 
# full data set (test_data=False) or only the test set (test_data=True)----#
def backtest(asset_df, test_data=True):
    '''
    -Takes the a dataframe and returns an output of model statistics
    '''
    # run the model and return a results dataframe with close, 
    trading_signals_df, X_train, y_train, X_test, y_test = rfm_crypto1.construct_X_y(asset_df)
    training_start = train_start(X_train)
    training_end = train_end(X_train)
    testing_start = test_start(X_test)
    testing_end = test_end(X_test)

    model = model_output(X_train, X_test, y_train, y_test)
    results = results_df(asset_df, X_train, X_test, y_train, y_test)
    # print(results)

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

    def stats():
        #print out the stats
        print(f'Training Start Date: {training_start}')
        print(f'Training End Date: {training_end}')
        print(f'Testing Start Date: {testing_start}')
        print(f'Testing End Date: {testing_end}')
        print()
        print(f'Starting Capital = ${initial_capital}')
        print(f'Ending Capital = ${format(float(final_capital), "0.2f")}')
        print(f'Percent Return = {format(float(((final_capital - initial_capital) / initial_capital) * 100), "0.0f")}%')
        print(f'Beat Asset by {format(float(((final_capital - asset_final_price) / asset_final_price) * 100), "0.0f")}%')
        print(f'Model Score = {format(float(model_score(X_train, X_test, y_train, y_test)), "0.2f")}')
        print(f'Model MSE = {format(float(mse(results.actual_returns, results.predicted_return)), "0.4f")}')
        print()
        print(f'Standard Dev Asset Return = {format(float(results.actual_returns.std()), "0.2f")}')
        print(f'Sharpe Ratio Asset= {format(float(((results.actual_returns.mean() - risk_free_rate) * 365)/(results.actual_returns.std() * np.sqrt(365))), "0.2f")}')
        print()
        print(f'Standard Dev Prediction Return = {format(float(results.predicted_return.std()), "0.2f")}')
        print(f'Sharpe Ratio Prediction = {format(float(((results.predicted_return.mean() - risk_free_rate) * 365)/(results.predicted_return.std() * np.sqrt(365))), "0.2f")}')
        print()
        print(classification_report(results.positive_return[testing_start:], results['predicted_value'][testing_start:]))

    def plot():
        # plot the data
        if test_data==True:
            cum_return_plot = cum_return[testing_start:].hvplot()
            asset_price = asset_df.close[testing_start:].hvplot()
            return (cum_return_plot * asset_price).opts(show_legend=False, ylabel="Asset/Portfolio Value, $s")
        
        else:
            cum_return_plot = cum_return.hvplot()
            asset_price = asset_df.close.hvplot()
            return (cum_return_plot * asset_price).hvplot().opts(show_legend=False, ylabel="Asset/Portfolio Value, $s")

    def fi():
        # x_var_list = ['ewma_x', 'macd', 'bollinger', 'rsi', 'psar', 'volume_ewma_x', 'Sentiment_Signal']
        x_var = rfm_crypto1.x_var_list()
        imp_list = model[0].feature_importances_
        
        new_dict = {}

        count = 0
        for var in x_var:
            new_dict[var] = imp_list[count]
            count += 1
        
        imp_matrix = pd.DataFrame(new_dict, index=[1])

        return imp_matrix.sort_values(by=[1], axis=1, ascending=False).hvplot(kind='bar', title='Feature Relative Importance Scores', invert=True, ylabel='Feature Realtive Importance Score', xlabel='Feature')

    return stats(), plot(), fi()