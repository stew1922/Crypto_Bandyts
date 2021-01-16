# add the current repository to the file path so that Python can find our libraries and then import the kraken_data function from data
import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from libs.data.kraken_data import kraken_data

# import pandas, numpy, datetime and Path
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

# ewmacrossoversignal
def ewma_crossover(data, period_fast=9, period_slow=13):

    ''' 
    'Exponential Weighted Moving Average Crossover Indicator'
    -Takes in a dataframe with at least with one column 'Close' and a datetime as the index
        *Optionally, takes the faster moving average window size ("period_fast"; default = 9)
        *Optionally, takes the slower moving average window size ("period_slow"; default = 13)
    -Returns a dataframe with 'close', 'fast_ewma', 'slow_ewma', 'ewma_diff', and 'signal'
    '''

    # build out an exponential moving average crossover signal generator
    # standard is to use windows of 9 and 13 for fast and slow, respectively
    
    # Check to make sure the fast EWMA is smaller than the slow EWMA
    if period_fast >= period_slow:
        return f'The fast EWMA signal (period_fast={period_fast}) is larger than or equal to the slow EWMA signal (period_slow={period_slow}).  The fast EWMA signal needs to be smaller than the slow EWMA signal.'

    # EWMA Fast
    ewma_fast = data.Close.ewm(span=period_fast).mean()

    # EWMA Slow
    ewma_slow = data.Close.ewm(span=period_slow).mean()

    # Create a dataframe that consolidates and generates signal data
    # the signal column of the dataframe will be binary: -1 = bearish, 1 = bullish
        # -1 is bearish in the condition that the fast ewma is below or equal to the slow ewma
        # 1 is bullish in the condiditon that the fast ewma is above the slow ewma
    cross_over_df = pd.DataFrame(
        {
        'close': data.Close, 
        'fast_ewma': ewma_fast, 
        'slow_ewma': ewma_slow, 
        'ewma_diff': ewma_fast - ewma_slow
        }
        )

    cross_over_df['signal'] = np.where(ewma_fast > ewma_slow, 1, -1)

    return cross_over_df

# ewmasignal
def ewma(data, period):

    '''
    'Exponential Weighted Moving Average'
    -Takes in a dataframe with at least one column 'Close' and a datetime as the index AND a moving average window
    -Returns a dataframe with 'close', 'ewma', 'ewma_diff', and 'signal'
    '''

    # build out an exponential moving average
    ewma = data.Close.ewm(span=period).mean()

    # Create a dataframe that consolidates and generates signal data
    # the signal column of the dataframe will be binary: -1 = bearish, 1 = bullish
        # -1 is bearish in the condition that the signal is above the closing price
        # 1 is bullish in the condiditon that the signal is below the closing price
    ewma_df = pd.DataFrame(
        {
        'close': data.Close, 
        'ewma': ewma,
        'ewma_diff': ewma - data.Close
        }
        )

    ewma_df['signal'] = np.where(ewma_diff > 0, 1, -1)

    return ewma_df

# bollingerbandsignal
def b_band(data, bb_period=20, std_dev=2):

    '''
    'Bollinger Bands'
    -Takes in a dataframe with at least one column 'Close' and a datetime as the index
        *Optionally, takes the moving average window size ("bb_period"; default = 20)
        *Optionally, takes the number of standard deviations away from the moving averge to create the bands ("std_dev"; default = 2)
    -Returns a dataframe with 'close', 'middle_band', 'upper_band', 'lower_band', 'band_delta', 'delta_ewma', 'band_signal', and 'signal'
    '''

    # Need 3 lines for Bollinger Bands:
        # Middle = simple moving average
        # Upper and lower = standard deviation of moving average
    # MA is typically set as 20 periods and the upper and lower bands are typically set as 2 std
    bb_mid = data.Close.rolling(window=bb_period).mean()

    bb_upper = bb_mid + (std_dev * data.Close.ewm(span=bb_period).std())
    bb_lower = bb_mid - (std_dev * data.Close.ewm(span=bb_period).std())

    # calculate the distance between the upper and lower band to determine if there is incoming volatility
    # the closer together the upper and lower bands are, the more likely volatility will hit soon (up or down)
    # will use a ewma to determine if the difference is above or below normal
    band_delta = bb_upper - bb_lower
    delta_ewma = band_delta.ewm(span=bb_period).mean()
    band_signal = band_delta - delta_ewma

    # Create a dataframe that consolidates and generates signal data
    # the signal column of the dataframe will be binary: -1 = currently high volatility, 1 = volatility incoming

    bb_df = pd.DataFrame(
        {'close': data.Close,
        'middle_band': bb_mid,
        'upper_band': bb_upper,
        'lower_band': bb_lower,
        'band_delta': band_delta,
        'delta_ewma': delta_ewma,
        'band_signal': band_signal}
    )

    bb_df['signal'] = np.where(band_signal > 0, -1, 1)

    return bb_df

# macdsignal
def macd(data, period_slow=26, period_fast=12, period_signal=9):

    '''
    'Moving Average Convergence/Divergence'
    -Takes in a dataframe with at least one column 'Close' and a datetime as the index
        *Optionally, takes the slow ewma window size ("period_slow"; default = 26)
        *Optionally, takes the fast ewma window size ("period_fast"; default = 12)
        *Optionally, takes signal line window size ("period_signal"; default = 9)
    -Returns a dataframe with 'close', 'slow_ewma', 'fast_ewma', 'macd', 'signal_line', 'con_div', 'macd_signal', 'condiv_signal', and 'signal'
    '''

    # The Moving Average Convergence/Divergence (MACD) indicator can be broken into three parts: the signal line, the MACD line, and the convergence/divergence between the two
    # THE MACD LINE
    # The MACD line is created by subtracting a slow EMA from a fast EMA.  
        # The defualts are: 26 for slow and 12 for fast.  Adjust to suit your fancy
    slow_ewma = data.Close.ewm(span=period_slow).mean()

    fast_ewma = data.Close.ewm(span=period_fast).mean()

    macd = fast_ewma - slow_ewma

    # THE SIGNAL LINE
    # The signal line is generated by taking an EWMA of the MACD line.  
        # The default EWMA to use is 9 periods, but adjust as you see fit
    signal_line = macd.ewm(span=period_signal).mean()

    # THE CONVERGENCE/DIVERGENCE
    # The convergence/divergence of the MACD and signal lines is simply the MACD minus the signal.  
        # When it is below zero, it is considered bearish and when it is above zero it is considered bullish
    condiv = macd - signal_line

    # build out a dataframe that houses all the MACD signal data
    macd_df = pd.DataFrame(
        {'close': data.Close,
        'slow_ewma': slow_ewma,
        'fast_ewma': fast_ewma,
        'macd': macd,
        'signal_line': signal_line,
        'con_div': condiv}
        )

    # add a column that returns a -1 or 1 for the MACD signal.  
        # -1 means the MACD is below the zero line and is bearish.  
        # 1 means the MACD is above the zero line and is bullish.
    macd_df['macd_signal'] = np.where(macd > 0, 1, -1)

    # add a column that return a -1 or 1 for the convergence/divergence.  
        # -1 means it is negative and bearish.  
        # 1 means it is positive and bullish. 
    macd_df['condiv_signal'] = np.where(condiv > 0, 1, -1)

    # add a column that returns a -1, 0, or 1.  This column is the sum of the previous two (so in order to get it to -1, 0, +1 we must divide by 2).  
        # -1 is bearish (all signals are bearish)
        # 0 is neutral (one signal is bullish and one is bearish) 
        # 1 is bullish (all signals are bullish)
    macd_df['signal'] = (macd_df.macd_signal + macd_df.condiv_signal)/2

    return macd_df

# smasignal
def sma(data, period):

    '''
    'Simple Moving Average'
    -Takes in a dataframe with at least one column 'Close' and a datetime index AND a moving average window ("period")
    -Returns a dataframe with 'close', 'sma', 'sma_delta', and 'signal'
    '''

    # create the SMA using an input period.  No default period will be provided and user must provide one.
    sma = data.Close.rolling(window=period).mean()

    # create a dataframe to house the sma signal generator
    sma_df = pd.DataFrame({
        'close': data.Close,
        'sma': sma,
        'sma_delta': sma - data.Close
    })

    # add a column that return -1 or 1: -1 = SMA above the asset price and is bearish, 1 = SMA below the asset price and is bullish
    sma_df['signal'] = np.where(sma_delta > 0, 1, -1)

    return sma_df

# rsisignal
def rsi(data, period=14, overbought=70, oversold=30):

    '''
    'Relative Strength Index'
    -Takes in a dataframe with at least one column 'Close' and a datetime index
        *Optionally, it can take exponential moving average window ("period"; default = 14)
        *Optionally, it can take overbought value ("overbought"; default = 70)
        *Optionally, it can take oversold value ("oversold"; default = 30)
    -Returns a dataframe with 'close', 'rsi' and 'signal'
    '''

        # RSI indicator formula:
        # RSI = 100 - [100/(1 + RS)]
        # RS = RS_gain / abs(RS_loss)
        # RS_gain = {[(average gain from previous period - 1) * 13] + current gain} / 14
        # RS_loss = {[(average loss from previous period - 1) * 13] + current loss} / 14
    # Default period = 14

    # add a few new columns to the data frame to track the RSI parameters:
    data['close_chg'] = data['Close'].diff()

    data['gain'] = data.close_chg.apply(lambda x: x if x >= 0 else 0)
    data['avg_gain'] = data.gain.ewm(span=period).mean()
    data['prev_avg_gain'] = data.avg_gain.shift()
    data['rs_gain'] = ((data.prev_avg_gain * (period - 1)) + data.gain) / period

    data['loss'] = data.close_chg.apply(lambda x: x if x < 0 else 0)
    data['avg_loss'] = data.loss.ewm(span=period).mean()
    data['prev_avg_loss'] = data.avg_loss.shift()
    data['rs_loss'] = ((data.prev_avg_loss * (period - 1)) + data.loss) / period

    data['rs'] = data.rs_gain / abs(data.rs_loss)
    data['rsi'] = 100 - (100 / (1 + data.rs))

    data = data[['Close', 'rsi']]

    # add a signal column that will help identify the most basic trend
    # -1 means the RSI is over the overbought value and considered bearish potential (default = 70)
    # 0 means the RSI is between the overbought and the oversold values and considered neutral
    # 1 means the RSI is below the oversold value and considered bullish potential (defualt = 30)
    
    # create a sub-function that will determine if the RSI is overbought, oversold or neutral
    def rsi_level(x):
        if x >= overbought:
            return -1
        elif overbought > x > oversold:
            return 0
        elif x <= oversold:
            return 1

    # using the rsi_level function above, add a new column to the dataframe that returns the signal
    data['signal'] = data.rsi.apply(lambda x: rsi_level(x))
    return data

# psarsignal
def psar(data, af_start=0.02, af_step=0.02, af_max=0.20):

    '''
    'Parabolic Stop and Reverse'
    -Takes in a dataframe with at least the following columns included: 'Close', 'High', and 'Low'
        *Optionally, takes the acceleration factor starting point ("af_start"; default = 0.02)
        *Optionally, takes the acceleration factor step size ("af_step"; default = 0.02)
        *Optionally, takes the acceleration factor maximum value ("af_max"; default = 0.20)
    -Returns 'af', 'trend', 'trend_high', 'trend_low', 'ep', 'psar_init', 'psar_final', and 'signal' appended to the original dataframe
    '''

    # set your initial values (there will be no PSAR for the very first point in a dataset since there is no prior PSAR)
    # AF_start = 0.02 by default, AF_step = 0.02 by default, AF_max = 0.20 by defualt
    # Trend = close[0] - close[1].  If trend > 0, then the trend is 'down', else it is 'up'
    # EP:
        # if trend is 'up', EP = low[0]
        # if trend is 'down', EP = high[0]
    # PSAR_init and PSAR_final are equal:
        # if trend is 'up', PSAR_init = high[0]
        # if trend is 'down', PSAR_init = low[0]    

    data['af'] = af_start
    data['trend'] = 0 if data.Close[0] - data.Close[1] > 0 else 1
    data['trend_high'] = data.High[0]
    data['trend_low'] = data.Low[0]
    data['ep'] = data.High[0] if data.trend[1] == 1 else data.Low[0]
    data['psar_init'] = data.High[0] if data.trend[1] == 0 else data.Low[0]
    data['psar_final'] = data['psar_init']
    data['signal'] = -1 if data.trend[0] == 0 else 1

    # loop through the whole dataframe
    for iloc in range(1, len(data)):

        # If the previous trend was DOWN set the initial current trend to 0 and perform the down trend PSAR Calculations:
        if data.trend.iloc[iloc-1] == 0:
            data.trend.iloc[iloc] = 0

            # 1) Initial PSAR = Previous PSAR - (Previous AF * (Previous PSAR - Previous EP))
            data.psar_init.iloc[iloc] = data.psar_final.iloc[iloc-1] - (data.af.iloc[iloc-1] * (data.psar_final.iloc[iloc-1] - data.ep.iloc[iloc-1]))
            # 2) If the dataframe positional ('iloc' from the loop above) is at the 3rd position or higher, the initial PSAR is the GREATER of the intial PSAR calculated in step 1 or the highest of the previous two candles.  Otherwise, it is the GREATER of the initial PSAR calculated in step 1 or the previous candle's high.
            if iloc < 1:
                data.psar_init.iloc[iloc] = max(data.psar_init.iloc[iloc], data.High.iloc[iloc-1])
            else:
                data.psar_init.iloc[iloc] = max(data.psar_init.iloc[iloc], data.High.iloc[iloc-1], data.High.iloc[iloc-2])
            # 3) If the current Low is is LESS THAN the previous EP, then the current EP == current Low, otherwise the the current EP == previous EP
            data.ep.iloc[iloc] = data.ep.iloc[iloc-1] if data.ep.iloc[iloc-1] < data.Low.iloc[iloc] else data.Low.iloc[iloc]
            # 4) If the current EP is updated to the current low, then the AF needs to be increased by the `af_step`, otherwise current AF == previous AF
            if data.ep.iloc[iloc] == data.Low.iloc[iloc]:
                data.af.iloc[iloc] = (data.af.iloc[iloc-1] + af_step) if data.af.iloc[iloc-1] < af_max else af_max
            else:
                data.af.iloc[iloc] = data.af.iloc[iloc-1]
            # 5) Update the low and high for the current trend, if necessary
            data.trend_low.iloc[iloc] = data.Low.iloc[iloc] if data.trend_low.iloc[iloc-1] > data.Low.iloc[iloc] else data.trend_low.iloc[iloc-1]
            data.trend_high.iloc[iloc] = data.High.iloc[iloc] if data.trend_high.iloc[iloc-1] < data.High.iloc[iloc] else data.trend_high.iloc[iloc-1]
            #6) If the initial PSAR is GREATER THAN the current High, the psar_final == psar_intial, otherwise the trend flips and the parameters reset -- trend == 1, PSAR == EP, EP == current High, AF == af_start, and trend_high and trend_low reset
            if data.psar_init.iloc[iloc] > data.High.iloc[iloc]:
                data.psar_final.iloc[iloc] = data.psar_init.iloc[iloc]
            else:
                data.trend.iloc[iloc] = 1
                data.psar_final.iloc[iloc] = data.ep.iloc[iloc]
                data.af.iloc[iloc] = af_start
                data.ep.iloc[iloc] = data.High.iloc[iloc]
                data.trend_high.iloc[iloc] = data.High.iloc[iloc]
                data.trend_low.iloc[iloc] = data.Low.iloc[iloc]

        # If the previous trend was UP set the initial current trend to 1 and perform the up trend PSAR Calculations:
        else:
            data.trend.iloc[iloc] = 1

            # 1) Initial PSAR = Previous PSAR + (Previous AF * (Previous EP - Previous PSAR))
            data.psar_init.iloc[iloc] = data.psar_final.iloc[iloc-1] + (data.af.iloc[iloc-1] * (data.ep.iloc[iloc-1] - data.psar_final.iloc[iloc-1]))
            # 2) If the dataframe positional ('iloc' from the loop above) is at the 3rd position or higher, the initial PSAR is the LEAST of the intial PSAR calculated in step 1 or the lowest of the previous two candles.  Otherwise, it is the LEAST of the initial PSAR calculated in step 1 or the previous candle's low.
            if iloc < 2:
                data.psar_init.iloc[iloc] = min(data.psar_init.iloc[iloc], data.Low.iloc[iloc-1])
            else:
                data.psar_init.iloc[iloc] = min(data.psar_init.iloc[iloc], data.Low.iloc[iloc-1], data.Low.iloc[iloc-2])
            # 3) If the current High is is GREATER THAN the previous EP, then the current EP == current High, otherwise the the current EP == previous EP
            data.ep.iloc[iloc] = data.ep.iloc[iloc-1] if data.ep.iloc[iloc-1] > data.High.iloc[iloc] else data.High.iloc[iloc]
            # 4) If the current EP is updated to the current High, then the AF needs to be increased by the `af_step`, otherwise current AF == previous AF
            if data.ep.iloc[iloc] == data.High.iloc[iloc]:
                data.af.iloc[iloc] = (data.af.iloc[iloc-1] + af_step) if data.af.iloc[iloc-1] < af_max else af_max
            else:
                data.af.iloc[iloc] = data.af.iloc[iloc-1]
            # 5) Update the low and high for the current trend, if necessary
            data.trend_low.iloc[iloc] = data.Low.iloc[iloc] if data.trend_low.iloc[iloc-1] > data.Low.iloc[iloc] else data.trend_low.iloc[iloc-1]
            data.trend_high.iloc[iloc] = data.High.iloc[iloc] if data.trend_high.iloc[iloc-1] < data.High.iloc[iloc] else data.trend_high.iloc[iloc-1]
            #6) If the initial PSAR is LESS THAN the current Low, the psar_final == psar_intial, otherwise the trend flips and the parameters reset -- trend == 0, PSAR == EP, EP == current Low, AF == af_start, and trend_high and trend_low reset
            if data.psar_init.iloc[iloc] < data.Low.iloc[iloc]:
                data.psar_final.iloc[iloc] = data.psar_init.iloc[iloc]
            else:
                data.trend.iloc[iloc] = 0
                data.psar_final.iloc[iloc] = data.ep.iloc[iloc-1]
                data.af.iloc[iloc] = af_start
                data.ep.iloc[iloc] = data.Low.iloc[iloc]
                data.trend_high.iloc[iloc] = data.High.iloc[iloc]
                data.trend_low.iloc[iloc] = data.Low.iloc[iloc]
        
        if data.trend.iloc[iloc] == 0:
            data['signal'].iloc[iloc] = -1
        else:
            data['signal'].iloc[iloc] = 1

    return data

# vwapsignal
def vwap(data):

    '''
    'Volume Weighted Average Price'
    -Takes in a dataframe with at least the following columns included: 'Close', 'High', 'Low', and 'Volume' and a datetime index
    -Returns 'avg_price', 'current_day', 'prev_day', 'daily_cum_vol', 'vwap' and 'signal' added to the original dataframe
    '''

    # VWAP = volume weighted average price
    # VWAP = sumS(volume) * sum(avg. price) / sum(volume)
        # avg price = (high + close + low) / 3
        # the volume and price is a running cumulative for the DAY (restarts everyday)
            # Since it restarts everyday, this is not a great tool on the daily timeframe and above (on the daily timeframe, VWAP is just equal to the avg price), so it is better to use this tool only on *INTRADAY* time frames.  

    data['avg_price'] = (data.Close + data.High + data.Low) / 3
    data['curent_day'] = data.index.weekday
    data['prev_day'] = data.curent_day.shift()
    data['daily_cum_vol'] = data.Volume

    for day in range(1, len(data)):
        if data.curent_day.iloc[day] == data.prev_day.iloc[day]:
            data['daily_cum_vol'].iloc[day] += data.Volume.iloc[day]
        else:
            data['daily_cum_vol'].iloc[day] = data.Volume.iloc[day]

    data['vwap'] = (data.daily_cum_vol * data.avg_price) / data.daily_cum_vol
    data.signal = np.where(data.vwap > data.Close, 1, 0)
    return data

# volumeewmasignal
def volume_ewma(data, period):

    '''
    'Volume Exponential Weighted Average'
    -Takes in a dataframe with at least one column 'Volume' and a datetime index AND a moving average window 'period'
    -Returns a dataframe with 'volume_ewma', and 'signal' added to the original dataframe
    '''

    # create a column with the exponential weighted moving average using the period input from the function call
    data['volume_ewma'] = data.Volume.ewm(span=period).mean()

    # add the signal column: 1 if the volume is higher than the ewma and -1 if it is lower
    data['signal'] = np.where(data.volume_ewma <= data.Volume, 1, -1)

    return data

# volumeewma_crossoversignal
def volume_ewma_crossover(data, period_fast=9, period_slow=13):

    '''
    'Volume Exponential Weighted Average Crossover'
    -Takes in a dataframe with at least one column 'Volume' and a datetime index
        *Optionally, takes the slow ewma window size ("period_slow"; default = 13)
        *Optionally, takes the fast ewma window size ("period_fast"; default = 9)
    -Returns a dataframe with 'slow_ewma', 'fast_ewam', and 'signal' added to the original dataframe
    '''

    # create a column for the fast and the slow ewma
    data['slow_ewma'] = data.Volume.ewm(span=period_slow).mean()
    data['fast_ewma'] = data.Volume.ewm(span=period_fast).mean()

    # create a column for the signal: 1 if the fast_ewma is above the slow_ewma, otherwise -1
    data['signal'] = np.where(data.fast_ewma > data.slow_ewma, 1, -1)

    return data

# tradingsignal
def technical_indicator_signal(asset):

    '''
    Input: Asset symbol - 'btc', 'eth', 'ltc', 'dot', 'xmr', 'xdg', 'xlm', 'xrp', 'zec', 'nano', 'trx', 'bch', 'xtz', 'ada','oxt'
    Returns:  Dataframe of technial indicator signals

    NOTICE: When analyzing on the daily timeframe or greater, VWAP will not apply as it is ONLY an intraday indicator.
    '''

    # create a dataframe to house the technical trading signals from the kraken_data function found in libs/data/kraken_data.py
    asset_df = kraken_data(asset)

    technical_signals = pd.DataFrame({
        'close': asset_df.Close,
        'volume': asset_df.Volume,
        'ewma_x': ewma_crossover(asset_df, period_fast=6, period_slow=9).signal,
        'macd': macd(asset_df, period_fast=7, period_slow=9, period_signal=5).signal,
        'bollinger': b_band(asset_df, bb_period=9).signal,
        'rsi': rsi(asset_df, oversold=35).signal,
        'psar': psar(asset_df).signal,
        'vwap': vwap(asset_df).signal,
        'volume_ewma_x': volume_ewma_crossover(asset_df, period_fast=1, period_slow=7).signal
    })

    # since VWAP won't work on daily time intervals and greater, we need to check the interval to see if we should include vwap as a column or not
    daily_seconds = 86400
    delta_seconds = timedelta.total_seconds(technical_signals.index[1] - technical_signals.index[0])
    interval = delta_seconds / daily_seconds

    if interval >= 1:
        # if the interval is greater than or equal to 1, then do not include VWAP
        technical_signals = technical_signals.drop(columns='vwap')

    # sum the various technical signals together to return a trade 'grade' or signal
    technical_signals['signal'] = technical_signals.drop(columns=['close']).sum(axis='columns')

    return technical_signals
