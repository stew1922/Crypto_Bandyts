# Signal Generator Library
### A library of technical trading signals.

## Table of Contents
* [Bollinger Bands](#Bollinger-Bands)
* [MACD](#MACD)
* [EMA](#EMA)
* [EMA Crossovers](#EMA-Crossovers)
* [SMA](#SMA)
* [RSI](#RSI)
* [PSAR](#PSAR)
* [VWAP](#VWAP)

#### Bollinger Bands
* [Code](signals.py#bollingerbandsignal)
* Takes a dataframe with a single datetime index that contains, at least, a column labeled 'Close'
    * optionally, can enter ***bb_period*** which is the window size for the SMA's used in the middle, lower and upper bands as well as the SMA used in the signal calculation.  
        * Default: `bb_period=20`
    * optionally, can enter ***std_dev*** which is the number of standard deviations that the lower and upper bands are away from the middle band.  
        * Default: `std_dev=2`
* Returns a dataframe with 'close', 'middle_band', 'upper_band', 'lower_band', 'band_delta', 'delta_ewma', 'band_signal', and 'signal' columns 
    * The 'signal' column contains either a -1 or 1:
        * -1 = a period of relative volatility
        * 1 = a period of relative stability
* A reading of 1 can indicate a period of relative calm that would insinuate that a period of volatility is just around the corner.

#### MACD
* [Code](signals.py#macdsignal)
* Takes a dataframe that has a datetime index and contains, at least, a column labeled 'Close'
    * optionally, takes ***period_slow*** which is the slower window of the two EWMAs being compared for the MACD line
        * Default: `period_slow=26`
    * optionally, takes ***period_fast*** which is the faster window of the two EWMAs being compared for the MACD line
        * Default: `period_fast=12`
    * optionally, takes ***period_signal*** which is the window for the EWMA of the Signal line
        * Default: `period_signal=9`
* Returns a dataframe with 'close', 'slow_ewma', 'fast_ewma', 'macd', 'signal_line', 'con_div', 'macd_signal', 'condiv_signal', and 'signal' columns
    * The 'signal' column contains a -1, 0, or 1:
        * -1 means bearish as both the MACD and convergence/divergence are bearish
        * 0 means neutral as either the MACD or convergence/divergence is bearish and the other is bullish
        * 1 means bullish as both the MACD and convergence/divergence are bullish

#### EMA
* [Code](signals.py#ewmasignal)
* Takes a dataframe with a single datetime index that contains, at least, a column labeled 'Close'
* Takes ***period*** which is the window that is used for the EWMA
    * This is not optional, and no default value exists
* Returns a dataframe with 'close', 'ewma', 'ewma_diff', and 'signal' columns
    * The 'signal' column contains either a -1 or 1:
        * -1 means that the current closing price is below the EWMA and is considered bearish from a momentum standpoint
        * 1 means that the current closing price is above the EWMA and is considered bullish from a momentum standpoint

#### EMA Crossovers
* [Code](signals.py#ewmacrossoversignal)
* Takes a dataframe with a single datetime index that contains, at least, a column labeled 'Close'
    * optionally, takes ***period_fast*** which is the faster window of the two EMWAs being compared
        * Default: `period_fast=9`
    * optionally, takes ***period_slow*** which is the slower window of the two EMWAs being compared
        * Default: `period_slow=13`
    * NOTE: the fast period *must* be smaller than the slow period.
* Returns a dataframe with 'close', 'fast_ewma', 'slow_ewma', 'ewma_diff', and 'signal' columns
    * The 'signal' columns contains either a -1 or 1:
        * -1 means the fast EWMA has crossed down below the slow EWMA and is generally considered bearish
        * 1 means the fast EWMA has crossed up above the slow EWMA and is generally considered bullish

#### SMA
* [Code](signals.py#smasignal)
* Takes a dataframe with a single datetime index that contains, at least, a column labeled 'Close'
* Takes ***period*** which is the window that is used for the SMA
    * This is not optional, and no default value exists
* Returns a dataframe with 'close', 'sma', 'sma_delta', and 'signal' columns
    * The 'signal' column contains either a -1 or 1:
        * -1 means that the current closing price is below the SMA and is considered bearish from a momentum standpoint
        * 1 means that the current closing price is above the SMA and is considered bullish from a momentum standpoint

#### RSI
* [Code](signals.py#rsisignal)
* Takes a dataframe with a single datetime index that contains, at least, a column labeled 'Close'
    * optionally, takes ***period*** which is the window that is used for the EMA
        * Default: `period=14`
    * optionally, takes ***overbought*** which is the level that the trader considers the asset to be overbought
        * Default: `overbought=70`
    * optionally, takes ***oversold*** which is the level that the trader considers the asset to be oversold
        * Default: `oversold=30`
* Returns a dataframe with 'close', 'rsi' and 'signal' columns
    * The 'signal' column contains either a -1, 0, or 1:
        * -1 means that the current closing price is above the overbought value and is considered bearish
        * 0 means that the current closing price is between the overbought and oversold values and is considered neutral
        * 1 means that the current closing price is below the oversold value and is considered bullish

#### PSAR
* [Code](signals.py#psarsignal)
* Takes a dataframe with a single datetime index that contains, at least, columns labeled 'Close', 'Low', and 'High'
    * optionally, takes ***af_start*** which is the beginning point for the Acceleration Factoer (AF)
        * Default: `af_start=0.02`
    * optionally, takes ***af_step*** which is the step size for when AF is changed
        * Default: `af_step=0.02`
    * optionally, takes ***af_max*** which is the maximum step size allowed
        * Default: `af_max`
* Returns a dataframe with 'Close', 'Low', 'High', 'af', 'trend', 'trend_high', 'trend_low', 'ep', 'psar_init', 'psar_final', and 'signal' columns
    * The 'signal' column contains either a -1 or 0:
        * -1 means that the trend is downward and could be considered bearish
        * 1 means that the trend is upward and could be considered bullish
* Other data can be obtained from this signal, i.e. - the 'psar_final' column could be considered a stop for any closing period as a risk mitigation strategy (if you are in an uptrend and you take out a long position, and then the price drops to the PSAR; that is the stop that would signal to exit the trade as the trend is flipping and becoming bearish)

#### VWAP
* STILL TESTING -- CONSIDER THIS IN BETA
* [Code](signals.py#vwapsignal)
* Takes a dataframe with a single datetime index and, at least, columns 'Close', 'High', 'Low', and 'Volume'
* Returns a dataframe with 'avg_price', 'current_day', 'prev_day', 'daily_cum_vol', 'vwap' and 'signal' added to the original df
    * The signal column contains either a 1 or -1
        * -1 means that the current price is below the VWAP and can be considered bearish _from a momentum standpoint_
        * 1 means that the current price is above VWAP and can be considered bullish _from a momentum standpoint_
* You could also use VWAP to pick an entry/exit price --> if the current price is below VWAP, then you could think of it as getting in at a 'below average' price.  Vice versa when it is above.  So use this signal in accordance with your strategy.

#### Technical Indicator Signals
* [Code](signals.py#tradingsignal)
* Takes a dataframe with a single datetime index and, at least, columns 'Close', 'High', 'Low', and 'Volume'
* Returns a dataframe with a 'Close' column and the technical trading indicators EWMA Crossover, MACD, Bollinger Bands, RSI, PSAR, VWAP (*note: only valid in the case of timeframes less than 1 day)

