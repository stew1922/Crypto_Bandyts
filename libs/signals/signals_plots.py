import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from libs.signals import signals
import hvplot.pandas
import pandas as pd
import numpy as np
from libs.data.kraken_data import kraken_data

def b_band_plot(asset, bb_period=20, std_dev=2):
    b_band = signals.b_band(asset, bb_period=bb_period, std_dev=std_dev)
    upper_band_plot = b_band.upper_band.hvplot(color='magenta', legend=False)
    lower_band_plot = b_band.lower_band.hvplot(color='magenta', legend=False)
    middle_band_plot = b_band.middle_band.hvplot(color='lightcoral', legend=False)

    return (upper_band_plot * lower_band_plot * middle_band_plot * b_band.close.hvplot(color='green')).opts(title='Bollinger Bands', ylabel="Asset Price, $'s", width=800, height=400, legend_position='top_left')

def macd_plot(asset, period_slow=26, period_fast=12, period_signal=9):
    macd_df = signals.macd(asset, period_slow=period_slow, period_fast=period_fast, period_signal=period_signal)

    close_plot = macd_df.close.hvplot()
    macd_plot = macd_df.macd.hvplot(color='magenta')
    signal_plot = macd_df.signal_line.hvplot(color='darksalmon')

    macd_df['macd_histogram_p'] = np.where(macd_df.con_div >= 0, macd_df.con_div, 0)
    macd_histogram_p = macd_df.macd_histogram_p.hvplot(kind='area', color='green', legend=False)

    macd_df['macd_histogram_n'] = np.where(macd_df.con_div < 0, macd_df.con_div, 0)
    macd_histogram_n = macd_df.macd_histogram_n.hvplot(kind='area', color='red', legend=False)

    return (macd_plot * signal_plot * macd_histogram_p * macd_histogram_n).opts(title='Moving Average Convergence/Divergence (MACD)', legend_position='top_left',  ylabel='MACD', width=800, height=400)

def ewma_x_plot(asset, period_fast=9, period_slow=13):
    ewma_x_df = signals.ewma_crossover(asset, period_fast=period_fast, period_slow=period_slow)

    close_plot = ewma_x_df.close.hvplot(color='blue')
    fast_plot = ewma_x_df.fast_ewma.hvplot(color='magenta')
    slow_plot = ewma_x_df.slow_ewma.hvplot(color='orange')
    

    return (close_plot * fast_plot * slow_plot).opts(title='Price Exponential Moving Average Crossover', legend_position='top_left',  ylabel="Asset Price, $'s", width=800, height=400)

def rsi_plot(asset, period=14, overbought=70, oversold=30):
    rsi_df = signals.rsi(asset, period=period, overbought=overbought, oversold=oversold)

    rsi_plot = rsi_df.rsi.hvplot(color='magenta')
    rsi_df['overbought'] = overbought
    overbought_plot = rsi_df.overbought.hvplot(legend=False, color='black')
    rsi_df['oversold'] = oversold
    oversold_plot = rsi_df.oversold.hvplot(legend=False, color='black')    

    return (rsi_plot * overbought_plot * oversold_plot).opts(title='Relative Strength Index (RSI)', legend_position='top_left', ylabel='RSI Value', width=800, height=400)

def psar_plot(asset, af_start=0.02, af_step=0.02, af_max=0.20):
    psar_df = signals.psar(asset, af_start=af_start, af_step=af_step, af_max=af_max)

    close_plot = psar_df.Close.hvplot(color='mediumblue')
    psar_plot = psar_df.psar_final.hvplot(kind='scatter', color='magenta', size=5)

    return (close_plot * psar_plot).opts(title='Parabolic Stop and Reverse (PSAR)', legend_position='top_left', ylabel="Asset Value, $'s", width=800, height=400)

def ewma_vol_plot(asset, period_fast=9, period_slow=13):
    vol_df = signals.volume_ewma_crossover(asset, period_fast=period_fast, period_slow=period_slow)

    vol_plot = vol_df.Volume.hvplot(kind='bar', xticks=10)
    slow_plot = vol_df.slow_ewma.hvplot(color='yellow', xticks=10)
    fast_plot = vol_df.fast_ewma.hvplot(color='magenta', xticks=10)

    return (vol_plot * slow_plot * fast_plot).opts(title='Volume Exponential Moving Average Crossover', legend_position='top_left', ylabel='Volume, shares/day', width=800, height=400, xlabel="Days (start date, t0 = 01/22/2019)")