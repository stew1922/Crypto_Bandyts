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
%matplotlib inline

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

trading_signals_df.head()

