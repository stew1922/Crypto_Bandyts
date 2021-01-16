'''
Basic:
initialize - data
S
get signal -
  buy -  if not holding, buy
  sell - if holding, sell
  '''


# Initialize
def init_bandyts():
    #get account data - btc current hodl, cash (assumes btc/usd is the target market currently in v.5)
    #if success, return hodl amount
    #if not, no trade, msg to twilio,exit

    return (hodl)
 
 # --get signal from backtest
def get_signal(signal):
 #   update_crypto_action() #why - line in the sand, can see action between here and trade palced
  #  eval_metrics() #take - uca (above), if price dropped more than 
# std of coin, avg std is, then if it goes below that stop out -- could be a useful way in a portfolio
    return signal

def decide_to_trade():
    #yes
    #no
    return signal

def signal_negative():
    return signal

# --get latest price & volume from kraken
def update_crypto_action():
    #get btc info from kraken
    return signal

# --get latest price & volume from kraken
def get_signal(signal):
    return signal

# --get latest price & volume from kraken
def get_signal(signal):
    return signal


'''
------------
all_columns_list = []
    for column in data.columns:
        all_columns_list.append(column)
    return all_columns_list 
    '''