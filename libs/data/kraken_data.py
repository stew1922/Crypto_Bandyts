from pathlib import Path
import pandas as pd

# Crypto name-pair dictionary to easily look up the different pairs
# the dictionary keys are the 'common' representation of the crypto abbreviation, while the values are the Kraken representation

crypto_names = {
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

data_file = Path('../data/crypto_prices2.csv')
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
    for crypto in crypto_names:
        columns_dict[crypto_names[crypto]] = []
    
    # populate each list with column names from data
    all_columns_list = all_columns(data)
    for crypto in crypto_names:
        for column in all_columns_list:
            if crypto_names[crypto] in column:
                columns_dict[crypto_names[crypto]].append(column)
    
    return columns_dict

# create a function that now creates individual dataframes for each crypto
def dataframe(data, asset):
    # create a dictionary of empty dataframes for each crypto
    dataframes = {}
    for crypto in crypto_names:
        dataframes[crypto] = pd.DataFrame({})
    # populate the dataframes from above with their respective columns from 'data'
    for crypto in crypto_names:
        dataframes[crypto] = data[crypto_columns(data)[crypto_names[crypto]]]

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