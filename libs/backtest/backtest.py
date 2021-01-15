# create a results dataframe with returns
X_test = []
y_test = []

def results(data):
    '''
    -Takes a dataframe from the machine learning module of predicted results with the actual returns and return signal.
    -Returns a dataframe of the original with the predicted returns added on
    '''

    data['predicted_returns'] = abs(data.returns) * data.positive_returns * data.predicted_signal
    data.dropna(subset=['returns'], inplace=True)
    
    return data

def model_stats(data, X_test=None, y_test=None):
    '''
    -Takes the results dataframe and returns an output of model statistics
        *Optionally, takes in the X_test and y_test of the model input to return the model score
    '''

    # capture the inital price of the asset and set the initial capital to the same.  
    asset_initial_price = data.close[0]
    initial_capital = data.close[0]

    # calculate the cumulative return of the model
    cum_return = initial_capital * (1 + results['predicted_return']).cumprod()

    # capture the final price of the asset and the end price of the capital
    final_capital = cum_return[-1]
    asset_final_price = data.close[-1]

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
    print(f'Standard Dev Asset = {format(float(data.positive_returns.std()), "0.2f")}')
    print(f'Sharpe Ratio Asset= {format(float(((data.positive_returns.mean() - risk_free_rate) * 365)/(data.positive_returns.std() * np.sqrt(365))), "0.2f")}')
    print()
    print(f'Standard Dev Prediction = {format(float(data.predicted_signal.std()), "0.2f")}')
    print(f'Sharpe Ratio Prediction = {format(float(((data.predicted_signal.mean() - risk_free_rate) * 365)/(data.predicted_signal.std() * np.sqrt(365))), "0.2f")}')

    return cum_return

def model_plot(data, start_date="", end_date=""):
    '''
    -Plots the model output versus the asset price to see how the model performs versus a buy and hold
        *Optionally, takes a starting and ending data for the plot
    '''
    cum_return_plot = model_stats(data)[start_date:end_date].hvplot()
    asset_price = data.close[start_date:end_date].hvplot()
    return (cum_return_plot * asset_price).opts(show_legend=False, ylabel="Asset/Portfolio Value, $s")