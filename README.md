# Crypto_Bandyts
Crypto trading algorithm

![](images/crypto_image.jpg "Source: https://www.telegraph.co.uk/technology/digital-money/how-to-understand-cryptocurrency-terminology/")  
<sup><sup>Credit: Getty Images</sup></sup>

  
## Group Members:
* Olan
* Phillipe
* James
* Gonzalo
* Travis

## Table of Contents
* [Proposal](#Proposal)
* [Tasks](#Tasks)
* [Libraries](#Libraries)
* [Scratch Notebooks](libs/notebooks)
* [Resources](#Resources)
* ...

### Proposal
* **Problem**: The cryptocurrency markets are very volatile and do not trade on fundamentals, so we would like an alternative model to be able to capture the volatility.  
* **Solution**: Create a cryptocurrency trading bot that utilizes technical trading signals and NLP to execute a profitable trading strategy.
    * Pull data from Kraken into a SQL database
    * Use the data from the database to generate trading signals based on technical trading indicators
    * Pull in news sources from the CryptoControl or other API and perform a sentiment analysis to generate a trading signal
    * Run all the trading signals through a Random Forest to determine the weights for the generated signals
    * Run the weighted signals through an LSTM model to predict future pricing
    * Perform a back test on the model outputs and perform a risk analysis
    * If the risk analysis is acceptable, send the buy/sell information to Kraken
    * ***Stretch Goal***: Send the data and metrics to a dashboard to be able to track trades, profit/loss, etc.
    * Profit!!!

### Tasks
- [x] Pull historic Kraken data and store in a SQL db
- [x] Create a trading signal with the historical Kraken data and technical trading indicators
- [x] Create a trading signal based on a crypto news sentiment analysis
- [ ] Create a Random Forest model to determine the trading signals' weights
- [ ] Create a back test module to analyze the risk
- [ ] Create a trading module to buy or sell based on the risk analysis

### Libraries
* [Data](libs/data)
* [Signals](libs/signals)
* [Trading](libs/trading)

### Resources 
* [Google Docs](https://docs.google.com/document/d/1GrOYwcoCp7ZqUtgUvB9V7iqvB39htRWjOiBAbJ3R6TM/edit) which houses a lot of useful links
* [Kraken Python SDK](https://github.com/veox/python3-krakenex)
