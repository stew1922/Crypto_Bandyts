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
* Create a cryptocurrency trading bot that utilizes technical trading signals and NLP.
    * Pull data from Kraken into a SQL database
    * Use the data from the database to generate trading signals based on technical trading indicators
    * Pull in new sources from the CryptoControl API and perform a sentiment analysis to generate a trading signal
    * Run all the trading signals through a Random Forest to determine the weights for the generated signals
    * Run the weighted signals through an LSTM model to predict future pricing
    * Perform a backtest on the model outputs and perform a risk analysis
    * If the risk analysis is acceptable, send the buy/sell information to Kraken
    * Send the data and metrics to a dashboard to be able to track trades, profit/loss, etc.
    * Profit


### Libraries
* [Data](libs/data)
* [Signals](libs/signals)
* [Trading](libs/trading)

### Resources 
* [Google Docs](https://docs.google.com/document/d/1GrOYwcoCp7ZqUtgUvB9V7iqvB39htRWjOiBAbJ3R6TM/edit) which houses a lot of useful links
* [Kraken Python SDK](https://github.com/veox/python3-krakenex)
