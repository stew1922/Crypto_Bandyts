<<<<<<< HEAD
##TBD


# BTC Trading Strategy
### Goal of project 2 - systematically enter/exit trades profitably via algo bot
---

### HOW - Execution Strategy - Placing the trade
#### 1. Signal Generation = direction
#### 2. Risk Allocation = how much to buy/sell, allocation % of portfolio
#### 3. Execution on kraken
---

##TBD
# Trading Strategy
### Goal - //make money, but lots to consider aside from how to actually do it - at what risk, what timeframe, what portfolio percentage, crypto arbitrage, fees (m/t), portflios, risk of portfolios, current holdings, other allocated risk going on for other securities in TOS/other platform, hedging btc trades, buying and selling 'contract for difference' btc derrivatives, buying EFTs - gld/slv/other metals, etc
---

##TBD
# Future Upgrades
### Gaussian-ize the signals
### Allocate risk according to weakness/strength of signal
### Scale in/out parameters
### Select and implement trading strategy
### Update results to db to use during analysis and optimization (?)
---

##TBD (uncatergorized)
# Possible Considerations
### Exploration of statistical tools for dealing with non-normal distributions, specifically generalized hyperbolic distributions that have semi-heavy tails
### Dealing the events that can ocurr within these semi-heavy tails will be important to take into consideration
### Learn more about the skewed t
### Seeing where the GARCH model could fit in this system
### We'll need a useful understanding of the statistical behavior of specific types of ratios of random qualities
### Managing the active to passive trade ratio
### Trading dynamically across different exchanges based on optional transaction costs
### Setting "subjective" parameter thresholds to alert human (twilio) before, during, or after a trade that something's is going on or has already happened that the human will want to look at and make a decision
### Securing fastest, reliable internet possible while also incorporating failovers/safeties
### Security review program - accounts, passwords, networks, physical
=======
#TB Written up, on paper now



# BTC Trading Strategy
### Goal - enter/exit trades profitably via algo bot
### Outlines strategy to be coded
---

### Strategy How-to
#### 1. Signal Generation = direction
#### 2. Risk Allocation = how much to buy, allocation % of portfolio
#### 3. Execution on kraken

#### Liquidity Check
#### Vol Check
#### GARCH


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
>>>>>>> d1a150502015085d174e827a656ff822d96daaf0
