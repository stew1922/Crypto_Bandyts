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
