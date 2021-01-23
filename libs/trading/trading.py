# Add the current repository to the file path so that Python can find our libraries and then import the kraken_data function from data
import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import libraries and dependencies
import ccxt
import os
import pandas as pd
from dotenv import load_dotenv
from crypto_bandyts import *

#Load environment variables
load_dotenv()

# Import environment variables
kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")  #SET WEEK OF 1/11 - bandyt5 - expiry 1/17/21 0:0:0:0UTC

# Set the public and private keys for the API
exchange = ccxt.kraken({
    'apiKey': kraken_public_key,
    'secret': kraken_secret_key,
})

#init_bandyts(secret)




#-------------------- main Crypto_Bandyts application
def main():
    #hodl_amt = 
 return rfm_signals



   

#if __name__== "__main__":
 # main()
  #g = input("End Program .... Press any key : "); print (g)