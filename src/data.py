"""
This is our data module - it's pretty simple right now but super important!
It gets us the list of stocks we want to trade (S&P 500 in this case).

I chose the S&P 500 because:
1. They're big, stable companies
2. Easy to trade (lots of volume)
3. Tons of data available
4. Less likely to have weird price movements

We could add more data sources later, like getting stocks from other indexes
or maybe even crypto!
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_sp500_tickers() -> list:
    """
    This function is cool because it scrapes Wikipedia to get our stock list.
    Instead of maintaining our own list (which would get outdated), we always
    have the latest S&P 500 companies.
    
    It also cleans up the tickers - sometimes they have weird stuff like '.B'
    at the end that we don't want.
    """
    try:
        # Grab the S&P 500 table from Wikipedia
        # The [0] is because read_html returns a list of all tables on the page
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].tolist()
        
        # Clean up the tickers - get rid of any weird suffixes
        tickers = [ticker.split('.')[0] for ticker in tickers]
        
        return tickers
        
    except Exception as e:
        logger.error(f"Couldn't get S&P 500 tickers: {str(e)}")
        return [] 