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
import numpy as np
import requests
import logging
import bs4 as bs
import yfinance as yf
from typing import List, Optional, Dict
import time
from datetime import datetime, timedelta
import pickle
import os.path
import random

logger = logging.getLogger(__name__)

def get_sp500_tickers() -> List[str]:
    """
    Get current S&P 500 tickers using Wikipedia.
    Returns a list of ticker symbols.
    """
    try:
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        logger.error(f"Error getting S&P 500 tickers: {str(e)}")
        # Fallback to a reliable subset of tickers
        fallback_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'BRK-B', 'JPM', 'V', 'XOM'
        ]
        logger.info(f"Using fallback list of {len(fallback_tickers)} tickers")
        return fallback_tickers

def get_cached_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Get data from cache if available and not expired."""
    cache_file = f'data/cache/{ticker}.pkl'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_date = cached_data.get('date')
                # Cache is valid for 24 hours
                if cache_date and datetime.now() - cache_date < timedelta(hours=24):
                    return cached_data.get('data')
        except Exception as e:
            logger.warning(f"Error reading cache for {ticker}: {str(e)}")
    return None

def save_to_cache(ticker: str, data: pd.DataFrame) -> None:
    """Save data to cache with timestamp."""
    try:
        cache_file = f'data/cache/{ticker}.pkl'
        cache_data = {
            'date': datetime.now(),
            'data': data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Error saving cache for {ticker}: {str(e)}")

def get_stock_data(ticker: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Get historical stock data with retries and caching.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with stock data or None if retrieval fails
    """
    # Check cache first
    cached_data = get_cached_data(ticker, start_date, end_date)
    if cached_data is not None:
        return cached_data
        
    # If not in cache, fetch from API
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid synchronized retries
            if attempt > 0:
                jitter = random.uniform(0, 2)
                time.sleep(5 * attempt + jitter)
                
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, timeout=15)
            
            if not data.empty:
                # Save successful response to cache
                save_to_cache(ticker, data)
                return data
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if "rate limit" in str(e).lower():
                # Add extra delay for rate limits
                time.sleep(10)
            
    logger.error(f"Failed to get data for {ticker} after {max_retries} attempts")
    return None

def get_batch_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 2,
    delay: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Get stock data in batches to handle rate limits.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        batch_size: Number of stocks to process in each batch
        delay: Delay in seconds between batches
        
    Returns:
        Dictionary mapping tickers to their historical data
    """
    stock_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        for ticker in batch:
            data = get_stock_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                stock_data[ticker] = data
                
        if i + batch_size < len(tickers):
            # Add jitter to delay
            jitter = random.uniform(0, 2)
            time.sleep(delay + jitter)
            
    return stock_data

def validate_stock_data(data: pd.DataFrame) -> bool:
    """
    Validate stock data meets minimum requirements.
    
    Args:
        data: DataFrame with stock price history
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check for minimum data points (1 year)
        if len(data) < 252:
            return False
            
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return False
            
        # Check for too many missing values
        if data[required_cols].isnull().sum().max() > len(data) * 0.1:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating stock data: {str(e)}")
        return False

def get_market_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get market index (S&P 500) data for comparison.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        
    Returns:
        DataFrame with market data
    """
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        spy = yf.Ticker('SPY')
        data = spy.history(start=start_date, end=end_date)
        
        if data.empty:
            logger.warning("No market data available")
            return pd.DataFrame()
            
        return data
        
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return pd.DataFrame() 