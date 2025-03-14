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

import os
import pandas as pd
import numpy as np
import requests
import logging
import bs4 as bs
import yfinance as yf
from typing import List, Optional, Dict, Any
import time
from datetime import datetime, timedelta
import pickle
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
Path("data/cache").mkdir(parents=True, exist_ok=True)

RELIABLE_TICKERS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet (Google)
    'AMZN',  # Amazon
    'META',  # Meta (Facebook)
    'NVDA',  # NVIDIA
    'TSLA',  # Tesla
    'JPM',   # JPMorgan Chase
    'V',     # Visa
    'WMT'    # Walmart
]

# Global rate limiter
last_request_time = {}
MIN_REQUEST_INTERVAL = 0.5  # seconds between requests per ticker

def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers."""
    try:
        # For testing, use a small set of reliable tickers
        if os.getenv('TEST_MODE', 'false').lower() == 'true':
            return RELIABLE_TICKERS[:5]
            
        # Try to get S&P 500 tickers from Wikipedia
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get S&P 500 tickers from Wikipedia: {e}")
            return RELIABLE_TICKERS
    except Exception as e:
        logger.error(f"Error getting S&P 500 tickers: {str(e)}")
        return RELIABLE_TICKERS

def get_cached_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get data from cache if available and not expired."""
    cache_file = f'data/cache/{ticker}.pkl'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_date = cached_data.get('date')
                # Cache is valid for 4 hours
                if cache_date and datetime.now() - cache_date < timedelta(hours=4):
                    return cached_data
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

async def _enforce_rate_limit(ticker: str) -> None:
    """Enforce rate limiting for API requests."""
    global last_request_time
    current_time = time.time()
    if ticker in last_request_time:
        elapsed = current_time - last_request_time[ticker]
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
    last_request_time[ticker] = time.time()

async def get_stock_data_async(ticker: str) -> Optional[pd.DataFrame]:
    """Get historical data for a single stock."""
    try:
        # Check cache first
        cached = get_cached_data(ticker)
        if cached:
            logger.info(f"Using cached data for {ticker}")
            return cached['data']

        # Enforce rate limit
        await _enforce_rate_limit(ticker)
        
        # Create a yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data with retries
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Use a session to manage connections
                with requests.Session() as session:
                    stock.session = session
                    data = stock.history(period='2y')
                
                if data.empty:
                    logger.error(f"Empty data received for {ticker}")
                    return None
                    
                # Add ticker column and validate
                data['Ticker'] = ticker
                if validate_stock_data(data):
                    # Save to cache
                    save_to_cache(ticker, data)
                    return data
                else:
                    logger.error(f"Invalid data received for {ticker}")
                    return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to get data for {ticker} after {max_retries} attempts: {str(e)}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {str(e)}")
        return None

async def get_batch_data_async(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Get historical data for multiple stocks concurrently."""
    try:
        # Limit concurrency to avoid overwhelming the API
        semaphore = asyncio.Semaphore(3)  # Reduced to 3 concurrent requests
        
        async def get_with_semaphore(ticker):
            async with semaphore:
                return ticker, await get_stock_data_async(ticker)
        
        # Create tasks for each ticker
        tasks = [get_with_semaphore(ticker) for ticker in tickers]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = {}
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                valid_results[result[0]] = result[1]
        
        return valid_results
        
    except Exception as e:
        logger.error(f"Error in batch data retrieval: {str(e)}")
        return {}

def get_batch_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper for get_batch_data_async."""
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async function
            results = loop.run_until_complete(get_batch_data_async(tickers))
        finally:
            # Always close the loop
            loop.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in get_batch_data: {str(e)}")
        return {}

def validate_stock_data(data: pd.DataFrame) -> bool:
    """
    Validate stock data meets minimum requirements.
    
    Args:
        data: DataFrame with stock price history
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check for minimum data points (6 months)
        if len(data) < 126:
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
            
        # Check cache first
        cached = get_cached_data('SPY')
        if cached:
            return cached['data']
            
        # Get fresh data
        with requests.Session() as session:
            spy = yf.Ticker('SPY')
            spy.session = session
            data = spy.history(start=start_date, end=end_date)
        
        if data.empty:
            logger.warning("No market data available")
            return pd.DataFrame()
            
        # Save to cache
        save_to_cache('SPY', data)
        return data
        
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return pd.DataFrame() 