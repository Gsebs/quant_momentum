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
from .cache import redis_cache

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
Path("data/cache").mkdir(parents=True, exist_ok=True)

# Constants for rate limiting
MIN_DELAY = 2.0  # Minimum delay between requests
MAX_DELAY = 5.0  # Maximum delay between requests
MAX_RETRIES = 3  # Maximum number of retries per request
BATCH_SIZE = 3   # Number of tickers to process in each batch

# Test mode tickers (reduced set for development)
RELIABLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# Global rate limiter
last_request_time = {}
MIN_REQUEST_INTERVAL = 5.0  # seconds between requests per ticker
BASE_DELAY = 10.0  # increased base delay for exponential backoff
MAX_DELAY = 20.0  # reduced max delay due to Heroku timeout
CACHE_DURATION = timedelta(hours=12)  # increased cache duration

# User agent headers with more realistic browser info
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
    'Sec-CH-UA': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Sec-CH-UA-Mobile': '?0',
    'Sec-CH-UA-Platform': '"Windows"'
}

def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers."""
    try:
        # For testing, use a small set of reliable tickers
        if os.getenv('TEST_MODE', 'false').lower() == 'true':
            return RELIABLE_TICKERS[:5]
            
        # Try to get S&P 500 tickers from Wikipedia
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options={'User-Agent': HEADERS['User-Agent']})[0]
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
                # Cache is valid for CACHE_DURATION
                if cache_date and datetime.now() - cache_date < CACHE_DURATION:
                    logger.info(f"Using cached data for {ticker} from {cache_date}")
                    return cached_data
                else:
                    logger.info(f"Cache expired for {ticker}, last updated {cache_date}")
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

def get_stock_data_sync(ticker: str) -> Optional[pd.DataFrame]:
    """Get historical data for a single stock synchronously."""
    try:
        # Check cache first
        cached = get_cached_data(ticker)
        if cached:
            logger.info(f"Using cached data for {ticker}")
            return cached['data']

        # Create a session with robust headers
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # Get historical data with retries and exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                # Add jitter to avoid synchronized requests
                jitter = random.uniform(0.1, 1.0)
                delay = min(BASE_DELAY * (2 ** attempt) + jitter, MAX_DELAY)
                
                if attempt > 0:
                    logger.info(f"Waiting {delay:.2f} seconds before retry {attempt + 1} for {ticker}")
                    time.sleep(delay)
                
                # Enforce rate limiting
                current_time = time.time()
                if ticker in last_request_time:
                    elapsed = current_time - last_request_time[ticker]
                    if elapsed < MIN_REQUEST_INTERVAL:
                        time.sleep(MIN_REQUEST_INTERVAL - elapsed + jitter)
                last_request_time[ticker] = time.time()
                
                # Create a new Ticker object for each attempt
                stock = yf.Ticker(ticker, session=session)
                
                # Try to get info first to validate the ticker
                info = stock.info
                if not info:
                    logger.error(f"No info available for {ticker}")
                    return None
                
                # Get historical data with specific parameters
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2 years
                
                data = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    auto_adjust=True,
                    timeout=30
                )
                
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
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit hit for {ticker} on attempt {attempt + 1}")
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Max retries reached for {ticker} due to rate limiting")
                        return None
                    continue
                else:
                    logger.error(f"HTTP error for {ticker}: {str(e)}")
                    return None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                else:
                    logger.error(f"Failed to get data for {ticker} after {MAX_RETRIES} attempts: {str(e)}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {str(e)}")
        return None

def get_batch_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Get historical data for multiple stocks."""
    try:
        # Process tickers in smaller batches
        results = {}
        batch_size = 3  # Process only 3 stocks at a time
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(tickers) + batch_size - 1)//batch_size}")
            
            for j, ticker in enumerate(batch):
                try:
                    # Add delay between requests with jitter
                    if j > 0:  # Don't delay first request in batch
                        jitter = random.uniform(1.0, 2.0)
                        delay = MIN_REQUEST_INTERVAL * jitter
                        logger.info(f"Waiting {delay:.2f} seconds before processing {ticker}")
                        time.sleep(delay)
                    
                    data = get_stock_data_sync(ticker)
                    if data is not None:
                        results[ticker] = data
                        logger.info(f"Successfully retrieved data for {ticker} ({i + j + 1}/{len(tickers)})")
                    else:
                        logger.warning(f"No data retrieved for {ticker} ({i + j + 1}/{len(tickers)})")
                    
                except Exception as e:
                    logger.error(f"Error getting data for {ticker}: {str(e)}")
                    continue
            
            # Add longer delay between batches
            if i + batch_size < len(tickers):
                delay = random.uniform(15, 20)
                logger.info(f"Taking a longer break of {delay:.2f} seconds after processing batch {i//batch_size + 1}")
                time.sleep(delay)
        
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
            
        # Get fresh data with improved rate limiting
        session = requests.Session()
        session.headers.update(HEADERS)
        
        for attempt in range(MAX_RETRIES):
            try:
                # Add jitter to avoid synchronized requests
                jitter = random.uniform(0.1, 1.0)
                delay = min(BASE_DELAY * (2 ** attempt) + jitter, MAX_DELAY)
                
                if attempt > 0:
                    logger.info(f"Waiting {delay:.2f} seconds before retry {attempt + 1} for SPY")
                    time.sleep(delay)
                
                # Enforce rate limiting
                current_time = time.time()
                if 'SPY' in last_request_time:
                    elapsed = current_time - last_request_time['SPY']
                    if elapsed < MIN_REQUEST_INTERVAL:
                        time.sleep(MIN_REQUEST_INTERVAL - elapsed + jitter)
                last_request_time['SPY'] = time.time()
                
                spy = yf.Ticker('SPY', session=session)
                data = spy.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    auto_adjust=True,
                    timeout=30
                )
                
                if data.empty:
                    logger.warning("No market data available")
                    if attempt < MAX_RETRIES - 1:
                        continue
                    return pd.DataFrame()
                
                # Save to cache
                save_to_cache('SPY', data)
                return data
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit hit for SPY on attempt {attempt + 1}")
                    if attempt == MAX_RETRIES - 1:
                        logger.error("Max retries reached for SPY due to rate limiting")
                        return pd.DataFrame()
                    continue
                else:
                    logger.error(f"HTTP error for SPY: {str(e)}")
                    return pd.DataFrame()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for SPY: {str(e)}")
                else:
                    logger.error(f"Failed to get SPY data after {MAX_RETRIES} attempts: {str(e)}")
                    return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return pd.DataFrame()

@redis_cache(expire_time=21600)  # Cache for 6 hours
def get_stock_data(ticker, period="1mo"):
    """
    Fetch stock data for a given ticker with caching and rate limiting.
    """
    logger.info(f"Fetching data for {ticker}")
    
    for attempt in range(MAX_RETRIES):
        try:
            # Add jitter to delay
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            if attempt > 0:
                logger.info(f"Waiting {delay:.2f} seconds before retry {attempt + 1} for {ticker}")
                time.sleep(delay)

            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval="1d")
            
            if hist.empty:
                raise ValueError(f"No data returned for {ticker}")
                
            # Process the data
            data = {
                'ticker': ticker,
                'current_price': hist['Close'][-1] if not hist.empty else None,
                'volume': hist['Volume'][-1] if not hist.empty else None,
                'price_change': ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) if not hist.empty else None,
                'last_updated': datetime.now().isoformat()
            }
            
            # Validate data
            if data['current_price'] is None or data['volume'] is None:
                raise ValueError(f"Invalid data returned for {ticker}")
            
            return data

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise

def get_batch_data(tickers):
    """
    Process tickers in batches with rate limiting.
    """
    results = []
    num_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        batch_start = i * BATCH_SIZE
        batch_end = min((i + 1) * BATCH_SIZE, len(tickers))
        batch = tickers[batch_start:batch_end]
        
        logger.info(f"Processing batch {i + 1} of {num_batches}")
        
        # Process each ticker in the batch
        for ticker in batch:
            try:
                data = get_stock_data(ticker)
                results.append(data)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Add delay between batches
        if i < num_batches - 1:
            delay = random.uniform(MIN_DELAY * 2, MAX_DELAY * 2)
            logger.info(f"Waiting {delay:.2f} seconds before next batch")
            time.sleep(delay)
    
    return results

def get_test_data():
    """
    Get data for test mode using a reduced set of reliable tickers.
    """
    return get_batch_data(RELIABLE_TICKERS)
