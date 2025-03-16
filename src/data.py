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
from typing import List, Optional, Dict, Any, Union
import time
from datetime import datetime, timedelta
import pickle
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .cache import redis_cache
import functools
import redis
import json
from urllib3.exceptions import MaxRetryError
from requests.exceptions import RequestException
from functools import wraps
import requests_cache

# Custom error classes
class RetryableError(Exception):
    """Error that should trigger a retry."""
    pass

class DataFetchError(Exception):
    """Error that indicates a permanent failure in data fetching."""
    pass

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
Path("data/cache").mkdir(parents=True, exist_ok=True)

# Constants for rate limiting
MIN_DELAY = 5
MAX_DELAY = 20
MAX_RETRIES = 7
BATCH_SIZE = 1    # Process one ticker at a time

# Test mode tickers (reduced set for development)
RELIABLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# Global rate limiter
last_request_time = {}
MIN_REQUEST_INTERVAL = 60.0  # seconds between requests per ticker
BASE_DELAY = 60.0  # base delay for exponential backoff
CACHE_DURATION = timedelta(hours=12)

# Constants for rate limiting and retries
INITIAL_BACKOFF = 10
MAX_BACKOFF = 120

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Initialize Redis client with SSL verification disabled
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url, ssl_cert_reqs=None)

# Initialize the cache with a 1-hour expiration
session = requests_cache.CachedSession(
    'yfinance.cache',
    backend='sqlite',
    expire_after=timedelta(hours=1)
)
yf.pdr_override()

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

def redis_cache(expire_time=300):
    """Redis cache decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Use Redis TLS URL if available, otherwise fallback to standard URL
                redis_url = os.getenv('REDIS_TLS_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
                redis_client = redis.from_url(redis_url, ssl_cert_reqs=None)
                
                # Convert any Timestamp objects in args to strings
                processed_args = []
                for arg in args:
                    if isinstance(arg, pd.Timestamp):
                        processed_args.append(arg.strftime('%Y-%m-%d'))
                    else:
                        processed_args.append(arg)
                
                # Convert any Timestamp objects in kwargs to strings
                processed_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, pd.Timestamp):
                        processed_kwargs[key] = value.strftime('%Y-%m-%d')
                    else:
                        processed_kwargs[key] = value
                
                # Generate cache key using processed arguments
                cache_key = f"{func.__name__}:{str(processed_args)}:{str(processed_kwargs)}"
                
                # Try to get cached result
                cached_result = redis_client.get(cache_key)
                if cached_result is not None:
                    return pd.read_json(cached_result)
                
                # If not cached, execute function and cache result
                result = func(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    # Convert index to strings if they are timestamps
                    if isinstance(result.index, pd.DatetimeIndex):
                        result.index = result.index.strftime('%Y-%m-%d')
                    redis_client.setex(cache_key, expire_time, result.to_json())
                return result
            except redis.RedisError as e:
                logging.error(f"Redis error in redis_cache: {str(e)}")
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in redis_cache: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def get_stock_data(ticker: str) -> Optional[Dict]:
    """
    Get stock data for a given ticker using Redis caching and yfinance.download.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        Optional[Dict]: Dictionary containing stock data and metrics, or None if data cannot be retrieved
    """
    cache_key = f"stock_data:{ticker}"
    
    try:
        # Try to get cached data first
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return pickle.loads(cached_data)
            
        # Implement exponential backoff with jitter for data retrieval
        max_retries = 5
        base_delay = 1
        max_delay = 32
        
        for attempt in range(max_retries):
            try:
                # Get historical data using download function
                df = yf.download(
                    ticker,
                    period="1y",
                    interval="1d",
                    progress=False,
                    show_errors=False,
                    timeout=10
                )
                
                if df.empty:
                    if attempt == max_retries - 1:
                        logging.warning(f"No historical data available for {ticker}")
                        return None
                else:
                    break
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to get historical data for {ticker} after {max_retries} attempts: {str(e)}")
                    return None
                    
                delay = min(max_delay, base_delay * (2 ** attempt))
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter
                
                logging.info(f"Retrying {ticker} historical data after {total_delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(total_delay)
        
        # Verify required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns for {ticker}")
            return None
            
        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        avg_volume = df['Volume'].mean()
        price_change = (current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        result = {
            'data': df,
            'current_price': current_price,
            'avg_volume': avg_volume,
            'price_change': price_change
        }
        
        # Cache the result for 1 hour
        try:
            redis_client.setex(
                cache_key,
                3600,  # 1 hour in seconds
                pickle.dumps(result)
            )
        except Exception as e:
            logging.error(f"Failed to cache data for {ticker}: {str(e)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error getting data for {ticker}: {str(e)}")
        return None

def get_batch_data(tickers: List[str]) -> List[Dict]:
    """
    Process tickers in batches with improved rate limiting.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        List of dictionaries containing stock data
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
        
        # Add longer delay between batches with jitter
        if i < num_batches - 1:
            jitter = random.uniform(0.1, 1.0)
            delay = random.uniform(MIN_DELAY * 3, MAX_DELAY * 2) + jitter
            logger.info(f"Batch complete. Waiting {delay:.2f} seconds before next batch")
            time.sleep(delay)
    
    return results

def get_test_data():
    """
    Get data for test mode using a reduced set of reliable tickers.
    """
    return get_batch_data(RELIABLE_TICKERS)
