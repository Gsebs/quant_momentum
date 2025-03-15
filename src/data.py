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
import functools
import redis
import json
from urllib3.exceptions import MaxRetryError
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
Path("data/cache").mkdir(parents=True, exist_ok=True)

# Constants for rate limiting
MIN_DELAY = 60.0  # Minimum delay between requests
MAX_DELAY = 120.0  # Maximum delay between requests
MAX_RETRIES = 5   # Maximum number of retries per request
BATCH_SIZE = 1    # Process one ticker at a time

# Test mode tickers (reduced set for development)
RELIABLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# Global rate limiter
last_request_time = {}
MIN_REQUEST_INTERVAL = 60.0  # seconds between requests per ticker
BASE_DELAY = 60.0  # base delay for exponential backoff
CACHE_DURATION = timedelta(hours=12)

# Constants for rate limiting and retries
INITIAL_BACKOFF = 60  # Initial backoff time in seconds
MAX_BACKOFF = 300  # Maximum backoff time in seconds

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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

def redis_cache(expire_time=300):
    """
    A decorator that implements Redis caching with error handling and automatic retry logic.
    
    Args:
        expire_time (int): Time in seconds for the cache to expire. Defaults to 300 seconds (5 minutes).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create Redis client with SSL verification disabled
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            redis_client = redis.from_url(redis_url, ssl_cert_reqs=None, decode_responses=True)
            
            # Create a cache key based on function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            try:
                # Try to get cached value
                cached_value = redis_client.get(cache_key)
                if cached_value:
                    return eval(cached_value)  # Convert string back to dictionary
                
                # If no cached value, call the function
                result = func(*args, **kwargs)
                
                # Cache the result if it's valid
                if result:
                    redis_client.setex(cache_key, expire_time, str(result))
                
                return result
            
            except (redis.RedisError, Exception) as e:
                logger.error(f"Redis cache error: {str(e)}")
                # If cache fails, just execute the function
                return func(*args, **kwargs)
            
        return wrapper
    return decorator

@redis_cache(expire_time=300)  # Cache for 5 minutes
def get_stock_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch stock data with improved error handling and rate limiting.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
    
    Returns:
        dict: Dictionary containing ticker, price, volume, and price_change
    """
    try:
        # Convert dates to timestamps
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Construct URL with proper parameters
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        params = {
            "symbol": ticker,
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits"
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }

        # Add random delay to avoid rate limiting
        sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
        logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds before requesting {ticker}")
        time.sleep(sleep_time)

        # Make request with session
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(base_url + ticker, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract price data
            chart = data['chart']['result'][0]
            timestamps = chart['timestamp']
            quotes = chart['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime([datetime.fromtimestamp(x) for x in timestamps]),
                'Open': quotes['open'],
                'High': quotes['high'],
                'Low': quotes['low'],
                'Close': quotes['close'],
                'Volume': quotes['volume']
            })
            df.set_index('Date', inplace=True)

            # Calculate metrics
            current_price = df['Close'].iloc[-1]
            avg_volume = df['Volume'].mean()
            price_change = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

            return {
                'current_price': current_price,
                'avg_volume': avg_volume,
                'price_change': price_change,
                'data': df
            }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:  # Too Many Requests
            logger.warning(f"Rate limit hit for {ticker}, waiting {MAX_BACKOFF} seconds")
            time.sleep(MAX_BACKOFF)
            raise RetryableError(f"Rate limit exceeded for {ticker}")
        else:
            logger.error(f"HTTP error occurred while fetching {ticker}: {str(e)}")
            raise DataFetchError(f"Failed to fetch data for {ticker}: {str(e)}")
            
    except (KeyError, IndexError) as e:
        logger.error(f"Data parsing error for {ticker}: {str(e)}")
        raise DataFetchError(f"Failed to parse data for {ticker}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error while fetching {ticker}: {str(e)}")
        raise DataFetchError(f"Unexpected error for {ticker}: {str(e)}")

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
