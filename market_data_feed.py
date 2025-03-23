import asyncio
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store market data
latest_prices: Dict[str, float] = {}
price_history: Dict[str, List[Dict]] = {}

# List of reliable tickers to monitor
RELIABLE_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'TSLA', 'JPM', 'V', 'WMT'
]

async def fetch_ticker_data(ticker: str) -> Dict:
    """Fetch latest data for a single ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d', interval='1m')
        
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            latest_prices[ticker] = latest_price
            
            # Update price history
            if ticker not in price_history:
                price_history[ticker] = []
            
            price_history[ticker].append({
                'timestamp': datetime.now().isoformat(),
                'price': latest_price,
                'volume': hist['Volume'].iloc[-1]
            })
            
            # Keep only last 100 data points
            if len(price_history[ticker]) > 100:
                price_history[ticker] = price_history[ticker][-100:]
            
            return {
                'ticker': ticker,
                'price': latest_price,
                'volume': hist['Volume'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    return None

async def update_market_data():
    """Update market data for all tickers"""
    while True:
        try:
            tasks = [fetch_ticker_data(ticker) for ticker in RELIABLE_TICKERS]
            results = await asyncio.gather(*tasks)
            
            # Filter out None results and log successful updates
            valid_results = [r for r in results if r is not None]
            logger.info(f"Updated market data for {len(valid_results)} tickers")
            
        except Exception as e:
            logger.error(f"Error in market data update: {str(e)}")
        
        # Wait for 1 minute before next update
        await asyncio.sleep(60)

async def run_feeds():
    """Start the market data feed"""
    logger.info("Starting market data feed")
    await update_market_data()

def get_latest_prices() -> Dict[str, float]:
    """Get the latest prices for all tickers"""
    return latest_prices

def get_price_history(ticker: str) -> List[Dict]:
    """Get price history for a specific ticker"""
    return price_history.get(ticker, []) 