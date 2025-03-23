import asyncio
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Dict, List
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store market data
market_data: Dict[str, Dict] = {}
reliable_tickers = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT',  # Crypto pairs that trade 24/7
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Major tech stocks
    'SPY', 'QQQ', 'VOO'  # ETFs
]

async def fetch_ticker_data(symbol: str) -> Dict:
    """Fetch ticker data with error handling."""
    try:
        if '/' in symbol:  # Crypto pair
            exchange = ccxt.binance()
            ticker = exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'volume': ticker['quoteVolume'],
                'timestamp': datetime.now().isoformat()
            }
        else:  # Stock
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get ticker '{symbol}' reason: {str(e)}")
        return None

async def update_market_data():
    """Update market data for all reliable tickers."""
    while True:
        try:
            for symbol in reliable_tickers:
                data = await fetch_ticker_data(symbol)
                if data:
                    market_data[symbol] = data
                    logger.info(f"Updated {symbol}: {data['price']}")
            
            # Keep only the last 100 data points for each ticker
            for symbol in market_data:
                if 'history' in market_data[symbol]:
                    market_data[symbol]['history'] = market_data[symbol]['history'][-100:]
            
            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            await asyncio.sleep(1)  # Wait before retrying

def get_latest_prices() -> Dict:
    """Get the latest prices for all tickers."""
    return {symbol: data['price'] for symbol, data in market_data.items()}

async def run_feeds():
    """Run all market data feeds."""
    await update_market_data()

def get_price_history(ticker: str) -> List[Dict]:
    """Get price history for a specific ticker"""
    return market_data.get(ticker, {}).get('history', []) 