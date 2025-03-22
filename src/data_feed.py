"""
Simulated market data feed for testing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Callable, Any, Optional
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import time
import random
import numpy as np

logger = logging.getLogger(__name__)

class MarketDataFeed:
    def __init__(self, symbols):
        self.symbols = symbols
        self.tickers = {}
        self.order_books = {}
        self.last_trades = {}
        self.running = False
        self.update_interval = 1  # Update every second
        self.base_prices = {}  # Base price for each symbol
        self.volatilities = {}  # Volatility for each symbol
        
    async def start(self):
        """Start the market data feed"""
        try:
            self.running = True
            logger.info("Starting market data feed...")
            
            # Initialize simulated data
            for symbol in self.symbols:
                # Generate random base price between 10 and 1000
                base_price = random.uniform(10, 1000)
                self.base_prices[symbol] = base_price
                
                # Generate random volatility between 0.1% and 1%
                volatility = random.uniform(0.001, 0.01)
                self.volatilities[symbol] = volatility
                
                # Initialize ticker data
                self.tickers[symbol] = {
                    'last_price': base_price,
                    'volume': random.randint(1000, 10000),
                    'timestamp': datetime.now()
                }
                
                # Initialize order book
                await self._update_market_data(symbol)
                
            if not self.tickers:
                raise ValueError("No valid tickers initialized")
                
            # Start update loop
            while self.running:
                try:
                    update_tasks = []
                    for symbol in self.tickers:
                        update_tasks.append(self._update_market_data(symbol))
                    await asyncio.gather(*update_tasks)
                    await asyncio.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error in update loop: {str(e)}")
                    await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error starting market data feed: {str(e)}")
            self.running = False
            raise
            
    async def _update_market_data(self, symbol: str):
        """Update simulated market data for a symbol"""
        try:
            base_price = self.base_prices[symbol]
            volatility = self.volatilities[symbol]
            
            # Generate new price with random walk
            price_change = np.random.normal(0, volatility * base_price)
            current_price = self.tickers[symbol]['last_price'] + price_change
            
            # Ensure price doesn't go negative
            current_price = max(0.01, current_price)
            
            # Update base price slowly to avoid drift
            self.base_prices[symbol] = 0.999 * self.base_prices[symbol] + 0.001 * current_price
            
            # Generate simulated volume
            volume = random.randint(100, 10000)
            
            # Simulate bid/ask spread based on volatility
            spread = current_price * (volatility / 10)  # Spread is 1/10th of volatility
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            # Update order book
            self.order_books[symbol] = {
                'bids': [[bid, random.randint(100, 1000)]],  # Price, Size
                'asks': [[ask, random.randint(100, 1000)]],
                'timestamp': datetime.now().timestamp()
            }
            
            # Update last trades
            trade = {
                'price': current_price,
                'amount': volume / 10,  # Average trade size
                'timestamp': datetime.now().timestamp(),
                'side': 'buy' if random.random() > 0.5 else 'sell'
            }
            
            if symbol not in self.last_trades:
                self.last_trades[symbol] = []
            self.last_trades[symbol].append(trade)
            
            # Keep only last 100 trades
            self.last_trades[symbol] = self.last_trades[symbol][-100:]
            
            # Update ticker data
            self.tickers[symbol] = {
                'last_price': current_price,
                'volume': volume,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")
            
    async def get_market_data(self) -> Dict:
        """Get current market data for all symbols"""
        data = {}
        for symbol in self.tickers:
            book = self.order_books.get(symbol, {})
            trades = self.last_trades.get(symbol, [])
            ticker = self.tickers.get(symbol, {})
            
            if book and trades and ticker:
                data[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'bid': book['bids'][0][0] if book.get('bids') else None,
                    'ask': book['asks'][0][0] if book.get('asks') else None,
                    'last_price': ticker['last_price'],
                    'volume': ticker['volume'],
                    'trades': trades[-10:],  # Last 10 trades
                    'order_book': book
                }
                
        return data
        
    async def get_historical_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Get simulated historical data for a symbol"""
        try:
            base_price = self.base_prices[symbol]
            volatility = self.volatilities[symbol]
            
            # Generate timestamps
            now = datetime.now()
            timestamps = [now - timedelta(minutes=i) for i in range(lookback)]
            timestamps.reverse()
            
            # Generate price series with random walk
            prices = [base_price]
            for _ in range(lookback - 1):
                price_change = np.random.normal(0, volatility * prices[-1])
                new_price = max(0.01, prices[-1] + price_change)
                prices.append(new_price)
                
            # Generate volume
            volumes = [random.randint(100, 10000) for _ in range(lookback)]
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + random.uniform(0, volatility)) for p in prices],
                'Low': [p * (1 - random.uniform(0, volatility)) for p in prices],
                'Close': prices,
                'Volume': volumes
            }, index=timestamps)
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def get_current_book_state(self, symbol: str) -> dict:
        """Get current order book state for a symbol."""
        return self.order_books.get(symbol, {'bids': [], 'asks': []})
        
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[dict]:
        """Get recent trades for a symbol."""
        trades = self.last_trades.get(symbol, [])
        return trades[-limit:] if trades else []
        
    async def stop(self):
        """Stop the market data feed"""
        self.running = False
        
    async def close(self):
        """Close the market data feed"""
        await self.stop() 