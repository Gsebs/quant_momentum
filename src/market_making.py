import logging
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Use StreamHandler instead of FileHandler for Heroku compatibility
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MarketMakingStrategy:
    def __init__(self, exchange_id: str = "binance", symbol: str = "BTC/USDT"):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
        })
        
    async def get_order_book(self) -> Dict:
        """Get current order book."""
        try:
            order_book = await self.exchange.fetch_order_book(self.symbol)
            logger.info(f"Retrieved order book for {self.symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {str(e)}")
            return None

    async def calculate_optimal_spread(self, order_book: Dict) -> tuple:
        """Calculate optimal bid-ask spread based on order book depth."""
        try:
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            # Calculate volume-weighted average prices
            bid_vwap = np.average(bids[:5, 0], weights=bids[:5, 1])
            ask_vwap = np.average(asks[:5, 0], weights=asks[:5, 1])
            
            # Calculate optimal spread
            spread = (ask_vwap - bid_vwap) / ((ask_vwap + bid_vwap) / 2)
            
            logger.info(f"Calculated optimal spread: {spread:.4f}")
            return bid_vwap, ask_vwap, spread
        except Exception as e:
            logger.error(f"Error calculating optimal spread: {str(e)}")
            return None, None, None

    async def place_orders(self, bid_price: float, ask_price: float, amount: float) -> List:
        """Place limit orders at calculated prices."""
        try:
            # Place bid order
            bid_order = await self.exchange.create_limit_buy_order(
                self.symbol, amount, bid_price
            )
            
            # Place ask order
            ask_order = await self.exchange.create_limit_sell_order(
                self.symbol, amount, ask_price
            )
            
            logger.info(f"Placed orders - Bid: {bid_price}, Ask: {ask_price}")
            return [bid_order, ask_order]
        except Exception as e:
            logger.error(f"Error placing orders: {str(e)}")
            return []

    async def monitor_positions(self) -> None:
        """Monitor open positions and adjust if necessary."""
        try:
            positions = await self.exchange.fetch_positions([self.symbol])
            for position in positions:
                logger.info(f"Position: {position['info']}")
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")

    async def run(self) -> None:
        """Main strategy loop."""
        while True:
            try:
                order_book = await self.get_order_book()
                if order_book:
                    bid_vwap, ask_vwap, spread = await self.calculate_optimal_spread(order_book)
                    if all(v is not None for v in [bid_vwap, ask_vwap, spread]):
                        amount = 0.001  # Minimum trade amount
                        await self.place_orders(bid_vwap, ask_vwap, amount)
                    await self.monitor_positions()
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
            finally:
                await asyncio.sleep(60)  # Wait for 1 minute before next iteration 