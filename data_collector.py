import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from market_data_feed import latest_prices, reliable_tickers
from trade_simulator import execute_trade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitrageDataCollector:
    def __init__(self):
        self.price_history: Dict[str, List[Dict]] = {
            "binance": [],
            "coinbase": []
        }
        self.max_history_size = 1000  # Keep last 1000 price points
        self.opportunities: List[Dict] = []
        self.min_spread_threshold = 0.001  # 0.1% minimum spread to consider
        self.observation_window = 5  # seconds to observe after opportunity
        self.data_file = "arbitrage_data.csv"
        
    def update_price_history(self, exchange: str, price: float):
        """Update price history with timestamp."""
        timestamp = datetime.now()
        self.price_history[exchange].append({
            "timestamp": timestamp,
            "price": price
        })
        if len(self.price_history[exchange]) > self.max_history_size:
            self.price_history[exchange].pop(0)
            
    def calculate_features(self, timestamp: datetime) -> Optional[Dict]:
        """Calculate features for ML model at a given timestamp."""
        try:
            # Get price points around the timestamp
            binance_prices = [p["price"] for p in self.price_history["binance"] 
                            if abs((p["timestamp"] - timestamp).total_seconds()) <= 5]
            coinbase_prices = [p["price"] for p in self.price_history["coinbase"] 
                             if abs((p["timestamp"] - timestamp).total_seconds()) <= 5]
            
            if not binance_prices or not coinbase_prices:
                return None
                
            # Calculate basic features
            current_binance = binance_prices[-1]
            current_coinbase = coinbase_prices[-1]
            spread = current_coinbase - current_binance
            spread_pct = spread / current_binance
            
            # Calculate volatility (standard deviation of returns)
            binance_returns = np.diff(binance_prices) / binance_prices[:-1]
            coinbase_returns = np.diff(coinbase_prices) / coinbase_prices[:-1]
            
            binance_vol = np.std(binance_returns) if len(binance_returns) > 0 else 0
            coinbase_vol = np.std(coinbase_returns) if len(coinbase_returns) > 0 else 0
            
            # Calculate momentum (price change over last 5 points)
            binance_momentum = (binance_prices[-1] - binance_prices[0]) / binance_prices[0]
            coinbase_momentum = (coinbase_prices[-1] - coinbase_prices[0]) / coinbase_prices[0]
            
            # Calculate spread history
            spread_history = [c["price"] - b["price"] 
                            for b, c in zip(self.price_history["binance"][-10:], 
                                          self.price_history["coinbase"][-10:])]
            spread_std = np.std(spread_history) if spread_history else 0
            
            return {
                "timestamp": timestamp,
                "spread_pct": spread_pct,
                "binance_vol": binance_vol,
                "coinbase_vol": coinbase_vol,
                "binance_momentum": binance_momentum,
                "coinbase_momentum": coinbase_momentum,
                "spread_std": spread_std,
                "price_level": current_binance  # Normalize by price level
            }
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
            
    async def simulate_trade_outcome(self, timestamp: datetime, spread_pct: float) -> Optional[float]:
        """Simulate what would have happened if we traded at this moment."""
        try:
            # Get prices at the time
            binance_price = next(p["price"] for p in reversed(self.price_history["binance"]) 
                               if p["timestamp"] <= timestamp)
            coinbase_price = next(p["price"] for p in reversed(self.price_history["coinbase"]) 
                                if p["timestamp"] <= timestamp)
            
            # Determine trade direction
            if spread_pct > 0:
                # Coinbase higher: buy on Binance, sell on Coinbase
                profit = await execute_trade(
                    buy_exchange="binance",
                    sell_exchange="coinbase",
                    buy_price=binance_price,
                    sell_price=coinbase_price
                )
            else:
                # Binance higher: buy on Coinbase, sell on Binance
                profit = await execute_trade(
                    buy_exchange="coinbase",
                    sell_exchange="binance",
                    buy_price=coinbase_price,
                    sell_price=binance_price
                )
                
            return profit
            
        except Exception as e:
            logger.error(f"Error simulating trade outcome: {e}")
            return None
            
    def save_opportunity(self, features: Dict, profit: float):
        """Save an arbitrage opportunity with its outcome."""
        opportunity = features.copy()
        opportunity["profitable"] = 1 if profit > 0 else 0
        opportunity["profit"] = profit
        self.opportunities.append(opportunity)
        
        # Save to CSV periodically
        if len(self.opportunities) % 100 == 0:
            self.save_to_csv()
            
    def save_to_csv(self):
        """Save collected opportunities to CSV file."""
        try:
            df = pd.DataFrame(self.opportunities)
            df.to_csv(self.data_file, index=False)
            logger.info(f"Saved {len(self.opportunities)} opportunities to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            
    async def collect_data(self):
        """Main data collection loop."""
        while True:
            try:
                # Update price history
                if latest_prices["binance"]:
                    self.update_price_history("binance", latest_prices["binance"])
                if latest_prices["coinbase"]:
                    self.update_price_history("coinbase", latest_prices["coinbase"])
                    
                # Calculate current features
                features = self.calculate_features(datetime.now())
                if features:
                    spread_pct = features["spread_pct"]
                    
                    # Check if spread exceeds threshold
                    if abs(spread_pct) > self.min_spread_threshold:
                        # Simulate trade outcome
                        profit = await self.simulate_trade_outcome(
                            features["timestamp"],
                            spread_pct
                        )
                        
                        if profit is not None:
                            self.save_opportunity(features, profit)
                            
                # Short sleep to avoid excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(1)

# Create global collector instance
collector = ArbitrageDataCollector()

async def run_data_collection():
    """Run the data collection process."""
    await collector.collect_data() 