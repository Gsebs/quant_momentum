"""
Trading engine for Quantitative HFT Algorithm
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from src.ml_model import HFTModel
from src.cache import RedisCache

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, model: HFTModel, cache: RedisCache, initial_capital: float = 1000000.0):
        self.model = model
        self.cache = cache
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.is_running = False
        self.max_position_size = 0.1  # 10% of capital per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        
    async def process_market_data(self, data: Dict[str, Any]) -> None:
        """Process incoming market data and execute trades"""
        try:
            # Extract data
            timestamp = data.get('timestamp')
            symbol = data.get('symbol')
            bid = data.get('bid')
            ask = data.get('ask')
            last_price = data.get('last_price')
            volume = data.get('volume')
            
            if not all([timestamp, symbol, bid, ask, last_price]):
                logger.warning(f"Missing required data fields for {symbol}")
                return
            
            # Get features and prediction
            features = self.model.compute_features(data)
            if features is not None:
                prediction = self.model.predict(features)
                
                # Update cache
                await self.update_cache(symbol, timestamp, prediction, data)
                
                # Check trade signals
                await self.check_trade_signals(symbol, prediction, bid, ask)
                
                # Monitor existing positions
                await self.monitor_positions(symbol, bid, ask)
                
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            
    async def update_cache(self, symbol: str, timestamp: str, prediction: float, data: Dict) -> None:
        """Update Redis cache with current market state"""
        try:
            cache_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'prediction': float(prediction),
                'order_book': {
                    'bid': data.get('bid'),
                    'ask': data.get('ask'),
                    'bid_size': data.get('bid_size'),
                    'ask_size': data.get('ask_size')
                },
                'trades': data.get('trades', []),
                'position': self.positions.get(symbol)
            }
            
            await self.cache.set(f'market_state:{symbol}', cache_data)
            
        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")
            
    async def check_trade_signals(self, symbol: str, prediction: float, bid: float, ask: float) -> None:
        """Check for and execute trade signals"""
        try:
            current_position = self.positions.get(symbol, 0)
            
            # Strong buy signal
            if prediction > 0.7 and current_position <= 0:
                size = self.calculate_position_size(symbol, ask)
                if size > 0:
                    await self.place_order(symbol, 'BUY', size, ask)
                    
            # Strong sell signal
            elif prediction < 0.3 and current_position >= 0:
                size = self.calculate_position_size(symbol, bid)
                if size > 0:
                    await self.place_order(symbol, 'SELL', size, bid)
                    
        except Exception as e:
            logger.error(f"Error checking trade signals: {str(e)}")
            
    async def monitor_positions(self, symbol: str, bid: float, ask: float) -> None:
        """Monitor existing positions for stop loss and take profit"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
                
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            if side == 'LONG':
                pnl_pct = (bid - entry_price) / entry_price
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    await self.close_position(symbol, bid)
                    
            elif side == 'SHORT':
                pnl_pct = (entry_price - ask) / entry_price
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    await self.close_position(symbol, ask)
                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
            
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate appropriate position size based on available capital"""
        try:
            max_capital = self.cash * self.max_position_size
            size = max_capital / price
            return round(size, 2)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    async def place_order(self, symbol: str, side: str, size: float, price: float) -> None:
        """Execute a trade order"""
        try:
            cost = size * price
            if side == 'BUY' and cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = {
                    'side': 'LONG',
                    'size': size,
                    'entry_price': price,
                    'timestamp': datetime.now().isoformat()
                }
                
            elif side == 'SELL':
                self.cash += cost
                self.positions[symbol] = {
                    'side': 'SHORT',
                    'size': size,
                    'entry_price': price,
                    'timestamp': datetime.now().isoformat()
                }
                
            trade = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
            self.trades.append(trade)
            
            # Update cache
            await self.cache.set(f'trade:{symbol}:{datetime.now().isoformat()}', trade)
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            
    async def close_position(self, symbol: str, price: float) -> None:
        """Close an existing position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
                
            size = position['size']
            entry_price = position['entry_price']
            side = position['side']
            
            if side == 'LONG':
                pnl = size * (price - entry_price)
                self.cash += (size * price)
            else:  # SHORT
                pnl = size * (entry_price - price)
                self.cash += (size * entry_price)
                
            trade = {
                'symbol': symbol,
                'side': 'SELL' if side == 'LONG' else 'BUY',
                'size': size,
                'price': price,
                'pnl': pnl,
                'timestamp': datetime.now().isoformat()
            }
            self.trades.append(trade)
            
            # Update cache
            await self.cache.set(f'trade:{symbol}:{datetime.now().isoformat()}', trade)
            
            # Remove position
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            
    async def start(self) -> None:
        """Start the trading engine"""
        try:
            self.is_running = True
            logger.info("Trading engine started")
        except Exception as e:
            logger.error(f"Error starting trading engine: {str(e)}")
            self.is_running = False
            
    async def stop(self) -> None:
        """Stop the trading engine"""
        try:
            self.is_running = False
            # Close all positions
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                if position['side'] == 'LONG':
                    await self.close_position(symbol, position['entry_price'])
                else:
                    await self.close_position(symbol, position['entry_price'])
            logger.info("Trading engine stopped")
        except Exception as e:
            logger.error(f"Error stopping trading engine: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current trading engine status"""
        return {
            'running': self.is_running,
            'cash': self.cash,
            'positions': self.positions,
            'trade_count': len(self.trades),
            'active_trades': len(self.positions)
        } 