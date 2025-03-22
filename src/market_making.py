<<<<<<< HEAD
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
=======
"""
Market Making Strategy for Quantitative HFT Algorithm.
"""

import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .hft_engine import HFTEngine, OrderBook
from numba import njit
from collections import deque
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_making.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Quote:
    """Represents a market making quote."""
    symbol: str
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    timestamp: datetime
    order_id: Optional[str] = None

def calculate_dynamic_spread(base_spread: float, min_spread: float, max_spread: float,
                           volatility: float, imbalance: float) -> float:
    """Calculate dynamic spread based on market conditions."""
    # Adjust spread based on volatility
    vol_factor = 1 + volatility * 2  # Double spread at 100% volatility
    
    # Adjust spread based on order book imbalance
    imb_factor = 1 + abs(imbalance)
    
    # Calculate final spread
    spread = base_spread * vol_factor * imb_factor
    
    # Ensure spread is within bounds
    return max(min_spread, min(max_spread, spread))

def calculate_skew_factor(inventory_ratio: float) -> float:
    """Calculate price skew factor based on inventory position."""
    # Use sigmoid function for smooth skew
    skew = 2 / (1 + np.exp(-4 * inventory_ratio)) - 1
    return skew

class MarketMakingStrategy:
    """Market making strategy implementation."""
    
    def __init__(self, config: Dict):
        """Initialize the market making strategy."""
        self.base_spread = config.get('base_spread', 0.001)  # 10 bps default
        self.min_spread = config.get('min_spread', 0.0005)  # 5 bps minimum
        self.max_spread = config.get('max_spread', 0.005)   # 50 bps maximum
        self.position_limit = config.get('position_limit', 100)
        self.inventory_target = config.get('inventory_target', 0)
        self.quote_validity = timedelta(milliseconds=config.get('quote_validity_ms', 50))
        self.volatility_window = config.get('volatility_window', 100)
        self.cancel_threshold_ms = config.get('cancel_threshold_ms', 500)
        self.base_quote_size = config.get('base_quote_size', 1.0)
        
        # Internal state
        self.price_history = {}  # symbol -> List[Tuple[datetime, float]]
        self.active_quotes = {}  # symbol -> List[Quote]
        self.last_quotes = {}   # symbol -> List[Quote]
        self.inventory = {}  # symbol -> position
        self.last_update = {}  # symbol -> timestamp
        self.volatility = {}  # symbol -> float
        self.ob_history = {}  # symbol -> deque
        self.quotes = {}  # symbol -> order book state
        
        # PnL tracking
        self.pnl = {}  # symbol -> Dict[str, float]
        self.trades = {}  # symbol -> List[Dict]
        self.realized_pnl = {}  # symbol -> float
        self.unrealized_pnl = {}  # symbol -> float
        self.total_volume = {}  # symbol -> float
        self.trade_count = {}  # symbol -> int
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def update_market_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update market data and recalculate metrics."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volatility[symbol] = 0.0
            self.pnl[symbol] = {
                'realized': 0.0,
                'unrealized': 0.0,
                'total': 0.0
            }
            self.trades[symbol] = []
            self.realized_pnl[symbol] = 0.0
            self.unrealized_pnl[symbol] = 0.0
            self.total_volume[symbol] = 0.0
            self.trade_count[symbol] = 0
            
        self.price_history[symbol].append((timestamp, price))
        self.last_update[symbol] = timestamp
        
        # Update unrealized PnL
        if symbol in self.inventory and self.inventory[symbol] != 0:
            position = self.inventory[symbol]
            avg_price = sum(t['price'] * t['size'] for t in self.trades[symbol]) / sum(t['size'] for t in self.trades[symbol]) if self.trades[symbol] else price
            self.unrealized_pnl[symbol] = position * (price - avg_price)
            self.pnl[symbol]['unrealized'] = self.unrealized_pnl[symbol]
            self.pnl[symbol]['total'] = self.realized_pnl[symbol] + self.unrealized_pnl[symbol]
        
        # Keep only recent prices within window
        cutoff = timestamp - timedelta(seconds=self.volatility_window)
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol]
            if t > cutoff
        ]
        
        # Calculate volatility
        if len(self.price_history[symbol]) > 1:
            prices = [p for _, p in self.price_history[symbol]]
            returns = np.diff(prices) / prices[:-1]
            self.volatility[symbol] = float(np.std(returns))
        
    def calculate_quote_prices(self, symbol: str, mid_price: float,
                             imbalance: float, timestamp: datetime) -> Tuple[float, float]:
        """Calculate bid and ask prices."""
        # Get volatility
        volatility = self.volatility.get(symbol, 0)
            
        # Calculate dynamic spread
        spread = calculate_dynamic_spread(
            self.base_spread,
            self.min_spread,
            self.max_spread,
            volatility,
            imbalance
        )
        
        # Calculate inventory skew
        inventory_ratio = self.inventory.get(symbol, 0) / self.position_limit
        skew = calculate_skew_factor(inventory_ratio)
        
        # Apply skew to spread
        half_spread = spread / 2
        bid_price = mid_price * (1 - half_spread * (1 + skew))
        ask_price = mid_price * (1 + half_spread * (1 - skew))
        
        return bid_price, ask_price
        
    def calculate_quote_size(self, symbol: str, side: str) -> float:
        """Calculate quote size based on inventory."""
        current_inventory = self.inventory.get(symbol, 0)
        inventory_ratio = current_inventory / self.position_limit
        
        # Base size
        size = self.base_quote_size
        
        # Adjust size based on inventory
        if side == 'bid':
            if current_inventory > 0:
                # Reduce bid size when long
                size *= max(0, 1 - inventory_ratio)
            else:
                # Increase bid size when short
                size *= min(2, 1 - inventory_ratio)
        else:  # ask
            if current_inventory > 0:
                # Increase ask size when long
                size *= min(2, 1 + inventory_ratio)
            else:
                # Reduce ask size when short
                size *= max(0, 1 + inventory_ratio)
                
        return size
        
    def should_cancel_quotes(self, symbol: str, timestamp: datetime) -> bool:
        """Determine if quotes should be cancelled."""
        if symbol not in self.active_quotes:
            return False
            
        for quote in self.active_quotes[symbol]:
            age = (timestamp - quote.timestamp).total_seconds() * 1000
            if age > self.cancel_threshold_ms:
                return True
                
        return False
        
    def update_quotes(self, symbol: str, mid_price: float,
                     imbalance: float, timestamp: datetime) -> List[Quote]:
        """Update quotes for a symbol."""
        # Calculate new quote prices
        bid_price, ask_price = self.calculate_quote_prices(
            symbol, mid_price, imbalance, timestamp
        )
        
        # Calculate quote sizes
        bid_size = self.calculate_quote_size(symbol, 'bid')
        ask_size = self.calculate_quote_size(symbol, 'ask')
        
        # Create new quotes
        quotes = [
            Quote(symbol, 'bid', bid_price, bid_size, timestamp),
            Quote(symbol, 'ask', ask_price, ask_size, timestamp)
        ]
        
        # Update active quotes and last quotes
        self.active_quotes[symbol] = quotes
        self.last_quotes[symbol] = quotes.copy()
        return quotes
        
    def update_inventory(self, symbol: str, side: str, size: float, price: float, timestamp: datetime) -> None:
        """Update inventory and PnL after a trade."""
        if symbol not in self.inventory:
            self.inventory[symbol] = 0
            self.pnl[symbol] = {'realized': 0.0, 'unrealized': 0.0, 'total': 0.0}
            self.trades[symbol] = []
            self.realized_pnl[symbol] = 0.0
            self.unrealized_pnl[symbol] = 0.0
            self.total_volume[symbol] = 0.0
            self.trade_count[symbol] = 0
            
        # Record trade
        trade = {
            'timestamp': timestamp,
            'side': side,
            'size': size,
            'price': price
        }
        self.trades[symbol].append(trade)
        
        # Update position
        old_position = self.inventory[symbol]
        if side.lower() == 'buy':
            self.inventory[symbol] += size
        else:
            self.inventory[symbol] -= size
            
        # Update volume and trade count
        self.total_volume[symbol] += size
        self.trade_count[symbol] += 1
        
        # Calculate realized PnL
        if (old_position > 0 and side.lower() == 'sell') or (old_position < 0 and side.lower() == 'buy'):
            # Closing position
            avg_entry = sum(t['price'] * t['size'] for t in self.trades[symbol][:-1]) / sum(t['size'] for t in self.trades[symbol][:-1])
            if side.lower() == 'sell':
                self.realized_pnl[symbol] += size * (price - avg_entry)
            else:
                self.realized_pnl[symbol] += size * (avg_entry - price)
                
        # Update PnL dictionary
        self.pnl[symbol]['realized'] = self.realized_pnl[symbol]
        self.pnl[symbol]['unrealized'] = self.unrealized_pnl[symbol]
        self.pnl[symbol]['total'] = self.realized_pnl[symbol] + self.unrealized_pnl[symbol]
            
    def get_signals(self) -> Dict:
        """Get current trading signals."""
        signals = {}
        for symbol in self.price_history:
            if len(self.price_history[symbol]) < 2:
                continue
                
            # Calculate signal based on recent price movement and inventory
            prices = [p for _, p in self.price_history[symbol]]
            returns = np.diff(prices) / prices[:-1]
            momentum = np.mean(returns)
            
            # Combine momentum with inventory position
            inventory_ratio = self.inventory.get(symbol, 0) / self.position_limit
            signal = momentum * (1 - abs(inventory_ratio))  # Reduce signal when inventory is high
            
            signals[symbol] = {
                'momentum': momentum,
                'inventory_ratio': inventory_ratio,
                'signal': signal
            }
            
        return signals

    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        try:
            metrics = {
                'inventory': self.inventory.copy(),
                'active_quotes': {
                    symbol: len(quotes)
                    for symbol, quotes in self.active_quotes.items()
                },
                'pnl': {
                    symbol: pnl.copy()
                    for symbol, pnl in self.pnl.items()
                },
                'volume': self.total_volume.copy(),
                'trade_count': self.trade_count.copy(),
                'volatility': self.volatility.copy()
            }
            
            # Calculate total PnL across all symbols
            total_realized = sum(self.realized_pnl.values())
            total_unrealized = sum(self.unrealized_pnl.values())
            metrics['total_pnl'] = {
                'realized': total_realized,
                'unrealized': total_unrealized,
                'total': total_realized + total_unrealized
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def handle_order_book(self, symbol, book_update):
        """Process order book update and adjust quotes"""
        try:
            # Update local order book
            self.quotes[symbol] = book_update
            
            # Calculate order book imbalance
            if isinstance(book_update, dict):
                total_bid_size = sum(float(size) for _, size in book_update.get('bids', []))
                total_ask_size = sum(float(size) for _, size in book_update.get('asks', []))
                total_size = total_bid_size + total_ask_size
                
                if total_size > 0:
                    imbalance = (total_bid_size - total_ask_size) / total_size
                else:
                    imbalance = 0
                    
                # Store in history
                if symbol not in self.ob_history:
                    self.ob_history[symbol] = deque(maxlen=100)
                self.ob_history[symbol].append((datetime.now(), imbalance))
                
                # Update quotes if needed
                mid_price = (float(book_update['bids'][0][0]) + float(book_update['asks'][0][0])) / 2
                self.update_quotes(symbol, mid_price, imbalance, datetime.now())
                
        except Exception as e:
            self.logger.error(f"Error handling order book update: {str(e)}")
            
    def _calculate_imbalance(self, symbol):
        """Calculate order book imbalance metric"""
        book = self.quotes.get(symbol)
        if not book or 'bids' not in book or 'asks' not in book:
            return 0
            
        bid_volume = sum(bid[1] for bid in book['bids'][:self.ob_levels])
        ask_volume = sum(ask[1] for ask in book['asks'][:self.ob_levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume
        
    def _adjust_quotes(self, symbol, imbalance):
        """Adjust quotes based on order book imbalance and other factors"""
        now = datetime.now()
        last_quote = self.last_quote_time.get(symbol, datetime.min)
        
        # Check if enough time has passed since last quote
        if (now - last_quote).total_seconds() * 1000 < self.quote_validity.total_seconds() * 1000:
            return
            
        # Get mid price
        book = self.quotes.get(symbol)
        if not book or not book['bids'] or not book['asks']:
            return
            
        mid_price = (book['bids'][0][0] + book['asks'][0][0]) / 2
        
        # Calculate dynamic spread based on factors
        spread = self._calculate_spread(symbol, imbalance)
        
        # Adjust quotes based on inventory
        position = self.positions.get(symbol, 0)
        inventory_skew = position / self.position_limit if self.position_limit > 0 else 0
        
        # Skew quotes based on inventory and imbalance
        bid_offset = spread/2 * (1 - inventory_skew)
        ask_offset = spread/2 * (1 + inventory_skew)
        
        # Further adjust based on ML prediction if available
        if symbol in self.ml_predictions:
            pred = self.ml_predictions[symbol]
            pred_impact = 0.1  # Maximum impact of prediction
            bid_offset *= (1 + pred * pred_impact)
            ask_offset *= (1 - pred * pred_impact)
        
        # Place orders
        self._place_quotes(symbol, mid_price - bid_offset, mid_price + ask_offset)
        self.last_quote_time[symbol] = now
        
    def _calculate_spread(self, symbol, imbalance):
        """Calculate dynamic spread based on market conditions"""
        spread = self.base_spread
        
        # Widen spread in volatile markets
        if symbol in self.volatility:
            vol_impact = self.volatility[symbol] / 0.001  # Normalize to basis points
            spread *= (1 + vol_impact)
            
        # Adjust for order book imbalance
        imbalance_impact = abs(imbalance) * 0.5
        spread *= (1 + imbalance_impact)
        
        # Ensure spread is within bounds
        return max(min(spread, self.max_spread), self.min_spread)
        
    def _place_quotes(self, symbol, bid_price, ask_price):
        """Place or update quotes in the market"""
        # Cancel existing orders first
        self._cancel_existing_orders(symbol)
        
        # Calculate quote sizes based on inventory
        position = self.positions.get(symbol, 0)
        base_size = self.position_limit * 0.1  # 10% of max position per order
        
        # Adjust sizes based on inventory
        if abs(position) > self.position_limit * self.position_reduction_threshold:
            # Reduce position by quoting larger size on the reducing side
            if position > 0:
                ask_size = base_size * 1.5
                bid_size = base_size * 0.5
            else:
                ask_size = base_size * 0.5
                bid_size = base_size * 1.5
        else:
            ask_size = bid_size = base_size
            
        # Place new orders
        self._place_order(symbol, "BUY", bid_price, bid_size)
        self._place_order(symbol, "SELL", ask_price, ask_size)
        
    def _cancel_existing_orders(self, symbol):
        """Cancel existing orders for the symbol"""
        # Implementation depends on exchange interface
        pass
        
    def _place_order(self, symbol, side, price, size):
        """Place a new order in the market"""
        # Implementation depends on exchange interface
        pass
        
    def handle_trade(self, trade):
        """Process trade execution"""
        try:
            symbol = trade['symbol']
            size = trade['size'] * (1 if trade['side'] == "BUY" else -1)
            
            # Update position
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            
            # Update PnL
            trade_pnl = self._calculate_trade_pnl(trade)
            self.pnl += trade_pnl
            
            # Update statistics
            self.trades_count += 1
            if trade_pnl > 0:
                self.win_count += 1
                
            # Store trade
            self.trades.append(trade)
            
            # Adjust quotes after trade
            if symbol in self.quotes:
                imbalance = self._calculate_imbalance(symbol)
                self._adjust_quotes(symbol, imbalance)
                
        except Exception as e:
            logger.error(f"Error handling trade: {str(e)}")
            
    def _calculate_trade_pnl(self, trade):
        """Calculate PnL for a single trade"""
        # Simple implementation - can be enhanced with transaction costs
        return trade['price'] * trade['size'] * (-1 if trade['side'] == "BUY" else 1)
        
    def update_ml_prediction(self, symbol, prediction):
        """Update ML model prediction for a symbol"""
        self.ml_predictions[symbol] = prediction
        
    def get_metrics(self):
        """Get strategy performance metrics"""
        return {
            'pnl': self.pnl,
            'trades_count': self.trades_count,
            'win_rate': self.win_count / self.trades_count if self.trades_count > 0 else 0,
            'positions': self.positions.copy(),
            'active_orders': len(self.quotes)
        }

    def update(self, ticker: str):
        """Main update loop for the market making strategy"""
        try:
            # Check and update quotes
            if self.should_cancel_quotes(ticker, datetime.now()):
                self.cancel_quotes(ticker)
                
            self.update_quotes(ticker, self.engine.get_order_book(ticker).get_mid_price(), datetime.now())
            
            # Check for quote executions
            self._check_quote_execution(ticker)
            
        except Exception as e:
            logger.error(f"Error in strategy update: {str(e)}")
            
    def _should_make_market(self, ticker: str) -> bool:
        """Determine if we should make markets based on conditions"""
        try:
            # Check if market is active
            book = self.engine.get_order_book(ticker)
            if not book or not book.is_valid():
                return False
                
            # Check spread
            best_bid = book.get_best_bid()
            best_ask = book.get_best_ask()
            if not best_bid or not best_ask:
                return False
                
            spread = (best_ask - best_bid) / best_bid
            if spread > self.max_spread:
                return False
                
            # Check volatility
            if ticker in self.volatility and self.volatility[ticker] > 0.002:  # Too volatile
                return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking market making conditions: {str(e)}")
            return False
            
    def _check_quote_execution(self, ticker: str):
        """Check if our quotes would be hit by current market prices"""
        try:
            quotes = self.quotes.get(ticker)
            if not quotes:
                return
                
            book = self.engine.order_books.get(ticker)
            if not book:
                return
                
            # Get current market prices
            market_bid = book.bid_prices[0] if len(book.bid_prices) > 0 else None
            market_ask = book.ask_prices[0] if len(book.ask_prices) > 0 else None
            
            if not market_bid or not market_ask:
                return
                
            # Check if our quotes would be hit
            for quote in quotes:
                if quote.side == 'bid' and quote.price >= market_ask:
                    self._execute_trade(ticker, 'buy', quote.price, quote.size)
                elif quote.side == 'ask' and quote.price <= market_bid:
                    self._execute_trade(ticker, 'sell', quote.price, quote.size)
                
        except Exception as e:
            logger.error(f"Error checking quote execution for {ticker}: {str(e)}")
            
    def _execute_trade(self, ticker: str, side: str, price: float, size: float):
        """Execute a trade when our quote is hit"""
        try:
            trade = {
                'ticker': ticker,
                'side': side,
                'price': price,
                'size': size,
                'timestamp': datetime.now()
            }
            
            # Add trade to engine's queue
            self.engine.trade_queue.put(trade)
            
            # Cancel quotes after execution
            self.cancel_quotes(ticker)
            
        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {str(e)}")
            
    def cancel_quotes(self, ticker: str):
        """Cancel all active quotes for a ticker"""
        try:
            if ticker in self.quotes:
                self.engine.cancel_all_orders(ticker)
                del self.quotes[ticker]
                
        except Exception as e:
            logger.error(f"Error cancelling quotes: {str(e)}")
            
    def place_quotes(self, ticker: str, bid_price: float, ask_price: float, 
                    bid_size: float, ask_size: float):
        """Place new quotes in the market"""
        try:
            # Cancel existing quotes first
            self.cancel_quotes(ticker)
            
            # Place new quotes
            if bid_size > 0:
                self.engine.place_order(ticker, 'buy', bid_size, bid_price, 'limit')
            if ask_size > 0:
                self.engine.place_order(ticker, 'sell', ask_size, ask_price, 'limit')
                
            # Update tracking
            self.quotes[ticker] = [Quote(ticker, 'bid', bid_price, bid_size, datetime.now())]
            
        except Exception as e:
            logger.error(f"Error placing quotes: {str(e)}")
            
    def _should_make_market(self, ticker: str) -> bool:
        """Determine if we should make markets based on conditions"""
        try:
            # Check if market is active
            book = self.engine.get_order_book(ticker)
            if not book or not book.is_valid():
                return False
                
            # Check spread
            best_bid = book.get_best_bid()
            best_ask = book.get_best_ask()
            if not best_bid or not best_ask:
                return False
                
            spread = (best_ask - best_bid) / best_bid
            if spread > self.max_spread:
                return False
                
            # Check volatility
            if ticker in self.volatility and self.volatility[ticker] > 0.002:  # Too volatile
                return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking market making conditions: {str(e)}")
            return False 
>>>>>>> heroku/main
