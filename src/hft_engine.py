"""
High-Frequency Trading Engine Implementation
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
from dataclasses import dataclass
import numba
from numba import jit
from decimal import Decimal
import time
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Order:
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    order_type: str
    timestamp: float
    id: str = None
    status: str = 'new'
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

@dataclass
class Trade:
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    timestamp: float
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = str(uuid.uuid4())

@dataclass
class OrderBookLevel:
    price: Decimal
    size: Decimal
    orders: int = 1

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[int, OrderBookLevel] = {}  # Changed from Decimal to int for index-based access
        self.asks: Dict[int, OrderBookLevel] = {}
        self.last_update_time: float = time.time()
        self._bid_prices: List[Decimal] = []  # Maintain sorted list of bid prices
        self._ask_prices: List[Decimal] = []  # Maintain sorted list of ask prices

    def update(self, side: str, price: Decimal, size: Decimal) -> None:
        book = self.bids if side.lower() == 'buy' else self.asks
        prices = self._bid_prices if side.lower() == 'buy' else self._ask_prices
        
        if size == 0:
            if price in prices:
                idx = prices.index(price)
                book.pop(idx, None)
                prices.remove(price)
        else:
            if price not in prices:
                prices.append(price)
                prices.sort(reverse=(side.lower() == 'buy'))
                idx = prices.index(price)
                book[idx] = OrderBookLevel(price=price, size=size)
            else:
                idx = prices.index(price)
                book[idx].size = size
                book[idx].orders += 1
                
        self.last_update_time = time.time()

    def get_best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids.get(0) if self.bids else None

    def get_best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks.get(0) if self.asks else None

    def get_bids_array(self) -> np.ndarray:
        """Convert bids to numpy array format for calculations."""
        if not self.bids:
            return np.array([])
        return np.array([(float(level.price), float(level.size), level.orders) 
                        for level in self.bids.values()])

    def get_asks_array(self) -> np.ndarray:
        """Convert asks to numpy array format for calculations."""
        if not self.asks:
            return np.array([])
        return np.array([(float(level.price), float(level.size), level.orders)
                        for level in self.asks.values()])

    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return float(best_bid.price + best_ask.price) / 2
        return None

    def get_spread(self) -> Optional[float]:
        """Calculate spread from best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return float(best_ask.price - best_bid.price)
        return None

    def get_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        bids = self.get_bids_array()
        asks = self.get_asks_array()
        
        bid_volume = np.sum(bids[:levels, 1]) if len(bids) > 0 else 0
        ask_volume = np.sum(asks[:levels, 1]) if len(asks) > 0 else 0
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume

    def get_microprice(self) -> Optional[float]:
        """Calculate microprice (size-weighted mid price)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if not best_bid or not best_ask:
            return None
            
        return calculate_microprice(
            float(best_bid.price),
            float(best_bid.size),
            float(best_ask.price),
            float(best_ask.size)
        )

@jit(nopython=True)
def calculate_microprice(bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> float:
    """Calculate microprice (size-weighted mid price)"""
    total_size = bid_size + ask_size
    if total_size == 0:
        return (bid_price + ask_price) / 2
    return (bid_price * ask_size + ask_price * bid_size) / total_size

def calculate_mid_price(bids: np.ndarray, asks: np.ndarray) -> float:
    """Calculate mid price from order book arrays."""
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    return (bids[0][0] + asks[0][0]) / 2

def calculate_spread(bids: np.ndarray, asks: np.ndarray) -> float:
    """Calculate spread from order book arrays."""
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    return asks[0][0] - bids[0][0]

class HFTEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.positions: Dict[str, Decimal] = defaultdict(Decimal)
        self.pnl: Dict[str, Decimal] = defaultdict(Decimal)
        self.capital: Decimal = Decimal(str(config.get('initial_capital', '100000')))
        self.metrics: Dict[str, float] = {}
        self.last_signal_time: Dict[str, float] = defaultdict(float)
        self.active_orders: Dict[str, Order] = {}
        self.orders: Dict[str, Order] = {}  # All orders (active and filled)
        self.symbols = config.get('symbols', [])
        self.is_running = False
        self._stop_event = asyncio.Event()
        
        # Initialize order books for all symbols
        for symbol in self.symbols:
            self.order_books[symbol] = OrderBook(symbol)

    async def start(self):
        """Start the HFT engine."""
        if self.is_running:
            logger.warning("HFT Engine is already running")
            return False
            
        try:
            self.is_running = True
            self._stop_event.clear()
            logger.info("HFT Engine started")
            
            # Start background tasks
            asyncio.create_task(self._monitor_positions())
            asyncio.create_task(self._monitor_risk())
            asyncio.create_task(self._update_metrics_loop())
            
            return True
        except Exception as e:
            logger.error(f"Error starting HFT Engine: {str(e)}")
            self.is_running = False
            return False

    async def stop(self):
        """Stop the HFT engine."""
        if not self.is_running:
            logger.warning("HFT Engine is not running")
            return
            
        try:
            self._stop_event.set()
            self.is_running = False
            
            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
                
            logger.info("HFT Engine stopped")
        except Exception as e:
            logger.error(f"Error stopping HFT Engine: {str(e)}")

    async def _monitor_positions(self):
        """Monitor positions and risk limits."""
        while not self._stop_event.is_set():
            try:
                for symbol in self.symbols:
                    position = self.positions[symbol]
                    if abs(position) > self.config.get('max_position', 1000):
                        logger.warning(f"Position limit exceeded for {symbol}: {position}")
                        # Reduce position
                        await self._reduce_position(symbol)
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}")
            await asyncio.sleep(1)

    async def _monitor_risk(self):
        """Monitor risk metrics."""
        while not self._stop_event.is_set():
            try:
                total_exposure = sum(abs(pos) for pos in self.positions.values())
                if total_exposure > self.config.get('max_exposure', 10000):
                    logger.warning(f"Total exposure limit exceeded: {total_exposure}")
                    # Reduce exposure
                    await self._reduce_exposure()
            except Exception as e:
                logger.error(f"Error in risk monitoring: {str(e)}")
            await asyncio.sleep(1)

    async def _update_metrics_loop(self):
        """Continuously update metrics."""
        while not self._stop_event.is_set():
            try:
                for symbol in self.symbols:
                    self._update_metrics(symbol)
            except Exception as e:
                logger.error(f"Error updating metrics: {str(e)}")
            await asyncio.sleep(1)

    async def _reduce_position(self, symbol: str):
        """Reduce position for a symbol."""
        position = self.positions[symbol]
        if position == 0:
            return
            
        side = 'sell' if position > 0 else 'buy'
        size = abs(position) / 2  # Reduce position by half
        
        book = self.order_books.get(symbol)
        if not book:
            return
            
        # Get best price for immediate execution
        price = book.get_best_bid().price if side == 'sell' else book.get_best_ask().price
        if not price:
            return
            
        # Create and submit order
        order = Order(
            symbol=symbol,
            side=side,
            price=price,
            size=Decimal(str(size)),
            order_type='market',
            timestamp=time.time()
        )
        await self.submit_order(order)

    async def _reduce_exposure(self):
        """Reduce total exposure across all symbols."""
        for symbol in self.symbols:
            await self._reduce_position(symbol)

    async def submit_order(self, order: Order) -> bool:
        """Submit a new order."""
        if not self.is_running:
            logger.warning("Cannot submit order - engine is not running")
            return False
            
        try:
            # Validate order
            if not self._validate_order(order):
                return False
                
            # Add to active orders
            self.active_orders[order.id] = order
            self.orders[order.id] = order
            
            logger.info(f"Order submitted: {order}")
            return True
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return False
            
        try:
            order = self.active_orders.pop(order_id)
            order.status = 'cancelled'
            logger.info(f"Order cancelled: {order}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False

    def _validate_order(self, order: Order) -> bool:
        """Validate an order before submission."""
        try:
            # Check symbol
            if order.symbol not in self.symbols:
                logger.warning(f"Invalid symbol: {order.symbol}")
                return False
                
            # Check price and size
            if order.price <= 0 or order.size <= 0:
                logger.warning(f"Invalid price or size: {order}")
                return False
                
            # Check capital
            order_value = order.price * order.size
            if order_value > self.capital:
                logger.warning(f"Insufficient capital for order: {order}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating order: {str(e)}")
            return False

    def update_order_book(self, symbol: str, bids: List[Tuple], asks: List[Tuple], timestamp: datetime) -> None:
        """Update the order book for a symbol."""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
            
        book = self.order_books[symbol]
        
        # Clear existing book
        book.bids.clear()
        book.asks.clear()
        book._bid_prices.clear()
        book._ask_prices.clear()
        
        # Update with new levels
        for price, size, orders in bids:
            book.update('buy', Decimal(str(price)), Decimal(str(size)))
            
        for price, size, orders in asks:
            book.update('sell', Decimal(str(price)), Decimal(str(size)))
            
        book.last_update_time = timestamp.timestamp()
        self._update_metrics(symbol)

    def execute_trade(self, trade: Trade) -> None:
        """Execute a trade and update positions."""
        self.trades.append(trade)
        
        # Update position
        side_multiplier = Decimal('1') if trade.side == 'buy' else Decimal('-1')
        self.positions[trade.symbol] += side_multiplier * trade.size
        
        # Update PnL
        trade_value = trade.price * trade.size
        self.pnl[trade.symbol] -= side_multiplier * trade_value
        self.capital -= side_multiplier * trade_value
        
        # Update metrics
        self._update_metrics(trade.symbol)

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get the order book for a symbol."""
        return self.order_books.get(symbol)

    def calculate_pnl(self, symbol: str) -> Decimal:
        """Calculate realized and unrealized PnL for a symbol."""
        realized_pnl = self.pnl[symbol]
        
        # Calculate unrealized PnL
        position = self.positions[symbol]
        if position != 0:
            book = self.get_order_book(symbol)
            if book:
                mid_price = book.get_mid_price()
                if mid_price:
                    unrealized_pnl = position * Decimal(str(mid_price))
                    return realized_pnl + unrealized_pnl
                    
        return realized_pnl

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = {
            'total_trades': len(self.trades),
            'total_pnl': float(sum(self.pnl.values())),
            'current_capital': float(self.capital),
            'positions': {symbol: float(pos) for symbol, pos in self.positions.items()},
            'unrealized_pnl': {}
        }
        
        # Calculate unrealized PnL for each symbol
        for symbol in self.positions:
            if symbol in self.order_books:
                book = self.order_books[symbol]
                mid_price = book.get_mid_price()
                if mid_price:
                    unrealized = float(self.positions[symbol] * Decimal(str(mid_price)))
                    metrics['unrealized_pnl'][symbol] = unrealized
                    
        return metrics

    def _update_metrics(self, symbol: str) -> None:
        """Update trading metrics for a symbol."""
        book = self.get_order_book(symbol)
        if not book:
            return
            
        self.metrics[f"{symbol}_mid_price"] = book.get_mid_price() or 0.0
        self.metrics[f"{symbol}_spread"] = book.get_spread() or 0.0
        self.metrics[f"{symbol}_imbalance"] = book.get_imbalance()
        self.metrics[f"{symbol}_position"] = float(self.positions[symbol])
        self.metrics[f"{symbol}_pnl"] = float(self.pnl[symbol])

    def handle_order_book_update(self, symbol: str, side: str, price: Union[float, Decimal], size: Union[float, Decimal]) -> None:
        """Handle order book updates."""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
            
        # Convert price and size to Decimal
        price_dec = Decimal(str(price))
        size_dec = Decimal(str(size))
        
        self.order_books[symbol].update(side, price_dec, size_dec)
        self._update_metrics(symbol)
        self._generate_signals(symbol)

    def handle_trade(self, trade: Trade) -> None:
        """Record a trade and update positions."""
        self.trades.append(trade)
        
        # Update position
        side_multiplier = Decimal('1') if trade.side == 'buy' else Decimal('-1')
        self.positions[trade.symbol] += side_multiplier * trade.size
        
        # Update PnL
        trade_value = trade.price * trade.size
        self.pnl[trade.symbol] -= side_multiplier * trade_value
        self.capital -= side_multiplier * trade_value
        
        self._update_metrics(trade.symbol)

    def _generate_signals(self, symbol: str) -> Dict[str, float]:
        """Generate trading signals based on market data."""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return {}
            
        current_time = time.time()
        if current_time - self.last_signal_time[symbol] < self.config.get('signal_interval', 1.0):
            return {}
            
        signals = {}
        
        # Calculate order book imbalance signal
        imbalance = order_book.get_imbalance()
        signals['imbalance'] = float(imbalance)
        
        # Calculate price momentum
        mid_price = order_book.get_mid_price()
        if mid_price:
            signals['momentum'] = 0.0  # TODO: Implement price momentum calculation
            
        # Position-based signals
        position = float(self.positions[symbol])
        max_position = float(self.config.get('position_limit', 100))
        signals['position_ratio'] = position / max_position if max_position != 0 else 0.0
        
        self.last_signal_time[symbol] = current_time
        return signals

    def _execute_signal(self, symbol: str, signals: Dict[str, float]) -> Optional[Order]:
        """Execute trading decision based on signals."""
        if not signals:
            return None
            
        # Simple signal-based execution logic
        imbalance = signals.get('imbalance', 0.0)
        position_ratio = signals.get('position_ratio', 0.0)
        
        # Don't trade if position limit reached
        if abs(position_ratio) >= 1.0:
            return None
            
        order_book = self.order_books.get(symbol)
        if not order_book:
            return None
            
        # Determine order side based on signals
        side = 'buy' if imbalance > 0.2 and position_ratio < 0.8 else 'sell' if imbalance < -0.2 and position_ratio > -0.8 else None
        if not side:
            return None
            
        # Get reference price
        ref_price = order_book.get_mid_price()
        if not ref_price:
            return None
            
        # Calculate order price and size
        price = float(ref_price * (1 - 0.001 if side == 'buy' else 1 + 0.001))  # 10 bps from mid
        size = float(self.config.get('base_order_size', 1.0))
        
        # Create and return order
        order = Order(
            symbol=symbol,
            side=side,
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            order_type='limit',
            timestamp=time.time()
        )
        
        self.active_orders[order.id] = order
        return order

    def get_metrics(self) -> Dict[str, float]:
        """Get current trading metrics."""
        metrics = self.metrics.copy()
        metrics['total_pnl'] = float(sum(self.pnl.values()))
        metrics['total_capital'] = float(self.capital)
        return metrics

    def record_trade(self, symbol: str, side: str, price: Decimal, size: Decimal) -> None:
        """Record a trade and update positions."""
        trade = Trade(
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            timestamp=time.time(),
            order_id=None,
            trade_id=None
        )
        self.execute_trade(trade)

    async def run(self) -> None:
        """Run the HFT engine."""
        try:
            self.is_running = True
            for symbol in self.symbols:
                if symbol not in self.order_books:
                    self.order_books[symbol] = OrderBook(symbol)
            
            while self.is_running:
                for symbol in self.symbols:
                    signals = self._generate_signals(symbol)
                    if signals:
                        order = self._execute_signal(symbol, signals)
                        if order:
                            self.active_orders[order.id] = order
                await asyncio.sleep(self.config.get('update_interval_ms', 50) / 1000)
                
        except Exception as e:
            logging.error(f"Error running HFT engine: {e}")
            self.is_running = False
            raise
        finally:
            self.is_running = False 