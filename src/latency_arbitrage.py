import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class LatencyArbitrageStrategy:
    def __init__(self, 
                 symbols: List[str],
                 fee_rate: float = 0.001,  # 0.1% fee per trade
                 min_profit_threshold: float = 0.002,  # 0.2% minimum profit
                 max_position: float = 1.0,  # Maximum position size in BTC
                 cooldown_ms: int = 100):  # Cooldown between trades
        self.symbols = symbols
        self.fee_rate = fee_rate
        self.min_profit_threshold = min_profit_threshold
        self.max_position = max_position
        self.cooldown_ms = cooldown_ms
        
        # State variables
        self.last_trade_time: Dict[str, float] = {}
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.trades: List[Dict] = []
        self.market_state: Dict[str, Dict] = {}
        
        # Performance tracking
        self.total_profit = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.missed_opportunities = 0
        
        # Latency tracking
        self.execution_latencies: List[float] = []
        self.price_update_latencies: Dict[str, List[float]] = {
            symbol: [] for symbol in symbols
        }
    
    def update_market_state(self, exchange: str, symbol: str, data: Dict):
        """Update market state with new data from an exchange."""
        key = f"{exchange}_{symbol}"
        
        # Store previous state for comparison
        prev_state = self.market_state.get(key, {})
        
        # Update state
        self.market_state[key] = {
            'timestamp': data['timestamp'],
            'price': data['ticker']['last'],
            'bid': data['ticker']['bid'],
            'ask': data['ticker']['ask'],
            'bids': data['orderbook']['bids'],
            'asks': data['orderbook']['asks']
        }
        
        # Calculate and store update latency
        if prev_state:
            latency = (data['timestamp'] - prev_state['timestamp']) * 1000  # ms
            self.price_update_latencies[symbol].append(latency)
            
            # Keep only last 1000 latency measurements
            if len(self.price_update_latencies[symbol]) > 1000:
                self.price_update_latencies[symbol].pop(0)
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade."""
        current_time = time.time() * 1000  # Convert to ms
        last_trade = self.last_trade_time.get(symbol, 0)
        return (current_time - last_trade) >= self.cooldown_ms
    
    def _calculate_profit_potential(self, 
                                  fast_price: float, 
                                  slow_price: float,
                                  volume_available: float) -> Tuple[float, float]:
        """Calculate potential profit and optimal trade size."""
        # Calculate price gap
        gap = fast_price - slow_price
        gap_pct = gap / slow_price
        
        # Calculate fees
        total_fee_pct = 2 * self.fee_rate  # Fee for both trades
        
        # Calculate potential profit percentage
        profit_pct = gap_pct - total_fee_pct
        
        # Determine optimal trade size
        max_trade_size = min(
            volume_available,
            self.max_position,
            abs(self.positions[symbol])  # Don't exceed current position
        )
        
        # Calculate absolute profit
        potential_profit = profit_pct * slow_price * max_trade_size
        
        return potential_profit, max_trade_size
    
    async def check_opportunity(self, 
                              symbol: str,
                              fast_exchange: str,
                              slow_exchange: str,
                              ml_confidence: float) -> Optional[Dict]:
        """Check for and evaluate arbitrage opportunities."""
        try:
            # Get market state for both exchanges
            fast_state = self.market_state.get(f"{fast_exchange}_{symbol}")
            slow_state = self.market_state.get(f"{slow_exchange}_{symbol}")
            
            if not fast_state or not slow_state:
                return None
            
            # Get current prices
            fast_price = fast_state['price']
            slow_price = slow_state['price']
            
            # Calculate price gap
            gap_pct = (fast_price - slow_price) / slow_price
            
            # Check if gap exceeds minimum threshold (including fees)
            if abs(gap_pct) <= (self.min_profit_threshold + 2 * self.fee_rate):
                return None
            
            # Check cooldown period
            if not self._check_cooldown(symbol):
                self.missed_opportunities += 1
                return None
            
            # Get available volume
            volume_available = min(
                sum(qty for _, qty in slow_state['bids'][:5]),
                sum(qty for _, qty in fast_state['asks'][:5])
            )
            
            # Calculate potential profit
            potential_profit, trade_size = self._calculate_profit_potential(
                fast_price, slow_price, volume_available
            )
            
            # If ML confidence is high enough and profit potential exists
            if ml_confidence >= 0.7 and potential_profit > 0:
                # Create trade opportunity object
                opportunity = {
                    'symbol': symbol,
                    'timestamp': time.time(),
                    'fast_exchange': fast_exchange,
                    'slow_exchange': slow_exchange,
                    'fast_price': fast_price,
                    'slow_price': slow_price,
                    'gap_pct': gap_pct,
                    'potential_profit': potential_profit,
                    'trade_size': trade_size,
                    'ml_confidence': ml_confidence
                }
                
                return opportunity
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking opportunity: {str(e)}")
            return None
    
    async def execute_trade(self, opportunity: Dict) -> Optional[Dict]:
        """Execute an arbitrage trade based on the identified opportunity."""
        try:
            symbol = opportunity['symbol']
            trade_size = opportunity['trade_size']
            start_time = time.perf_counter()
            
            # Simulate trade execution
            # In a real system, this would place actual orders
            trade_result = {
                'symbol': symbol,
                'timestamp': time.time(),
                'fast_exchange': opportunity['fast_exchange'],
                'slow_exchange': opportunity['slow_exchange'],
                'buy_price': opportunity['slow_price'],
                'sell_price': opportunity['fast_price'],
                'size': trade_size,
                'fees': 2 * self.fee_rate * trade_size * opportunity['slow_price'],
                'profit': opportunity['potential_profit'],
                'execution_time': (time.perf_counter() - start_time) * 1000  # ms
            }
            
            # Update state
            self.last_trade_time[symbol] = time.time() * 1000
            self.trades.append(trade_result)
            self.total_trades += 1
            
            if trade_result['profit'] > 0:
                self.successful_trades += 1
                self.total_profit += trade_result['profit']
            
            # Track execution latency
            self.execution_latencies.append(trade_result['execution_time'])
            if len(self.execution_latencies) > 1000:
                self.execution_latencies.pop(0)
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        return {
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'success_rate': self.successful_trades / max(1, self.total_trades),
            'missed_opportunities': self.missed_opportunities,
            'avg_execution_latency': np.mean(self.execution_latencies) if self.execution_latencies else 0,
            'avg_price_update_latency': {
                symbol: np.mean(latencies) if latencies else 0
                for symbol, latencies in self.price_update_latencies.items()
            }
        } 