import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ExecutionSimulator:
    def __init__(self,
                 base_latency_ms: float = 5.0,  # Base latency in milliseconds
                 latency_std_ms: float = 1.0,   # Standard deviation of latency
                 slippage_bps: float = 0.5,     # Average slippage in basis points
                 fill_probability: float = 0.95, # Probability of getting filled
                 exchange_fees: Dict[str, float] = None):  # Fee rate per exchange
        
        self.base_latency_ms = base_latency_ms
        self.latency_std_ms = latency_std_ms
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.exchange_fees = exchange_fees or {'default': 0.001}  # Default 0.1% fee
        
        # Execution statistics
        self.executions: List[Dict] = []
        self.failed_executions: List[Dict] = []
        self.latencies: List[float] = []
        self.slippages: List[float] = []
        
        # Market state cache
        self.market_state: Dict[str, Dict] = {}
    
    def update_market_state(self, exchange: str, symbol: str, data: Dict):
        """Update internal market state cache."""
        key = f"{exchange}_{symbol}"
        self.market_state[key] = {
            'timestamp': data['timestamp'],
            'orderbook': data['orderbook'],
            'ticker': data['ticker']
        }
    
    def _simulate_latency(self) -> float:
        """Generate a realistic latency value."""
        # Use truncated normal distribution to avoid negative latencies
        latency = max(0.1, np.random.normal(
            self.base_latency_ms,
            self.latency_std_ms
        ))
        return latency
    
    def _simulate_slippage(self, base_price: float, side: str) -> float:
        """Simulate price slippage based on market conditions."""
        # Convert basis points to percentage
        slippage_pct = self.slippage_bps / 10000.0
        
        # Add some randomness to slippage
        actual_slippage = np.random.normal(slippage_pct, slippage_pct/2)
        
        # Slippage is positive for buys (price goes up), negative for sells
        direction = 1 if side == 'buy' else -1
        
        return base_price * (1 + direction * actual_slippage)
    
    def _check_liquidity(self, 
                        exchange: str,
                        symbol: str,
                        side: str,
                        size: float,
                        max_price: float) -> Tuple[bool, float]:
        """Check if there's enough liquidity for the trade."""
        key = f"{exchange}_{symbol}"
        market_data = self.market_state.get(key)
        
        if not market_data:
            return False, 0.0
        
        orderbook = market_data['orderbook']
        orders = orderbook['asks'] if side == 'buy' else orderbook['bids']
        
        # Calculate available volume within price limit
        available_volume = 0.0
        for price, volume in orders:
            if (side == 'buy' and price <= max_price) or \
               (side == 'sell' and price >= max_price):
                available_volume += volume
                if available_volume >= size:
                    return True, available_volume
        
        return False, available_volume
    
    async def simulate_execution(self,
                               exchange: str,
                               symbol: str,
                               side: str,
                               size: float,
                               price: float,
                               max_slippage_bps: float = 10.0) -> Optional[Dict]:
        """Simulate execution of a trade with realistic conditions."""
        try:
            execution_start = time.perf_counter()
            
            # Simulate network latency
            latency = self._simulate_latency()
            await asyncio.sleep(latency / 1000.0)  # Convert to seconds
            
            # Get current market state after latency
            key = f"{exchange}_{symbol}"
            current_state = self.market_state.get(key)
            
            if not current_state:
                logger.error(f"No market data available for {exchange} {symbol}")
                return None
            
            # Calculate maximum acceptable price with slippage
            max_slippage = price * (max_slippage_bps / 10000.0)
            max_price = price * (1 + max_slippage) if side == 'buy' else price * (1 - max_slippage)
            
            # Check liquidity
            has_liquidity, available_volume = self._check_liquidity(
                exchange, symbol, side, size, max_price
            )
            
            if not has_liquidity:
                logger.warning(f"Insufficient liquidity for {side} {size} {symbol} on {exchange}")
                self.failed_executions.append({
                    'timestamp': time.time(),
                    'exchange': exchange,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'intended_price': price,
                    'reason': 'insufficient_liquidity',
                    'available_volume': available_volume
                })
                return None
            
            # Simulate fill probability
            if random.random() > self.fill_probability:
                logger.warning(f"Order not filled for {side} {size} {symbol} on {exchange}")
                self.failed_executions.append({
                    'timestamp': time.time(),
                    'exchange': exchange,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'intended_price': price,
                    'reason': 'not_filled'
                })
                return None
            
            # Calculate execution price with slippage
            executed_price = self._simulate_slippage(price, side)
            
            # Ensure price is within acceptable range
            if (side == 'buy' and executed_price > max_price) or \
               (side == 'sell' and executed_price < max_price):
                logger.warning(f"Price slippage too high for {side} {size} {symbol} on {exchange}")
                self.failed_executions.append({
                    'timestamp': time.time(),
                    'exchange': exchange,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'intended_price': price,
                    'actual_price': executed_price,
                    'reason': 'excessive_slippage'
                })
                return None
            
            # Calculate fees
            fee_rate = self.exchange_fees.get(exchange, self.exchange_fees['default'])
            fees = size * executed_price * fee_rate
            
            # Record execution time
            execution_time = (time.perf_counter() - execution_start) * 1000  # ms
            
            # Create execution report
            execution = {
                'timestamp': time.time(),
                'exchange': exchange,
                'symbol': symbol,
                'side': side,
                'size': size,
                'intended_price': price,
                'executed_price': executed_price,
                'slippage_bps': ((executed_price - price) / price) * 10000,
                'fees': fees,
                'latency_ms': latency,
                'execution_time_ms': execution_time
            }
            
            # Update statistics
            self.executions.append(execution)
            self.latencies.append(latency)
            self.slippages.append(execution['slippage_bps'])
            
            # Keep only last 1000 measurements
            if len(self.latencies) > 1000:
                self.latencies.pop(0)
            if len(self.slippages) > 1000:
                self.slippages.pop(0)
            
            return execution
            
        except Exception as e:
            logger.error(f"Error simulating execution: {str(e)}")
            return None
    
    def get_metrics(self) -> Dict:
        """Get execution simulator metrics."""
        total_executions = len(self.executions) + len(self.failed_executions)
        
        return {
            'total_executions': total_executions,
            'successful_executions': len(self.executions),
            'failed_executions': len(self.failed_executions),
            'fill_rate': len(self.executions) / max(1, total_executions),
            'avg_latency_ms': np.mean(self.latencies) if self.latencies else 0,
            'avg_slippage_bps': np.mean(self.slippages) if self.slippages else 0,
            'latency_std_ms': np.std(self.latencies) if len(self.latencies) > 1 else 0,
            'slippage_std_bps': np.std(self.slippages) if len(self.slippages) > 1 else 0,
            'max_latency_ms': max(self.latencies) if self.latencies else 0,
            'max_slippage_bps': max(self.slippages) if self.slippages else 0
        } 