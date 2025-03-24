import asyncio
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import json
from market_data_feed import get_latest_prices
from strategy_engine import get_cached_signals

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for tracking trades and PnL
trades_log: List[Dict] = []
cumulative_pnl = 0.0
positions: Dict[str, float] = {
    "binance": 0.0,
    "coinbase": 0.0
}

# Initial capital and current capital
initial_capital = 10000.0  # Starting with 10,000 USDT
current_capital = initial_capital

# Simulation parameters
LATENCY = 0.05  # 50ms latency
SLIPPAGE_RATE = 0.0005  # 0.05% slippage
FEE_RATE = 0.001  # 0.1% fee on each trade leg
MAX_POSITION_SIZE = 1.0  # Maximum position size in base currency

async def execute_trade(
    buy_exchange: str,
    sell_exchange: str,
    buy_price: float,
    sell_price: float,
    quantity: float = 1.0
) -> float:
    """Simulate executing a buy on one exchange and a sell on another."""
    global cumulative_pnl, current_capital, positions

    try:
        start_time = time.time()
        
        # Simulate network/exchange latency
        await asyncio.sleep(LATENCY)
        
        # Apply slippage: price moves unfavorably during latency
        # If buying, price increases; if selling, price decreases
        exec_buy_price = buy_price * (1 + SLIPPAGE_RATE)
        exec_sell_price = sell_price * (1 - SLIPPAGE_RATE)
        
        # Apply fees on each leg
        buy_fee = exec_buy_price * FEE_RATE * quantity
        sell_fee = exec_sell_price * FEE_RATE * quantity
        
        # Calculate costs and proceeds
        buy_cost = exec_buy_price * quantity + buy_fee
        sell_proceeds = exec_sell_price * quantity - sell_fee
        
        # Calculate profit
        gross_profit = sell_proceeds - buy_cost
        net_profit = gross_profit
        
        # Update positions
        positions[buy_exchange] += quantity
        positions[sell_exchange] -= quantity
        
        # Update capital and PnL
        current_capital += net_profit
        cumulative_pnl += net_profit
        
        # Record trade details
        trade_record = {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "buy_price": exec_buy_price,
            "sell_price": exec_sell_price,
            "quantity": quantity,
            "buy_fee": buy_fee,
            "sell_fee": sell_fee,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "latency": LATENCY,
            "slippage": SLIPPAGE_RATE
        }
        trades_log.append(trade_record)
        
        # Log trade execution
        logger.info(
            f"Executed trade: Buy {quantity} on {buy_exchange} at {exec_buy_price:.2f}, "
            f"Sell on {sell_exchange} at {exec_sell_price:.2f}, "
            f"Net Profit: {net_profit:.2f}"
        )
        
        return net_profit
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return 0.0

def get_trades_log() -> List[Dict]:
    """Return the list of executed trades."""
    return trades_log

def get_cumulative_pnl() -> float:
    """Return the total accumulated profit/loss."""
    return cumulative_pnl

def get_current_positions() -> Dict[str, float]:
    """Return current positions on each exchange."""
    return positions.copy()

def get_portfolio_value() -> float:
    """Return current portfolio value (capital + unrealized PnL)."""
    return current_capital

def get_trade_statistics() -> Dict:
    """Return statistics about executed trades."""
    if not trades_log:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0
        }
    
    winning_trades = [t for t in trades_log if t["net_profit"] > 0]
    losing_trades = [t for t in trades_log if t["net_profit"] < 0]
    
    return {
        "total_trades": len(trades_log),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades_log),
        "avg_profit": sum(t["net_profit"] for t in trades_log) / len(trades_log),
        "max_profit": max(t["net_profit"] for t in trades_log),
        "max_loss": min(t["net_profit"] for t in trades_log)
    } 