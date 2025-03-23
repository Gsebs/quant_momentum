import logging
from typing import Dict, List
from datetime import datetime
import json
from market_data_feed import get_latest_prices
from strategy_engine import get_cached_signals

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store trading state
trades_log: List[Dict] = []
cumulative_pnl: float = 0.0
positions: Dict[str, Dict] = {}
initial_capital: float = 1000000.0
current_capital: float = initial_capital

def execute_trade(ticker: str, signal: str, momentum_score: float) -> Dict:
    """Execute a simulated trade"""
    global positions, current_capital, cumulative_pnl
    
    try:
        latest_prices = get_latest_prices()
        if ticker not in latest_prices:
            return None
        
        current_price = latest_prices[ticker]
        position_size = current_capital * 0.1  # Use 10% of capital per trade
        
        if signal == 'BUY' and ticker not in positions:
            # Calculate number of shares to buy
            shares = position_size / current_price
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'value': shares * current_price,
                'momentum_score': momentum_score
            }
            
            # Update positions and capital
            positions[ticker] = {
                'shares': shares,
                'entry_price': current_price,
                'entry_time': datetime.now().isoformat()
            }
            current_capital -= shares * current_price
            
            trades_log.append(trade)
            logger.info(f"Executed BUY trade for {ticker}: {trade}")
            return trade
            
        elif signal == 'SELL' and ticker in positions:
            position = positions[ticker]
            shares = position['shares']
            
            # Calculate PnL
            pnl = shares * (current_price - position['entry_price'])
            cumulative_pnl += pnl
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'SELL',
                'shares': shares,
                'price': current_price,
                'value': shares * current_price,
                'pnl': pnl,
                'momentum_score': momentum_score
            }
            
            # Update capital and remove position
            current_capital += shares * current_price
            del positions[ticker]
            
            trades_log.append(trade)
            logger.info(f"Executed SELL trade for {ticker}: {trade}")
            return trade
    
    except Exception as e:
        logger.error(f"Error executing trade for {ticker}: {str(e)}")
    
    return None

def get_trades_log() -> List[Dict]:
    """Get the list of executed trades"""
    return trades_log

def get_cumulative_pnl() -> float:
    """Get the cumulative PnL"""
    return cumulative_pnl

def get_positions() -> Dict[str, Dict]:
    """Get current positions"""
    return positions

def get_portfolio_value() -> float:
    """Get current portfolio value"""
    return current_capital + sum(
        pos['shares'] * get_latest_prices().get(ticker, pos['entry_price'])
        for ticker, pos in positions.items()
    ) 