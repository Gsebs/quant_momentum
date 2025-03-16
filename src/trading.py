"""
Trading module for executing trades and managing portfolio
"""

import logging
import json
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Optional
import numpy as np
from .cache import redis_client
import traceback

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.load_portfolio_state()
        
    def load_portfolio_state(self):
        """Load portfolio state from Redis or initialize new state"""
        try:
            state = redis_client.get('portfolio_state')
            if state:
                state = json.loads(state)
                self.cash = float(state.get('cash', self.initial_capital))
                self.positions = state.get('positions', {})
                self.trades = state.get('trades', [])
                self.portfolio_history = state.get('portfolio_history', [])
                self.total_trades = state.get('total_trades', 0)
                self.winning_trades = state.get('winning_trades', 0)
            else:
                self.initialize_portfolio()
        except Exception as e:
            logger.error(f"Error loading portfolio state: {str(e)}")
            self.initialize_portfolio()
    
    def initialize_portfolio(self):
        """Initialize a new portfolio state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.save_portfolio_state()
    
    def save_portfolio_state(self):
        """Save portfolio state to Redis"""
        try:
            state = {
                'cash': self.cash,
                'positions': self.positions,
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'last_update': datetime.now().isoformat()
            }
            redis_client.set('portfolio_state', json.dumps(state))
        except Exception as e:
            logger.error(f"Error saving portfolio state: {str(e)}")
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for a list of tickers using real-time data"""
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Get real-time quote data
                quote = stock.history(period='1d', interval='1m')
                if not quote.empty:
                    current_price = float(quote['Close'].iloc[-1])
                    prices[ticker] = current_price
                    logger.info(f"Got real-time price for {ticker}: ${current_price:.2f}")
                else:
                    # Fallback to regular market price if real-time data unavailable
                    current_price = stock.info.get('regularMarketPrice')
                    if current_price:
                        prices[ticker] = float(current_price)
                        logger.info(f"Using regular market price for {ticker}: ${current_price:.2f}")
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {str(e)}")
        return prices
    
    def execute_trade(self, ticker: str, signal: str, momentum_score: float) -> Optional[Dict]:
        """Execute a trade based on the signal"""
        try:
            # Get current price
            current_price = self.get_current_prices([ticker]).get(ticker)
            if not current_price:
                logger.error(f"Could not get price for {ticker}")
                return None
            
            # Calculate position size (5-15% of portfolio based on momentum score)
            portfolio_value = self.get_portfolio_value()
            base_position_size = 0.05  # 5% minimum position size
            additional_size = min(abs(momentum_score) * 0.10, 0.10)  # Up to additional 10%
            position_size = (base_position_size + additional_size) * portfolio_value
            quantity = max(int(position_size / current_price), 1)  # Ensure at least 1 share
            
            trade = None
            if signal == 'BUY':
                # Always try to buy if we have cash
                if self.cash >= current_price * quantity:
                    trade = self.execute_buy(ticker, quantity, current_price)
                    logger.info(f"Executed BUY trade for {ticker}: {quantity} shares at ${current_price:.2f}")
                else:
                    # Try with smaller quantity if not enough cash
                    max_quantity = int(self.cash / current_price)
                    if max_quantity > 0:
                        trade = self.execute_buy(ticker, max_quantity, current_price)
                        logger.info(f"Executed smaller BUY trade for {ticker}: {max_quantity} shares at ${current_price:.2f}")
            elif signal == 'SELL':
                if ticker in self.positions:
                    # Sell existing position
                    quantity_to_sell = self.positions[ticker]['quantity']
                    trade = self.execute_sell(ticker, quantity_to_sell, current_price)
                    logger.info(f"Executed SELL trade for {ticker}: {quantity_to_sell} shares at ${current_price:.2f}")
                else:
                    # Short selling with available cash as collateral
                    if self.cash >= current_price * quantity:
                        trade = self.execute_short(ticker, quantity, current_price)
                        logger.info(f"Executed SHORT trade for {ticker}: {quantity} shares at ${current_price:.2f}")
            
            if trade:
                # Calculate P&L if it's a closing trade
                if signal == 'SELL' and ticker in self.positions:
                    entry_price = self.positions[ticker]['price']
                    pnl = (current_price - entry_price) * quantity
                    trade['pnl'] = pnl
                
                self.trades.append(trade)
                self.total_trades += 1
                if trade.get('pnl', 0) > 0:
                    self.winning_trades += 1
                
                # Force immediate updates
                self.update_positions()
                self.update_portfolio_history()
                self.save_portfolio_state()
                
                logger.info(f"Trade executed successfully: {trade}")
                return trade
            else:
                logger.warning(f"No trade executed for {ticker} with signal {signal}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return None
    
    def execute_buy(self, ticker: str, quantity: int, price: float) -> Dict:
        """Execute a buy trade"""
        total_cost = quantity * price
        self.cash -= total_cost
        
        if ticker in self.positions:
            # Average down
            current_position = self.positions[ticker]
            new_quantity = current_position['quantity'] + quantity
            new_cost = (current_position['price'] * current_position['quantity'] + total_cost)
            self.positions[ticker] = {
                'quantity': new_quantity,
                'price': new_cost / new_quantity,
                'market_value': new_quantity * price,
                'unrealized_pnl': 0
            }
        else:
            # New position
            self.positions[ticker] = {
                'quantity': quantity,
                'price': price,
                'market_value': total_cost,
                'unrealized_pnl': 0
            }
        
        return {
            'time': datetime.now().isoformat(),
            'ticker': ticker,
            'type': 'BUY',
            'price': price,
            'quantity': quantity,
            'total': total_cost,
            'status': 'FILLED'
        }
    
    def execute_sell(self, ticker: str, quantity: int, price: float) -> Dict:
        """Execute a sell trade"""
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        total_value = quantity * price
        self.cash += total_value
        
        pnl = (price - position['price']) * quantity
        
        if quantity >= position['quantity']:
            # Close position
            del self.positions[ticker]
        else:
            # Reduce position
            position['quantity'] -= quantity
            position['market_value'] = position['quantity'] * price
        
        return {
            'time': datetime.now().isoformat(),
            'ticker': ticker,
            'type': 'SELL',
            'price': price,
            'quantity': quantity,
            'total': total_value,
            'pnl': pnl,
            'status': 'FILLED'
        }
    
    def execute_short(self, ticker: str, quantity: int, price: float) -> Dict:
        """Execute a short sell trade"""
        total_value = quantity * price
        self.cash -= total_value  # Set aside cash as collateral
        
        self.positions[ticker] = {
            'quantity': -quantity,  # Negative quantity indicates short position
            'price': price,
            'market_value': total_value,
            'unrealized_pnl': 0
        }
        
        return {
            'time': datetime.now().isoformat(),
            'ticker': ticker,
            'type': 'SHORT',
            'price': price,
            'quantity': quantity,
            'total': total_value,
            'status': 'FILLED'
        }
    
    def update_positions(self):
        """Update position values and P&L"""
        if not self.positions:
            return
        
        current_prices = self.get_current_prices(list(self.positions.keys()))
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                quantity = abs(position['quantity'])
                position['market_value'] = quantity * current_price
                position['unrealized_pnl'] = (
                    (current_price - position['price']) * quantity
                    if position['quantity'] > 0
                    else (position['price'] - current_price) * quantity
                )
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        self.update_positions()
        return self.cash + sum(
            position['market_value'] for position in self.positions.values()
        )
    
    def update_portfolio_history(self):
        """Update portfolio value history"""
        current_value = self.get_portfolio_value()
        self.portfolio_history.append({
            'timestamp': datetime.now().isoformat(),
            'value': current_value
        })
        
        # Keep only last 30 days of history
        if len(self.portfolio_history) > 30 * 24 * 12:  # 30 days of 5-minute intervals
            self.portfolio_history = self.portfolio_history[-30 * 24 * 12:]
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio metrics"""
        try:
            portfolio_value = self.get_portfolio_value()
            
            # Calculate daily return
            if len(self.portfolio_history) >= 2:
                prev_value = self.portfolio_history[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
            else:
                daily_return = 0
            
            # Calculate win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Calculate max drawdown
            if len(self.portfolio_history) > 0:
                values = [h['value'] for h in self.portfolio_history]
                peak = values[0]
                max_drawdown = 0
                
                for value in values[1:]:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0
            
            return {
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': self.positions,
                'trades': self.trades[-50:],  # Return last 50 trades
                'portfolio_history': self.portfolio_history,
                'daily_return': daily_return,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown * 100,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'portfolio_value': self.initial_capital,
                'cash': self.initial_capital,
                'positions': {},
                'trades': [],
                'portfolio_history': [],
                'daily_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'winning_trades': 0
            } 