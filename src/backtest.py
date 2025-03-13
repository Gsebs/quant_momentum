"""
This is my backtesting system - it's like a time machine for testing trading strategies!
Instead of risking real money, we can see how our strategy would have performed in the past.

What's cool about this backtest is it:
1. Simulates real trading with stuff like transaction costs
2. Tracks important metrics (returns, drawdowns, etc.)
3. Shows us if our strategy actually works
4. Helps catch potential issues before trading real money
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf
from . import risk
from . import indicators
import matplotlib.pyplot as plt
import seaborn as sns
from . import config
import os

logger = logging.getLogger(__name__)

class BacktestResult:
    """Class to store and analyze backtest results."""
    
    def __init__(self):
        """Initialize backtest result object."""
        self.dates = []
        self.portfolio_values = []
        self.trades = []
        self.initial_capital = 0
        self.metrics = {}
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        try:
            if not self.portfolio_values:
                self.metrics = {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'max_drawdown': 0.0,
                    'avg_turnover': 0.0,
                    'num_trades': 0
                }
                return
            
            # Calculate returns
            final_value = self.portfolio_values[-1]
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate annualized return
            days = (self.dates[-1] - self.dates[0]).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Calculate max drawdown
            peak = self.portfolio_values[0]
            max_drawdown = 0
            for value in self.portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate turnover
            total_turnover = sum(trade['shares'] * trade['price'] for trade in self.trades)
            avg_turnover = total_turnover / self.initial_capital if self.trades else 0
            
            # Store metrics
            self.metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'avg_turnover': avg_turnover,
                'num_trades': len(self.trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self.metrics = {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'avg_turnover': 0.0,
                'num_trades': 0
            }
    
    def summary(self) -> Dict[str, float]:
        """Return summary of backtest results."""
        return self.metrics
    
    def plot_performance(self, title: str = "Portfolio Performance"):
        """Plot portfolio value over time."""
        try:
            if not self.portfolio_values:
                logger.error("No portfolio values to plot")
                return
                
            plt.figure(figsize=(12, 6))
            plt.plot(self.dates, self.portfolio_values)
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            
            # Save plot
            os.makedirs("data/reports", exist_ok=True)
            plt.savefig("data/reports/performance_plot.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}")
            raise

class MomentumBacktest:
    def __init__(self, 
                 initial_capital: float = 1000000.0,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 rebalance_freq: str = 'ME',
                 max_position_size: float = 0.05):
        """
        Initialize momentum strategy backtester.
        
        Args:
            initial_capital: Initial portfolio value
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            rebalance_freq: Rebalancing frequency ('ME' for month end, 'WE' for week end)
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.rebalance_freq = rebalance_freq
        self.max_position_size = max_position_size
        
    def _calculate_transaction_costs(self, position_changes: pd.Series, prices: pd.DataFrame) -> pd.Series:
        """Calculate transaction costs for position changes.
        
        Args:
            position_changes: Series of position changes
            prices: DataFrame with price data
            
        Returns:
            Series of transaction costs
        """
        try:
            # Commission costs (0.1% per trade)
            commission = abs(position_changes) * self.commission
            
            # Market impact costs (0.2% for large trades)
            market_impact = abs(position_changes) * self.slippage
            
            # Total transaction costs
            transaction_costs = commission + market_impact
            
            return transaction_costs
        except Exception as e:
            logging.error(f"Error calculating transaction costs: {str(e)}")
            return pd.Series(0, index=position_changes.index)

    def _get_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Get the closing price for a ticker on a specific date."""
        ticker_data = self.data[ticker]
        if date in ticker_data.index:
            return float(ticker_data.loc[date, 'Close'].iloc[0])
        return None

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate the total portfolio value on a specific date."""
        portfolio_value = self.cash
        for ticker, shares in self.positions.items():
            if shares > 0:
                price = float(self.data[ticker].loc[date, 'Close'].iloc[0])
                portfolio_value += shares * price
        return portfolio_value
    
    def _generate_signals(self, data_dict: Dict[str, pd.DataFrame], 
                         date: pd.Timestamp) -> pd.DataFrame:
        """Generate trading signals for a given date."""
        try:
            signals = pd.DataFrame()
            
            # Convert date to timezone-naive if it's timezone-aware
            if pd.api.types.is_datetime64tz_dtype(date):
                date = date.tz_localize(None)
            
            for ticker, data in data_dict.items():
                # Convert data index to timezone-naive if it's timezone-aware
                if pd.api.types.is_datetime64tz_dtype(data.index):
                    data.index = data.index.tz_localize(None)
                
                # Get data up to the given date
                hist_data = data[data.index <= date]
                
                if not hist_data.empty:
                    # Calculate momentum indicators
                    indicators_dict = indicators.calculate_momentum_indicators(hist_data)
                    
                    # Store last values
                    signals.loc[ticker, 'rsi'] = indicators_dict['rsi'].iloc[-1]
                    signals.loc[ticker, 'macd'] = indicators_dict['macd'].iloc[-1]
                    signals.loc[ticker, 'volatility'] = indicators_dict['volatility'].iloc[-1]
                    
                    # Calculate returns for different periods
                    for period in [5, 10, 20, 60, 120, 252]:
                        returns = hist_data['Close'].pct_change(period).iloc[-1]
                        signals.loc[ticker, f'return_{period}d'] = returns
                    
                    # Store last price
                    signals.loc[ticker, 'price'] = hist_data['Close'].iloc[-1]
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_target_positions(self, prices: Dict[str, float], portfolio_value: float) -> Dict[str, float]:
        """
        Calculate target positions based on signals and portfolio value.
        
        Args:
            prices: Dictionary of current prices for each ticker
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of target positions in shares
        """
        try:
            target_positions = {}
            
            # Keep a cash buffer
            available_value = portfolio_value * 0.95  # Keep 5% cash buffer
            
            for ticker, price in prices.items():
                if ticker in self.signals and price > 0:
                    # Get position size from signals (as decimal)
                    position_size = self.signals[ticker]
                    
                    # Calculate target value for this position
                    target_value = portfolio_value * position_size
                    
                    # Limit position value to available value and max position size
                    max_position_value = min(
                        target_value,
                        portfolio_value * self.max_position_size,
                        available_value
                    )
                    
                    # Calculate target shares
                    target_shares = max_position_value / price
                    
                    # Calculate transaction costs
                    current_shares = self.positions.get(ticker, 0)
                    shares_diff = target_shares - current_shares
                    transaction_value = abs(shares_diff * price)
                    cost = transaction_value * self.commission
                    
                    # Only update if there's a meaningful change and we can afford it
                    if abs(shares_diff) > 0.01 and transaction_value + cost <= available_value:
                        target_positions[ticker] = target_shares
                        available_value -= (transaction_value + cost)
                        self.num_trades += 1
                        self.total_turnover += transaction_value / portfolio_value
                    else:
                        target_positions[ticker] = current_shares
                else:
                    target_positions[ticker] = 0.0
            
            return target_positions
            
        except Exception as e:
            logger.error(f"Error calculating target positions: {str(e)}")
            return {ticker: 0.0 for ticker in prices}
    
    def run(self):
        """Run the backtest simulation."""
        try:
            portfolio_values = pd.Series(index=self.dates)
            positions = {}  # Dictionary to track positions {ticker: shares}
            
            # Initialize with starting cash
            portfolio_values.iloc[0] = self.initial_capital
            self.cash = self.initial_capital
            
            for i in range(len(self.dates)):
                date = self.dates[i]
                
                # Get signals for the current date
                current_signals = self.signals.loc[date] if date in self.signals.index else pd.Series()
                
                # Rebalance portfolio on Mondays
                if date.weekday() == 0:
                    self._rebalance_portfolio(date, current_signals, positions)
                
                # Calculate and store portfolio value
                portfolio_value = self._calculate_portfolio_value(date)
                if portfolio_value is not None:
                    portfolio_values[date] = portfolio_value
            
            # Forward fill any missing values
            portfolio_values = portfolio_values.ffill()
            
            # Calculate returns and risk metrics
            self.calculate_metrics()
            
            return portfolio_values
            
        except Exception as e:
            logging.error(f"Error running backtest: {str(e)}")
            raise

def backtest_strategy(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    position_sizes: Optional[Dict[str, float]] = None,
    commission: float = 0.001,  # 0.1% commission
    slippage: float = 0.001  # 0.1% slippage
) -> BacktestResult:
    """Run backtest simulation."""
    try:
        # Initialize variables
        cash = initial_capital
        positions = {}  # {ticker: shares}
        portfolio_values = []
        dates = []
        trades = []
        
        # Download historical data
        data = {}
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    data[ticker] = stock_data
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {str(e)}")
        
        if not data:
            logger.error("No valid historical data available")
            return BacktestResult()
        
        # Get common dates across all stocks
        common_dates = pd.DatetimeIndex([])
        for ticker_data in data.values():
            if common_dates.empty:
                common_dates = ticker_data.index
            else:
                common_dates = common_dates.intersection(ticker_data.index)
        
        # Rebalance portfolio weekly
        for date in common_dates:
            current_prices = {}
            for ticker in tickers:
                if ticker in data and date in data[ticker].index:
                    current_prices[ticker] = float(data[ticker].loc[date, 'Close'])
            
            # Calculate current portfolio value
            portfolio_value = cash
            for ticker, shares in positions.items():
                if ticker in current_prices:
                    portfolio_value += shares * current_prices[ticker]
            
            # Rebalance on Mondays
            if date.weekday() == 0:
                # Calculate target positions
                for ticker in tickers:
                    if ticker not in current_prices:
                        continue
                        
                    # Get target position size
                    target_size = position_sizes.get(ticker, 0) if position_sizes else (1.0 / len(tickers))
                    target_value = portfolio_value * target_size
                    
                    # Calculate target shares
                    price = current_prices[ticker]
                    target_shares = int(target_value / price)
                    
                    # Calculate trade cost
                    trade_cost = target_shares * price * (commission + slippage)
                    
                    # Check if we have enough cash for the trade
                    if target_shares > 0 and target_shares * price + trade_cost <= cash:
                        # Record trade
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'shares': target_shares,
                            'price': price,
                            'cost': trade_cost
                        })
                        
                        # Update positions and cash
                        positions[ticker] = target_shares
                        cash -= (target_shares * price + trade_cost)
            
            # Record portfolio value
            portfolio_values.append(portfolio_value)
            dates.append(date)
        
        # Create result object
        result = BacktestResult()
        result.dates = dates
        result.portfolio_values = portfolio_values
        result.trades = trades
        result.initial_capital = initial_capital
        result.calculate_metrics()
        
        return result
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

def run_backtest_from_recommendations(
    recommendations_file: str,
    start_date: str,
    end_date: str,
    top_n: int = 10,
    initial_capital: float = 100000
) -> BacktestResult:
    """Run backtest using recommendations from Excel file."""
    try:
        # Read recommendations
        df = pd.read_excel(recommendations_file)
        df = df.rename(columns={'Unnamed: 0': 'Ticker'})  # Rename Unnamed: 0 column to Ticker
        top_stocks = df.nlargest(top_n, 'composite_score')
        
        # Run backtest with position sizes
        result = backtest_strategy(
            tickers=top_stocks['Ticker'].tolist(),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_sizes=dict(zip(top_stocks['Ticker'], top_stocks['position_size']))
        )
        
        # Print and plot results
        logger.info("\nBacktest Results:")
        for metric, value in result.summary().items():
            logger.info(f"{metric}: {value}")
        
        result.plot_performance(f"Portfolio Performance (Top {top_n} Momentum Stocks)")
        return result
    
    except Exception as e:
        logger.error(f"Error running backtest from recommendations: {str(e)}")
        raise

if __name__ == "__main__":
    # Run backtest from recommendations
    result = run_backtest_from_recommendations(
        "data/momentum_signals.xlsx",
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        top_n=10
    ) 