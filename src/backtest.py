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

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self):
        """
        This class is like our report card - it keeps track of everything that happened
        during our backtest: how much money we made/lost, what trades we did, etc.
        """
        self.portfolio_values = pd.Series()  # How our money grew over time
        self.positions = pd.DataFrame()      # What stocks we held
        self.trades = pd.DataFrame()         # All our buy/sell moves
        self.metrics = {}                    # Performance stats
        self.turnover = pd.Series()          # How much trading we did
    
    def summary(self) -> Dict[str, float]:
        """
        This is where we crunch the numbers to see how well we did. We look at:
        - Total returns (how much money we made)
        - Risk-adjusted returns (Sharpe ratio - returns vs risk)
        - Maximum drawdown (biggest loss from peak)
        - Trading activity (how much we bought/sold)
        
        These metrics help us understand if the strategy is actually good or just lucky.
        """
        try:
            if self.portfolio_values.empty:
                return {}
            
            # Calculate our daily returns
            returns = self.portfolio_values.pct_change().dropna()
            
            metrics = {}
            
            # Total return - this is the bottom line
            metrics['total_return'] = float((1 + returns).prod() - 1)
            
            # Turn it into a yearly number - easier to compare with other investments
            days = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
            metrics['annualized_return'] = float((1 + metrics['total_return']) ** (365 / days) - 1)
            
            # How risky was it? Higher volatility means more risk
            metrics['annualized_volatility'] = float(returns.std() * np.sqrt(252))
            
            # Sharpe ratio - the holy grail of metrics
            # Shows how much return we got for the risk we took
            risk_free_rate = 0.02  # Using 2% as risk-free rate
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = returns - daily_rf
            if metrics['annualized_volatility'] > 0:
                metrics['sharpe_ratio'] = float(metrics['annualized_return'] / metrics['annualized_volatility'])
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Maximum drawdown - our biggest losing streak
            # This is super important - shows how bad things can get
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            metrics['max_drawdown'] = float(drawdowns.min())
            
            # How much trading we did - important for considering costs
            if not self.turnover.empty:
                metrics['avg_turnover'] = float(self.turnover.mean())
            else:
                metrics['avg_turnover'] = 0.0
            
            # Total number of trades - helps understand strategy activity
            if not self.trades.empty:
                metrics['num_trades'] = len(self.trades)
            else:
                metrics['num_trades'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Oops, couldn't calculate summary stats: {str(e)}")
            return {}

    def plot_performance(self, title: str = "Portfolio Performance") -> None:
        """
        Creates some cool charts to visualize how we did. Shows:
        1. Portfolio value over time - the growth of our money
        2. Drawdowns - when and how badly we lost money
        
        These visuals really help spot patterns and potential issues.
        """
        try:
            plt.style.use('seaborn-v0_8')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Top chart - how our money grew
            ax1.plot(self.portfolio_values.index, self.portfolio_values.values, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.grid(True)
            ax1.legend()
            
            # Bottom chart - our losses from peaks (drawdowns)
            drawdown = (self.portfolio_values / self.portfolio_values.expanding().max() - 1)
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            ax2.plot(drawdown.index, drawdown.values, color='red', label='Drawdown')
            ax2.set_title('Portfolio Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting performance: {str(e)}")

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
            # Get signals for each ticker
            signals = {}
            for ticker in prices:
                if ticker in self.signals:
                    signals[ticker] = self.signals[ticker]
                else:
                    signals[ticker] = 0.0
            
            # Calculate position sizes
            total_signal = sum(abs(signal) for signal in signals.values())
            if total_signal == 0:
                return {ticker: 0.0 for ticker in prices}
            
            # Calculate target positions
            target_positions = {}
            for ticker, signal in signals.items():
                if ticker in prices and prices[ticker] > 0:
                    position_size = (signal / total_signal) * portfolio_value
                    shares = position_size / prices[ticker]
                    target_positions[ticker] = float(shares)
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
            self.calculate_metrics(portfolio_values)
            
            return portfolio_values
            
        except Exception as e:
            logging.error(f"Error running backtest: {str(e)}")
            raise

def backtest_strategy(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    rebalance_freq: int = 20,  # Rebalance every N business days
    position_sizes: Optional[Dict[str, float]] = None
) -> BacktestResult:
    """
    Run backtest on a list of stocks.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Initial portfolio value
        transaction_cost: Transaction cost per trade
        rebalance_freq: Number of business days between rebalances
        position_sizes: Dictionary mapping tickers to target position sizes
        
    Returns:
        BacktestResult object with backtest results
    """
    try:
        # Download historical data
        logger.info("Downloading historical data for backtesting...")
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = yf.download(ticker, start=start_date, end=end_date)
            except Exception as e:
                logger.warning(f"Error downloading data for {ticker}: {str(e)}")
                continue
        
        if not data:
            raise ValueError("No valid data downloaded")
        
        # Initialize portfolio
        portfolio_value = initial_capital
        cash = portfolio_value
        positions = {ticker: 0 for ticker in data.keys()}
        
        # Create date range
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        portfolio_values = pd.Series(index=dates, dtype=float)
        portfolio_values.iloc[0] = initial_capital
        
        # Track days since last rebalance and trading metrics
        days_since_rebalance = 0
        num_trades = 0
        total_turnover = 0.0
        
        # Run backtest
        for i in range(1, len(dates)):
            date = dates[i]
            portfolio_value = cash
            
            # Update positions with current prices
            for ticker, ticker_data in data.items():
                if date in ticker_data.index:
                    price = float(ticker_data.loc[date, 'Close'].iloc[0])
                    position_value = positions[ticker] * price
                    portfolio_value += position_value
            
            # Rebalance portfolio if needed
            days_since_rebalance += 1
            if days_since_rebalance >= rebalance_freq:
                active_tickers = [ticker for ticker in tickers if date in data[ticker].index]
                if active_tickers:
                    days_since_rebalance = 0
                    old_positions = positions.copy()
                    turnover = 0.0
                    
                    # Calculate target positions using provided position sizes
                    for ticker in active_tickers:
                        try:
                            price = float(data[ticker].loc[date, 'Close'].iloc[0])
                            current_shares = positions[ticker]
                            
                            # Use position size from momentum signals if available
                            if position_sizes and ticker in position_sizes:
                                target_value = portfolio_value * position_sizes[ticker]
                            else:
                                # Fallback to equal weight with max position size limit
                                base_weight = 1.0 / len(active_tickers)
                                target_value = portfolio_value * base_weight
                            
                            target_shares = (target_value / price)
                            
                            # Calculate transaction cost only on the change in position
                            shares_diff = target_shares - current_shares
                            transaction_value = abs(shares_diff * price)
                            cost = transaction_value * transaction_cost
                            
                            # Update position and cash if there's a meaningful change
                            if abs(shares_diff) > 0.01:  # Only count trades above 1% of a share
                                positions[ticker] = target_shares
                                cash -= (shares_diff * price + cost)
                                num_trades += 1
                                turnover += transaction_value
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error rebalancing {ticker}: {str(e)}")
                            continue
                    
                    # Update total turnover
                    if portfolio_value > 0:
                        total_turnover += turnover / portfolio_value
            
            # Update portfolio value
            portfolio_values.iloc[i] = portfolio_value if portfolio_value > 0 else portfolio_values.iloc[i-1]
        
        # Clean up any missing values
        portfolio_values = portfolio_values.ffill()
        
        # Calculate average turnover
        avg_turnover = total_turnover / (len(dates) / rebalance_freq) if len(dates) > 0 else 0
        
        # Create BacktestResult object
        result = BacktestResult()
        result.portfolio_values = portfolio_values
        result.positions = pd.DataFrame(positions, index=[dates[-1]])
        result.metrics = {
            'initial_capital': initial_capital,
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] - initial_capital) / initial_capital,
            'annualized_return': (1 + (portfolio_values.iloc[-1] - initial_capital) / initial_capital) ** (252 / len(dates)) - 1,
            'max_drawdown': (portfolio_values / portfolio_values.cummax() - 1).min(),
            'avg_turnover': avg_turnover,
            'num_trades': num_trades
        }
        
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