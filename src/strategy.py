"""
Hey! This is my momentum trading strategy that I built. It looks for stocks that are going up
and have good momentum behind them. The cool part is it uses both traditional momentum
indicators and some machine learning to make better picks.

The basic idea is:
1. Get S&P 500 stocks
2. Calculate momentum scores using returns, volatility, and technical indicators
3. Rank the stocks and pick the best ones
4. Use ML to enhance our picks
5. Figure out how much to invest in each stock
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
from . import config
import logging
import os
import yfinance as yf
from src.ml_model import enhance_momentum_signals
from src.db import MomentumDB
from src.reporting import MomentumReport
from src.indicators import calculate_momentum_indicators
from src.risk import calculate_risk_metrics, calculate_position_sizes, analyze_sector_exposure
from src.backtest import MomentumBacktest
from src.data import get_sp500_tickers

# Just basic logging setup to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_date_str(date: datetime) -> str:
    """Just converts a date to YYYY-MM-DD format"""
    return date.strftime("%Y-%m-%d")

def calculate_return(data: pd.DataFrame, lookback_days: int) -> float:
    """
    This is where we calculate returns over different time periods. For example, if we want
    to know how a stock did over the last month, we'd use lookback_days=21 (trading days).
    
    The cool thing about this function is it handles all the edge cases - like what if we
    don't have enough data, or what if there are missing prices? It just returns 0 in those
    cases so our strategy doesn't break.
    """
    try:
        if len(data) < lookback_days:
            return 0.0
        
        start_price = float(data['Close'].iloc[-lookback_days].iloc[0])
        end_price = float(data['Close'].iloc[-1].iloc[0])
        
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return 0.0
            
        return (end_price - start_price) / start_price
    except Exception as e:
        logger.error(f"Oops, couldn't calculate return: {str(e)}")
        return 0.0

def compute_momentum(data: pd.DataFrame, periods: Dict[str, int]) -> Dict[str, float]:
    """
    This is the heart of our strategy - it looks at a bunch of different ways to measure
    momentum. I picked these indicators because they each tell us something different:
    
    1. Returns over different timeframes - shows if momentum is consistent
    2. RSI - tells us if the stock is overbought/oversold
    3. MACD - shows if momentum is accelerating or slowing down
    4. Volume - confirms if the price moves have strong backing
    5. Volatility - helps avoid stocks that are too risky
    
    The function combines all these to get a complete picture of the stock's momentum.
    Each indicator helps catch different types of momentum moves.
    """
    momentum = {}
    
    try:
        # First get basic returns - this is the simplest momentum measure
        returns = data['Close'].pct_change()
        momentum['returns'] = float(returns.iloc[-1].iloc[0])
        
        # Check volatility - we don't want stocks that are too wild
        momentum['volatility'] = float(returns.rolling(window=20).std().iloc[-1].iloc[0])
        
        # Volume is super important - price moves mean more when volume is high
        volume_sma = data['Volume'].rolling(window=20).mean()
        momentum['volume_ratio'] = float(data['Volume'].iloc[-1].iloc[0] / volume_sma.iloc[-1].iloc[0])
        
        # RSI - one of my favorite indicators. It's like a momentum thermometer
        # Values over 70 mean overbought, under 30 mean oversold
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        momentum['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1].iloc[0])
        
        # MACD - great for catching trend changes
        # When the MACD crosses its signal line, it often means momentum is shifting
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        momentum['macd'] = float(macd.iloc[-1].iloc[0])
        momentum['histogram'] = float((macd - signal).iloc[-1].iloc[0])
        
        # Look at returns over different time periods
        # This helps us see if momentum is building up or dying down
        for period_name, lookback_days in periods.items():
            momentum[f"{period_name}_momentum"] = calculate_return(data, lookback_days)
        
        # Rate of change - shows how fast the price is moving
        for period in [5, 10, 20]:
            momentum[f'roc_{period}'] = float(data['Close'].pct_change(periods=period).iloc[-1].iloc[0])
        
        # Trend strength - bigger number means stronger trend
        momentum['trend_strength'] = abs(momentum['roc_20'])
        
        # Final momentum signal combines RSI and trend
        # This is cool because it considers both momentum and overbought/oversold conditions
        momentum['momentum_signal'] = momentum['rsi'] * momentum['roc_20']
        
        # Adjust returns for volatility - like a Sharpe ratio for momentum
        if momentum['volatility'] != 0:
            momentum['volatility_adjusted_returns'] = momentum['returns'] / momentum['volatility']
        else:
            momentum['volatility_adjusted_returns'] = 0.0
            
    except Exception as e:
        logger.error(f"Had trouble computing momentum: {str(e)}")
            
    return momentum

def rank_stocks(momentum_data: List[Dict]) -> pd.DataFrame:
    """
    This is where we rank all the stocks to find the best ones. The cool part about
    this ranking system is that it:
    
    1. Looks at momentum over multiple timeframes (1M, 3M, 6M, 12M)
    2. Considers technical indicators (RSI, MACD)
    3. Weights recent momentum more heavily (30% to 12M, 25% to 6M, etc.)
    
    I designed it this way because I found that stocks with good momentum across
    different timeframes tend to keep performing well.
    """
    try:
        # Turn our data into a DataFrame for easier ranking
        df = pd.DataFrame(momentum_data)
        
        # Rank each momentum period separately
        momentum_cols = [col for col in df.columns if col.endswith('_momentum')]
        for col in momentum_cols:
            rank_col = f"{col[:-9]}_momentum_rank"
            df[rank_col] = df[col].rank(pct=True)
        
        # Add ranks for technical indicators
        tech_cols = ['rsi', 'macd', 'volatility']
        for col in tech_cols:
            if col in df.columns:
                df[f"{col}_rank"] = df[col].rank(pct=True)
            else:
                df[f"{col}_rank"] = 0.5  # Neutral if missing
        
        # Weight the components - this is key to our strategy
        # We care more about recent momentum but still consider longer-term trends
        weights = {
            '12m_momentum_rank': 0.3,   # 30% weight to 1-year momentum
            '6m_momentum_rank': 0.25,    # 25% to 6-month
            '3m_momentum_rank': 0.2,     # 20% to 3-month
            '1m_momentum_rank': 0.15,    # 15% to 1-month
            'rsi_rank': 0.05,           # 5% each to RSI and MACD
            'macd_rank': 0.05
        }
        
        # Calculate final score
        df['composite_score'] = 0.0
        for col, weight in weights.items():
            if col in df.columns:
                df['composite_score'] += df[col] * weight
            else:
                # Handle missing indicators by redistributing their weight
                remaining_cols = [c for c in weights.keys() if c in df.columns]
                if remaining_cols:
                    total_remaining_weight = sum(weights[c] for c in remaining_cols)
                    if total_remaining_weight > 0:
                        for c in remaining_cols:
                            df['composite_score'] += df[c] * (weights[c] / total_remaining_weight)
        
        return df
    except Exception as e:
        logger.error(f"Error ranking stocks: {str(e)}")
        return pd.DataFrame()

def filter_universe(data: pd.DataFrame, min_price: float = 5.0, min_volume: float = 100000) -> bool:
    """
    Filter stocks based on price and volume criteria.
    
    Args:
        data: DataFrame with price data
        min_price: Minimum price threshold
        min_volume: Minimum volume threshold
        
    Returns:
        bool: True if stock passes filters
    """
    try:
        # Get latest price and average volume
        last_price = float(data['Close'].iloc[-1].iloc[0])
        avg_volume = float(data['Volume'].rolling(window=20).mean().iloc[-1].iloc[0])
        
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = float(returns.std().iloc[0] * np.sqrt(252))
        
        # Check if values are valid
        if pd.isna(last_price) or pd.isna(avg_volume) or pd.isna(volatility):
            return False
            
        # Apply filters
        price_check = last_price >= min_price
        volume_check = avg_volume >= min_volume
        volatility_check = volatility <= 0.5  # Filter out extremely volatile stocks
        
        return bool(price_check and volume_check and volatility_check)
        
    except Exception as e:
        logger.error(f"Error filtering universe: {str(e)}")
        return False

def build_momentum_strategy(tickers: List[str], data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build momentum strategy for a list of stocks.
    
    Args:
        tickers: List of stock tickers
        data_dict: Dictionary of stock data frames
        
    Returns:
        DataFrame: Momentum metrics and ranks for all stocks
    """
    momentum_data = []
    
    # Define lookback periods (in trading days)
    periods = {
        '1m': 21,    # 1 month
        '3m': 63,    # 3 months
        '6m': 126,   # 6 months
        '12m': 252  # 12 months
    }
    
    for ticker in tickers:
        try:
            data = data_dict.get(ticker)
            if data is None or data.empty:
                continue
                
            # Check if stock passes universe filters
            if not filter_universe(data):
                continue
            
            # Compute momentum metrics and technical indicators
            momentum = compute_momentum(data, periods)
            
            if momentum:
                momentum['Ticker'] = ticker
                momentum['Last_Price'] = float(data['Close'].iloc[-1].iloc[0])
                momentum['Avg_Volume'] = float(data['Volume'].rolling(window=20).mean().iloc[-1].iloc[0])
                momentum_data.append(momentum)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not momentum_data:
        logger.warning("No valid momentum data generated")
        return pd.DataFrame()
    
    # Convert to DataFrame
    momentum_df = pd.DataFrame(momentum_data)
    
    # Rank stocks with technical indicators
    momentum_df = rank_stocks(momentum_df)
    
    # Enhance signals with ML model
    momentum_df = enhance_momentum_signals(momentum_df)
    
    # Calculate position sizes
    momentum_df = calculate_position_sizes(momentum_df)
    
    # Calculate risk metrics
    momentum_df = calculate_risk_metrics(momentum_df)
    
    # Log columns before storing
    logger.info(f"Momentum DataFrame columns before storing: {momentum_df.columns.tolist()}")
    
    return momentum_df

def generate_trade_recommendations(momentum_df: pd.DataFrame, top_n: int = config.TOP_N_STOCKS, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate and optionally save trade recommendations.
    
    Args:
        momentum_df: DataFrame with momentum metrics and ranks
        top_n: Number of top stocks to recommend
        save_path: Path to save recommendations (optional)
        
    Returns:
        DataFrame: Trade recommendations
    """
    try:
        if momentum_df.empty:
            logger.warning("No momentum data available for generating recommendations")
            return pd.DataFrame()
        
        # Select top N stocks
        top_stocks = momentum_df.head(top_n)
        
        # Prepare recommendations DataFrame
        recommendations = pd.DataFrame({
            'Ticker': top_stocks.index,
            'Last_Price': top_stocks['Last_Price'],
            'Position_Size': top_stocks['position_size'],
            'Momentum_Score': top_stocks['composite_score'],
            'ML_Score': top_stocks.get('ml_score', 0),
            'Enhanced_Score': top_stocks.get('enhanced_score', top_stocks['composite_score']),
            'Sharpe_Ratio': top_stocks['risk_sharpe_ratio'],
            'Max_Drawdown': top_stocks['risk_max_drawdown'],
            '1M_Return': top_stocks['1m_momentum'],
            '3M_Return': top_stocks['3m_momentum'],
            '6M_Return': top_stocks['6m_momentum'],
            '12M_Return': top_stocks['12m_momentum'],
            'Volatility': top_stocks['risk_annualized_volatility'],
            'Avg_Volume': top_stocks['Avg_Volume']
        })
        
        # Format percentage columns
        pct_columns = ['Momentum_Score', 'ML_Score', 'Enhanced_Score', 'Max_Drawdown',
                      '1M_Return', '3M_Return', '6M_Return', '12M_Return', 'Volatility']
        for col in pct_columns:
            recommendations[col] = recommendations[col].map('{:.2%}'.format)
        
        # Format other columns
        recommendations['Last_Price'] = recommendations['Last_Price'].map('${:.2f}'.format)
        recommendations['Position_Size'] = recommendations['Position_Size'].map('${:,.0f}'.format)
        recommendations['Avg_Volume'] = recommendations['Avg_Volume'].map('{:,.0f}'.format)
        recommendations['Sharpe_Ratio'] = recommendations['Sharpe_Ratio'].map('{:.2f}'.format)
        
        # Save recommendations if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            recommendations.to_excel(save_path, index=False)
            logger.info(f"Trade recommendations saved to {save_path}")
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()

class MomentumStrategy:
    """Momentum trading strategy implementation."""
    
    def __init__(self):
        """Initialize strategy."""
        self.data = None
        self.signals = None
        self.positions = None
        
    def run(self):
        """Run the momentum strategy."""
        try:
            # Get S&P 500 tickers
            tickers = get_sp500_tickers()
            logging.info(f"Retrieved {len(tickers)} tickers")
            if not tickers:
                raise ValueError("No tickers retrieved")
            
            # Download data and calculate signals for each ticker
            all_signals = []
            for ticker in tickers:
                try:
                    logging.info(f"Processing ticker: {ticker}")
                    # Download data for this ticker
                    data = yf.download(ticker, start=config.START_DATE, progress=False)
                    if data.empty:
                        logging.warning(f"No data available for {ticker}")
                        continue
                    
                    logging.info(f"Downloaded {len(data)} rows of data for {ticker}")
                    
                    # Calculate signals
                    ticker_signals = self._calculate_momentum_signals(data)
                    if ticker_signals is not None and not ticker_signals.empty:
                        ticker_signals.index = [ticker]  # Set the index to the ticker
                        all_signals.append(ticker_signals)
                        logging.info(f"Successfully calculated signals for {ticker}")
                    else:
                        logging.warning(f"No signals calculated for {ticker}")
                except Exception as e:
                    logging.error(f"Error calculating signals for {ticker}: {str(e)}")
                    continue
            
            logging.info(f"Calculated signals for {len(all_signals)} tickers")
            
            if not all_signals:
                raise ValueError("No signals calculated for any tickers")
            
            # Combine all signals
            signals = pd.concat(all_signals)
            
            # Sort by composite score
            signals = signals.sort_values('composite_score', ascending=False)
            
            # Save signals to Excel
            signals.to_excel('data/momentum_signals.xlsx')
            logging.info(f"Saved signals for {len(signals)} tickers to Excel")
            
            return signals
            
        except Exception as e:
            logging.error(f"Error running momentum strategy: {str(e)}")
            raise

    def _download_data(self, ticker: str) -> pd.DataFrame:
        """Download historical data for the given ticker."""
        try:
            stock = yf.download(ticker, start='2020-01-01', progress=True)
            if not stock.empty:
                return stock
            else:
                logger.warning(f"No data available for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None

    def calculate_return(self, data, periods):
        """
        Calculate the return over a specified period.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data
            periods (int): Number of periods to calculate return over
            
        Returns:
            pd.Series: Series containing returns
        """
        return data['Close'].pct_change(periods=periods)

    def _calculate_momentum_signals(self, data):
        """
        Calculate momentum signals for a given stock.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data for a stock
            
        Returns:
            pd.DataFrame: DataFrame containing momentum signals
        """
        try:
            # Calculate returns for different periods
            returns_1m = self.calculate_return(data, periods=21)  # 1 month
            returns_3m = self.calculate_return(data, periods=63)  # 3 months
            returns_6m = self.calculate_return(data, periods=126)  # 6 months
            returns_12m = self.calculate_return(data, periods=252)  # 12 months
            
            # Calculate volatility using a rolling window
            volatility = data['Close'].pct_change().rolling(window=252).std()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.rolling(window=14).mean()
            avg_loss = losses.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate trend strength
            sma_50 = data['Close'].rolling(window=50).mean()
            sma_200 = data['Close'].rolling(window=200).mean()
            trend_strength = (sma_50 - sma_200) / sma_200
            
            # Get the latest values (properly extract scalar values)
            latest_returns = {
                '1M': float(returns_1m.iloc[-1]),
                '3M': float(returns_3m.iloc[-1]),
                '6M': float(returns_6m.iloc[-1]),
                '12M': float(returns_12m.iloc[-1])
            }
            
            latest_volatility = float(volatility.iloc[-1])
            latest_rsi = float(rsi.iloc[-1])
            latest_trend = float(trend_strength.iloc[-1])
            
            # Log the latest values for debugging
            logging.info(f"Latest returns: 1M={returns_1m.iloc[-1:]}, 3M={returns_3m.iloc[-1:]}, 6M={returns_6m.iloc[-1:]}, 12M={returns_12m.iloc[-1:]}")
            logging.info(f"Latest volatility: {volatility.iloc[-1:]}")
            logging.info(f"Latest RSI: {rsi.iloc[-1:]}")
            logging.info(f"Latest trend strength: {trend_strength.iloc[-1:]}")
            
            # Calculate momentum score (weighted average of returns)
            weights = {'1M': 0.4, '3M': 0.3, '6M': 0.2, '12M': 0.1}
            momentum_score = 0
            valid_periods = 0
            
            for period, weight in weights.items():
                if not pd.isna(latest_returns[period]):
                    momentum_score += latest_returns[period] * weight
                    valid_periods += weight
            
            if valid_periods > 0:
                momentum_score = momentum_score / valid_periods
            else:
                momentum_score = 0
            
            # Adjust for volatility
            volatility_adjustment = 0
            if not pd.isna(latest_volatility):
                if latest_volatility > 0.3:  # High volatility threshold
                    volatility_adjustment = -0.2
                elif latest_volatility < 0.15:  # Low volatility threshold
                    volatility_adjustment = 0.1
                
            # Adjust for trend
            trend_adjustment = 0
            if not pd.isna(latest_trend):
                if latest_trend > 0.05:  # Strong uptrend
                    trend_adjustment = 0.2
                elif latest_trend < -0.05:  # Strong downtrend
                    trend_adjustment = -0.2
                
            # Calculate composite score
            composite_score = momentum_score + volatility_adjustment + trend_adjustment
            
            # Calculate position size (normalized between 0 and 1)
            position_size = max(0, min(1, (composite_score + 1) / 2))
            
            # Create signals DataFrame with ticker as index
            signals = pd.DataFrame({
                'momentum_score': [momentum_score],
                'volatility': [latest_volatility],
                'trend_strength': [latest_trend],
                'composite_score': [composite_score],
                'position_size': [position_size]
            })
            
            return signals
            
        except Exception as e:
            logging.error(f"Error calculating signals: {str(e)}")
            return None

if __name__ == "__main__":
    # Initialize database and reporting
    db = MomentumDB('data/momentum.db')
    reporter = MomentumReport()
    
    # Get S&P 500 tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    
    # Download historical data
    data_dict = {}
    for ticker in sp500_tickers:
        try:
            # Clean ticker symbol
            clean_ticker = ticker.replace('.', '-')
            
            # Try to download data with different periods if needed
            for period in ["2y", "1y", "6mo"]:
                try:
                    stock = yf.Ticker(clean_ticker)
                    data = stock.history(period=period)
                    if not data.empty and len(data) > 20:  # Ensure we have at least 20 days of data
                        data_dict[ticker] = data
                        data.to_csv(f"data/{ticker}_historical.csv")
                        logger.info(f"Downloaded data for {ticker}")
                        break
                    else:
                        continue
                except Exception as e:
                    continue
            
            if ticker not in data_dict:
                logger.warning(f"No data available for {ticker}")
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            continue
    
    if not data_dict:
        logger.error("No data downloaded for any tickers")
        exit(1)
    
    # Build momentum strategy
    momentum_df = build_momentum_strategy(sp500_tickers, data_dict)
    
    # Generate recommendations
    recommendations = generate_trade_recommendations(
        momentum_df,
        top_n=10,
        save_path="data/momentum_signals.xlsx"
    )
    
    # Store momentum metrics in database
    if not momentum_df.empty:
        db.store_momentum_metrics(momentum_df)
        
        # Generate detailed report
        reporter.generate_report(
            momentum_df=momentum_df,
            recommendations=recommendations,
            output_file="momentum_report.xlsx"
        )
        
        # Run backtest
        backtest = MomentumBacktest()
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        try:
            backtest_result = backtest.run(data_dict, start_date, end_date)
            
            # Print backtest results
            print("\nBacktest Results:")
            if backtest_result and hasattr(backtest_result, 'metrics'):
                print(f"Total Return: {backtest_result.metrics.get('total_return', 0.0):.2f}%")
                print(f"Annualized Return: {backtest_result.metrics.get('annualized_return', 0.0):.2f}%")
                print(f"Annualized Volatility: {backtest_result.metrics.get('annualized_volatility', 0.0):.2f}%")
                print(f"Sharpe Ratio: {backtest_result.metrics.get('sharpe_ratio', 0.0):.2f}")
                print(f"Sortino Ratio: {backtest_result.metrics.get('sortino_ratio', 0.0):.2f}")
                print(f"Max Drawdown: {backtest_result.metrics.get('max_drawdown', 0.0):.2f}%")
                print(f"Average Turnover: {backtest_result.metrics.get('avg_turnover', 0.0):.2f}%")
            else:
                print("Backtest failed to generate valid results")
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            print("\nBacktest Results:")
            print("Error: Failed to run backtest successfully")
    
    # Print top recommendations
    if not recommendations.empty:
        print("\nTop 10 Momentum Stocks:")
        print(recommendations.to_string(index=False))
    else:
        print("No recommendations generated") 