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
from src.ml_model import enhance_signals
from src.db import MomentumDB, store_momentum_metrics
from src.reporting import generate_report
from src.indicators import calculate_momentum_indicators, calculate_rsi, calculate_macd
from src.risk import calculate_risk_metrics, calculate_position_sizes, analyze_sector_exposure
from src.backtest import MomentumBacktest, backtest_strategy, BacktestResult, run_backtest_from_recommendations
from src.data import get_sp500_tickers
from src.momentum import calculate_momentum_metrics
import time
from . import momentum
from . import ml_model
from . import risk
from . import reporting
from . import backtest
from . import db
from joblib import load
import concurrent.futures
import requests.exceptions

# Just basic logging setup to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

def get_stock_data(ticker: str, start_date: str, end_date: str, max_retries: int = 3, timeout: int = 5) -> Optional[pd.DataFrame]:
    """
    Get stock data with retries and timeout.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt
        
    Returns:
        DataFrame with stock data or None if failed
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, timeout=timeout)
            if not data.empty:
                return data
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to get data for {ticker} after {max_retries} attempts: {str(e)}")
            else:
                time.sleep(1)  # Wait before retrying
        except Exception as e:
            logger.warning(f"Error getting data for {ticker}: {str(e)}")
            break
    return None

def get_date_str(date: datetime) -> str:
    """Just converts a date to YYYY-MM-DD format"""
    return date.strftime("%Y-%m-%d")

def calculate_return(data: pd.DataFrame, lookback_days: int) -> float:
    """
    Calculate returns over a lookback period.
    
    Args:
        data: DataFrame with price data
        lookback_days: Number of days to look back
        
    Returns:
        float: Return over the period
    """
    try:
        if data is None or data.empty or len(data) < lookback_days:
            return 0.0
        
        # Get start and end prices using tail to ensure we have data
        prices = data['Close'].tail(lookback_days)
        if len(prices) < lookback_days:
            return 0.0
            
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return 0.0
            
        return (end_price - start_price) / start_price
    except Exception as e:
        logger.error(f"Oops, couldn't calculate return: {str(e)}")
        return 0.0

def compute_momentum(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate momentum metrics for a stock.
    
    Args:
        data: DataFrame with price and volume data
        
    Returns:
        Dictionary with momentum metrics
    """
    try:
        if data is None or data.empty:
            return {}
            
        momentum = {}
        
        # Calculate returns for different periods
        momentum['1m_return'] = calculate_return(data, 21)  # ~1 month
        momentum['3m_return'] = calculate_return(data, 63)  # ~3 months  
        momentum['6m_return'] = calculate_return(data, 126)  # ~6 months
        momentum['12m_return'] = calculate_return(data, 252)  # ~12 months
        
        # Calculate volatility (annualized)
        momentum['volatility'] = float(data['Close'].pct_change().std() * np.sqrt(252))
        
        # Calculate volume ratio (current vs average)
        avg_volume = data['Volume'].rolling(window=20).mean()
        momentum['volume_ratio'] = float(data['Volume'].iloc[-1] / avg_volume.iloc[-1])
        
        # Calculate RSI
        momentum['rsi'] = float(calculate_rsi(data).iloc[-1])
        
        # Calculate MACD
        macd_data = calculate_macd(data)
        momentum['macd'] = float(macd_data['macd'].iloc[-1])
        momentum['macd_signal'] = float(macd_data['signal'].iloc[-1])
        momentum['macd_hist'] = float(macd_data['histogram'].iloc[-1])
        
        # Calculate ROC for different periods
        momentum['roc_5'] = float(data['Close'].pct_change(5).iloc[-1])
        momentum['roc_10'] = float(data['Close'].pct_change(10).iloc[-1])
        momentum['roc_20'] = float(data['Close'].pct_change(20).iloc[-1])
        
        # Trend strength
        sma_50 = data['Close'].rolling(window=50).mean()
        sma_200 = data['Close'].rolling(window=200).mean()
        momentum['trend_strength'] = float((sma_50.tail(1).values[0] - sma_200.tail(1).values[0]) / sma_200.tail(1).values[0])
        
        # Calculate weighted momentum score
        momentum_score = (
            momentum['1m_return'] * config.MOMENTUM_WEIGHTS['1M'] +
            momentum['3m_return'] * config.MOMENTUM_WEIGHTS['3M'] +
            momentum['6m_return'] * config.MOMENTUM_WEIGHTS['6M'] +
            momentum['12m_return'] * config.MOMENTUM_WEIGHTS['12M']
        )
        
        # Normalize momentum score to range [-1, 1]
        momentum_score = np.clip(momentum_score / 0.5, -1, 1)  # Scale by 0.5 (50% return threshold)
        momentum['momentum_score'] = momentum_score
        
        # Calculate volatility adjustment (-0.25 to 0)
        vol_score = -0.25 * (momentum['volatility'] / 0.3)  # 0.3 = 30% vol threshold
        momentum['volatility_score'] = np.clip(vol_score, -0.25, 0)
        
        # Calculate trend adjustment (-0.25 to 0.25)
        trend_score = momentum['trend_strength']
        if abs(trend_score) > 0.05:  # Strong trend threshold
            trend_score = np.sign(trend_score) * 0.25
        else:
            trend_score *= 5  # Scale trend to -0.25 to 0.25
        momentum['trend_score'] = np.clip(trend_score, -0.25, 0.25)
        
        # Calculate composite score (-1.5 to 1.5)
        momentum['composite_score'] = momentum_score + momentum['volatility_score'] + momentum['trend_score']
        
        # Calculate position size (0 to MAX_POSITION_SIZE)
        # Base position is 0 for negative scores, scales up linearly for positive scores
        if momentum['composite_score'] > 0:
            position_size = config.MAX_POSITION_SIZE * (momentum['composite_score'] / 1.5)
        else:
            position_size = 0
        
        momentum['position_size'] = np.clip(position_size, 0, config.MAX_POSITION_SIZE)
        
        return momentum
        
    except Exception as e:
        logger.error(f"Error computing momentum: {str(e)}")
        return {}

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
        
        if df.empty:
            logger.warning("No momentum data available for ranking")
            return pd.DataFrame()
            
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

def filter_universe(data: pd.DataFrame) -> bool:
    """
    Filter out stocks that don't meet our criteria:
    1. Price > $5 (avoid penny stocks)
    2. Volume > 100000 (ensure enough liquidity)
    3. Not too volatile (< 50% annual volatility)
    """
    try:
        if data is None or data.empty:
            return False
            
        # Get the latest price and volume
        last_price = float(data['Close'].tail(1).values[0])
        avg_volume = float(data['Volume'].rolling(window=20).mean().tail(1).values[0])
        
        # Calculate volatility (annualized)
        returns = data['Close'].pct_change()
        volatility = float(returns.std() * np.sqrt(252))  # Annualize daily volatility
        
        # Check if the stock meets our criteria
        price_ok = last_price > 5.0
        volume_ok = avg_volume > 100000
        volatility_ok = volatility < 0.5
        
        return price_ok and volume_ok and volatility_ok
    except Exception as e:
        logger.error(f"Error filtering universe: {str(e)}")
        return False

def run_strategy() -> None:
    """
    Run the momentum strategy:
    1. Get universe of stocks
    2. Calculate momentum metrics
    3. Rank stocks
    4. Generate signals and reports
    """
    try:
        # Get S&P 500 tickers
        tickers = get_sp500_tickers()
        if not tickers:
            logger.error("Failed to get S&P 500 tickers")
            return
            
        # Calculate lookback period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # Get more than a year of data
        
        # Get data and calculate momentum for each stock
        momentum_data = []
        stock_data = {}
        
        # Use ThreadPoolExecutor for parallel data fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(
                    get_stock_data, 
                    ticker, 
                    get_date_str(start_date), 
                    get_date_str(end_date)
                ): ticker for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty and filter_universe(data):
                        stock_data[ticker] = data
                        metrics = compute_momentum(data)
                        if metrics:
                            metrics['ticker'] = ticker
                            momentum_data.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
        
        if not momentum_data:
            logger.error("No valid momentum data calculated")
            return
            
        # Rank stocks
        ranked_stocks = rank_stocks(momentum_data)
        if ranked_stocks.empty:
            logger.error("Failed to rank stocks")
            return
            
        # Sort by composite score and get top stocks
        ranked_stocks = ranked_stocks.sort_values('composite_score', ascending=False)
        top_stocks = ranked_stocks.head(config.TOP_N_STOCKS)
        
        # Save signals to Excel
        signals_file = 'data/momentum_signals.xlsx'
        top_stocks.to_excel(signals_file, index=False)
        logger.info(f"Saved momentum signals to {signals_file}")
        
        # Generate performance report
        report_file = 'data/reports/momentum_report.xlsx'
        generate_report(top_stocks, stock_data, report_file)
        logger.info(f"Generated performance report at {report_file}")
        
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")

if __name__ == '__main__':
    run_strategy() 