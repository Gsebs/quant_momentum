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
from src.data import get_sp500_tickers, get_batch_data, validate_stock_data, get_stock_data, RetryableError, DataFetchError
from src.momentum import calculate_momentum_metrics, calculate_momentum_score
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
import pickle
import os.path
import threading
from queue import Queue
import redis
import json
from .cache import redis_client, get_redis

# Just basic logging setup to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/cache', exist_ok=True)
os.makedirs('models', exist_ok=True)

# List of reliable tickers to trade
RELIABLE_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'NVDA', 'BRK-B', 'UNH', 'JNJ', 'XOM'
]

# Constants
BATCH_SIZE = 2  # Reduced batch size
BATCH_DELAY = 30  # Delay between batches in seconds
CACHE_KEY = "momentum_signals"
CACHE_EXPIRY = 300  # 5 minutes

def calculate_momentum_score(prices: pd.Series) -> float:
    """
    Calculate momentum score based on multiple timeframes
    """
    try:
        # Calculate returns for different timeframes
        returns_1d = prices.pct_change(1).fillna(0)
        returns_5d = prices.pct_change(5).fillna(0)
        returns_20d = prices.pct_change(20).fillna(0)
        
        # Calculate momentum indicators
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        
        # Combine signals into a momentum score
        momentum_score = (
            0.3 * returns_1d.iloc[-1] +  # 30% weight to daily return
            0.3 * returns_5d.iloc[-1] +   # 30% weight to weekly return
            0.2 * returns_20d.iloc[-1] +  # 20% weight to monthly return
            0.1 * (rsi - 50) / 50 +       # 10% weight to RSI
            0.1 * macd                     # 10% weight to MACD
        )
        
        # Normalize score between -1 and 1
        return np.clip(momentum_score, -1, 1)
    except Exception as e:
        logger.error(f"Error calculating momentum score: {str(e)}")
        return 0.0

def calculate_rsi(prices: pd.Series, periods: int = 14) -> float:
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return 50.0

def calculate_macd(prices: pd.Series) -> float:
    """Calculate MACD indicator"""
    try:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return 0.0

def get_historical_data(ticker: str) -> Optional[pd.DataFrame]:
    """Get historical price data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        # Try to get data with retries and different periods
        periods = ['1mo', '5d', '1d']  # Try different periods
        
        for period in periods:
            for attempt in range(3):
                try:
                    data = stock.history(period=period)
                    if not data.empty and len(data) > 0:
                        logger.info(f"Successfully fetched {period} data for {ticker}")
                        return data
                    logger.warning(f"Empty data received for {ticker} with period {period}, attempt {attempt + 1}")
                    time.sleep(1)  # Wait before retry
                except Exception as e:
                    if attempt == 2 and period == periods[-1]:  # Last attempt of last period
                        raise
                    logger.warning(f"Error fetching {period} data for {ticker}, attempt {attempt + 1}: {str(e)}")
                    time.sleep(1)  # Wait before retry
        
        # If all attempts fail, try to get quote data as fallback
        try:
            info = stock.info
            if info and 'regularMarketPrice' in info:
                # Create a minimal DataFrame with just the current price
                current_time = pd.Timestamp.now()
                data = pd.DataFrame({
                    'Open': [info['regularMarketPrice']],
                    'High': [info['regularMarketPrice']],
                    'Low': [info['regularMarketPrice']],
                    'Close': [info['regularMarketPrice']],
                    'Volume': [info.get('regularMarketVolume', 0)]
                }, index=[current_time])
                logger.info(f"Using quote data for {ticker} as fallback")
                return data
        except Exception as e:
            logger.error(f"Failed to get quote data for {ticker}: {str(e)}")
        
        logger.error(f"Failed to get any valid data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def determine_signal(momentum_score: float) -> str:
    """Determine trading signal based on momentum score"""
    if momentum_score >= 0.3:
        return 'BUY'
    elif momentum_score <= -0.3:
        return 'SELL'
    else:
        return 'HOLD'

def get_cached_signals() -> Optional[Dict]:
    """Get cached signals from Redis"""
    try:
        client = get_redis()
        if client is None:
            logger.error("Redis client not available")
            return None
            
        signals = client.get('momentum_signals')
        if signals:
            return json.loads(signals)
    except Exception as e:
        logger.error(f"Error getting cached signals: {str(e)}")
    return None

def cache_signals(signals: Dict):
    """Cache signals to Redis"""
    try:
        client = get_redis()
        if client is None:
            logger.error("Redis client not available")
            return
            
        client.set('momentum_signals', json.dumps(signals))
        client.expire('momentum_signals', 300)  # Expire after 5 minutes
    except Exception as e:
        logger.error(f"Error caching signals: {str(e)}")

def process_batch(tickers: List[str]) -> Dict[str, Dict]:
    """Process a batch of tickers and calculate their momentum scores."""
    results = {}
    
    for ticker in tickers:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            data = stock.history(period='1y')
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                continue
            
            # Calculate momentum score
            momentum_score = calculate_momentum_score(data['Close'])
            
            # Get current price and calculate price change
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2])
            price_change = (current_price - prev_price) / prev_price
            
            results[ticker] = {
                'momentum_score': momentum_score,
                'current_price': current_price,
                'price_change': price_change,
                'signal': determine_signal(momentum_score)
            }
            
            logger.info(f"Processed {ticker}: Score={momentum_score:.2f}, Signal={results[ticker]['signal']}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
            
    return results

def update_signals(tickers: List[str]):
    """Update momentum signals for the given tickers.
    
    Args:
        tickers: List of stock tickers to process
    """
    if not tickers:
        logging.error("No tickers provided")
        return
    
    try:
        all_signals = []
        total_batches = (len(tickers) + 9) // 10  # Round up division
        
        for batch_num in range(total_batches):
            logging.info(f"Processing batch {batch_num + 1}/{total_batches}")
            start_idx = batch_num * 10
            end_idx = min(start_idx + 10, len(tickers))
            batch = tickers[start_idx:end_idx]
            
            batch_signals = process_batch(batch)
            all_signals.extend(batch_signals)
            
            if batch_num < total_batches - 1:  # Don't wait after the last batch
                logging.info("Waiting 5 seconds before next batch")
                time.sleep(5)
        
        # Sort signals by momentum score in descending order
        all_signals.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        # Cache the signals
        try:
            redis_client.set('momentum_signals', pickle.dumps(all_signals), ex=3600)  # Cache for 1 hour
        except Exception as e:
            logging.error(f"Failed to cache signals: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error updating signals: {str(e)}")

def run_strategy(tickers: List[str]) -> Dict[str, Dict]:
    """
    Run momentum strategy and generate trading signals
    """
    signals = {}
    
    try:
        # Check cache first
        cached_signals = get_cached_signals()
        if cached_signals and isinstance(cached_signals, dict):
            # Only use cache if less than 5 minutes old
            cache_time = cached_signals.get('timestamp')
            if cache_time and (datetime.now() - datetime.fromisoformat(cache_time)).seconds < 300:
                cached_data = cached_signals.get('signals', {})
                if cached_data:  # Only return if we have actual signals
                    logger.info(f"Using cached signals for {len(cached_data)} tickers")
                    return cached_data
        
        # Process tickers in smaller batches to avoid rate limits
        batch_size = 2  # Reduced batch size
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(tickers) + batch_size - 1)//batch_size}")
            
            for ticker in batch:
                try:
                    # Get historical data
                    data = get_historical_data(ticker)
                    if data is None or data.empty:
                        logger.warning(f"Skipping {ticker} due to missing data")
                        continue
                    
                    # Calculate momentum score
                    try:
                        momentum_score = calculate_momentum_score(data['Close'])
                    except Exception as e:
                        logger.error(f"Error calculating momentum for {ticker}: {str(e)}")
                        momentum_score = 0
                    
                    try:
                        # Get current price and daily change
                        current_price = float(data['Close'].iloc[-1])
                        if len(data) > 1:
                            prev_price = float(data['Close'].iloc[-2])
                            daily_change = ((current_price - prev_price) / prev_price) * 100
                        else:
                            daily_change = 0
                    except Exception as e:
                        logger.error(f"Error calculating price change for {ticker}: {str(e)}")
                        current_price = 0
                        daily_change = 0
                    
                    # Determine trading signal
                    signal = determine_signal(momentum_score)
                    
                    signals[ticker] = {
                        'momentum_score': round(momentum_score, 2),
                        'signal': signal,
                        'price': round(current_price, 2),
                        'change': round(daily_change, 1)
                    }
                    
                    logger.info(f"Generated signal for {ticker}: {signal} (score: {momentum_score:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    continue
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(tickers):
                time.sleep(3)  # Increased delay
        
        if signals:  # Only cache if we have signals
            # Cache the signals with timestamp
            signals_with_timestamp = {
                'timestamp': datetime.now().isoformat(),
                'signals': signals
            }
            cache_signals(signals_with_timestamp)
            logger.info(f"Cached signals for {len(signals)} tickers")
        else:
            logger.warning("No valid signals generated")
        
        return signals
        
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")
        return {}

def get_cached_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Get data from cache if available and not expired."""
    cache_file = f'data/cache/{ticker}.pkl'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_date = cached_data.get('date')
                # Cache is valid for 4 hours
                if cache_date and datetime.now() - cache_date < timedelta(hours=4):
                    return cached_data.get('data')
        except Exception as e:
            logger.warning(f"Error reading cache for {ticker}: {str(e)}")
    return None

def save_to_cache(ticker: str, data: pd.DataFrame) -> None:
    """Save data to cache with timestamp."""
    try:
        cache_file = f'data/cache/{ticker}.pkl'
        cache_data = {
            'date': datetime.now(),
            'data': data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Error saving cache for {ticker}: {str(e)}")

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
        
        position_size = np.clip(position_size, 0, config.MAX_POSITION_SIZE)
        momentum['position_size'] = position_size
        
        return momentum
    except Exception as e:
        logger.error(f"Error computing momentum: {str(e)}")
        return {}

def rank_stocks(momentum_data: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Rank stocks based on momentum metrics.
    
    Args:
        momentum_data: List of dictionaries with momentum metrics
        
    Returns:
        DataFrame with ranked stocks
    """
    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(momentum_data)
        
        if df.empty:
            logger.warning("No momentum data available for ranking")
            return pd.DataFrame()
            
        # Rank each momentum period separately
        momentum_cols = [col for col in df.columns if col.endswith('_momentum')]
        for col in momentum_cols:
            rank_col = f"{col}_rank"
            df[rank_col] = df[col].rank(ascending=False)
            
        # Calculate composite rank
        rank_cols = [col for col in df.columns if col.endswith('_rank')]
        df['composite_rank'] = df[rank_cols].mean(axis=1)
        
        # Sort by composite rank
        df = df.sort_values('composite_rank')
        
        return df
        
    except Exception as e:
        logger.error(f"Error ranking stocks: {str(e)}")
        return pd.DataFrame()

def filter_universe(data: pd.DataFrame) -> bool:
    """
    Filter universe based on price, volume, and volatility thresholds.
    
    Args:
        data: DataFrame with price and volume data
        
    Returns:
        bool: True if stock passes filters, False otherwise
    """
    try:
        if data is None or data.empty:
            return False
            
        # Get the latest price and volume
        last_price = float(data['Close'].tail(1).values[0])
        avg_volume = float(data['Volume'].rolling(window=20).mean().tail(1).values[0])
        volatility = float(data['Close'].pct_change().std() * np.sqrt(252))
        
        # Check price threshold
        price_ok = last_price >= config.MIN_PRICE
        
        # Check volume threshold
        volume_ok = avg_volume >= config.MIN_VOLUME
        
        # Check volatility threshold
        volatility_ok = volatility <= config.MAX_VOLATILITY
        
        return price_ok and volume_ok and volatility_ok
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
            momentum = compute_momentum(data)
            
            if momentum:
                momentum['Ticker'] = ticker
                # Get the last price and average volume directly
                momentum['Last_Price'] = float(data['Close'].tail(1).values[0])
                momentum['Avg_Volume'] = float(data['Volume'].rolling(window=20).mean().tail(1).values[0])
                momentum_data.append(momentum)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not momentum_data:
        logger.warning("No valid momentum data generated")
        return pd.DataFrame()
    
    # Convert to DataFrame
    momentum_df = pd.DataFrame(momentum_data)
    
    # Set Ticker as index
    if 'Ticker' in momentum_df.columns:
        momentum_df.set_index('Ticker', inplace=True)
    
    # Rank stocks with technical indicators
    momentum_df = rank_stocks(momentum_df)
    
    # Enhance signals with ML model
    momentum_df = enhance_signals(momentum_df)
    
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
            '1M_Return': top_stocks['1m_return'],
            '3M_Return': top_stocks['3m_return'],
            '6M_Return': top_stocks['6m_return'],
            '12M_Return': top_stocks['12m_return'],
            'Volatility': top_stocks['volatility'],
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

def main():
    """Main function to run the momentum strategy."""
    try:
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Load historical data
        historical_data = load_sp500_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Calculate momentum metrics
        momentum_df = calculate_momentum_metrics(historical_data)
        
        # Load ML model and enhance signals
        try:
            model = load('models/momentum_model.joblib')
            logger.info("Model loaded from models/momentum_model.joblib")
            momentum_df = enhance_signals(momentum_df)
        except Exception as e:
            logger.warning(f"Error loading/using ML model: {str(e)}")
        
        # Calculate risk metrics
        risk_adjusted_data = momentum_df.copy()
        
        # Log DataFrame columns before storing
        logger.info(f"Momentum DataFrame columns before storing: {risk_adjusted_data.columns.tolist()}")
        
        # Save trade recommendations
        risk_adjusted_data.to_excel('data/momentum_signals.xlsx')
        logger.info("Trade recommendations saved to data/momentum_signals.xlsx")
        
        # Store momentum metrics in database
        store_momentum_metrics(risk_adjusted_data)
        
        # Generate report
        generate_report(risk_adjusted_data, "data/reports/momentum_report.xlsx")
        
        # Run backtest
        backtest = BacktestResult()
        result = run_backtest_from_recommendations(
            recommendations_file='data/momentum_signals.xlsx',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            top_n=10,
            initial_capital=100000
        )
        result.plot_performance()
        
        # Print backtest results
        metrics = result.summary()
        print("\nBacktest Results:")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Average Turnover: {metrics.get('avg_turnover', 0):.2%}")
        print(f"Number of Trades: {metrics.get('num_trades', 0)}")
        
        # Open report files automatically
        logger.info("Opening report files...")
        os.system("./open_reports.sh")
        
        logger.info("Strategy run completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
