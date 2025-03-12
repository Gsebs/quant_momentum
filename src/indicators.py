"""
Technical indicators and advanced momentum metrics calculation module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with price data
        period: RSI period (default: 14)
        
    Returns:
        Series with RSI values
    """
    try:
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Get gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=data.index)

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Dictionary with MACD line and signal line
    """
    try:
        # Calculate EMAs
        fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with price data
        period: Moving average period
        num_std: Number of standard deviations
        
    Returns:
        Dictionary with upper band, middle band, and lower band
    """
    try:
        # Calculate middle band (SMA)
        middle_band = data['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}

def calculate_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate historical volatility.
    
    Args:
        data: DataFrame with price data
        period: Rolling window period
        
    Returns:
        Series with volatility values
    """
    try:
        # Calculate daily returns
        returns = data['Close'].pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        return volatility
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return pd.Series(index=data.index)

def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate comprehensive momentum indicators.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with various momentum indicators
    """
    try:
        indicators = {}
        
        # RSI
        indicators['rsi'] = calculate_rsi(data)
        
        # MACD
        macd_data = calculate_macd(data)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(data)
        indicators.update(bb_data)
        
        # Volatility
        indicators['volatility'] = calculate_volatility(data)
        
        # Rate of Change (ROC)
        indicators['roc_5'] = data['Close'].pct_change(periods=5)
        indicators['roc_10'] = data['Close'].pct_change(periods=10)
        indicators['roc_20'] = data['Close'].pct_change(periods=20)
        
        # Moving Average Crossovers
        indicators['sma_20'] = data['Close'].rolling(window=20).mean()
        indicators['sma_50'] = data['Close'].rolling(window=50).mean()
        indicators['sma_200'] = data['Close'].rolling(window=200).mean()
        
        # Volume-based indicators
        indicators['volume_sma'] = data['Volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
        
        return indicators
    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {str(e)}")
        return {} 