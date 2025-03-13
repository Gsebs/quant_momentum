"""
Module for calculating momentum metrics and indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

def calculate_momentum_score(data: pd.DataFrame) -> float:
    """
    Calculate momentum score for a single stock.
    
    Args:
        data (pd.DataFrame): DataFrame with historical price data for a single stock
        
    Returns:
        float: Momentum score
    """
    try:
        # Sort by date
        data = data.sort_values('Date')
        
        # Calculate returns for different periods
        last_price = data['Close'].iloc[-1]
        returns = {
            '1m_return': (data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) * 100,
            '3m_return': (data['Close'].iloc[-1] / data['Close'].iloc[-66] - 1) * 100,
            '6m_return': (data['Close'].iloc[-1] / data['Close'].iloc[-132] - 1) * 100,
            '12m_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
        }
        
        # Calculate momentum score
        momentum_score = (
            returns['1m_return'] * 0.4 +
            returns['3m_return'] * 0.3 +
            returns['6m_return'] * 0.2 +
            returns['12m_return'] * 0.1
        )
        
        return momentum_score
        
    except Exception as e:
        logging.error(f"Error calculating momentum score: {str(e)}")
        return None

def calculate_momentum_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum metrics for each stock.
    
    Args:
        data (pd.DataFrame): DataFrame with historical price data
        
    Returns:
        pd.DataFrame: DataFrame with momentum metrics
    """
    try:
        # Group by ticker
        grouped = data.groupby('Ticker')
        
        # Initialize results dictionary
        results = {}
        
        for ticker, group in grouped:
            try:
                # Sort by date
                group = group.sort_values('Date')
                
                # Calculate returns
                last_price = group['Close'].iloc[-1]
                returns = {
                    'Last_Price': last_price,
                    'Avg_Volume': group['Volume'].mean(),
                    '1m_return': (group['Close'].iloc[-1] / group['Close'].iloc[-22] - 1) * 100,
                    '3m_return': (group['Close'].iloc[-1] / group['Close'].iloc[-66] - 1) * 100,
                    '6m_return': (group['Close'].iloc[-1] / group['Close'].iloc[-132] - 1) * 100,
                    '12m_return': (group['Close'].iloc[-1] / group['Close'].iloc[0] - 1) * 100
                }
                
                # Calculate volatility
                returns_series = group['Close'].pct_change()
                returns['volatility'] = returns_series.std() * np.sqrt(252) * 100
                
                # Volume ratio (current vs average)
                returns['volume_ratio'] = group['Volume'].iloc[-1] / group['Volume'].mean()
                
                # RSI
                delta = group['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                returns['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # MACD
                exp1 = group['Close'].ewm(span=12, adjust=False).mean()
                exp2 = group['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                returns['macd'] = macd.iloc[-1]
                returns['macd_signal'] = signal.iloc[-1]
                returns['macd_hist'] = macd.iloc[-1] - signal.iloc[-1]
                
                # Rate of Change
                returns['roc_5'] = (group['Close'].iloc[-1] / group['Close'].iloc[-5] - 1) * 100
                returns['roc_10'] = (group['Close'].iloc[-1] / group['Close'].iloc[-10] - 1) * 100
                returns['roc_20'] = (group['Close'].iloc[-1] / group['Close'].iloc[-20] - 1) * 100
                
                # Trend strength
                sma_20 = group['Close'].rolling(window=20).mean()
                sma_50 = group['Close'].rolling(window=50).mean()
                returns['trend_strength'] = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) * 100
                
                # Composite scores
                returns['momentum_score'] = (
                    returns['1m_return'] * 0.4 +
                    returns['3m_return'] * 0.3 +
                    returns['6m_return'] * 0.2 +
                    returns['12m_return'] * 0.1
                )
                
                returns['volatility_score'] = 100 - (returns['volatility'] / 2)  # Lower volatility is better
                returns['trend_score'] = returns['trend_strength'] * 100
                
                # Final composite score
                returns['composite_score'] = (
                    returns['momentum_score'] * 0.5 +
                    returns['volatility_score'] * 0.3 +
                    returns['trend_score'] * 0.2
                )
                
                # Position size based on volatility
                max_position = 0.1  # 10% max position
                vol_factor = np.exp(-returns['volatility'] / 100)  # Reduce size for high volatility
                returns['position_size'] = max_position * vol_factor
                
                # Rankings
                results[ticker] = returns
                
            except Exception as e:
                logging.warning(f"Error calculating momentum metrics for {ticker}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        momentum_df = pd.DataFrame.from_dict(results, orient='index')
        
        # Calculate rankings
        momentum_df['rsi_rank'] = momentum_df['rsi'].rank(ascending=False)
        momentum_df['macd_rank'] = momentum_df['macd'].rank(ascending=False)
        momentum_df['volatility_rank'] = momentum_df['volatility'].rank()
        
        # Initialize ML score and enhanced score columns
        momentum_df['ml_score'] = 0.0
        momentum_df['enhanced_score'] = momentum_df['composite_score']
        
        # Initialize risk metrics
        momentum_df['risk_total_return'] = momentum_df['12m_return']
        momentum_df['risk_annualized_return'] = momentum_df['12m_return']
        momentum_df['risk_annualized_volatility'] = momentum_df['volatility']
        momentum_df['risk_sharpe_ratio'] = momentum_df['12m_return'] / momentum_df['volatility']
        momentum_df['risk_sortino_ratio'] = momentum_df['12m_return'] / (momentum_df['volatility'] * 0.7)  # Approximate
        momentum_df['risk_max_drawdown'] = 0.0  # Will be calculated later
        
        return momentum_df
        
    except Exception as e:
        logging.error(f"Error in calculate_momentum_metrics: {str(e)}")
        raise 