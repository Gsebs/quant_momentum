"""
This is where I handle all the risk stuff in my strategy. It's super important because
even a great strategy can blow up if you don't manage risk properly.

What this module does:
1. Calculates risk metrics (Sharpe ratio, drawdowns, etc.)
2. Makes sure we don't put too much money in one stock
3. Keeps track of sector exposure (don't want all tech stocks!)
4. Adjusts position sizes based on risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    The Sharpe ratio is like a report card for your strategy - it tells you how much return
    you're getting for the risk you're taking. I use 2% as the risk-free rate (like T-bills).
    
    A Sharpe ratio above 1 is decent, above 2 is great, and above 3 is amazing.
    This helps us compare different strategies and see if we're actually good or just lucky.
    """
    try:
        # Convert yearly rate to daily (there are 252 trading days in a year)
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate how much we beat the risk-free rate
        excess_returns = returns - daily_rf
        
        # Get yearly Sharpe ratio (multiply by sqrt(252) to annualize)
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        return sharpe
    except Exception as e:
        logger.error(f"Couldn't calculate Sharpe ratio: {str(e)}")
        return 0.0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    The Sortino ratio is like a smarter Sharpe ratio - it only looks at downside risk
    (when we lose money). This is cool because most investors care more about losses
    than gains.
    
    I like using both Sharpe and Sortino because they tell different stories:
    - Sharpe tells us about overall risk-adjusted returns
    - Sortino focuses on the bad stuff (losses)
    """
    try:
        # Convert yearly rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # How much we beat the risk-free rate
        excess_returns = returns - daily_rf
        
        # Only look at the losses (negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        # Calculate yearly Sortino ratio
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_std)
        
        return sortino
    except Exception as e:
        logger.error(f"Couldn't calculate Sortino ratio: {str(e)}")
        return 0.0

def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    This finds our worst losing streak - how much money we lost from peak to bottom.
    Super important because it tells us:
    1. How bad things can get (helps set stop losses)
    2. If we can handle the psychological pain
    3. How much capital we need to survive the worst times
    """
    try:
        # Track the highest price we've seen
        running_max = prices.expanding(min_periods=1).max()
        drawdowns = prices / running_max - 1
        
        # Find the worst drawdown
        max_drawdown = drawdowns.min()
        end_idx = drawdowns.idxmin()
        
        # When did the drawdown start?
        peak_idx = running_max.loc[:end_idx].idxmax()
        
        return max_drawdown, peak_idx, end_idx
    except Exception as e:
        logger.error(f"Couldn't calculate max drawdown: {str(e)}")
        return 0.0, None, None

def calculate_position_sizes(data: pd.DataFrame) -> pd.DataFrame:
    """
    This is where we decide how much to bet on each stock. The key ideas are:
    1. Don't put too much in any one stock (max 20%)
    2. Put more money in our highest conviction picks
    3. Make sure all position sizes add up to 100%
    
    The cool part is we use both momentum and risk metrics to size positions.
    Stocks with better momentum get bigger positions, but we cap them for safety.
    """
    try:
        # Make a copy so we don't mess up the original data
        result = data.copy()
        
        # Use enhanced score if we have it, otherwise use basic momentum
        if 'enhanced_score' in result.columns:
            score = result['enhanced_score']
        else:
            score = result['composite_score']
            
        # Convert scores to position sizes (making sure they add to 1)
        total_score = score.sum()
        if total_score > 0:
            result['position_size'] = score / total_score
        else:
            result['position_size'] = 0
            
        # Set minimum and maximum position sizes
        max_position = 0.2  # No more than 20% in one stock
        min_position = 0.02  # At least 2% if we're going to bother
        
        # Apply the limits
        result['position_size'] = np.clip(result['position_size'], min_position, max_position)
        
        # Make sure it all adds up to 100%
        total_position = result['position_size'].sum()
        if total_position > 0:
            result['position_size'] = result['position_size'] / total_position
            
        return result
        
    except Exception as e:
        logging.error(f"Error calculating position sizes: {str(e)}")
        return data

def analyze_sector_exposure(momentum_df: pd.DataFrame, sector_data: Dict[str, str]) -> pd.Series:
    """
    Makes sure we're not too heavy in any one sector. For example, during the
    tech bubble, you didn't want all tech stocks!
    
    This helps us:
    1. Stay diversified across sectors
    2. Avoid sector-specific risks
    3. Catch sector rotation opportunities
    """
    try:
        # Map stocks to their sectors
        sectors = pd.Series(sector_data)
        portfolio_sectors = sectors[momentum_df.index]
        
        # Calculate how much we have in each sector
        sector_weights = momentum_df['enhanced_score'].groupby(portfolio_sectors).sum()
        sector_weights = sector_weights / sector_weights.sum()
        
        return sector_weights
    except Exception as e:
        logger.error(f"Error analyzing sector exposure: {str(e)}")
        return pd.Series()

def calculate_risk_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk metrics for each stock."""
    try:
        # Create a copy to avoid modifying the original
        risk_data = data.copy()
        
        # Calculate risk metrics
        for ticker in risk_data.index:
            try:
                # Total return
                risk_data.loc[ticker, 'risk_total_return'] = risk_data.loc[ticker, '12m_return']
                
                # Annualized return (already annualized)
                risk_data.loc[ticker, 'risk_annualized_return'] = risk_data.loc[ticker, '12m_return']
                
                # Annualized volatility (already annualized)
                risk_data.loc[ticker, 'risk_annualized_volatility'] = risk_data.loc[ticker, 'volatility']
                
                # Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                excess_return = risk_data.loc[ticker, 'risk_annualized_return'] - risk_free_rate
                if risk_data.loc[ticker, 'risk_annualized_volatility'] > 0:
                    risk_data.loc[ticker, 'risk_sharpe_ratio'] = excess_return / risk_data.loc[ticker, 'risk_annualized_volatility']
                else:
                    risk_data.loc[ticker, 'risk_sharpe_ratio'] = 0.0
                
                # Sortino ratio (using downside deviation)
                downside_returns = min(0, risk_data.loc[ticker, 'risk_annualized_return'] - risk_free_rate)
                if downside_returns < 0:
                    risk_data.loc[ticker, 'risk_sortino_ratio'] = excess_return / abs(downside_returns)
                else:
                    risk_data.loc[ticker, 'risk_sortino_ratio'] = risk_data.loc[ticker, 'risk_sharpe_ratio']
                
                # Maximum drawdown (using 12-month return as proxy)
                risk_data.loc[ticker, 'risk_max_drawdown'] = min(0, risk_data.loc[ticker, '12m_return'])
                
            except Exception as e:
                logger.error(f"Error calculating risk metrics for {ticker}: {str(e)}")
                continue
        
        return risk_data
        
    except Exception as e:
        logger.error(f"Error in calculate_risk_metrics: {str(e)}")
        return data 