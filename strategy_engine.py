import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime, timedelta
from market_data_feed import get_latest_prices, get_price_history, reliable_tickers
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store strategy state
cached_signals: Dict[str, Dict] = {}
last_confidence: float = 0.0

def calculate_momentum_score(prices: List[float], window: int = 20) -> float:
    """Calculate momentum score using price changes"""
    if len(prices) < window:
        return 0.0
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate momentum score (weighted average of returns)
    weights = np.exp(np.linspace(-1, 0, len(returns)))
    weights /= weights.sum()
    
    momentum_score = np.sum(returns * weights)
    return momentum_score

def generate_signals() -> Dict[str, Dict]:
    """Generate trading signals based on momentum"""
    global cached_signals, last_confidence
    
    try:
        signals = {}
        confidence_scores = []
        
        for ticker in reliable_tickers:
            price_history = get_price_history(ticker)
            if not price_history:
                continue
            
            # Extract prices
            prices = [p['price'] for p in price_history]
            
            # Calculate momentum score
            momentum_score = calculate_momentum_score(prices)
            
            # Generate signal based on momentum score
            if momentum_score > 0.3:
                signal = 'BUY'
            elif momentum_score < -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Calculate confidence score (absolute value of momentum)
            confidence = abs(momentum_score)
            confidence_scores.append(confidence)
            
            signals[ticker] = {
                'momentum_score': momentum_score,
                'signal': signal,
                'price': prices[-1],
                'change': (prices[-1] - prices[0]) / prices[0] * 100,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        
        # Update global state
        cached_signals = signals
        last_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return signals
    
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        return {}

def get_cached_signals() -> Dict[str, Dict]:
    """Get the last cached signals"""
    return cached_signals

async def run_strategy():
    """Run the trading strategy"""
    logger.info("Starting trading strategy")
    
    while True:
        try:
            # Generate new signals
            signals = generate_signals()
            
            # Log strategy status
            logger.info(f"Generated signals for {len(signals)} tickers")
            logger.info(f"Average confidence: {last_confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {str(e)}")
        
        # Wait for 5 minutes before next update
        await asyncio.sleep(300) 