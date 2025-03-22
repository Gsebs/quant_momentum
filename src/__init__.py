"""
<<<<<<< HEAD
Market Making Strategy Package
""" 
=======
Quantitative HFT Algorithm Package
"""

from . import strategy
from . import cache
from . import trading
from . import ml_model

__all__ = ['strategy', 'cache', 'trading', 'ml_model']

from .strategy import run_strategy, get_cached_signals, RELIABLE_TICKERS
from .cache import RedisCache, clear_cache
from .trading import TradingEngine
from .ml_model import HFTModel, HFTFeatureEngine

__version__ = '1.0.0'

from .data import get_stock_data
from .momentum import calculate_momentum_score 
>>>>>>> heroku/main
