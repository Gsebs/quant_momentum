"""
Quant Momentum Trading Algorithm Package
"""

from .strategy import run_strategy, get_cached_signals, RELIABLE_TICKERS
from .cache import redis_client, clear_cache

__version__ = '1.0.0'

from .data import get_stock_data
from .momentum import calculate_momentum_score 