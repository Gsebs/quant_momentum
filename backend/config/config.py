import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange configurations
EXCHANGES = ['binance', 'coinbase']

# Trading pairs to monitor
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'ADA/USDT'
]

# API Keys (loaded from environment variables)
API_KEYS = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'secret': os.getenv('BINANCE_API_SECRET', '')
    },
    'coinbase': {
        'api_key': os.getenv('COINBASE_API_KEY', ''),
        'secret': os.getenv('COINBASE_API_SECRET', '')
    }
}

# Risk management settings
RISK_LIMITS = {
    'max_position_size': 1.0,  # Maximum position size in BTC
    'max_daily_loss': 1000.0,  # Maximum daily loss in USD
    'max_drawdown': 0.1,       # Maximum drawdown (10%)
    'min_profit_threshold': 0.1 # Minimum profit threshold in USD
}

# Trading parameters
TRADING_PARAMS = {
    'min_profit_threshold': 0.1,  # Minimum profit threshold in USD
    'max_slippage': 0.001,        # Maximum allowed slippage (0.1%)
    'order_timeout': 5,           # Order timeout in seconds
    'retry_attempts': 3,          # Number of retry attempts for failed orders
    'retry_delay': 1,             # Delay between retries in seconds
}

# Market data settings
MARKET_DATA = {
    'update_interval': 1,         # Market data update interval in seconds
    'price_history_size': 1000,   # Number of historical prices to maintain
    'volume_threshold': 1000.0,   # Minimum volume threshold in USD
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'trading.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# WebSocket settings
WEBSOCKET_CONFIG = {
    'ping_interval': 20,          # Ping interval in seconds
    'ping_timeout': 10,           # Ping timeout in seconds
    'reconnect_interval': 5,      # Reconnect interval in seconds
    'max_reconnect_attempts': 5,  # Maximum number of reconnect attempts
}

# Database settings
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'trading_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Frontend settings
FRONTEND_CONFIG = {
    'host': '0.0.0.0',
    'port': 3000,
    'debug': True,
}

# Main configuration dictionary
CONFIG = {
    'exchanges': EXCHANGES,
    'trading_pairs': TRADING_PAIRS,
    'api_keys': API_KEYS,
    'risk_limits': RISK_LIMITS,
    'trading_params': TRADING_PARAMS,
    'market_data': MARKET_DATA,
    'logging': LOGGING_CONFIG,
    'websocket': WEBSOCKET_CONFIG,
    'database': DATABASE_CONFIG,
    'frontend': FRONTEND_CONFIG,
}

def get_config() -> Dict:
    """Get the complete configuration dictionary"""
    return CONFIG

def validate_config() -> bool:
    """Validate the configuration settings"""
    try:
        # Validate API keys
        for exchange, keys in API_KEYS.items():
            if not keys['api_key'] or not keys['secret']:
                print(f"Warning: Missing API keys for {exchange}")

        # Validate trading pairs
        if not TRADING_PAIRS:
            print("Error: No trading pairs specified")
            return False

        # Validate risk limits
        if RISK_LIMITS['max_position_size'] <= 0:
            print("Error: Invalid max position size")
            return False

        # Validate trading parameters
        if TRADING_PARAMS['min_profit_threshold'] <= 0:
            print("Error: Invalid minimum profit threshold")
            return False

        return True

    except Exception as e:
        print(f"Error validating configuration: {str(e)}")
        return False 