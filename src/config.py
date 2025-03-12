"""
Configuration settings for the momentum strategy
"""

# Data settings
START_DATE = '2020-01-01'  # Default start date for historical data
END_DATE = None  # None means up to current date

# Strategy parameters
MOMENTUM_WINDOW = 12  # Number of months for momentum calculation
MIN_PRICE = 5.0  # Minimum stock price for universe filtering
MIN_VOLUME = 100000  # Minimum average daily volume
TOP_N_STOCKS = 10  # Number of top stocks to recommend

# Stock universe settings
MARKET_INDEX = '^GSPC'  # S&P 500 index

# Backtest settings
REBALANCE_FREQUENCY = 'M'  # Monthly rebalancing
TRANSACTION_COSTS = 0.001  # 0.1% transaction costs

# File paths
DATA_DIR = "data"
RESULTS_DIR = "data/results"

# Database configuration
DB_PATH = "data/momentum.db"

# Data paths
SIGNALS_FILE = "momentum_signals.xlsx"

# Model parameters
MODEL_PATH = "models/momentum_model.joblib"
LOOKBACK_PERIODS = {
    "1m": 21,    # 1 month
    "3m": 63,    # 3 months
    "6m": 126,   # 6 months
    "12m": 252   # 12 months
}

# ML model parameters
ML_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42
} 