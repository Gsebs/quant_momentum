"""
Configuration parameters for the momentum strategy.
"""

from datetime import datetime, timedelta

# Data parameters
START_DATE = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years of data
END_DATE = datetime.now().strftime('%Y-%m-%d')
MIN_PRICE = 5.0  # Minimum stock price
MIN_VOLUME = 100000  # Minimum average daily volume

# Strategy parameters
TOP_N_STOCKS = 10  # Number of stocks to hold in portfolio
MOMENTUM_WEIGHTS = {
    '1M': 0.4,
    '3M': 0.3,
    '6M': 0.2,
    '12M': 0.1
}

# Risk management parameters
MAX_POSITION_SIZE = 0.20  # Maximum position size as fraction of portfolio
STOP_LOSS = 0.10  # Stop loss percentage
MAX_SECTOR_EXPOSURE = 0.30  # Maximum exposure to any sector

# Technical indicator parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOW = 252  # One year of trading days

# Backtest parameters
INITIAL_CAPITAL = 100000
COMMISSION = 0.001  # 0.1% commission per trade
SLIPPAGE = 0.001  # 0.1% slippage per trade

# Machine learning parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
N_ESTIMATORS = 100

# File paths
DATA_DIR = 'data'
SIGNALS_FILE = f'{DATA_DIR}/momentum_signals.xlsx'
REPORT_FILE = f'{DATA_DIR}/momentum_report.xlsx'
DB_FILE = f'{DATA_DIR}/momentum.db'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 