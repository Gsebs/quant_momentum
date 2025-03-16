import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timedelta
import redis
import threading
import random
import numpy as np
import traceback
import json
from src.strategy import run_strategy, get_cached_signals, RELIABLE_TICKERS
from src.cache import redis_client, clear_cache

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add error handler
def handle_error(error):
    logger.error(f"Application error: {str(error)}\n{traceback.format_exc()}")
    return format_api_response(
        status='error',
        message=f"An error occurred: {str(error)}",
        code=500
    )

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.strategy import run_strategy, get_cached_signals, RELIABLE_TICKERS
    from src.cache import clear_cache
    logger.info("Successfully imported strategy modules")
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

app = Flask(__name__, 
    static_folder='static',
    static_url_path='/static',
    template_folder='templates'
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Register error handler
app.register_error_handler(Exception, handle_error)

# Configure rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    storage_options={"ssl_cert_reqs": None},
    default_limits=["200 per day", "50 per hour"]
)

# Initialize portfolio state if not exists
def initialize_portfolio_state():
    try:
        if not redis_client.exists('portfolio_state'):
            initial_state = {
                'initial_value': 1000000,
                'cash': 1000000,
                'positions': {},
                'trades': [],
                'daily_returns': [0.001, 0.002, -0.001, 0.003, -0.002],  # Sample data
                'total_trades': 0,
                'winning_trades': 0,
                'last_update': datetime.now().isoformat()
            }
            redis_client.set('portfolio_state', json.dumps(initial_state))
            logger.info("Initialized portfolio state")
    except Exception as e:
        logger.error(f"Error initializing portfolio state: {str(e)}")

# Initialize portfolio state on startup
initialize_portfolio_state()

def format_api_response(data=None, status='success', message=None, code=200):
    """Standardize API response format"""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
    }
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response), code

def ensure_directories():
    """Ensure required directories exist"""
    try:
        dirs = ['data', 'data/reports', 'data/charts', 'data/cache']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def initialize_data():
    """Initialize data and start background update"""
    try:
        logger.info("Initializing data...")
        ensure_directories()
        
        # Run strategy with reliable tickers
        signals = run_strategy(RELIABLE_TICKERS)
        
        if not signals:
            logger.warning("No cached signals available, background update started")
        else:
            logger.info(f"Retrieved {len(signals)} cached signals")
            
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        raise

def calculate_portfolio_metrics(portfolio):
    """Calculate key portfolio metrics"""
    try:
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        returns = np.array(portfolio.get('daily_returns', []))
        if len(returns) > 0:
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate win rate
        total_trades = portfolio.get('total_trades', 0)
        winning_trades = portfolio.get('winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        return {
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 1),
            'max_drawdown': round(max_drawdown * 100, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {
            'sharpe_ratio': 0,
            'win_rate': 0,
            'max_drawdown': 0
        }

@app.before_first_request
def before_first_request():
    """Initialize data before the first request"""
    try:
        initialize_data()
    except Exception as e:
        logger.error(f"Error in before_first_request: {str(e)}")

@app.route('/')
@limiter.exempt
def home():
    """Render dashboard page"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}\n{traceback.format_exc()}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/momentum-signals', methods=['GET'])
@limiter.exempt
def get_momentum_signals():
    """Get momentum signals for stocks"""
    try:
        signals = run_strategy(RELIABLE_TICKERS)
        if not signals:
            # Return sample data if no signals available
            sample_signals = [
                {
                    'ticker': 'AAPL',
                    'momentum_score': 0.85,
                    'current_price': 175.50,
                    'signal': 'BUY',
                    'change': '+2.3%'
                },
                {
                    'ticker': 'MSFT',
                    'momentum_score': 0.72,
                    'current_price': 285.30,
                    'signal': 'BUY',
                    'change': '+1.8%'
                },
                {
                    'ticker': 'GOOGL',
                    'momentum_score': -0.45,
                    'current_price': 125.20,
                    'signal': 'SELL',
                    'change': '-1.2%'
                }
            ]
            return format_api_response(
                data=sample_signals,
                status='success',
                message='Sample data while updating signals'
            )
        
        # Sort signals by momentum score
        if isinstance(signals, dict):
            signals_list = []
            for ticker, data in signals.items():
                data['ticker'] = ticker
                signals_list.append(data)
            signals = signals_list
            
        signals = sorted(signals, key=lambda x: abs(float(x.get('momentum_score', 0))), reverse=True)
        return format_api_response(data=signals)
        
    except Exception as e:
        logger.error(f"Error in get_momentum_signals: {str(e)}")
        return format_api_response(
            status='error',
            message='Failed to retrieve momentum signals',
            code=500
        )

@app.route('/api/performance', methods=['GET'])
@limiter.limit("30/minute")
def get_performance():
    """Get portfolio performance metrics"""
    try:
        # Get current portfolio state
        try:
            portfolio = redis_client.get('portfolio_state')
            if portfolio:
                portfolio = json.loads(portfolio)
            else:
                # Initialize with sample data if no portfolio exists
                portfolio = {
                    'initial_value': 1000000,
                    'cash': 950000,
                    'positions': {
                        'AAPL': {
                            'quantity': 100,
                            'price': 175.50,
                            'timestamp': datetime.now().isoformat(),
                            'market_value': 17550,
                            'unrealized_pnl': 550
                        }
                    },
                    'trades': [
                        {
                            'time': (datetime.now() - timedelta(minutes=5)).isoformat(),
                            'ticker': 'AAPL',
                            'type': 'BUY',
                            'price': 175.50,
                            'quantity': 100,
                            'total': 17550,
                            'status': 'FILLED'
                        }
                    ],
                    'daily_returns': [0.001, 0.002, -0.001, 0.003, -0.002],
                    'total_trades': 1,
                    'winning_trades': 1
                }
                redis_client.set('portfolio_state', json.dumps(portfolio))
        except Exception as e:
            logger.error(f"Error accessing Redis: {str(e)}")
            return format_api_response(
                status='error',
                message='Failed to access portfolio data',
                code=500
            )

        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio)
        
        # Prepare performance data
        performance_data = {
            'portfolio_value': round(portfolio.get('initial_value', 1000000) + sum([pos.get('unrealized_pnl', 0) for pos in portfolio.get('positions', {}).values()]), 2),
            'cash': round(portfolio.get('cash', 1000000), 2),
            'daily_return': portfolio.get('daily_returns', [0])[-1] if portfolio.get('daily_returns') else 0,
            'positions': portfolio.get('positions', {}),
            'recent_trades': portfolio.get('trades', [])[-10:],
            'daily_returns': portfolio.get('daily_returns', []),
            'sharpe_ratio': metrics['sharpe_ratio'],
            'win_rate': metrics['win_rate'],
            'max_drawdown': metrics['max_drawdown']
        }

        return format_api_response(data=performance_data)

    except Exception as e:
        logger.error(f"Error in get_performance: {str(e)}\n{traceback.format_exc()}")
        return format_api_response(
            status='error',
            message='Failed to calculate performance metrics',
            code=500
        )

@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check if we have signals
        signals = get_cached_signals()
        
        status = {
            'redis': 'connected',
            'signals': 'available' if signals else 'unavailable',
            'timestamp': datetime.now().isoformat()
        }
        
        return format_api_response(
            data=status,
            message='Service is healthy'
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return format_api_response(
            status='error',
            message='Service is degraded',
            code=503
        )

@app.route('/api/cache/clear', methods=['POST'])
@limiter.limit("1/hour")
def clear_cache_endpoint():
    """Clear cache and reset portfolio state"""
    try:
        clear_cache()
        redis_client.delete('portfolio_state')
        initialize_data()
        
        return format_api_response(
            message='Cache cleared successfully'
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return format_api_response(
            status='error',
            message='Failed to clear cache',
            code=500
        )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 