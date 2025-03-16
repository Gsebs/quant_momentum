import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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

from datetime import datetime, timedelta
import redis
import threading
import random
import numpy as np
import traceback
import json
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure Redis with better error handling and retries
def get_redis_client():
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    retry = Retry(ExponentialBackoff(), 3)  # Retry 3 times with exponential backoff
    
    return redis.from_url(
        redis_url,
        ssl_cert_reqs=None,
        decode_responses=True,
        socket_timeout=5,  # 5 seconds timeout
        socket_connect_timeout=5,
        retry=retry
    )

try:
    redis_client = get_redis_client()
    redis_client.ping()  # Test connection
    logger.info("Successfully connected to Redis")
except (ConnectionError, TimeoutError) as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    # Initialize in-memory fallback
    class FallbackCache:
        def __init__(self):
            self._data = {}
        def get(self, key):
            return self._data.get(key)
        def set(self, key, value, ex=None):
            self._data[key] = value
        def delete(self, key):
            self._data.pop(key, None)
    
    redis_client = FallbackCache()
    logger.info("Using in-memory fallback cache")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    storage_options={"ssl_cert_reqs": None},
    default_limits=["200 per day", "50 per hour"]
)

# Create necessary directories on startup
def ensure_directories():
    try:
        dirs = ['data', 'data/reports', 'data/charts']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def initialize_data():
    """Initialize data and start background update."""
    try:
        logger.info("Initializing data...")
        ensure_directories()
        
        # Run strategy with reliable tickers
        signals = run_strategy(RELIABLE_TICKERS)
        
        if not signals:
            logger.info("No cached signals available, background update started")
        else:
            logger.info(f"Retrieved {len(signals)} cached signals")
            
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        raise

@app.before_first_request
def before_first_request():
    """Initialize data before the first request."""
    try:
        initialize_data()
    except Exception as e:
        logger.error(f"Error in before_first_request: {str(e)}")

@app.route('/')
@limiter.exempt
def home():
    """Home page route."""
    return render_template('index.html')

@app.route('/api/momentum-signals', methods=['GET'])
@limiter.exempt
def get_momentum_signals():
    """
    Get momentum signals for stocks.
    Returns cached data immediately and updates in background.
    """
    try:
        # Initialize data if needed
        signals = run_strategy(RELIABLE_TICKERS)
        
        return jsonify({
            'status': 'updating' if not signals else 'success',
            'data': signals,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_momentum_signals: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/performance', methods=['GET'])
@limiter.limit("30/minute")
def get_performance():
    try:
        signals = get_cached_signals()
        if not signals:
            return jsonify({'status': 'error', 'message': 'No signals available'})

        # Get current portfolio state
        try:
            portfolio = redis_client.get('portfolio_state')
            if portfolio:
                portfolio = json.loads(portfolio)
            else:
                # Initialize portfolio if not exists
                portfolio = {
                    'initial_value': 1000000,
                    'cash': 1000000,
                    'positions': {},
                    'trades': [],
                    'daily_returns': [],
                    'total_trades': 0,
                    'winning_trades': 0
                }
                redis_client.set('portfolio_state', json.dumps(portfolio))
        except Exception as e:
            logger.error(f"Error accessing Redis: {str(e)}")
            portfolio = {
                'initial_value': 1000000,
                'cash': 1000000,
                'positions': {},
                'trades': [],
                'daily_returns': [],
                'total_trades': 0,
                'winning_trades': 0
            }

        # Update positions based on current prices
        portfolio_value = portfolio['cash']
        daily_pnl = 0
        
        for signal in signals:
            try:
                ticker = signal['ticker']
                current_price = float(signal['current_price'])
                
                # Update position values
                if ticker in portfolio['positions']:
                    position = portfolio['positions'][ticker]
                    old_value = position['quantity'] * position['price']
                    new_value = position['quantity'] * current_price
                    portfolio_value += new_value
                    daily_pnl += new_value - old_value
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing signal {signal}: {str(e)}")
                continue

        # Process new signals for trades
        for signal in signals:
            try:
                momentum_score = float(signal.get('momentum_score', 0))
                ticker = signal['ticker']
                current_price = float(signal['current_price'])
                
                # Execute trades based on momentum signals
                if abs(momentum_score) > 0.1:
                    position_size = min(abs(momentum_score) * portfolio_value * 0.1, portfolio['cash'])  # Size based on signal strength
                    quantity = int(position_size / current_price)
                    
                    if momentum_score > 0.1 and ticker not in portfolio['positions'] and quantity > 0:  # BUY
                        if portfolio['cash'] >= position_size:
                            portfolio['positions'][ticker] = {
                                'quantity': quantity,
                                'price': current_price,
                                'timestamp': datetime.now().isoformat()
                            }
                            portfolio['cash'] -= quantity * current_price
                            portfolio['trades'].append({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'ticker': ticker,
                                'type': 'BUY',
                                'price': current_price,
                                'quantity': quantity
                            })
                            portfolio['total_trades'] += 1
                    
                    elif momentum_score < -0.1 and ticker in portfolio['positions']:  # SELL
                        position = portfolio['positions'][ticker]
                        sell_value = position['quantity'] * current_price
                        portfolio['cash'] += sell_value
                        
                        # Calculate if winning trade
                        if current_price > position['price']:
                            portfolio['winning_trades'] += 1
                        
                        portfolio['trades'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'ticker': ticker,
                            'type': 'SELL',
                            'price': current_price,
                            'quantity': position['quantity']
                        })
                        portfolio['total_trades'] += 1
                        del portfolio['positions'][ticker]
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing trade for signal {signal}: {str(e)}")
                continue

        # Calculate daily return
        try:
            daily_return = daily_pnl / (portfolio_value - daily_pnl) if (portfolio_value - daily_pnl) > 0 else 0
            portfolio['daily_returns'].append(daily_return)
            
            # Keep only last 30 days of returns
            if len(portfolio['daily_returns']) > 30:
                portfolio['daily_returns'] = portfolio['daily_returns'][-30:]
            
            # Calculate performance metrics
            returns = np.array(portfolio['daily_returns'])
            avg_daily_return = float(np.mean(returns)) if len(returns) > 0 else 0
            volatility = float(np.std(returns)) if len(returns) > 0 else 0
            sharpe_ratio = float((avg_daily_return / volatility) * np.sqrt(252)) if volatility > 0 else 0
            max_drawdown = float(min(returns)) if len(returns) > 0 else 0
            win_rate = float(portfolio['winning_trades'] / portfolio['total_trades'] * 100) if portfolio['total_trades'] > 0 else 0
            
            # Save updated portfolio state
            try:
                redis_client.set('portfolio_state', json.dumps(portfolio))
            except Exception as e:
                logger.error(f"Error saving portfolio state: {str(e)}")

            # Return performance data
            return jsonify({
                'status': 'success',
                'data': {
                    'portfolio_value': portfolio_value,
                    'cash': portfolio['cash'],
                    'positions': portfolio['positions'],
                    'daily_return': daily_return,
                    'avg_daily_return': avg_daily_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': portfolio['total_trades'],
                    'winning_trades': portfolio['winning_trades'],
                    'recent_trades': portfolio['trades'][-10:],  # Last 10 trades
                    'daily_returns': portfolio['daily_returns']
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error calculating performance metrics',
                'error': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in get_performance: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

@app.route('/api/charts/<filename>', methods=['GET'])
@limiter.limit("60/minute")
def get_chart(filename):
    try:
        return send_from_directory('data/charts', filename)
    except Exception as e:
        logger.error(f"Error retrieving chart {filename}: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Chart not found',
            'message': f'Chart {filename} could not be retrieved'
        }), 404

@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint."""
    try:
        initialize_data()
        return jsonify({
            'status': 'healthy',
            'environment': os.getenv('FLASK_ENV', 'production'),
            'data_directory': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/cache/clear', methods=['POST'])
@limiter.limit("1/hour")
def clear_cache_endpoint():
    try:
        clear_cache()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Cache clear failed',
            'message': str(e)
        }), 500

# Initialize data on startup
initialize_data()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 