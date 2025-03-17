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
from src.trading import PortfolioManager

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

# Initialize portfolio manager
portfolio_manager = PortfolioManager(initial_capital=1000000.0)

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
    """Get momentum signals for stocks and execute trades"""
    try:
        # First try to get cached signals
        cached = get_cached_signals()
        if cached and isinstance(cached, dict) and cached.get('signals'):
            logger.info("Using cached signals")
            signals = cached.get('signals')
        else:
            logger.info("No valid cached signals, fetching new data")
            # Try to get fresh signals
            signals = run_strategy(RELIABLE_TICKERS[:5])  # Start with just 5 tickers to reduce load
        
        if not signals or not isinstance(signals, dict):
            # Provide fallback data if no signals available
            logger.warning("No signals available, using fallback data")
            signals = {
                'AAPL': {
                    'momentum_score': 0.1,
                    'signal': 'HOLD',
                    'price': 100.0,
                    'change': 0.0
                },
                'MSFT': {
                    'momentum_score': 0.2,
                    'signal': 'HOLD',
                    'price': 200.0,
                    'change': 0.0
                }
            }
        
        # Sort signals by absolute momentum score and execute trades
        signals_list = []
        trades_executed = []
        
        try:
            # First sort by absolute momentum score
            sorted_signals = sorted(
                [(ticker, data) for ticker, data in signals.items()],
                key=lambda x: abs(float(x[1].get('momentum_score', 0))),
                reverse=True
            )
            
            for ticker, data in sorted_signals:
                try:
                    data['ticker'] = ticker
                    momentum_score = float(data.get('momentum_score', 0))
                    signal = data.get('signal', 'HOLD')
                    
                    # Execute trade if signal is strong enough
                    if signal in ['BUY', 'SELL'] and abs(momentum_score) >= 0.3:
                        try:
                            trade = portfolio_manager.execute_trade(
                                ticker=ticker,
                                signal=signal,
                                momentum_score=momentum_score
                            )
                            if trade:
                                trades_executed.append(trade)
                                logger.info(f"Executed trade: {trade}")
                        except Exception as e:
                            logger.error(f"Error executing trade for {ticker}: {str(e)}")
                            
                    signals_list.append(data)
                except Exception as e:
                    logger.error(f"Error processing signal for {ticker}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error sorting/processing signals: {str(e)}")
        
        # Get updated portfolio metrics after all trades
        try:
            portfolio_manager.update_positions()
            portfolio_manager.update_portfolio_history()
            metrics = portfolio_manager.get_portfolio_metrics()
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
            metrics = {
                'total_value': 1000000.0,
                'cash': 1000000.0,
                'positions': {},
                'returns': 0.0
            }
        
        # Add trade information to response
        response_data = {
            'signals': signals_list or [],
            'portfolio': metrics,
            'trades_executed': trades_executed,
            'last_update': datetime.now().isoformat()
        }
        
        return format_api_response(data=response_data)
        
    except Exception as e:
        logger.error(f"Error in get_momentum_signals: {str(e)}\n{traceback.format_exc()}")
        # Return a basic response with fallback data
        fallback_data = {
            'signals': [
                {'ticker': 'AAPL', 'momentum_score': 0.1, 'signal': 'HOLD', 'price': 100.0, 'change': 0.0},
                {'ticker': 'MSFT', 'momentum_score': 0.2, 'signal': 'HOLD', 'price': 200.0, 'change': 0.0}
            ],
            'portfolio': {
                'total_value': 1000000.0,
                'cash': 1000000.0,
                'positions': {},
                'returns': 0.0
            },
            'trades_executed': [],
            'last_update': datetime.now().isoformat()
        }
        return format_api_response(
            data=fallback_data,
            status='partial',
            message='Using fallback data due to error',
            code=200  # Return 200 with fallback data instead of error
        )

@app.route('/api/performance', methods=['GET'])
@limiter.limit("30/minute")
def get_performance():
    """Get portfolio performance metrics"""
    try:
        if redis_client is None:
            logger.error("Redis connection not available")
            return format_api_response(
                status='error',
                message='Service temporarily unavailable',
                code=503
            )
            
        # Get portfolio metrics
        metrics = portfolio_manager.get_portfolio_metrics()
        return format_api_response(data=metrics)
            
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Redis connection error in get_performance: {str(e)}")
        return format_api_response(
            status='error',
            message='Service temporarily unavailable',
            code=503
        )
    except Exception as e:
        logger.error(f"Error in get_performance: {str(e)}")
        return format_api_response(
            status='error',
            message='Failed to retrieve performance metrics',
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