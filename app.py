import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.strategy import run_strategy, get_cached_signals, RELIABLE_TICKERS
from src.cache import clear_cache
from datetime import datetime, timedelta
import redis
import threading
import random
import numpy as np
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure Redis for rate limiting
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(
    redis_url,
    ssl_cert_reqs=None,  # Disable SSL certificate verification
    decode_responses=True  # Decode responses to UTF-8 strings
)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    storage_options={"ssl_cert_reqs": None},  # Disable SSL certificate verification
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

        # Initialize portfolio metrics
        portfolio_value = 1000000  # Starting with $1M
        total_trades = 0
        win_count = 0
        daily_returns = []
        
        # Process signals for portfolio stats
        for signal in signals:  # Changed from signals.values() to signals
            momentum_score = float(signal['momentum_score'])
            price_change = float(signal['price_change'])
            
            # Count trades and wins
            if abs(momentum_score) > 0.1:  # Only count significant signals as trades
                total_trades += 1
                if (momentum_score > 0 and price_change > 0) or (momentum_score < 0 and price_change < 0):
                    win_count += 1
            
            # Calculate daily return impact
            position_size = abs(momentum_score) * 0.1  # Size position based on signal strength
            daily_return = position_size * price_change
            daily_returns.append(daily_return)
        
        # Calculate portfolio statistics
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        volatility = np.std(daily_returns) if daily_returns else 0
        sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252) if volatility else 0
        max_drawdown = min(daily_returns) if daily_returns else 0
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Generate recent trades
        recent_trades = []
        for signal in signals:  # Changed from signals.values() to signals
            if abs(float(signal['momentum_score'])) > 0.1:
                trade = {
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'ticker': signal['ticker'],
                    'type': 'BUY' if float(signal['momentum_score']) > 0 else 'SELL',
                    'price': float(signal['current_price']),
                    'quantity': int(100000 * abs(float(signal['momentum_score'])))  # Position size based on signal strength
                }
                recent_trades.append(trade)
        
        # Sort trades by time (most recent first)
        recent_trades.sort(key=lambda x: x['time'], reverse=True)
        
        # Generate performance data for the last 30 days
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
        cumulative_return = 1.0
        values = []
        for _ in dates:
            daily_change = np.random.normal(avg_daily_return, volatility)
            cumulative_return *= (1 + daily_change)
            values.append(portfolio_value * cumulative_return)
        
        return jsonify({
            'status': 'success',
            'portfolio_stats': {
                'portfolio_value': portfolio_value,
                'daily_return': avg_daily_return * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100  # Convert to percentage
            },
            'strategy_performance': {
                'win_rate': win_rate,
                'profit_factor': (win_count / (total_trades - win_count)) if (total_trades - win_count) > 0 else 1.0
            },
            'model_metrics': {
                'prediction_accuracy': win_rate,  # Using win rate as a proxy for prediction accuracy
                'signal_strength': np.mean([abs(float(s['momentum_score'])) for s in signals]) * 100  # Average signal strength
            },
            'recent_trades': recent_trades[:10],  # Show only the 10 most recent trades
            'performance_data': {
                'dates': dates,
                'values': values
            }
        })
    except Exception as e:
        app.logger.error(f"Error retrieving performance data: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

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