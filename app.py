import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.strategy import run_strategy
from src.cache import clear_cache
from datetime import datetime, timedelta
import redis
import threading
import random

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

# List of reliable tickers for momentum strategy
RELIABLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "UNH",
    "BAC", "XOM", "PFE", "CSCO", "VZ"
]

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
        report_file = 'data/reports/momentum_report.xlsx'
        signals_file = 'data/momentum_signals.xlsx'
        
        # Initialize data if files don't exist
        if not os.path.exists(report_file) or not os.path.exists(signals_file):
            initialize_data()
            return jsonify({
                'status': 'initializing',
                'message': 'Data is being generated',
                'portfolio_stats': {
                    'portfolio_value': 1000000.00,  # Initial portfolio value
                    'daily_return': 0.00,
                    'sharpe_ratio': 0.00,
                    'max_drawdown': 0.00
                },
                'recent_trades': [],
                'performance_data': {
                    'dates': [],
                    'values': []
                }
            })

        # Get momentum signals
        signals = run_strategy(RELIABLE_TICKERS)
        
        if not signals:
            signals = {}
            
        # Calculate portfolio statistics
        total_value = sum(float(signal.get('current_price', 0)) * 100 for signal in signals.values())
        daily_returns = [float(signal.get('1d_return', 0)) for signal in signals.values()]
        avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        returns_std = pd.Series(daily_returns).std() if daily_returns else 0
        sharpe_ratio = ((avg_daily_return - risk_free_rate) / returns_std) if returns_std != 0 else 0
        
        # Calculate max drawdown from signals
        max_drawdown = max([abs(float(signal.get('drawdown', 0))) for signal in signals.values()], default=0)
        
        # Generate recent trades from signals
        recent_trades = []
        for ticker, signal in signals.items():
            if abs(float(signal.get('momentum_score', 0))) > 0.5:  # Threshold for trade signals
                trade_type = 'BUY' if float(signal.get('momentum_score', 0)) > 0 else 'SELL'
                recent_trades.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'ticker': ticker,
                    'type': trade_type,
                    'price': float(signal.get('current_price', 0)),
                    'quantity': int(100000 / float(signal.get('current_price', 1)))  # Assuming $100k per trade
                })
        
        # Sort trades by time (most recent first) and limit to last 10
        recent_trades = sorted(recent_trades, key=lambda x: x['time'], reverse=True)[:10]
        
        # Generate performance data (simulated historical values)
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
        base_value = total_value if total_value > 0 else 1000000
        
        # Generate realistic performance values with momentum trend
        values = []
        current_value = base_value
        for i in range(30):
            daily_change = avg_daily_return + (random.uniform(-0.02, 0.02) if avg_daily_return != 0 else random.uniform(-0.005, 0.015))
            current_value *= (1 + daily_change)
            values.append(round(current_value, 2))

        return jsonify({
            'status': 'success',
            'portfolio_stats': {
                'portfolio_value': round(total_value, 2),
                'daily_return': round(avg_daily_return * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown * 100, 2)
            },
            'recent_trades': recent_trades,
            'performance_data': {
                'dates': dates,
                'values': values
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving performance data: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
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