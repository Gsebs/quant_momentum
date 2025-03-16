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
@limiter.limit("30/minute")
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
        
        if not os.path.exists(report_file) or not os.path.exists(signals_file):
            initialize_data()
            if not os.path.exists(report_file) or not os.path.exists(signals_file):
                return jsonify({
                    'error': 'Performance data not available',
                    'message': 'Data is being generated'
                }), 404

        # Read performance data from Excel files
        report_data = pd.read_excel(report_file, sheet_name=None)  # Read all sheets
        signals_data = pd.read_excel(signals_file)
        
        # Convert DataFrame to dict for each sheet
        performance_data = {
            'overview': report_data['Overview'].to_dict(orient='records'),
            'returns': report_data['Returns'].to_dict(orient='records'),
            'risk_metrics': report_data['Risk Metrics'].to_dict(orient='records'),
            'technical': report_data['Technical Indicators'].to_dict(orient='records'),
            'signals': signals_data.to_dict(orient='records')
        }
        
        # Calculate portfolio statistics
        latest_data = signals_data.iloc[-1] if not signals_data.empty else {}
        portfolio_stats = {
            'portfolio_value': float(latest_data.get('Last_Price', 0) * 100),  # Assuming $100 per position
            'daily_return': float(latest_data.get('1m_return', 0)),
            'sharpe_ratio': float(latest_data.get('risk_sharpe_ratio', 0)),
            'max_drawdown': float(latest_data.get('risk_max_drawdown', 0)),
        }
        
        # Get recent trades (last 10 trades)
        recent_trades = []
        for _, row in signals_data.iterrows():
            if row.get('position_size', 0) > 0:
                recent_trades.append({
                    'date': datetime.now().isoformat(),
                    'ticker': row.get('Ticker', ''),
                    'type': 'BUY' if row.get('momentum_score', 0) > 0 else 'SELL',
                    'price': float(row.get('Last_Price', 0)),
                    'quantity': int(row.get('position_size', 0) / float(row.get('Last_Price', 1)))
                })
        recent_trades = recent_trades[-10:]  # Get last 10 trades
        
        return jsonify({
            'status': 'success',
            'data': performance_data,
            'portfolio_stats': portfolio_stats,
            'recent_trades': recent_trades,
            'cached': True,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving performance data: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
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