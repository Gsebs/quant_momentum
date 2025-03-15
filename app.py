import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.strategy import run_strategy
from src.cache import clear_cache
from datetime import datetime, timedelta
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
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
    try:
        logger.info("Initializing data...")
        ensure_directories()
        # Run the strategy to generate data files
        run_strategy()
        logger.info("Data initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}", exc_info=True)
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
    return jsonify({
        'status': 'ok',
        'message': 'Momentum Trading API',
        'endpoints': [
            '/api/momentum-signals',
            '/api/performance',
            '/api/charts/<filename>'
        ]
    })

@app.route('/api/momentum-signals')
@limiter.limit("5 per minute")
def get_momentum_signals():
    """Get momentum signals for stocks."""
    try:
        initialize_data()
        signals = run_strategy()
        return jsonify({
            'status': 'success',
            'data': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting momentum signals: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/performance', methods=['GET'])
@limiter.limit("30/minute")
def get_performance():
    try:
        report_file = 'data/reports/momentum_report.xlsx'
        if not os.path.exists(report_file):
            initialize_data()
            if not os.path.exists(report_file):
                return jsonify({
                    'error': 'Performance report not available',
                    'message': 'Report is being generated'
                }), 404
            
        df = pd.read_excel(report_file)
        performance = df.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'performance': performance,
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 