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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379")
)

# Create necessary directories on startup
os.makedirs('data', exist_ok=True)
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/charts', exist_ok=True)

def initialize_data():
    try:
        logger.info("Initializing data...")
        # Run the strategy to generate data files
        run_strategy()
        logger.info("Data initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}", exc_info=True)
        raise

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
@limiter.limit("30/minute")
def get_momentum_signals():
    """Get momentum trading signals."""
    try:
        signals = run_strategy()
        
        if not signals:
            return jsonify({'error': 'No momentum signals available'}), 404
            
        return jsonify({
            'status': 'success',
            'signals': signals,
            'cached': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting momentum signals: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
@limiter.limit("30/minute")
def get_performance():
    try:
        report_file = 'data/reports/momentum_report.xlsx'
        if not os.path.exists(report_file):
            initialize_data()
            if not os.path.exists(report_file):
                return jsonify({'error': 'Performance report not available'}), 404
            
        df = pd.read_excel(report_file)
        performance = df.to_dict(orient='records')
        
        return jsonify({
            'performance': performance,
            'cached': True,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving performance data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/<filename>', methods=['GET'])
@limiter.limit("60/minute")
def get_chart(filename):
    try:
        return send_from_directory('data/charts', filename)
    except Exception as e:
        logger.error(f"Error retrieving chart {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Chart {filename} not found'}), 404

@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/cache/clear', methods=['POST'])
@limiter.limit("1/hour")
def clear_cache_endpoint():
    try:
        clear_cache()
        return jsonify({'status': 'success', 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 