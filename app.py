import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
from src.strategy import run_strategy
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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

@app.route('/api/momentum-signals')
def get_momentum_signals():
    """Get momentum trading signals."""
    logger.info("Running strategy to generate fresh signals...")
    logger.info("Initializing data...")
    
    try:
        results = run_strategy()
        if not results:
            return jsonify({'error': 'No momentum signals available'}), 404
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error generating momentum signals: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    try:
        report_file = 'data/reports/momentum_report.xlsx'
        if not os.path.exists(report_file):
            initialize_data()  # Try to generate the report if it doesn't exist
            if not os.path.exists(report_file):
                return jsonify({'error': 'Performance report not available'}), 404
            
        df = pd.read_excel(report_file)
        performance = df.to_dict(orient='records')
        
        return jsonify({
            'performance': performance,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving performance data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/<filename>', methods=['GET'])
def get_chart(filename):
    try:
        return send_from_directory('data/charts', filename)
    except Exception as e:
        logger.error(f"Error retrieving chart {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Chart {filename} not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 