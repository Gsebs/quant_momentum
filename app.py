import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from src.strategy import run_strategy

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
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/charts', exist_ok=True)

# Initialize data on startup
@app.before_first_request
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
    try:
        logger.info("Fetching momentum signals...")
        signals_file = 'data/momentum_signals.xlsx'
        if not os.path.exists(signals_file):
            logger.info("Momentum signals file not found, generating data...")
            run_strategy()
            if not os.path.exists(signals_file):
                logger.error("Failed to generate momentum signals file")
                return jsonify({'error': 'Failed to generate momentum signals'}), 500
        
        df = pd.read_excel(signals_file)
        logger.info("Successfully retrieved momentum signals")
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error fetching momentum signals: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    try:
        logger.info("Fetching performance data...")
        report_file = 'data/reports/momentum_report.xlsx'
        if not os.path.exists(report_file):
            logger.info("Performance report not found, generating data...")
            run_strategy()
            if not os.path.exists(report_file):
                logger.error("Failed to generate performance report")
                return jsonify({'error': 'Failed to generate performance report'}), 500
            
        df = pd.read_excel(report_file)
        logger.info("Successfully retrieved performance data")
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error fetching performance data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/<filename>')
def get_chart(filename):
    try:
        logger.info(f"Fetching chart: {filename}")
        chart_path = f'data/reports/{filename}'
        if not os.path.exists(chart_path):
            logger.error(f"Chart not found: {filename}")
            return jsonify({'error': 'Chart not found'}), 404
            
        logger.info(f"Successfully retrieved chart: {filename}")
        return send_file(chart_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error fetching chart: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 