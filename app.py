import os
import pandas as pd
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from src.strategy import run_strategy

app = Flask(__name__)
CORS(app)

# Create necessary directories on startup
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/charts', exist_ok=True)

# Initialize data on startup
@app.before_first_request
def initialize_data():
    try:
        # Run the strategy to generate data files
        run_strategy()
    except Exception as e:
        print(f"Error initializing data: {str(e)}")

@app.route('/api/momentum-signals')
def get_momentum_signals():
    try:
        signals_file = 'data/momentum_signals.xlsx'
        if not os.path.exists(signals_file):
            # Generate data if it doesn't exist
            run_strategy()
        
        df = pd.read_excel(signals_file)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    try:
        report_file = 'data/reports/momentum_report.xlsx'
        if not os.path.exists(report_file):
            # Generate data if it doesn't exist
            run_strategy()
            
        df = pd.read_excel(report_file)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/<filename>')
def get_chart(filename):
    try:
        chart_path = f'data/reports/{filename}'
        if not os.path.exists(chart_path):
            return jsonify({'error': 'Chart not found'}), 404
            
        return send_file(chart_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 