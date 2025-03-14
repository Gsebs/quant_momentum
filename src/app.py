from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.strategy import MomentumStrategy
from src.cache import cache, init_cache
from src.data import fetch_sp500_tickers
from src.backtest import run_momentum_backtest
from src.reporting import generate_performance_metrics
from src.risk import calculate_risk_metrics
import logging
import os
import redis
import json
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Initialize Redis cache
init_cache()

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

strategy = MomentumStrategy()

@app.route('/api/metrics', methods=['GET'])
@limiter.limit("50/hour")
def get_metrics():
    try:
        # Get cached metrics or calculate new ones
        metrics = cache.get('metrics')
        if not metrics:
            portfolio = strategy.get_current_portfolio()
            performance_data = strategy.get_performance_history()
            
            # Calculate core metrics
            total_value = portfolio['total_value'] if portfolio else 0
            total_return = portfolio.get('total_return', 0) if portfolio else 0
            daily_return = portfolio.get('daily_return', 0) if portfolio else 0
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(performance_data)
            
            metrics = {
                'totalValue': total_value,
                'totalReturn': total_return,
                'dailyReturn': daily_return,
                'sharpeRatio': risk_metrics['sharpe_ratio'],
                'maxDrawdown': risk_metrics['max_drawdown'],
                'winRate': risk_metrics['win_rate'],
                'volatility': risk_metrics['volatility'],
                'momentumScore': strategy.calculate_market_momentum_score(),
                'activePositions': len(portfolio['positions']) if portfolio and 'positions' in portfolio else 0,
                'turnoverRate': portfolio.get('turnover_rate', 0) if portfolio else 0
            }
            
            # Cache metrics for 5 minutes
            cache.set('metrics', json.dumps(metrics), ex=300)
        else:
            metrics = json.loads(metrics)
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': 'Failed to fetch metrics'}), 500

@app.route('/api/performance', methods=['GET'])
@limiter.limit("50/hour")
def get_performance():
    try:
        performance = cache.get('performance')
        if not performance:
            # Get historical performance data
            performance_data = strategy.get_performance_history()
            
            # Calculate daily returns and cumulative returns
            performance = [{
                'date': entry['date'].strftime('%Y-%m-%d'),
                'value': entry['portfolio_value'],
                'return': entry['return'],
                'cumulative_return': entry['cumulative_return']
            } for entry in performance_data]
            
            # Cache performance data for 5 minutes
            cache.set('performance', json.dumps(performance), ex=300)
        else:
            performance = json.loads(performance)
        
        return jsonify(performance)
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        return jsonify({'error': 'Failed to fetch performance data'}), 500

@app.route('/api/signals', methods=['GET'])
@limiter.limit("50/hour")
def get_signals():
    try:
        signals = cache.get('signals')
        if not signals:
            # Get current momentum signals
            current_signals = strategy.get_momentum_signals()
            
            # Format signals for frontend
            signals = [{
                'ticker': signal['ticker'],
                'momentum_score': signal['momentum_score'],
                'rank': signal['rank'],
                'signal_strength': signal['signal_strength'],
                'last_price': signal['last_price'],
                'volume': signal['volume'],
                'recommendation': signal['recommendation']
            } for signal in current_signals]
            
            # Cache signals for 5 minutes
            cache.set('signals', json.dumps(signals), ex=300)
        else:
            signals = json.loads(signals)
        
        return jsonify(signals)
    except Exception as e:
        logger.error(f"Error getting momentum signals: {str(e)}")
        return jsonify({'error': 'Failed to fetch momentum signals'}), 500

@app.route('/api/alerts', methods=['GET'])
@limiter.limit("50/hour")
def get_alerts():
    try:
        alerts = cache.get('alerts')
        if not alerts:
            # Get system alerts and notifications
            portfolio = strategy.get_current_portfolio()
            risk_metrics = calculate_risk_metrics(strategy.get_performance_history())
            
            alerts = []
            
            # Check for various alert conditions
            if risk_metrics['volatility'] > 0.2:  # 20% volatility threshold
                alerts.append({
                    'type': 'warning',
                    'message': 'High portfolio volatility detected',
                    'timestamp': datetime.now().isoformat()
                })
            
            if risk_metrics['max_drawdown'] < -0.1:  # 10% drawdown threshold
                alerts.append({
                    'type': 'danger',
                    'message': f"Maximum drawdown of {risk_metrics['max_drawdown']:.1%} exceeded threshold",
                    'timestamp': datetime.now().isoformat()
                })
            
            if portfolio and portfolio.get('cash_position', 0) > 0.3:  # 30% cash threshold
                alerts.append({
                    'type': 'info',
                    'message': 'High cash position may indicate limited opportunities',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Cache alerts for 5 minutes
            cache.set('alerts', json.dumps(alerts), ex=300)
        else:
            alerts = json.loads(alerts)
        
        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        return jsonify({'error': 'Failed to fetch alerts'}), 500

@app.route('/api/strategy/config', methods=['GET'])
@limiter.limit("50/hour")
def get_strategy_config():
    try:
        return jsonify(strategy.get_config())
    except Exception as e:
        logger.error(f"Error getting strategy config: {str(e)}")
        return jsonify({'error': 'Failed to fetch strategy configuration'}), 500

@app.route('/api/strategy/update', methods=['POST'])
@limiter.limit("10/hour")
def update_strategy():
    try:
        params = request.get_json()
        strategy.update_config(params)
        return jsonify({'message': 'Strategy configuration updated successfully'})
    except Exception as e:
        logger.error(f"Error updating strategy config: {str(e)}")
        return jsonify({'error': 'Failed to update strategy configuration'}), 500

@app.route('/api/backtest', methods=['POST'])
@limiter.limit("5/hour")
def backtest():
    try:
        params = request.get_json()
        results = run_momentum_backtest(params)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return jsonify({'error': 'Failed to run backtest'}), 500

@app.route('/api/cache/clear', methods=['POST'])
@limiter.limit("1/hour")
def clear_cache():
    try:
        cache.flushdb()
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': 'Failed to clear cache'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 