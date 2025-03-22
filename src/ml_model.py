"""
Machine Learning Model for Quantitative HFT Algorithm
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Dict, List, Optional, Tuple
import logging
from numba import njit
import time
from datetime import datetime, timedelta
from collections import deque
import threading
import xgboost as xgb
import os

logger = logging.getLogger(__name__)

@njit
def compute_features(
    prices: np.ndarray,
    volumes: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    window: int = 100
) -> np.ndarray:
    """Compute features using Numba for microsecond-level performance"""
    n = len(prices)
    if n < window:
        return np.zeros(8)  # Number of features
        
    # Price-based features
    returns = np.diff(prices[-window:]) / prices[-window-1:-1]
    volatility = np.std(returns)
    momentum = returns[-1]  # Most recent return
    
    # Volume-based features
    volume_ma = np.mean(volumes[-window:])
    volume_std = np.std(volumes[-window:])
    
    # Order book features
    imbalance = (np.sum(bid_sizes[-window:]) - np.sum(ask_sizes[-window:])) / \
                (np.sum(bid_sizes[-window:]) + np.sum(ask_sizes[-window:]))
    
    # Trend features
    short_ma = np.mean(prices[-20:])
    long_ma = np.mean(prices[-window:])
    
    return np.array([
        volatility,
        momentum,
        volume_ma,
        volume_std,
        imbalance,
        short_ma / long_ma - 1,  # Trend indicator
        np.percentile(returns, 95),  # Tail risk
        np.sum(returns > 0) / len(returns)  # Buy pressure
    ])

@njit
def compute_orderbook_features(
    bids: np.ndarray,  # shape: (N, 2) for price, volume
    asks: np.ndarray,  # shape: (N, 2) for price, volume
    depth: int = 5
) -> np.ndarray:
    """Compute order book features using Numba for microsecond-level performance."""
    if len(bids) == 0 or len(asks) == 0:
        return np.zeros(4)
        
    # Calculate order book imbalance
    bid_volume = np.sum(bids[:depth, 1])
    ask_volume = np.sum(asks[:depth, 1])
    total_volume = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
    
    # Calculate weighted average prices
    bid_wap = np.sum(bids[:depth, 0] * bids[:depth, 1]) / bid_volume if bid_volume > 0 else bids[0, 0]
    ask_wap = np.sum(asks[:depth, 0] * asks[:depth, 1]) / ask_volume if ask_volume > 0 else asks[0, 0]
    
    # Calculate spread
    spread = asks[0, 0] - bids[0, 0]
    
    return np.array([
        imbalance,
        bid_wap,
        ask_wap,
        spread
    ])

@njit
def compute_price_features(
    prices: np.ndarray,  # shape: (N,) for price series
    volumes: np.ndarray,  # shape: (N,) for volume series
    windows: np.ndarray  # shape: (K,) for different lookback periods
) -> np.ndarray:
    """Compute price-based features using Numba."""
    n_windows = len(windows)
    n_features = 4  # returns, volatility, volume_ratio, momentum
    features = np.zeros(n_windows * n_features)
    
    if len(prices) < 2:
        return features
        
    # Compute returns
    returns = np.diff(prices) / prices[:-1]
    
    for i, window in enumerate(windows):
        if window >= len(returns):
            continue
            
        window_returns = returns[-window:]
        window_volumes = volumes[-window:]
        
        # Calculate features for this window
        features[i*n_features + 0] = np.mean(window_returns)  # Average return
        features[i*n_features + 1] = np.std(window_returns)   # Volatility
        features[i*n_features + 2] = window_volumes[-1] / np.mean(window_volumes)  # Volume ratio
        features[i*n_features + 3] = np.sum(window_returns > 0) / len(window_returns)  # Up momentum
        
    return features

class HFTFeatureEngine:
    def __init__(self, lookback_periods=[10, 30, 60]):
        self.lookback_periods = lookback_periods
        self.price_history = {}
        self.volume_history = {}
        self.imbalance_history = {}
        
    def initialize_symbol(self, symbol):
        """Initialize data structures for a new symbol"""
        max_lookback = max(self.lookback_periods)
        self.price_history[symbol] = deque(maxlen=max_lookback)
        self.volume_history[symbol] = deque(maxlen=max_lookback)
        self.imbalance_history[symbol] = deque(maxlen=max_lookback)
        
    def update(self, symbol, price, volume, imbalance):
        """Update feature data with new market data"""
        if symbol not in self.price_history:
            self.initialize_symbol(symbol)
            
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.imbalance_history[symbol].append(imbalance)
        
    def compute_features(self, symbol):
        """Compute features for ML model"""
        if symbol not in self.price_history:
            return None
            
        features = []
        
        # Price momentum features
        for period in self.lookback_periods:
            if len(self.price_history[symbol]) >= period:
                price_array = list(self.price_history[symbol])[-period:]
                returns = np.diff(price_array) / price_array[:-1]
                features.extend([
                    np.mean(returns),
                    np.std(returns),
                    np.sum(returns > 0) / period  # Fraction of positive returns
                ])
            else:
                features.extend([0, 0, 0])
                
        # Volume features
        for period in self.lookback_periods:
            if len(self.volume_history[symbol]) >= period:
                volume_array = list(self.volume_history[symbol])[-period:]
                features.extend([
                    np.mean(volume_array),
                    np.std(volume_array),
                    volume_array[-1] / np.mean(volume_array)  # Relative volume
                ])
            else:
                features.extend([0, 0, 1])
                
        # Order book imbalance features
        for period in self.lookback_periods:
            if len(self.imbalance_history[symbol]) >= period:
                imbalance_array = list(self.imbalance_history[symbol])[-period:]
                features.extend([
                    np.mean(imbalance_array),
                    np.std(imbalance_array),
                    imbalance_array[-1]  # Current imbalance
                ])
            else:
                features.extend([0, 0, 0])
                
        return np.array(features).reshape(1, -1)

class MLPredictor:
    """Machine learning predictor for market movements."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.predictions = {}
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
    async def load_model(self):
        """Load the trained model and scaler."""
        try:
            # For now, we'll just initialize with dummy values
            # In production, load actual model from disk/cloud
            self.model = None
            self.scaler = None
            self.feature_columns = []
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    async def predict(self, symbol: str, data: Dict) -> float:
        """Generate predictions for the given market data."""
        try:
            # For now, return a random prediction between -1 and 1
            # In production, use actual model predictions
            prediction = np.random.uniform(-1, 1)
            self.predictions[symbol] = prediction
            return prediction
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return 0.0
            
    async def get_predictions(self) -> Dict:
        """Get current predictions for all symbols."""
        return self.predictions
        
    async def get_metrics(self) -> Dict:
        """Get model performance metrics."""
        return self.metrics
        
    async def update_metrics(self, actual_movements: Dict):
        """Update model performance metrics based on actual market movements."""
        try:
            # In production, calculate actual metrics
            # For now, use random values
            self.metrics = {
                'accuracy': np.random.uniform(0.5, 0.8),
                'precision': np.random.uniform(0.5, 0.8),
                'recall': np.random.uniform(0.5, 0.8),
                'f1_score': np.random.uniform(0.5, 0.8)
            }
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    async def retrain(self, training_data: Dict):
        """Retrain the model with new data."""
        try:
            # In production, implement actual retraining logic
            return True
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False

class HFTModel:
    def __init__(self, model_path=None):
        self.model = None
        self.feature_engine = HFTFeatureEngine()
        self.predictions = {}
        self.last_update = {}
        self.update_interval_ms = 100  # Minimum time between predictions
        
        if model_path:
            self.load_model(model_path)
            
        # Start prediction thread
        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            
    def train(self, X, y):
        """Train a new model"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=100,
                n_jobs=-1
            )
            self.model.fit(X, y)
            logger.info("Successfully trained new model")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
            
    def save_model(self, model_path):
        """Save trained model to file"""
        if self.model is None:
            logger.error("No model to save")
            return False
            
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Successfully saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
            
    def update_data(self, symbol, price, volume, imbalance):
        """Update market data for predictions"""
        self.feature_engine.update(symbol, price, volume, imbalance)
        
    def _prediction_loop(self):
        """Continuous prediction loop"""
        while self.running:
            try:
                self._update_predictions()
                time.sleep(0.01)  # 10ms sleep
            except Exception as e:
                logger.error(f"Error in prediction loop: {str(e)}")
                
    def _update_predictions(self):
        """Update predictions for all symbols"""
        if self.model is None:
            return
            
        now = time.time() * 1000  # Current time in milliseconds
        
        for symbol in self.feature_engine.price_history.keys():
            last_update = self.last_update.get(symbol, 0)
            
            # Check if enough time has passed since last update
            if now - last_update < self.update_interval_ms:
                continue
                
            # Compute features and make prediction
            features = self.feature_engine.compute_features(symbol)
            if features is not None:
                try:
                    pred = self.model.predict_proba(features)[0]
                    # Convert to directional signal: -1 (down), 0 (neutral), 1 (up)
                    if pred[1] > 0.6:  # Strong up signal
                        signal = 1
                    elif pred[1] < 0.4:  # Strong down signal
                        signal = -1
                    else:
                        signal = 0
                        
                    self.predictions[symbol] = signal
                    self.last_update[symbol] = now
                    
                except Exception as e:
                    logger.error(f"Error making prediction for {symbol}: {str(e)}")
                    
    def get_prediction(self, symbol):
        """Get latest prediction for a symbol"""
        return self.predictions.get(symbol, 0)
        
    def get_metrics(self):
        """Get model performance metrics"""
        return {
            'symbols_covered': len(self.predictions),
            'prediction_latency_ms': self.update_interval_ms,
            'model_type': type(self.model).__name__ if self.model else None
        }
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.running = False
        if hasattr(self, 'prediction_thread'):
            self.prediction_thread.join(timeout=1.0) 