"""
Machine Learning Manager for HFT strategies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import xgboost as xgb
from dataclasses import dataclass
import threading
import queue
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    """Container for market features used in ML predictions."""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_trade_price: float
    last_trade_size: float
    imbalance: float
    volatility: float
    spread: float
    mid_price_change: float
    volume_weighted_price: float
    trade_flow_imbalance: float
    
class MLManager:
    """Manages machine learning models for HFT strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_queue = queue.Queue(maxsize=10000)
        self.training_data = {
            'features': [],
            'labels': []
        }
        
        # Feature history for rolling calculations
        self.feature_history = {
            'price_changes': [],
            'volumes': [],
            'imbalances': [],
            'spreads': []
        }
        
        # Performance metrics
        self.prediction_latencies = []
        self.prediction_accuracy = []
        
        # Initialize models
        self._initialize_models()
        
        # Start feature collection thread
        self.is_running = True
        threading.Thread(target=self._collect_features, daemon=True).start()
        
    def _initialize_models(self):
        """Initialize ML models for different prediction tasks."""
        # Price direction prediction model (Random Forest)
        self.models['price_direction'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            n_jobs=-1
        )
        
        # Order flow prediction model (XGBoost)
        self.models['order_flow'] = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            tree_method='hist',  # For faster training
            predictor='cpu_predictor'  # Ensure CPU prediction for low latency
        )
        
        # Volatility prediction model (Gradient Boosting)
        self.models['volatility'] = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        # Initialize scalers for each model
        for model_name in self.models:
            self.scalers[model_name] = StandardScaler()
            
    def compute_features(self, order_book: Dict, trades: List[Dict]) -> MarketFeatures:
        """Compute features from current market state."""
        try:
            # Basic price and size features
            bid_price = order_book['bids'][0].price if order_book['bids'] else 0
            ask_price = order_book['asks'][0].price if order_book['asks'] else 0
            bid_size = order_book['bids'][0].size if order_book['bids'] else 0
            ask_size = order_book['asks'][0].size if order_book['asks'] else 0
            
            # Calculate derived features
            mid_price = (bid_price + ask_price) / 2 if bid_price and ask_price else 0
            spread = ask_price - bid_price if bid_price and ask_price else 0
            
            # Order book imbalance
            total_bid_size = sum(level.size for level in order_book['bids'][:5])
            total_ask_size = sum(level.size for level in order_book['asks'][:5])
            imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size) \
                       if (total_bid_size + total_ask_size) > 0 else 0
                       
            # Recent trade features
            recent_trades = [t for t in trades if (datetime.now() - t['timestamp']).total_seconds() <= 1]
            last_trade = recent_trades[-1] if recent_trades else {'price': mid_price, 'size': 0}
            
            # Calculate trade flow imbalance
            buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'buy')
            sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'sell')
            trade_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) \
                                 if (buy_volume + sell_volume) > 0 else 0
                                 
            # Volatility estimation
            price_changes = self.feature_history['price_changes']
            volatility = np.std(price_changes) if len(price_changes) > 1 else 0
            
            # Volume-weighted average price (VWAP)
            if recent_trades:
                vwap = sum(t['price'] * t['size'] for t in recent_trades) / sum(t['size'] for t in recent_trades)
            else:
                vwap = mid_price
                
            # Mid price change
            if len(self.feature_history['price_changes']) > 0:
                mid_price_change = self.feature_history['price_changes'][-1]
            else:
                mid_price_change = 0
                
            # Create feature object
            features = MarketFeatures(
                timestamp=datetime.now(),
                symbol=order_book['symbol'],
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                last_trade_price=last_trade['price'],
                last_trade_size=last_trade['size'],
                imbalance=imbalance,
                volatility=volatility,
                spread=spread,
                mid_price_change=mid_price_change,
                volume_weighted_price=vwap,
                trade_flow_imbalance=trade_flow_imbalance
            )
            
            # Update feature history
            self._update_feature_history(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {str(e)}")
            return None
            
    def _update_feature_history(self, features: MarketFeatures):
        """Update rolling feature history."""
        # Keep last 100 observations for each feature
        max_history = 100
        
        self.feature_history['price_changes'].append(features.mid_price_change)
        self.feature_history['volumes'].append(features.last_trade_size)
        self.feature_history['imbalances'].append(features.imbalance)
        self.feature_history['spreads'].append(features.spread)
        
        # Trim histories
        for key in self.feature_history:
            if len(self.feature_history[key]) > max_history:
                self.feature_history[key] = self.feature_history[key][-max_history:]
                
    def _collect_features(self):
        """Background thread to collect and store features for training."""
        while self.is_running:
            try:
                features = self.feature_queue.get(timeout=1)
                
                # Convert features to array format
                feature_array = self._features_to_array(features)
                
                # Store features for training
                self.training_data['features'].append(feature_array)
                
                # Generate label (simplified example - you would want to look ahead in real implementation)
                if len(self.training_data['features']) > 1:
                    prev_mid = (self.training_data['features'][-2][2] + self.training_data['features'][-2][3]) / 2
                    curr_mid = (feature_array[2] + feature_array[3]) / 2
                    label = 1 if curr_mid > prev_mid else -1 if curr_mid < prev_mid else 0
                    self.training_data['labels'].append(label)
                    
                # Limit training data size
                max_training_samples = 100000
                if len(self.training_data['features']) > max_training_samples:
                    self.training_data['features'] = self.training_data['features'][-max_training_samples:]
                    self.training_data['labels'] = self.training_data['labels'][-max_training_samples:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in feature collection: {str(e)}")
                time.sleep(1)
                
    def _features_to_array(self, features: MarketFeatures) -> np.ndarray:
        """Convert feature object to numpy array."""
        return np.array([
            features.bid_price,
            features.ask_price,
            features.bid_size,
            features.ask_size,
            features.last_trade_price,
            features.last_trade_size,
            features.imbalance,
            features.volatility,
            features.spread,
            features.mid_price_change,
            features.volume_weighted_price,
            features.trade_flow_imbalance
        ])
        
    def train_models(self):
        """Train all models with collected data."""
        try:
            if len(self.training_data['features']) < 1000:
                logger.warning("Insufficient training data")
                return False
                
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])
            
            # Split into training and validation sets
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_name, model in self.models.items():
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_val_scaled = self.scalers[model_name].transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                accuracy = model.score(X_val_scaled, y_val)
                logger.info(f"Model {model_name} validation accuracy: {accuracy:.4f}")
                
            # Save models
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
            
    def predict(self, features: MarketFeatures, model_name: str) -> Tuple[int, float]:
        """Get prediction from specified model."""
        try:
            start_time = time.time()
            
            # Convert features to array
            feature_array = self._features_to_array(features)
            
            # Scale features
            feature_array_scaled = self.scalers[model_name].transform(feature_array.reshape(1, -1))
            
            # Get prediction
            prediction = self.models[model_name].predict(feature_array_scaled)[0]
            probability = np.max(self.models[model_name].predict_proba(feature_array_scaled)[0])
            
            # Record latency
            latency = (time.time() - start_time) * 1000  # ms
            self.prediction_latencies.append(latency)
            
            return prediction, probability
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0, 0.0
            
    def _save_models(self):
        """Save trained models and scalers."""
        try:
            for model_name, model in self.models.items():
                joblib.dump(model, f"models/{model_name}_model.pkl")
                joblib.dump(self.scalers[model_name], f"models/{model_name}_scaler.pkl")
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            
    def load_models(self):
        """Load trained models and scalers."""
        try:
            for model_name in self.models:
                self.models[model_name] = joblib.load(f"models/{model_name}_model.pkl")
                self.scalers[model_name] = joblib.load(f"models/{model_name}_scaler.pkl")
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
            
    def get_metrics(self) -> Dict:
        """Get performance metrics for ML models."""
        return {
            'avg_prediction_latency': np.mean(self.prediction_latencies[-100:]) if self.prediction_latencies else 0,
            'prediction_accuracy': np.mean(self.prediction_accuracy[-100:]) if self.prediction_accuracy else 0,
            'training_samples': len(self.training_data['features']),
            'model_metrics': {
                name: {
                    'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
                } for name, model in self.models.items()
            }
        } 