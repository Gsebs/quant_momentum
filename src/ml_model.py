"""
Machine learning model for enhancing momentum signals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Optional, Tuple
import joblib
import os
import yfinance as yf
from joblib import load

logger = logging.getLogger(__name__)

class MomentumMLModel:
    def __init__(self, model_path: str = "models/momentum_model.joblib"):
        """Initialize the ML model for momentum prediction."""
        self.model_path = model_path
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Could not load existing model: {str(e)}")
    
    def _calculate_technical_indicators(self, data):
        """Calculate technical indicators for each ticker."""
        indicators = pd.DataFrame()
        
        # Get unique tickers excluding empty strings
        tickers = [t for t in data.columns.get_level_values('Ticker').unique() if t]
        
        for ticker in tickers:
            # Get price data for the current ticker
            price_data = pd.DataFrame()
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                try:
                    price_data[col] = data[col, ticker]
                except KeyError:
                    continue
                
            if price_data.empty:
                continue
            
            # Calculate indicators for this ticker
            ticker_indicators = {}
            
            # Returns and volatility
            price_data['returns'] = price_data['Close'].pct_change()
            price_data['volatility'] = price_data['returns'].rolling(window=20).std()
            
            # Volume indicators
            price_data['volume_sma'] = price_data['Volume'].rolling(window=20).mean()
            price_data['volume_ratio'] = price_data['Volume'] / price_data['volume_sma']
            
            # RSI
            delta = price_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            price_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
            price_data['macd'] = exp1 - exp2
            price_data['signal'] = price_data['macd'].ewm(span=9, adjust=False).mean()
            price_data['histogram'] = price_data['macd'] - price_data['signal']
            
            # Bollinger Bands
            sma = price_data['Close'].rolling(window=20).mean()
            std = price_data['Close'].rolling(window=20).std()
            price_data['upper'] = sma + (std * 2)
            price_data['middle'] = sma
            price_data['lower'] = sma - (std * 2)
            
            # Channel Position
            channel_width = price_data['upper'] - price_data['lower']
            close_minus_lower = price_data['Close'] - price_data['lower']
            price_data['channel_position'] = close_minus_lower / channel_width
            
            # Rate of Change
            for period in [5, 10, 20]:
                price_data[f'roc_{period}'] = price_data['Close'].pct_change(periods=period)
            
            # Moving Averages
            for period in [50, 200]:
                price_data[f'sma_{period}'] = price_data['Close'].rolling(window=period).mean()
            
            # Trend Strength
            price_data['trend_strength'] = abs(price_data['Close'].pct_change(20))
            
            # Momentum Signal (improved version)
            price_data['momentum_signal'] = (
                0.3 * price_data['rsi'] +  # RSI weight
                0.3 * price_data['channel_position'] +  # Channel position weight
                0.2 * (price_data['Close'] > price_data['sma_50']).astype(float) +  # Above 50-day MA
                0.2 * (price_data['Close'] > price_data['sma_200']).astype(float)  # Above 200-day MA
            )
            
            # Volatility Adjusted Returns
            price_data['volatility_adjusted_returns'] = price_data['returns'] / price_data['volatility']
            
            # Add all indicators to the main DataFrame with MultiIndex
            for col in price_data.columns:
                indicators[(col, ticker)] = price_data[col]
        
        # Set the index name
        indicators.columns.names = ['Price', 'Ticker']
        
        return indicators.fillna(method='ffill').fillna(0)  # Forward fill then fill remaining NaNs with 0

    def _prepare_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for training or prediction."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Create a copy to avoid modifying the original data
        X = X.copy()

        # Ensure all required columns exist
        required_columns = [
            'returns', 'volatility', 'volume_ratio', 'rsi', 'macd', 'histogram',
            'roc_5', 'roc_10', 'roc_20', 'trend_strength', 'momentum_signal',
            'volatility_adjusted_returns'
        ]
        
        for col in required_columns:
            if col not in X.columns:
                logger.warning(f"Missing column {col}, adding with zeros")
                X[col] = 0.0

        # Convert all columns to float
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill any NaN values
        X = X.ffill().bfill().fillna(0)

        # Winsorize extreme values
        for col in X.columns:
            lower_bound = X[col].quantile(0.01)
            upper_bound = X[col].quantile(0.99)
            X[col] = np.clip(X[col], lower_bound, upper_bound)

        # Normalize all features to [0, 1] range
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        return X, y

    def train(self, X, y):
        """Train the model on the given data."""
        try:
            # Reset index to avoid duplicate index issues
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Train both models
            self.rf_model.fit(X, y)
            self.gb_model.fit(X, y)
            
            # Save the trained models
            self._save_model()
            
            logging.info("Model training completed successfully")
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions for the input data."""
        try:
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Model not trained - no feature columns available")
                
            # Prepare features
            X, _ = self._prepare_features(data)
            
            if len(X) == 0:
                raise ValueError("No valid features for prediction")
            
            # Ensure we use the same features as training
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                logger.warning(f"Missing columns in prediction data: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0.0
            
            X = X[self.feature_columns]
            
            # Generate predictions from both models
            rf_pred = self.rf_model.predict(X)
            gb_pred = self.gb_model.predict(X)
            
            # Ensemble predictions (simple average)
            predictions = (rf_pred + gb_pred) / 2
            
            # Convert to series with proper index
            predictions = pd.Series(predictions, index=X.index)
            
            # Clip predictions to reasonable range
            predictions = np.clip(predictions, -0.5, 0.5)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return pd.Series()
    
    def _save_model(self) -> bool:
        """Save the trained model to disk."""
        try:
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'feature_columns': self.feature_columns
            }
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _load_model(self) -> bool:
        """Load a trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.rf_model = model_data['rf_model']
                self.gb_model = model_data['gb_model']
                self.feature_columns = model_data['feature_columns']
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning("Model file does not exist")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

def enhance_signals(momentum_data: pd.DataFrame) -> pd.DataFrame:
    """Enhance momentum signals with ML predictions."""
    try:
        # Create a copy to avoid modifying the original
        enhanced_data = momentum_data.copy()
        
        # Calculate ranks for technical indicators
        enhanced_data['rsi_rank'] = enhanced_data['rsi'].rank(pct=True)
        enhanced_data['macd_rank'] = enhanced_data['macd'].rank(pct=True)
        enhanced_data['volatility_rank'] = enhanced_data['volatility'].rank(pct=True)
        
        # Try to load the ML model
        model_path = 'models/momentum_model.joblib'
        if os.path.exists(model_path):
            try:
                model = load(model_path)
                logger.info(f"Model loaded from {model_path}")
                
                # Prepare features for prediction
                features = enhanced_data[[
                    'rsi_rank', 'macd_rank', 'volatility_rank',
                    'volume_ratio', 'trend_strength'
                ]].fillna(0)
                
                # Make predictions
                predictions = model.predict_proba(features)[:, 1]
                enhanced_data['ml_score'] = predictions * 100
                
            except Exception as e:
                logger.warning(f"Error loading/using ML model: {str(e)}")
                enhanced_data['ml_score'] = 0.0
        else:
            logger.warning("No ML model found")
            enhanced_data['ml_score'] = 0.0
        
        # Calculate enhanced score
        enhanced_data['enhanced_score'] = enhanced_data['composite_score']
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Error in enhance_signals: {str(e)}")
        return momentum_data 