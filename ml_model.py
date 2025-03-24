import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import logging
import joblib
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitragePredictor:
    def __init__(self, model_path='models'):
        """Initialize the ML model for predicting profitable arbitrage opportunities."""
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.feature_columns = [
            'price_diff',           # Current price difference between exchanges
            'binance_price',        # Current Binance price
            'coinbase_price',       # Current Coinbase price
            'binance_volume',       # Recent trading volume on Binance
            'coinbase_volume',      # Recent trading volume on Coinbase
            'price_volatility',     # Recent price volatility
            'spread',              # Current bid-ask spread
            'binance_latency',     # Current latency to Binance
            'coinbase_latency',    # Current latency to Coinbase
            'time_of_day',         # Hour of day (0-23)
            'day_of_week',         # Day of week (0-6)
        ]
        
        # Create models directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    
    def prepare_features(self, data):
        """Prepare features for the ML model."""
        df = pd.DataFrame(data)
        
        # Add time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        df['time_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Calculate volatility
        df['price_volatility'] = df['price_diff'].rolling(window=10).std()
        
        # Fill missing values
        df = df.fillna(method='ffill')
        
        return df[self.feature_columns]
    
    def prepare_labels(self, data):
        """Prepare labels (profitable opportunities) for training."""
        df = pd.DataFrame(data)
        
        # Define profitable trades (considering fees and slippage)
        threshold = 0.001  # 0.1% minimum profit after fees
        df['profitable'] = (abs(df['price_diff']) > threshold).astype(float)
        
        # Calculate actual profit potential
        df['profit_potential'] = abs(df['price_diff']) - threshold
        
        return df['profit_potential']
    
    def train(self, training_data):
        """Train the ML model on historical arbitrage data."""
        logger.info("Preparing training data...")
        
        # Prepare features and labels
        X = self.prepare_features(training_data)
        y = self.prepare_labels(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        logger.info("Training model...")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model Training Score: {train_score:.4f}")
        logger.info(f"Model Test Score: {test_score:.4f}")
        
        # Save model
        self.save_model()
        
        return train_score, test_score
    
    def predict(self, market_data):
        """Predict profit potential for new market data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        X = self.prepare_features(market_data)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def save_model(self):
        """Save the trained model and scaler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(self.model_path, f'arbitrage_model_{timestamp}.joblib')
        scaler_file = os.path.join(self.model_path, f'scaler_{timestamp}.joblib')
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_timestamp):
        """Load a previously trained model and scaler."""
        model_file = os.path.join(self.model_path, f'arbitrage_model_{model_timestamp}.joblib')
        scaler_file = os.path.join(self.model_path, f'scaler_{model_timestamp}.joblib')
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        logger.info(f"Model loaded from {model_file}")
    
    def get_feature_importance(self):
        """Get the importance of each feature in making predictions."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False) 