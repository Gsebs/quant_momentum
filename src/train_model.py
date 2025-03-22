"""
Script to train initial HFT model with simulated data.
"""

import numpy as np
import logging
from ml_model import HFTModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data(n_samples=10000):
    """Generate simulated training data"""
    # Generate random features
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate random features in reasonable ranges
        bid_ask_spread = np.random.uniform(0.0001, 0.01)  # 0.01% to 1%
        volume_imbalance = np.random.uniform(-1, 1)  # -100% to 100%
        price_trend = np.random.uniform(-0.01, 0.01)  # -1% to 1%
        volatility = np.random.uniform(0, 0.02)  # 0% to 2%
        order_book_imbalance = np.random.uniform(-1, 1)  # -100% to 100%
        
        features.append([
            bid_ask_spread,
            volume_imbalance,
            price_trend,
            volatility,
            order_book_imbalance
        ])
        
        # Generate label based on features
        # Simple rule: if volume imbalance and order book imbalance agree
        # and price trend is in the same direction, it's a strong signal
        signal = (
            np.sign(volume_imbalance) == np.sign(order_book_imbalance) and
            np.sign(price_trend) == np.sign(volume_imbalance)
        )
        labels.append(1 if signal else 0)
    
    return np.array(features), np.array(labels)

def main():
    try:
        logger.info("Generating training data...")
        features, labels = generate_training_data()
        
        logger.info("Creating and training model...")
        model = HFTModel()
        model.add_training_sample(features, labels)
        
        if model.train():
            logger.info("Model trained successfully")
            model.save('models/hft_model.pkl')
            logger.info("Model saved to models/hft_model.pkl")
            
            # Print feature importance
            importance = model.get_feature_importance()
            logger.info("Feature importance:")
            for feature, score in importance.items():
                logger.info(f"{feature}: {score:.4f}")
        else:
            logger.error("Failed to train model")
            
    except Exception as e:
        logger.error(f"Error in training script: {str(e)}")

if __name__ == "__main__":
    main() 