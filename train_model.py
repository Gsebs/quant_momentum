import asyncio
import logging
import os
import pandas as pd
from price_feed import PriceFeed
from ml_model import ArbitragePredictor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, binance_api_key, binance_secret_key, symbols=['BTC/USDT', 'ETH/USDT']):
        self.price_feed = PriceFeed(
            binance_api_key=binance_api_key,
            binance_secret_key=binance_secret_key,
            symbols=symbols
        )
        self.data = []
        self.collection_start = None
    
    async def collect_training_data(self, duration_seconds=3600):  # 1 hour by default
        """Collect training data from both exchanges."""
        logger.info(f"Starting data collection for {duration_seconds} seconds...")
        self.collection_start = time.time()
        
        # Start price feed in background
        feed_task = asyncio.create_task(self.price_feed.start())
        
        try:
            while time.time() - self.collection_start < duration_seconds:
                # Log progress every minute
                elapsed = int(time.time() - self.collection_start)
                if elapsed % 60 == 0:
                    logger.info(f"Data collection in progress... {elapsed//60} minutes elapsed")
                
                # Collect data from price feed
                for symbol in self.price_feed.symbols:
                    if symbol in self.price_feed.price_differences:
                        self.data.extend(self.price_feed.price_differences[symbol])
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
        finally:
            # Stop price feed
            self.price_feed.running = False
            await feed_task
            await self.price_feed.close()
        
        return self.data

async def main():
    # Initialize data collector with your API keys
    collector = DataCollector(
        binance_api_key=os.getenv('BINANCE_API_KEY'),
        binance_secret_key=os.getenv('BINANCE_SECRET_KEY')
    )
    
    try:
        # Collect training data
        logger.info("Starting training data collection...")
        training_data = await collector.collect_training_data(duration_seconds=3600)  # 1 hour
        
        if not training_data:
            logger.error("No training data collected!")
            return
        
        # Save raw data
        df = pd.DataFrame(training_data)
        df.to_csv('training_data.csv', index=False)
        logger.info(f"Saved {len(df)} training samples to training_data.csv")
        
        # Initialize and train the ML model
        logger.info("Training ML model...")
        predictor = ArbitragePredictor()
        train_score, test_score = predictor.train(training_data)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        logger.info("\nFeature Importance:")
        for idx, row in importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        logger.info("\nTraining complete!")
        logger.info(f"Training Score: {train_score:.4f}")
        logger.info(f"Test Score: {test_score:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Training stopped by user") 