import asyncio
import logging
import os
from price_feed import PriceFeed
from ml_model import ArbitragePredictor
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFTSystem:
    def __init__(self, binance_api_key, binance_secret_key, symbols=['BTC/USDT', 'ETH/USDT']):
        self.price_feed = PriceFeed(
            binance_api_key=binance_api_key,
            binance_secret_key=binance_secret_key,
            symbols=symbols
        )
        self.predictor = ArbitragePredictor()
        self.running = False
        self.trade_history = []
        
        # Load the latest trained model
        self.predictor.load_latest_model()
    
    async def process_opportunities(self):
        """Process potential arbitrage opportunities using the ML model."""
        while self.running:
            try:
                for symbol in self.price_feed.symbols:
                    if symbol in self.price_feed.price_differences:
                        # Get latest price differences
                        opportunities = self.price_feed.price_differences[symbol]
                        if not opportunities:
                            continue
                        
                        # Convert to DataFrame for prediction
                        df = pd.DataFrame(opportunities)
                        
                        # Make predictions
                        predictions = self.predictor.predict(df)
                        
                        # Log high probability opportunities
                        for idx, prob in enumerate(predictions):
                            if prob > 0.8:  # High confidence threshold
                                opp = opportunities[idx]
                                logger.info(f"High probability opportunity detected!")
                                logger.info(f"Symbol: {symbol}")
                                logger.info(f"Price difference: {opp['price_diff']}")
                                logger.info(f"Probability: {prob:.4f}")
                                
                                # Record opportunity
                                self.trade_history.append({
                                    'timestamp': time.time(),
                                    'symbol': symbol,
                                    'price_diff': opp['price_diff'],
                                    'probability': prob,
                                    'binance_price': opp['binance_price'],
                                    'coinbase_price': opp['coinbase_price']
                                })
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                logger.error(f"Error processing opportunities: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    def save_trade_history(self):
        """Save trade history to CSV file."""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            filename = f"trade_history_{int(time.time())}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Trade history saved to {filename}")
    
    async def run(self, duration_seconds=None):
        """Run the HFT system."""
        try:
            self.running = True
            
            # Start price feed and opportunity processor
            feed_task = asyncio.create_task(self.price_feed.start())
            processor_task = asyncio.create_task(self.process_opportunities())
            
            start_time = time.time()
            logger.info("HFT system started...")
            
            while self.running:
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Duration {duration_seconds}s reached. Stopping...")
                    break
                
                # Log status every minute
                elapsed = int(time.time() - start_time)
                if elapsed % 60 == 0:
                    logger.info(f"System running for {elapsed//60} minutes...")
                
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in HFT system: {e}")
        finally:
            self.running = False
            self.price_feed.running = False
            
            # Wait for tasks to complete
            await feed_task
            await processor_task
            await self.price_feed.close()
            
            # Save trade history
            self.save_trade_history()
            logger.info("HFT system stopped.")

async def main():
    # Initialize HFT system with your API keys
    system = HFTSystem(
        binance_api_key=os.getenv('BINANCE_API_KEY'),
        binance_secret_key=os.getenv('BINANCE_SECRET_KEY')
    )
    
    try:
        # Run system for 1 hour
        await system.run(duration_seconds=3600)
    except KeyboardInterrupt:
        logger.info("Stopping HFT system...")
        system.running = False

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System stopped by user") 