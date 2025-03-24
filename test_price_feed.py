import asyncio
import os
import logging
import signal
import time
from price_feed import PriceFeed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for the price feed
price_feed = None

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, closing connections...")
    if price_feed:
        # Schedule the close coroutine
        loop = asyncio.get_event_loop()
        loop.create_task(price_feed.close())

async def main():
    global price_feed
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Initialize price feed with your testnet credentials
    price_feed = PriceFeed(
        binance_api_key=os.getenv('BINANCE_API_KEY'),
        binance_secret=os.getenv('BINANCE_SECRET_KEY'),
        symbols=['BTC/USDT', 'ETH/USDT']  # Monitor multiple pairs
    )
    
    try:
        logger.info("\n=== Starting HFT System Test (30 second run) ===")
        logger.info("1. Connecting to exchanges...")
        logger.info("2. Monitoring price differences...")
        logger.info("3. Analyzing arbitrage opportunities...")
        logger.info("4. Tracking latencies...")
        logger.info("============================================\n")
        
        # Start price feed in background task
        feed_task = asyncio.create_task(price_feed.start())
        
        # Run for 30 seconds
        start_time = time.time()
        test_duration = 30  # Run for 30 seconds
        
        while time.time() - start_time < test_duration:
            remaining = int(test_duration - (time.time() - start_time))
            if remaining % 5 == 0:  # Show countdown every 5 seconds
                logger.info(f"Test running... {remaining} seconds remaining")
            await asyncio.sleep(1)
        
        # Stop the price feed
        price_feed.running = False
        await feed_task
            
        logger.info("\n=== Test Complete ===")
        logger.info("Summary of findings:")
        for symbol in price_feed.symbols:
            if symbol in price_feed.price_differences and len(price_feed.price_differences[symbol]) > 0:
                diffs = [d['difference'] for d in price_feed.price_differences[symbol]]
                logger.info(f"\nPair: {symbol}")
                logger.info(f"Total opportunities detected: {len(diffs)}")
                logger.info(f"Average price difference: ${sum(diffs)/len(diffs):.2f}")
                logger.info(f"Maximum price difference: ${max(abs(d) for d in diffs):.2f}")
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        if price_feed:
            await price_feed.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test stopped by user") 