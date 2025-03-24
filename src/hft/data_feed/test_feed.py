import asyncio
import logging
from market_data import MarketDataFeed, market_data
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def monitor_prices():
    """Monitor and print price updates from both exchanges."""
    start_time = asyncio.get_event_loop().time()
    test_duration = 30  # Run for 30 seconds
    
    while True:
        current_time = asyncio.get_event_loop().time()
        if current_time - start_time > test_duration:
            logger.info("Test duration completed")
            return
            
        binance_data = market_data["binance"]
        coinbase_data = market_data["coinbase"]
        
        if binance_data and coinbase_data and binance_data.get("price") and coinbase_data.get("price"):
            price_diff = abs(binance_data["price"] - coinbase_data["price"])
            price_diff_pct = price_diff / binance_data["price"] * 100
            
            logger.info(f"""
Market Data Update:
------------------
Binance:
  Price: ${binance_data["price"]:,.2f}
  Last Update: {datetime.fromtimestamp(binance_data["last_update"]/1000).strftime('%H:%M:%S.%f')}

Coinbase:
  Price: ${coinbase_data["price"]:,.2f}
  Last Update: {datetime.fromtimestamp(coinbase_data["last_update"]/1000).strftime('%H:%M:%S.%f')}

Price Difference: ${price_diff:,.2f} ({price_diff_pct:.3f}%)
            """)
        
        await asyncio.sleep(1)

async def main():
    # Create market data feed instance
    feed = MarketDataFeed("BTC-USD")
    
    try:
        # Start feed and price monitor as concurrent tasks
        feed_task = asyncio.create_task(feed.start())
        monitor_task = asyncio.create_task(monitor_prices())
        
        # Wait for the monitor to complete
        await monitor_task
        
        # Stop the feed
        await feed.stop()
        feed_task.cancel()
        
        logger.info("Market data feed stopped")
        
    except KeyboardInterrupt:
        await feed.stop()
        logger.info("Market data feed stopped")

if __name__ == "__main__":
    asyncio.run(main()) 