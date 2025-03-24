import os
import asyncio
import logging
from datetime import datetime
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_binance_connection():
    try:
        # Initialize Binance client with testnet
        binance = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': True  # Enable testnet
            }
        })
        
        logger.info("Testing Binance testnet connection...")
        
        # Test market data fetching
        markets = await binance.load_markets()
        logger.info(f"Successfully loaded {len(markets)} markets")
        
        # Test specific symbol price
        symbol = 'BTC/USDT'
        ticker = await binance.fetch_ticker(symbol)
        logger.info(f"{symbol} Price: ${ticker['last']:.2f}")
        logger.info(f"24h Volume: {ticker['quoteVolume']:.2f} USDT")
        
        # Test account information (if API key has permissions)
        try:
            balance = await binance.fetch_balance()
            logger.info("Successfully fetched account balance")
            logger.info(f"Free USDT: {balance.get('USDT', {}).get('free', 0):.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch balance (this is normal for read-only API keys): {e}")
        
        logger.info("Binance testnet connection test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing Binance connection: {e}")
        raise
    finally:
        if 'binance' in locals():
            await binance.close()

async def main():
    await test_binance_connection()

if __name__ == "__main__":
    asyncio.run(main()) 