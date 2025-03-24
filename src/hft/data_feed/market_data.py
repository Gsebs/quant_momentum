import asyncio
import json
import logging
import ssl
from datetime import datetime
from typing import Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global market data store
market_data: Dict[str, Dict[str, Any]] = {
    "binance": {},
    "coinbase": {}
}

class MarketDataFeed:
    def __init__(self, symbol: str = "BTC-USD"):
        self.symbol = symbol
        # Convert BTC-USD to BTCUSDT for Binance.US
        self.binance_symbol = "btcusdt"  # Fixed symbol for Binance.US
        self.running = False
        logger.info(f"Initialized with symbol: {symbol} (Binance.US: {self.binance_symbol})")
        
    async def listen_binance(self):
        """Listen to Binance.US WebSocket for real-time trade data."""
        url = f"wss://stream.binance.us:9443/ws/{self.binance_symbol}@trade"
        logger.info(f"Attempting to connect to Binance.US WebSocket at: {url}")
        
        # Create SSL context for Binance.US
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        while self.running:
            try:
                async with websockets.connect(url, ssl=ssl_context) as ws:
                    logger.info(f"Connected to Binance.US WebSocket for {self.binance_symbol}")
                    async for msg in ws:
                        data = json.loads(msg)
                        logger.debug(f"Binance.US raw message: {data}")
                        
                        if "p" in data:  # Price field
                            price = float(data["p"])
                            timestamp = data["E"]  # Event time
                            
                            market_data["binance"] = {
                                "price": price,
                                "timestamp": timestamp,
                                "last_update": datetime.now().timestamp() * 1000
                            }
                            
                            logger.info(f"Binance.US {self.binance_symbol}: ${price:,.2f}")
                        else:
                            logger.debug(f"Received non-price message from Binance.US: {data}")
                        
            except ConnectionClosed as e:
                logger.warning(f"Binance.US WebSocket connection closed: {str(e)}")
                logger.info("Attempting to reconnect in 1 second...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in Binance.US feed: {str(e)}")
                logger.info("Attempting to reconnect in 1 second...")
                await asyncio.sleep(1)

    async def listen_coinbase(self):
        """Listen to Coinbase WebSocket for real-time trade data."""
        url = "wss://ws-feed.exchange.coinbase.com"
        logger.info(f"Attempting to connect to Coinbase WebSocket at: {url}")
        
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        subscribe_message = {
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": ["matches"]
        }
        
        while self.running:
            try:
                async with websockets.connect(url, ssl=ssl_context) as ws:
                    logger.info(f"Connected to Coinbase WebSocket for {self.symbol}")
                    await ws.send(json.dumps(subscribe_message))
                    logger.debug(f"Sent subscription message: {subscribe_message}")
                    
                    async for msg in ws:
                        data = json.loads(msg)
                        logger.debug(f"Coinbase raw message: {data}")
                        
                        if data.get("type") == "match":
                            price = float(data["price"])
                            timestamp = datetime.strptime(
                                data["time"], "%Y-%m-%dT%H:%M:%S.%fZ"
                            ).timestamp() * 1000
                            
                            market_data["coinbase"] = {
                                "price": price,
                                "timestamp": timestamp,
                                "last_update": datetime.now().timestamp() * 1000
                            }
                            
                            logger.info(f"Coinbase {self.symbol}: ${price:,.2f}")
                        else:
                            logger.debug(f"Received non-match message from Coinbase: {data}")
                            
            except ConnectionClosed as e:
                logger.warning(f"Coinbase WebSocket connection closed: {str(e)}")
                logger.info("Attempting to reconnect in 1 second...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in Coinbase feed: {str(e)}")
                logger.info("Attempting to reconnect in 1 second...")
                await asyncio.sleep(1)

    async def start(self):
        """Start both WebSocket feeds."""
        self.running = True
        await asyncio.gather(
            self.listen_binance(),
            self.listen_coinbase()
        )

    async def stop(self):
        """Stop both WebSocket feeds."""
        self.running = False
        logger.info("Stopping market data feed")

# Example usage
async def main():
    feed = MarketDataFeed("BTC-USD")
    try:
        await feed.start()
    except KeyboardInterrupt:
        await feed.stop()

if __name__ == "__main__":
    asyncio.run(main()) 