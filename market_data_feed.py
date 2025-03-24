import asyncio
import websockets
import json
import logging
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for latest prices
latest_prices: Dict[str, Optional[float]] = {
    "binance": None,
    "coinbase": None
}

# Define reliable crypto pairs
reliable_tickers = [
    "BTC/USDT",  # Bitcoin
    "ETH/USDT",  # Ethereum
    "BNB/USDT",  # Binance Coin
    "SOL/USDT",  # Solana
    "ADA/USDT"   # Cardano
]

# WebSocket URLs and subscription messages
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

def get_binance_subscribe_message(symbol: str) -> str:
    """Generate Binance WebSocket subscription message."""
    return json.dumps({
        "method": "SUBSCRIBE",
        "params": [f"{symbol.lower()}@trade"],
        "id": 1
    })

def get_coinbase_subscribe_message(symbol: str) -> str:
    """Generate Coinbase WebSocket subscription message."""
    return json.dumps({
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": [symbol.replace("/", "-")]}]
    })

async def connect_binance():
    """Connect to Binance WebSocket and maintain connection."""
    while True:
        try:
            async with websockets.connect(BINANCE_WS_URL) as ws:
                # Subscribe to all reliable tickers
                for symbol in reliable_tickers:
                    subscribe_msg = get_binance_subscribe_message(symbol)
                    await ws.send(subscribe_msg)
                    logger.info(f"Subscribed to Binance {symbol}")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        if 'e' in data and data['e'] == 'trade':
                            symbol = data['s']
                            price = float(data['p'])
                            latest_prices["binance"] = price
                            logger.debug(f"Binance {symbol}: {price}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding Binance message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Binance message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed. Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

async def connect_coinbase():
    """Connect to Coinbase WebSocket and maintain connection."""
    while True:
        try:
            async with websockets.connect(COINBASE_WS_URL) as ws:
                # Subscribe to all reliable tickers
                for symbol in reliable_tickers:
                    subscribe_msg = get_coinbase_subscribe_message(symbol)
                    await ws.send(subscribe_msg)
                    logger.info(f"Subscribed to Coinbase {symbol}")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "ticker" and "price" in data:
                            price = float(data["price"])
                            latest_prices["coinbase"] = price
                            logger.debug(f"Coinbase {data.get('product_id')}: {price}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding Coinbase message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Coinbase message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Coinbase WebSocket connection closed. Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Coinbase WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

async def update_market_data():
    """Continuously update market data for all reliable tickers."""
    while True:
        try:
            # Log current prices and spread
            if latest_prices["binance"] and latest_prices["coinbase"]:
                spread = latest_prices["coinbase"] - latest_prices["binance"]
                spread_pct = spread / latest_prices["binance"]
                logger.info(f"Spread: {spread_pct:.4%}")
            
            # Short sleep to avoid excessive logging
            await asyncio.sleep(0.1)  # 100ms between updates
            
        except Exception as e:
            logger.error(f"Error in market data update loop: {e}")
            await asyncio.sleep(1)  # Wait longer on error

async def run_feeds():
    """Initialize and start market data feeds."""
    try:
        # Start WebSocket connections and market data updates concurrently
        await asyncio.gather(
            connect_binance(),
            connect_coinbase(),
            update_market_data()
        )
    except Exception as e:
        logger.error(f"Error initializing feeds: {e}")
        raise

def get_latest_prices() -> Dict[str, Optional[float]]:
    """Get the latest prices from both exchanges."""
    return latest_prices.copy()  # Return a copy to prevent external modification 