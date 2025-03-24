import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional
import websockets
import ccxt.async_support as ccxt
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceFeed:
    def __init__(self, binance_api_key: str, binance_secret: str, symbols: list = ['BTC/USDT']):
        self.symbols = symbols
        self.binance_prices: Dict[str, float] = {}
        self.coinbase_prices: Dict[str, float] = {}
        self.price_differences: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        self.last_update_times: Dict[str, Dict[str, float]] = {
            symbol: {'binance': 0, 'coinbase': 0} for symbol in symbols
        }
        
        # Initialize exchange clients
        self.binance = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'testnet': True}
        })
        
        # WebSocket URLs
        self.binance_ws_url = 'wss://testnet.binance.vision/ws'
        self.coinbase_ws_url = 'wss://ws-feed.exchange.coinbase.com'  # Updated Coinbase URL
        
        # Performance metrics
        self.latencies: Dict[str, deque] = {
            'binance': deque(maxlen=1000),
            'coinbase': deque(maxlen=1000)
        }
        
        # Connection state
        self.running = False
        
    async def start(self):
        """Start the price feed connections."""
        self.running = True
        logger.info("\n========== HFT Arbitrage System Starting ==========")
        logger.info(f"Monitoring pairs: {', '.join(self.symbols)}")
        logger.info("Connecting to exchanges...")
        logger.info("Press Ctrl+C to stop the system")
        logger.info("=============================================\n")
        try:
            # Start all tasks concurrently
            tasks = [
                asyncio.create_task(self.binance_websocket()),
                asyncio.create_task(self.coinbase_websocket()),
                asyncio.create_task(self.monitor_arbitrage_opportunities())
            ]
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in price feed: {e}")
            raise
        finally:
            self.running = False

    async def binance_websocket(self):
        """Maintain Binance WebSocket connection with automatic reconnection."""
        while self.running:
            try:
                # Format symbols for Binance stream
                streams = [f"{symbol.lower().replace('/', '')}@trade" 
                          for symbol in self.symbols]
                ws_url = f"{self.binance_ws_url}/{'@'.join(streams)}"
                
                async with websockets.connect(ws_url) as websocket:
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": streams,
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info("Connected to Binance WebSocket")
                    
                    while self.running:
                        try:
                            message = await websocket.recv()
                            await self.process_binance_message(json.loads(message))
                        except websockets.ConnectionClosed:
                            logger.warning("Binance WebSocket connection closed, reconnecting...")
                            break
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Binance WebSocket error: {e}")
                    await asyncio.sleep(1)  # Wait before reconnecting
                else:
                    break

    async def coinbase_websocket(self):
        """Maintain Coinbase WebSocket connection with automatic reconnection."""
        while self.running:
            try:
                async with websockets.connect(self.coinbase_ws_url) as websocket:
                    # Convert symbols to Coinbase format
                    products = [symbol.replace('/', '-') for symbol in self.symbols]
                    
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": products,
                        "channels": ["matches"]
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info("Connected to Coinbase WebSocket")
                    
                    while self.running:
                        try:
                            message = await websocket.recv()
                            await self.process_coinbase_message(json.loads(message))
                        except websockets.ConnectionClosed:
                            logger.warning("Coinbase WebSocket connection closed, reconnecting...")
                            break
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Coinbase WebSocket error: {e}")
                    await asyncio.sleep(1)  # Wait before reconnecting
                else:
                    break

    async def process_binance_message(self, message: dict):
        """Process incoming Binance trade messages."""
        try:
            if 'e' in message and message['e'] == 'trade':
                symbol = message['s']
                price = float(message['p'])
                timestamp = time.time_ns() // 1000  # Convert to microseconds
                
                # Calculate latency
                trade_time = message['T'] * 1000  # Convert to microseconds
                latency = timestamp - trade_time
                self.latencies['binance'].append(latency)
                
                # Update price and timestamp
                normalized_symbol = f"{symbol[:3]}/{symbol[3:]}"
                self.binance_prices[normalized_symbol] = price
                self.last_update_times[normalized_symbol]['binance'] = timestamp
                
                # Check for arbitrage opportunity
                await self.check_arbitrage(normalized_symbol)
                
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")

    async def process_coinbase_message(self, message: dict):
        """Process incoming Coinbase trade messages."""
        try:
            if message['type'] == 'match':
                symbol = message['product_id'].replace('-', '/')
                price = float(message['price'])
                timestamp = time.time_ns() // 1000  # Convert to microseconds
                
                # Calculate latency
                trade_time = int(datetime.fromisoformat(message['time'].replace('Z', '+00:00')).timestamp() * 1_000_000)
                latency = timestamp - trade_time
                self.latencies['coinbase'].append(latency)
                
                # Update price and timestamp
                self.coinbase_prices[symbol] = price
                self.last_update_times[symbol]['coinbase'] = timestamp
                
                # Check for arbitrage opportunity
                await self.check_arbitrage(symbol)
                
        except Exception as e:
            logger.error(f"Error processing Coinbase message: {e}")

    async def check_arbitrage(self, symbol: str):
        """Check for arbitrage opportunities between exchanges."""
        try:
            if symbol in self.binance_prices and symbol in self.coinbase_prices:
                price_diff = self.binance_prices[symbol] - self.coinbase_prices[symbol]
                self.price_differences[symbol].append({
                    'timestamp': time.time_ns() // 1000,
                    'binance_price': self.binance_prices[symbol],
                    'coinbase_price': self.coinbase_prices[symbol],
                    'difference': price_diff,
                    'binance_latency': np.mean(list(self.latencies['binance'])),
                    'coinbase_latency': np.mean(list(self.latencies['coinbase']))
                })
                
                # Log significant price differences (potential arbitrage opportunities)
                if abs(price_diff) > (self.binance_prices[symbol] * 0.001):  # 0.1% threshold
                    logger.info(f"Potential arbitrage opportunity for {symbol}:")
                    logger.info(f"Binance: ${self.binance_prices[symbol]:.2f}")
                    logger.info(f"Coinbase: ${self.coinbase_prices[symbol]:.2f}")
                    logger.info(f"Difference: ${price_diff:.2f}")
                    
        except Exception as e:
            logger.error(f"Error checking arbitrage: {e}")

    async def monitor_arbitrage_opportunities(self):
        """Monitor and analyze arbitrage opportunities."""
        stats_counter = 0
        while self.running:
            try:
                for symbol in self.symbols:
                    if len(self.price_differences[symbol]) > 0:
                        recent_diffs = [d['difference'] for d in self.price_differences[symbol]]
                        mean_diff = np.mean(recent_diffs)
                        std_diff = np.std(recent_diffs)
                        
                        # Calculate average latencies
                        avg_binance_latency = np.mean(list(self.latencies['binance']))
                        avg_coinbase_latency = np.mean(list(self.latencies['coinbase']))
                        
                        # Print statistics every 5 seconds
                        stats_counter += 1
                        if stats_counter >= 5:  # Changed from 10 to 5 seconds
                            logger.info(f"\n=== {symbol} HFT Stats ===")
                            logger.info(f"Current Prices:")
                            logger.info(f"  Binance: ${self.binance_prices.get(symbol, 0):.2f}")
                            logger.info(f"  Coinbase: ${self.coinbase_prices.get(symbol, 0):.2f}")
                            logger.info(f"Performance Metrics:")
                            logger.info(f"  Mean price difference: ${mean_diff:.2f}")
                            logger.info(f"  Price volatility: ${std_diff:.2f}")
                            logger.info(f"Latency Analysis:")
                            logger.info(f"  Binance: {avg_binance_latency/1000:.2f}ms")
                            logger.info(f"  Coinbase: {avg_coinbase_latency/1000:.2f}ms")
                            logger.info(f"Opportunities found: {len(self.price_differences[symbol])}")
                            logger.info("==================\n")
                            stats_counter = 0
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring arbitrage: {e}")
                await asyncio.sleep(1)

    def get_price_data(self) -> Dict:
        """Get current price data and statistics for ML model training."""
        return {
            'price_differences': self.price_differences,
            'latencies': self.latencies,
            'current_prices': {
                'binance': self.binance_prices,
                'coinbase': self.coinbase_prices
            }
        }

    async def close(self):
        """Close all connections gracefully."""
        logger.info("\n========== Shutting Down HFT System ==========")
        self.running = False
        await self.binance.close()
        logger.info("Disconnected from exchanges")
        logger.info("System shutdown complete")
        logger.info("==========================================\n") 