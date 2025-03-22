"""
Binance WebSocket Test Client for BTC/USDT HFT Latency Arbitrage
"""

import websockets
import json
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, List
import time
import statistics
from collections import deque

# Configure logging with microsecond precision
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """Client for testing Binance WebSocket connection and data validation for HFT."""
    
    def __init__(self, max_latency_ms: float = 50.0):
        self.ws_url = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
        self.running = False
        self.last_update = None
        self.message_count = 0
        self.error_count = 0
        self.max_latency_ms = max_latency_ms
        self.latencies = deque(maxlen=1000)  # Store last 1000 latency measurements
        self.price_history = deque(maxlen=1000)  # Store last 1000 prices
        
    def calculate_statistics(self) -> Dict:
        """Calculate statistical measures for latency and price."""
        if not self.latencies:
            return {}
            
        latency_stats = {
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies),
            'avg_latency_ms': statistics.mean(self.latencies),
            'median_latency_ms': statistics.median(self.latencies),
            'stdev_latency_ms': statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
        }
        
        if self.price_history:
            price_stats = {
                'min_price': min(self.price_history),
                'max_price': max(self.price_history),
                'avg_price': statistics.mean(self.price_history),
                'price_volatility': statistics.stdev(self.price_history) if len(self.price_history) > 1 else 0
            }
            latency_stats.update(price_stats)
            
        return latency_stats
        
    def validate_message(self, message: Dict) -> bool:
        """
        Validate that the message contains all required fields with correct types.
        
        Args:
            message: Dictionary containing the parsed WebSocket message
            
        Returns:
            bool: True if message is valid, False otherwise
        """
        required_fields = {
            'e': ('Event type', str, lambda x: x == 'aggTrade'),
            'E': ('Event time', int, lambda x: x > 0),
            's': ('Symbol', str, lambda x: x == 'BTCUSDT'),
            'p': ('Price', str, lambda x: float(x) > 0),
            'q': ('Quantity', str, lambda x: float(x) > 0),
            'T': ('Trade time', int, lambda x: x > 0),
            'm': ('Is market maker', bool, None),
            'M': ('Ignore', bool, None)  # Additional field for market maker trade
        }
        
        # Check message structure
        if not isinstance(message, dict):
            logger.error(f"Invalid message format: expected dict, got {type(message)}")
            return False
            
        for field, (name, expected_type, validator) in required_fields.items():
            # Check if field exists
            if field not in message:
                logger.error(f"Missing required field: {name} ({field})")
                return False
                
            # Check field type
            if not isinstance(message[field], expected_type):
                logger.error(f"Invalid type for {name} ({field}): "
                           f"expected {expected_type}, got {type(message[field])}")
                return False
                
            # Run custom validator if provided
            if validator and not validator(message[field]):
                logger.error(f"Validation failed for {name} ({field}): {message[field]}")
                return False
                
        return True
        
    def process_message(self, message: Dict) -> Optional[Dict]:
        """
        Process and format the validated message with high-precision timestamps.
        
        Args:
            message: Dictionary containing the validated WebSocket message
            
        Returns:
            Dict: Processed message with converted values and latency metrics
        """
        try:
            # Get local timestamp as early as possible
            local_time = time.time()
            local_datetime = datetime.now()
            
            # Convert price and quantity with high precision
            price = Decimal(message['p'])
            quantity = Decimal(message['q'])
            
            # Calculate various latency metrics
            exchange_time = message['T'] / 1000  # Convert to seconds
            exchange_processing_time = message['E'] / 1000 - exchange_time
            network_latency = local_time - message['E'] / 1000
            total_latency = local_time - exchange_time
            
            processed_data = {
                'event_type': message['e'],
                'event_time': datetime.fromtimestamp(message['E'] / 1000),
                'symbol': message['s'],
                'price': price,
                'quantity': quantity,
                'trade_time': datetime.fromtimestamp(message['T'] / 1000),
                'is_market_maker': message['m'],
                'local_time': local_datetime,
                'latency_ms': total_latency * 1000,
                'exchange_processing_ms': exchange_processing_time * 1000,
                'network_latency_ms': network_latency * 1000
            }
            
            # Update statistics
            self.latencies.append(total_latency * 1000)
            self.price_history.append(float(price))
            
            # Check if latency exceeds threshold
            if total_latency * 1000 > self.max_latency_ms:
                logger.warning(
                    f"High latency detected: {total_latency * 1000:.2f}ms "
                    f"(threshold: {self.max_latency_ms}ms)"
                )
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
            
    async def connect_and_validate(self, num_messages: int = 5):
        """
        Connect to Binance WebSocket and validate a specified number of messages.
        
        Args:
            num_messages: Number of messages to validate before disconnecting
        """
        self.running = True
        self.message_count = 0
        self.error_count = 0
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info(f"Connected to Binance WebSocket: {self.ws_url}")
                connection_time = time.time()
                
                while self.running and self.message_count < num_messages:
                    try:
                        # Receive message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        
                        # Parse JSON
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse message: {e}")
                            self.error_count += 1
                            continue
                            
                        # Validate message
                        if not self.validate_message(data):
                            self.error_count += 1
                            continue
                            
                        # Process message
                        processed_data = self.process_message(data)
                        if processed_data:
                            self.message_count += 1
                            self.last_update = processed_data['local_time']
                            
                            # Log processed data
                            logger.info(
                                f"Message {self.message_count}/{num_messages} validated:\n"
                                f"Price: {processed_data['price']}\n"
                                f"Quantity: {processed_data['quantity']}\n"
                                f"Total Latency: {processed_data['latency_ms']:.3f}ms\n"
                                f"Network Latency: {processed_data['network_latency_ms']:.3f}ms\n"
                                f"Exchange Processing: {processed_data['exchange_processing_ms']:.3f}ms\n"
                                f"Local Time: {processed_data['local_time']}\n"
                                f"Trade Time: {processed_data['trade_time']}"
                            )
                            
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket message timeout")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        self.error_count += 1
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
            
        finally:
            self.running = False
            # Calculate and log statistics
            stats = self.calculate_statistics()
            logger.info(
                f"\nWebSocket test completed:"
                f"\nMessages processed: {self.message_count}"
                f"\nErrors encountered: {self.error_count}"
                f"\nConnection duration: {time.time() - connection_time:.2f}s"
                f"\n\nLatency Statistics:"
                f"\nMinimum: {stats.get('min_latency_ms', 0):.3f}ms"
                f"\nMaximum: {stats.get('max_latency_ms', 0):.3f}ms"
                f"\nAverage: {stats.get('avg_latency_ms', 0):.3f}ms"
                f"\nMedian: {stats.get('median_latency_ms', 0):.3f}ms"
                f"\nStandard Deviation: {stats.get('stdev_latency_ms', 0):.3f}ms"
                f"\n\nPrice Statistics:"
                f"\nMinimum: {stats.get('min_price', 0):.2f}"
                f"\nMaximum: {stats.get('max_price', 0):.2f}"
                f"\nAverage: {stats.get('avg_price', 0):.2f}"
                f"\nVolatility: {stats.get('price_volatility', 0):.2f}"
            )

async def main():
    """Main function to run the WebSocket test."""
    # Initialize client with 50ms max latency threshold
    client = BinanceWebSocketClient(max_latency_ms=50.0)
    await client.connect_and_validate(num_messages=100)  # Test with 100 messages

if __name__ == "__main__":
    asyncio.run(main()) 