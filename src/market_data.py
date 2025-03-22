import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
import time
import ccxt.async_support as ccxt
from decimal import Decimal
import os

logger = logging.getLogger(__name__)

class MarketDataFeed:
    """Market data feed handler for streaming market data."""
    
    def __init__(self, symbols: List[str] = None, on_data_callback: Optional[Callable] = None,
                 on_error_callback: Optional[Callable] = None):
        self.callbacks = []
        if on_data_callback:
            self.callbacks.append(on_data_callback)
            
        self.symbols = symbols or []
        self.exchanges = {}
        self.subscriptions = {}
        self.last_update = {}
        self.last_update_time = {}
        self.connections = {}
        self.market_data = {}
        self.metrics = {
            'messages_received': 0,
            'updates_per_second': 0,
            'latency_ms': 0,
            'errors': 0,
            'reconnects': 0
        }
        self.start_time = datetime.now()
        self.is_running = False
        self.on_error_callback = on_error_callback
        self._stop_event = asyncio.Event()
        self._reconnect_delay = 5  # Initial delay in seconds
        self._max_reconnect_delay = 300  # Maximum delay in seconds
        
    def subscribe(self, callback):
        """Subscribe to market data updates."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            
    def unsubscribe(self, callback):
        """Unsubscribe from market data updates."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    async def initialize(self):
        """Initialize market data connections."""
        try:
            # Initialize exchange connections with proxy settings if needed
            proxy = None
            if 'HTTPS_PROXY' in os.environ:
                proxy = os.environ['HTTPS_PROXY']
                
            # Initialize with multiple exchanges for redundancy
            self.exchanges = {
                'bybit': ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'},
                    'proxy': proxy
                }),
                'kucoin': ccxt.kucoin({
                    'enableRateLimit': True,
                    'proxy': proxy
                }),
                'huobi': ccxt.huobi({
                    'enableRateLimit': True,
                    'proxy': proxy
                })
            }
            
            # Initialize connections
            self.connections = {
                exchange: {
                    'status': 'disconnected',
                    'last_heartbeat': None,
                    'reconnect_count': 0
                }
                for exchange in self.exchanges
            }
            
            # Load markets
            success = False
            for name, exchange in self.exchanges.items():
                try:
                    await exchange.load_markets()
                    self.connections[name]['status'] = 'connected'
                    self.connections[name]['last_heartbeat'] = time.time()
                    success = True
                except Exception as e:
                    logger.error(f"Error loading markets for {name}: {e}")
                    continue
                    
            if not success:
                raise Exception("Failed to connect to any exchange")
                
            # Subscribe to initial symbols
            for symbol in self.symbols:
                await self.subscribe_symbol(symbol)
                self.last_update_time[symbol] = time.time()
                
            # Start monitoring tasks
            asyncio.create_task(self._monitor_connections())
            asyncio.create_task(self._update_metrics())
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing market data feed: {e}")
            if self.on_error_callback:
                await self.on_error_callback(e)
            raise
            
    async def subscribe_symbol(self, symbol: str):
        """Subscribe to market data for a specific symbol."""
        try:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {
                    exchange: False for exchange in self.exchanges
                }
                
                # Subscribe to WebSocket feeds for each exchange
                for name, exchange in self.exchanges.items():
                    try:
                        await exchange.watch_order_book(symbol)
                        await exchange.watch_trades(symbol)
                        self.subscriptions[symbol][name] = True
                    except Exception as e:
                        logger.error(f"Error subscribing to {symbol} on {name}: {e}")
                        continue
                        
                # Check if at least one subscription succeeded
                if not any(self.subscriptions[symbol].values()):
                    raise Exception(f"Failed to subscribe to {symbol} on any exchange")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False
            
    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from market data for a specific symbol."""
        try:
            if symbol in self.subscriptions:
                for name, exchange in self.exchanges.items():
                    try:
                        if self.subscriptions[symbol][name]:
                            await exchange.unsubscribe_order_book(symbol)
                            await exchange.unsubscribe_trades(symbol)
                    except Exception as e:
                        logger.error(f"Error unsubscribing from {symbol} on {name}: {e}")
                        
                del self.subscriptions[symbol]
                
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False
            
    async def _handle_error(self, error: Exception):
        """Handle errors in market data processing."""
        logger.error(f"Market data error: {error}")
        self.metrics['errors'] += 1
        if self.on_error_callback:
            await self.on_error_callback(error)
            
    async def _process_orderbook(self, exchange: str, symbol: str, orderbook: Dict):
        """Process order book updates."""
        try:
            timestamp = datetime.now()
            
            # Validate orderbook data
            if not isinstance(orderbook, dict) or 'bids' not in orderbook or 'asks' not in orderbook:
                logger.warning(f"Invalid orderbook data from {exchange}: {orderbook}")
                return
                
            # Convert string prices and sizes to Decimal
            bids = [(Decimal(str(price)), Decimal(str(size))) 
                   for price, size in orderbook['bids']]
            asks = [(Decimal(str(price)), Decimal(str(size))) 
                   for price, size in orderbook['asks']]
                   
            # Calculate latency
            exchange_time = orderbook.get('timestamp', timestamp.timestamp() * 1000)
            latency = (timestamp.timestamp() * 1000) - exchange_time
            
            # Update metrics
            self.metrics['messages_received'] += 1
            self.metrics['latency_ms'] = (self.metrics['latency_ms'] * 0.95 + latency * 0.05)
            
            # Update last update time
            self.last_update_time[symbol] = time.time()
            
            # Prepare market data
            market_data = {
                'exchange': exchange,
                'symbol': symbol,
                'timestamp': timestamp,
                'bids': bids,
                'asks': asks,
                'latency_ms': latency
            }
            
            # Update stored market data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            self.market_data[symbol][exchange] = market_data
            
            # Notify subscribers
            for callback in self.callbacks:
                await callback(symbol, market_data)
                
        except Exception as e:
            await self._handle_error(e)
            
    async def _process_trades(self, exchange: str, symbol: str, trades: List[Dict]):
        """Process trade updates."""
        try:
            timestamp = datetime.now()
            
            for trade in trades:
                # Validate trade data
                required_fields = {'price', 'amount', 'side', 'timestamp'}
                if not all(field in trade for field in required_fields):
                    logger.warning(f"Invalid trade message from {exchange}: {trade}")
                    continue
                    
                # Convert price and amount to Decimal
                try:
                    price = Decimal(str(trade['price']))
                    amount = Decimal(str(trade['amount']))
                except (TypeError, ValueError) as e:
                    logger.warning(f"Invalid price or amount in trade: {e}")
                    continue
                    
                # Calculate latency
                trade_time = trade.get('timestamp', timestamp.timestamp() * 1000)
                latency = (timestamp.timestamp() * 1000) - trade_time
                
                # Update metrics
                self.metrics['messages_received'] += 1
                self.metrics['latency_ms'] = (self.metrics['latency_ms'] * 0.95 + latency * 0.05)
                
                # Prepare trade data
                trade_data = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'price': price,
                    'amount': amount,
                    'side': trade['side'],
                    'latency_ms': latency
                }
                
                # Notify subscribers
                for callback in self.callbacks:
                    await callback(symbol, trade_data)
                    
        except Exception as e:
            await self._handle_error(e)
            
    async def _monitor_connections(self):
        """Monitor exchange connections and reconnect if needed."""
        while not self._stop_event.is_set():
            try:
                for name, conn in self.connections.items():
                    if conn['status'] == 'connected':
                        # Check heartbeat
                        if time.time() - conn['last_heartbeat'] > 30:  # 30 seconds timeout
                            logger.warning(f"Connection timeout for {name}")
                            conn['status'] = 'disconnected'
                            
                    if conn['status'] == 'disconnected':
                        # Calculate reconnect delay with exponential backoff
                        delay = min(self._reconnect_delay * (2 ** conn['reconnect_count']),
                                  self._max_reconnect_delay)
                        
                        logger.info(f"Attempting to reconnect to {name} after {delay}s")
                        try:
                            exchange = self.exchanges[name]
                            await exchange.load_markets()
                            
                            # Resubscribe to symbols
                            for symbol in self.symbols:
                                if symbol in self.subscriptions:
                                    await exchange.watch_order_book(symbol)
                                    await exchange.watch_trades(symbol)
                                    
                            conn['status'] = 'connected'
                            conn['last_heartbeat'] = time.time()
                            conn['reconnect_count'] = 0
                            self.metrics['reconnects'] += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to reconnect to {name}: {e}")
                            conn['reconnect_count'] += 1
                            
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                
            await asyncio.sleep(5)  # Check every 5 seconds
            
    async def _update_metrics(self):
        """Update metrics periodically."""
        while not self._stop_event.is_set():
            try:
                # Calculate updates per second
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed > 0:
                    self.metrics['updates_per_second'] = self.metrics['messages_received'] / elapsed
                    
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                
            await asyncio.sleep(1)
            
    async def start(self):
        """Start the market data feed."""
        self.is_running = True
        self._stop_event.clear()
        await self.initialize()
        
    async def stop(self):
        """Stop the market data feed."""
        self.is_running = False
        self._stop_event.set()
        await self.cleanup()
        
    async def cleanup(self):
        """Clean up resources."""
        try:
            for exchange in self.exchanges.values():
                await exchange.close()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def get_last_price(self, symbol: str, exchange: str = None) -> Optional[Decimal]:
        """Get the last known price for a symbol."""
        try:
            if exchange:
                return self.last_update.get(symbol, {}).get(exchange, {}).get('price')
            else:
                # Return the most recent price from any exchange
                updates = self.last_update.get(symbol, {})
                if not updates:
                    return None
                    
                latest_update = max(updates.values(), key=lambda x: x['timestamp'])
                return latest_update.get('price')
                
        except Exception as e:
            logger.error(f"Error getting last price: {e}")
            return None

    def get_market_data(self, symbol: Optional[str] = None) -> Dict:
        """Get current market data for one or all symbols."""
        if symbol:
            return self.market_data.get(symbol, {})
        return self.market_data
    
    def get_latency(self, symbol: str) -> float:
        """Get current latency for a symbol."""
        try:
            if symbol in self.market_data:
                latencies = [data.get('latency_ms', 0) 
                           for data in self.market_data[symbol].values()]
                if latencies:
                    return sum(latencies) / len(latencies)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating latency: {e}")
            return 0.0 