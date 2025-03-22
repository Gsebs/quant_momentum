"""
Live Trading Manager for HFT system.
Handles real-time data processing, monitoring, and trade execution.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import queue
import threading
import time
from dataclasses import dataclass
import asyncio
import websockets
import json
from collections import deque
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    network_latency: float
    process_time: float
    queue_sizes: Dict[str, int]
    
@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    execution_latency: float
    order_success_rate: float
    fill_ratio: float
    slippage: float
    trading_volume: float
    pnl: float
    sharpe_ratio: float
    
class LiveTradingManager:
    """Manages live trading operations and monitoring."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        
        # Performance monitoring
        self.system_metrics_history = deque(maxlen=1000)
        self.trading_metrics_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        
        # Data queues
        self.market_data_queue = queue.Queue(maxsize=10000)
        self.order_queue = queue.Queue(maxsize=1000)
        self.execution_queue = queue.Queue(maxsize=1000)
        
        # Websocket connections
        self.ws_connections = {}
        self.ws_lock = threading.Lock()
        
        # Performance thresholds
        self.max_latency = config.get('max_latency_ms', 1.0)  # 1ms default
        self.max_cpu_usage = config.get('max_cpu_usage', 80.0)  # 80% default
        self.max_memory_usage = config.get('max_memory_usage', 85.0)  # 85% default
        
        # Initialize monitoring threads
        self.monitor_threads = []
        
    async def start(self):
        """Start live trading operations."""
        try:
            self.is_running = True
            
            # Start monitoring threads
            self._start_system_monitoring()
            self._start_trading_monitoring()
            self._start_data_processing()
            
            # Start websocket connections
            await self._start_websocket_connections()
            
            logger.info("Live trading manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live trading manager: {str(e)}")
            return False
            
    def stop(self):
        """Stop live trading operations."""
        self.is_running = False
        self._cleanup_connections()
        logger.info("Live trading manager stopped")
        
    def _start_system_monitoring(self):
        """Start system performance monitoring thread."""
        def monitor_system():
            while self.is_running:
                try:
                    # Collect system metrics
                    metrics = SystemMetrics(
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        network_latency=self._measure_network_latency(),
                        process_time=time.process_time(),
                        queue_sizes={
                            'market_data': self.market_data_queue.qsize(),
                            'orders': self.order_queue.qsize(),
                            'execution': self.execution_queue.qsize()
                        }
                    )
                    
                    # Check thresholds
                    if metrics.cpu_usage > self.max_cpu_usage:
                        logger.warning(f"High CPU usage: {metrics.cpu_usage}%")
                    if metrics.memory_usage > self.max_memory_usage:
                        logger.warning(f"High memory usage: {metrics.memory_usage}%")
                        
                    self.system_metrics_history.append(metrics)
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {str(e)}")
                    time.sleep(1)
                    
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
        self.monitor_threads.append(thread)
        
    def _start_trading_monitoring(self):
        """Start trading performance monitoring thread."""
        def monitor_trading():
            while self.is_running:
                try:
                    # Calculate trading metrics
                    metrics = self._calculate_trading_metrics()
                    
                    # Check performance thresholds
                    if metrics.execution_latency > self.max_latency:
                        logger.warning(f"High execution latency: {metrics.execution_latency}ms")
                    if metrics.fill_ratio < 0.8:  # 80% fill ratio threshold
                        logger.warning(f"Low fill ratio: {metrics.fill_ratio}")
                        
                    self.trading_metrics_history.append(metrics)
                    time.sleep(0.1)  # Update every 100ms
                    
                except Exception as e:
                    logger.error(f"Error in trading monitoring: {str(e)}")
                    time.sleep(1)
                    
        thread = threading.Thread(target=monitor_trading, daemon=True)
        thread.start()
        self.monitor_threads.append(thread)
        
    def _start_data_processing(self):
        """Start real-time data processing thread."""
        def process_data():
            while self.is_running:
                try:
                    # Process market data
                    while not self.market_data_queue.empty():
                        start_time = time.time()
                        
                        data = self.market_data_queue.get_nowait()
                        self._process_market_data(data)
                        
                        # Record processing latency
                        latency = (time.time() - start_time) * 1000
                        self.latency_history.append(latency)
                        
                    # Process orders
                    while not self.order_queue.empty():
                        order = self.order_queue.get_nowait()
                        self._process_order(order)
                        
                    time.sleep(0.0001)  # 100μs sleep
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in data processing: {str(e)}")
                    time.sleep(0.1)
                    
        thread = threading.Thread(target=process_data, daemon=True)
        thread.start()
        self.monitor_threads.append(thread)
        
    async def _start_websocket_connections(self):
        """Initialize websocket connections for market data."""
        try:
            for exchange, url in self.config['websocket_urls'].items():
                asyncio.create_task(self._maintain_websocket(exchange, url))
                
        except Exception as e:
            logger.error(f"Error starting websocket connections: {str(e)}")
            
    async def _maintain_websocket(self, exchange: str, url: str):
        """Maintain websocket connection with automatic reconnection."""
        while self.is_running:
            try:
                async with websockets.connect(url) as ws:
                    with self.ws_lock:
                        self.ws_connections[exchange] = ws
                        
                    logger.info(f"Connected to {exchange} websocket")
                    
                    while self.is_running:
                        try:
                            message = await ws.recv()
                            await self._handle_websocket_message(exchange, message)
                        except websockets.ConnectionClosed:
                            break
                            
            except Exception as e:
                logger.error(f"Websocket error for {exchange}: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting
                
    async def _handle_websocket_message(self, exchange: str, message: str):
        """Process incoming websocket messages."""
        try:
            data = json.loads(message)
            
            # Add to market data queue
            self.market_data_queue.put({
                'exchange': exchange,
                'timestamp': datetime.now(),
                'data': data
            })
            
        except Exception as e:
            logger.error(f"Error handling websocket message: {str(e)}")
            
    def _process_market_data(self, data: Dict):
        """Process incoming market data."""
        try:
            # Extract relevant information
            exchange = data['exchange']
            timestamp = data['timestamp']
            market_data = data['data']
            
            # Update order book
            if 'book' in market_data:
                self._update_order_book(exchange, market_data['book'])
                
            # Process trades
            if 'trades' in market_data:
                self._process_trades(exchange, market_data['trades'])
                
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            
    def _process_order(self, order: Dict):
        """Process and execute trading orders."""
        try:
            start_time = time.time()
            
            # Validate order
            if not self._validate_order(order):
                logger.warning(f"Invalid order: {order}")
                return
                
            # Execute order
            execution_result = self._execute_order(order)
            
            # Record execution metrics
            latency = (time.time() - start_time) * 1000
            self.execution_queue.put({
                'order': order,
                'result': execution_result,
                'latency': latency
            })
            
        except Exception as e:
            logger.error(f"Error processing order: {str(e)}")
            
    def _validate_order(self, order: Dict) -> bool:
        """Validate order parameters."""
        required_fields = ['symbol', 'side', 'quantity', 'price', 'type']
        return all(field in order for field in required_fields)
        
    def _execute_order(self, order: Dict) -> Dict:
        """Execute trading order."""
        try:
            # Simulate order execution (replace with actual exchange API calls)
            execution_price = order['price']
            executed_quantity = order['quantity']
            
            return {
                'status': 'filled',
                'executed_price': execution_price,
                'executed_quantity': executed_quantity,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
            
    def _calculate_trading_metrics(self) -> TradingMetrics:
        """Calculate current trading performance metrics."""
        try:
            # Calculate execution metrics
            recent_executions = list(self.execution_queue.queue)
            if not recent_executions:
                return TradingMetrics(0, 0, 0, 0, 0, 0, 0)
                
            # Calculate metrics
            latencies = [exec['latency'] for exec in recent_executions]
            success_count = sum(1 for exec in recent_executions if exec['result']['status'] == 'filled')
            fill_ratios = [
                exec['result']['executed_quantity'] / exec['order']['quantity']
                for exec in recent_executions if exec['result']['status'] == 'filled'
            ]
            
            # Calculate slippage
            slippages = [
                abs(exec['result']['executed_price'] - exec['order']['price']) / exec['order']['price']
                for exec in recent_executions if exec['result']['status'] == 'filled'
            ]
            
            # Calculate metrics
            metrics = TradingMetrics(
                execution_latency=np.mean(latencies),
                order_success_rate=success_count / len(recent_executions),
                fill_ratio=np.mean(fill_ratios) if fill_ratios else 0,
                slippage=np.mean(slippages) if slippages else 0,
                trading_volume=sum(exec['result']['executed_quantity'] * exec['result']['executed_price']
                                 for exec in recent_executions if exec['result']['status'] == 'filled'),
                pnl=self._calculate_pnl(recent_executions),
                sharpe_ratio=self._calculate_sharpe_ratio()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {str(e)}")
            return TradingMetrics(0, 0, 0, 0, 0, 0, 0)
            
    def _calculate_pnl(self, executions: List[Dict]) -> float:
        """Calculate realized P&L from executions."""
        try:
            pnl = 0
            positions = {}
            
            for exec in executions:
                if exec['result']['status'] != 'filled':
                    continue
                    
                order = exec['order']
                result = exec['result']
                symbol = order['symbol']
                
                if symbol not in positions:
                    positions[symbol] = {'quantity': 0, 'cost_basis': 0}
                    
                # Update position
                if order['side'] == 'buy':
                    positions[symbol]['quantity'] += result['executed_quantity']
                    positions[symbol]['cost_basis'] += result['executed_quantity'] * result['executed_price']
                else:
                    # Calculate P&L for sells
                    avg_price = positions[symbol]['cost_basis'] / positions[symbol]['quantity']
                    pnl += result['executed_quantity'] * (result['executed_price'] - avg_price)
                    positions[symbol]['quantity'] -= result['executed_quantity']
                    
            return pnl
            
        except Exception as e:
            logger.error(f"Error calculating P&L: {str(e)}")
            return 0
            
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent trading performance."""
        try:
            # Get recent P&L history
            pnl_history = [metrics.pnl for metrics in self.trading_metrics_history]
            if len(pnl_history) < 2:
                return 0
                
            # Calculate returns
            returns = np.diff(pnl_history)
            if len(returns) == 0:
                return 0
                
            # Calculate Sharpe ratio (annualized)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return == 0:
                return 0
                
            sharpe = np.sqrt(252) * mean_return / std_return  # Annualized
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
            
    def _measure_network_latency(self) -> float:
        """Measure current network latency."""
        try:
            # Ping exchange servers (replace with actual exchange endpoints)
            start_time = time.time()
            # Add actual network latency measurement here
            latency = (time.time() - start_time) * 1000
            return latency
        except Exception as e:
            logger.error(f"Error measuring network latency: {str(e)}")
            return 0
            
    def _cleanup_connections(self):
        """Clean up websocket connections."""
        with self.ws_lock:
            for exchange, ws in self.ws_connections.items():
                try:
                    asyncio.create_task(ws.close())
                except Exception as e:
                    logger.error(f"Error closing {exchange} websocket: {str(e)}")
            self.ws_connections.clear()
            
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        try:
            # Get latest metrics
            system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            trading_metrics = self.trading_metrics_history[-1] if self.trading_metrics_history else None
            
            return {
                'system': {
                    'cpu_usage': system_metrics.cpu_usage if system_metrics else 0,
                    'memory_usage': system_metrics.memory_usage if system_metrics else 0,
                    'network_latency': system_metrics.network_latency if system_metrics else 0,
                    'queue_sizes': system_metrics.queue_sizes if system_metrics else {},
                },
                'trading': {
                    'execution_latency': trading_metrics.execution_latency if trading_metrics else 0,
                    'order_success_rate': trading_metrics.order_success_rate if trading_metrics else 0,
                    'fill_ratio': trading_metrics.fill_ratio if trading_metrics else 0,
                    'slippage': trading_metrics.slippage if trading_metrics else 0,
                    'trading_volume': trading_metrics.trading_volume if trading_metrics else 0,
                    'pnl': trading_metrics.pnl if trading_metrics else 0,
                    'sharpe_ratio': trading_metrics.sharpe_ratio if trading_metrics else 0
                },
                'latency': {
                    'avg_latency': np.mean(self.latency_history) if self.latency_history else 0,
                    'max_latency': np.max(self.latency_history) if self.latency_history else 0,
                    'min_latency': np.min(self.latency_history) if self.latency_history else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {} 