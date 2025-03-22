import asyncio
import logging
import time
from typing import Dict, List, Set, Optional
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiosqlite
from dataclasses import dataclass
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class MarketUpdate:
    exchange: str
    symbol: str
    timestamp: float
    price: float
    volume: float
    side: str
    orderbook: Dict
    
class DataManager:
    def __init__(self,
                 symbols: List[str],
                 max_queue_size: int = 10000,
                 db_path: str = "trades.db"):
        self.symbols = symbols
        self.db_path = db_path
        
        # High-performance queues for real-time data
        self.market_queues: Dict[str, asyncio.Queue] = {
            symbol: asyncio.Queue(maxsize=max_queue_size)
            for symbol in symbols
        }
        
        # Efficient circular buffers for recent data
        self.price_history: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        self.volume_history: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Active WebSocket connections
        self.active_connections: Set[str] = set()
        
        # Performance metrics
        self.processing_times: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        self.queue_sizes: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        
        # Database initialization
        self.db_initialized = False
        
    async def initialize_db(self):
        """Initialize SQLite database with WAL mode for better performance."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        timestamp REAL,
                        exchange TEXT,
                        symbol TEXT,
                        price REAL,
                        volume REAL,
                        side TEXT,
                        latency_ms REAL
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_time 
                    ON trades(timestamp)
                """)
                await db.commit()
            self.db_initialized = True
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            
    async def process_market_update(self, update: MarketUpdate):
        """Process incoming market data update with minimal latency."""
        try:
            start_time = time.perf_counter()
            
            # Update price history (lock-free operation)
            self.price_history[update.symbol].append(
                (update.timestamp, update.price)
            )
            
            # Update volume history
            self.volume_history[update.symbol].append(
                (update.timestamp, update.volume)
            )
            
            # Queue size monitoring
            queue_size = self.market_queues[update.symbol].qsize()
            self.queue_sizes[update.symbol].append(queue_size)
            
            # Log processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times[update.symbol].append(processing_time)
            
            # Async database write (if needed)
            if update.volume > 0:  # Only log actual trades
                await self._write_to_db(update)
                
        except Exception as e:
            logger.error(f"Error processing market update: {str(e)}")
            
    async def _write_to_db(self, update: MarketUpdate):
        """Write trade data to SQLite asynchronously."""
        if not self.db_initialized:
            return
            
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        update.timestamp,
                        update.exchange,
                        update.symbol,
                        update.price,
                        update.volume,
                        update.side,
                        0.0  # latency placeholder
                    )
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Error writing to database: {str(e)}")
            
    def get_recent_prices(self, 
                         symbol: str, 
                         lookback_seconds: float = 60.0) -> np.ndarray:
        """Get recent price data efficiently using NumPy."""
        try:
            if symbol not in self.price_history:
                return np.array([])
                
            current_time = time.time()
            cutoff_time = current_time - lookback_seconds
            
            # Convert deque to NumPy array for vectorized operations
            data = np.array(list(self.price_history[symbol]))
            if len(data) == 0:
                return np.array([])
                
            # Vectorized filtering
            mask = data[:, 0] >= cutoff_time
            return data[mask]
            
        except Exception as e:
            logger.error(f"Error getting recent prices: {str(e)}")
            return np.array([])
            
    async def run_cpu_intensive_task(self, func, *args):
        """Run CPU-intensive task in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(func, *args))
        
    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics."""
        metrics = {}
        for symbol in self.symbols:
            proc_times = list(self.processing_times[symbol])
            queue_sizes = list(self.queue_sizes[symbol])
            
            metrics[symbol] = {
                'avg_processing_time_ms': np.mean(proc_times) if proc_times else 0,
                'max_processing_time_ms': max(proc_times) if proc_times else 0,
                'avg_queue_size': np.mean(queue_sizes) if queue_sizes else 0,
                'max_queue_size': max(queue_sizes) if queue_sizes else 0,
                'current_queue_size': self.market_queues[symbol].qsize()
            }
            
        return metrics
        
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        # Close any active connections
        for conn in self.active_connections:
            try:
                # Implementation depends on WebSocket library used
                pass
            except Exception as e:
                logger.error(f"Error closing connection {conn}: {str(e)}") 