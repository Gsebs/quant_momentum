"""
Main script for running the Quantitative HFT Algorithm
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any
import ccxt.pro as ccxt
import pandas as pd
import numpy as np
from .hft_engine import HFTEngine
from .market_making import MarketMakingStrategy
from .ml_model import HFTModel, HFTFeatureEngine
import sqlite3
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .market_data import MarketDataFeed
from .strategy import LatencyArbitrageStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hft.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HFT Monitoring System")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
            "https://quant-momentum-hft.netlify.app",
            "http://localhost:3000"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HFTSystem:
    def __init__(self, 
                 exchange_id: str = 'binance',
                 symbols: List[str] = ['BTC/USDT', 'ETH/USDT'],
                 risk_limit: float = 1000000,
                 max_position: float = 100000,
                 latency_threshold_ms: float = 10):
        
        # Initialize components
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'apiKey': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_SECRET')
        })
        
        self.hft_engine = HFTEngine(
            symbols=symbols,
            risk_limit=risk_limit,
            max_position=max_position,
            latency_threshold_ms=latency_threshold_ms
        )
        
        self.market_maker = MarketMakingStrategy({
            'base_spread': 0.001,  # 10 bps
            'min_spread': 0.0005,  # 5 bps
            'max_spread': 0.002,   # 20 bps
            'max_position_size': max_position,
            'quote_validity_ms': 50
        })
        
        self.ml_model = HFTModel()
        self.feature_engine = HFTFeatureEngine()
        
        # Initialize database
        self.db_path = Path('data/hft.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
        
        self.symbols = symbols
        self.running = False
        
    def init_database(self):
        """Initialize SQLite database for storing trades and metrics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create trades table
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                latency_ms REAL NOT NULL,
                pnl REAL
            )
        ''')
        
        # Create metrics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DATETIME PRIMARY KEY,
                total_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_latency_ms REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_trade(self, trade):
        """Log trade to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO trades (symbol, side, price, size, timestamp, latency_ms, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.symbol,
            trade.side,
            trade.price,
            trade.size,
            trade.timestamp.isoformat(),
            trade.latency_ms,
            0  # PnL will be updated later
        ))
        
        conn.commit()
        conn.close()
        
    def update_metrics(self):
        """Update and log performance metrics"""
        metrics = self.hft_engine.metrics
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO metrics (timestamp, total_pnl, win_rate, sharpe_ratio, max_drawdown, avg_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            metrics['total_pnl'],
            metrics['winning_trades'] / max(metrics['total_trades'], 1),
            self.calculate_sharpe_ratio(),
            metrics['max_drawdown'],
            metrics['avg_latency_ms']
        ))
        
        conn.commit()
        conn.close()
        
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from trade history"""
        conn = sqlite3.connect(self.db_path)
        trades_df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1000",
            conn
        )
        conn.close()
        
        if len(trades_df) < 2:
            return 0
            
        # Calculate returns
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.set_index('timestamp')
        returns = trades_df['pnl'].resample('1Min').sum()
        
        if len(returns) < 2:
            return 0
            
        # Annualize Sharpe ratio (assuming minute returns)
        sharpe = np.sqrt(525600) * (returns.mean() / returns.std())
        return float(sharpe)
        
    async def handle_order_book(self, symbol: str, orderbook: Dict):
        """Process order book updates"""
        try:
            # Extract order book data
            bids = orderbook['bids']  # [[price, size], ...]
            asks = orderbook['asks']
            
            # Update HFT engine
            for bid in bids:
                await self.hft_engine.handle_order_book_update(symbol, 'BUY', bid[0], bid[1])
            for ask in asks:
                await self.hft_engine.handle_order_book_update(symbol, 'SELL', ask[0], ask[1])
                
            # Update market maker
            self.market_maker.handle_order_book(symbol, orderbook)
            
            # Update ML features
            mid_price = (bids[0][0] + asks[0][0]) / 2
            volume = sum(bid[1] for bid in bids) + sum(ask[1] for ask in asks)
            imbalance = self.market_maker._calculate_imbalance(symbol)
            
            self.feature_engine.update(symbol, mid_price, volume, imbalance)
            self.ml_model.update_data(symbol, mid_price, volume, imbalance)
            
        except Exception as e:
            logger.error(f"Error handling order book for {symbol}: {str(e)}")
            
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        try:
            while self.running:
                for symbol in self.symbols:
                    try:
                        # Fetch order book
                        orderbook = await self.exchange.watch_order_book(symbol)
                        await self.handle_order_book(symbol, orderbook)
                        
                        # Update metrics every minute
                        if datetime.now().second == 0:
                            self.update_metrics()
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            self.running = False
            
        finally:
            await self.exchange.close()
            
    def stop(self):
        """Stop the trading system"""
        self.running = False
            
async def main():
    """Initialize and run the HFT system"""
    try:
        # Initialize system
        system = HFTSystem(
            exchange_id='binance',
            symbols=['BTC/USDT', 'ETH/USDT'],
            risk_limit=1000000,
            max_position=100000,
            latency_threshold_ms=10
        )
        
        # Run the system
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "running", "message": "HFT Monitoring System API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Simulate market data and trading activity
            data = {
                "timestamp": datetime.now().isoformat(),
                "price": 50000 + np.random.normal(0, 10),
                "volume": np.random.exponential(100),
                "trades": [
                    {
                        "id": len(active_connections),
                        "side": "BUY" if np.random.random() > 0.5 else "SELL",
                        "price": 50000 + np.random.normal(0, 5),
                        "size": np.random.exponential(0.1),
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "status": "active",
        "connections": len(active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
        
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
