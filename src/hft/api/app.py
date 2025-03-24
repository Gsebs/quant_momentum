from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..data_feed.market_data import market_data, MarketDataFeed
from ..strategy.strategy_engine import HFTStrategy
from ..simulation.trade_simulator import TradeSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HFT Latency Arbitrage API")

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
market_feed = None
strategy = None
trade_simulator = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global market_feed, strategy, trade_simulator
    
    try:
        # Initialize components
        market_feed = MarketDataFeed("BTC-USD")
        trade_simulator = TradeSimulator()
        strategy = HFTStrategy()

        # Start background tasks
        asyncio.create_task(market_feed.start())
        asyncio.create_task(strategy.run())
        
        logger.info("Started market data feed and strategy")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global market_feed, strategy
    
    try:
        if market_feed:
            await market_feed.stop()
        if strategy:
            await strategy.stop()
        logger.info("Stopped market data feed and strategy")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get current system status including latest prices and trading metrics."""
    try:
        return {
            "status": "ok",
            "message": "HFT Strategy API is running",
            "timestamp": datetime.now().timestamp(),
            "market_data": market_data,
            "strategy": {
                "last_confidence": strategy.get_last_confidence() if strategy else 0.0,
                "threshold": strategy.threshold if strategy else 0.0,
                "min_confidence": strategy.min_confidence if strategy else 0.0
            },
            "trading": {
                "cumulative_pnl": trade_simulator.get_cumulative_pnl() if trade_simulator else 0.0,
                "balance": trade_simulator.get_balance() if trade_simulator else 0.0,
                "trade_stats": trade_simulator.get_trade_stats() if trade_simulator else {}
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades() -> List[Dict[str, Any]]:
    """Get the list of executed trades."""
    try:
        return trade_simulator.get_trades() if trade_simulator else []
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/stats")
async def get_trade_stats() -> Dict[str, Any]:
    """Get trading statistics."""
    try:
        return trade_simulator.get_trade_stats() if trade_simulator else {}
    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().timestamp()
    }

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
            "timestamp": datetime.now().timestamp()
        }
    ) 