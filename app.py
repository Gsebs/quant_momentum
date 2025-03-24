from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from datetime import datetime

from market_data_feed import run_feeds, latest_prices
from strategy_engine import run_strategy, last_confidence
from trade_simulator import get_trades_log, get_cumulative_pnl, get_portfolio_value, get_positions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HFT Latency Arbitrage API",
    description="API for monitoring and controlling the HFT Latency Arbitrage system",
    version="1.0.0"
)

# Enable CORS for all origins (adjust in production to specific domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Start background tasks for market data feeds and strategy execution."""
    try:
        # Launch background tasks
        asyncio.create_task(run_feeds())
        asyncio.create_task(run_strategy())
        logger.info("Background tasks for feeds and strategy started successfully")
    except Exception as e:
        logger.error(f"Failed to start background tasks: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    return {
        "status": "ok",
        "message": "HFT Latency Arbitrage API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get current system status including latest prices, trades, and performance metrics."""
    try:
        return {
            "status": "ok",
            "message": "HFT Strategy API is running",
            "latest_prices": latest_prices,
            "trades_count": len(get_trades_log()),
            "cumulative_pnl": get_cumulative_pnl(),
            "portfolio_value": get_portfolio_value(),
            "positions": get_positions(),
            "last_confidence": last_confidence,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades():
    """Get the complete trade log."""
    try:
        return get_trades_log()
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 