from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging

from market_data_feed import run_feeds, get_latest_prices
from strategy_engine import run_strategy, last_confidence
from trade_simulator import get_trades_log, get_cumulative_pnl, get_positions, get_portfolio_value

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HFT Latency Arbitrage API",
    description="High-Frequency Trading Latency Arbitrage System API",
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
    """Start background tasks for market data feeds and strategy."""
    try:
        # Launch background tasks
        asyncio.create_task(run_feeds())
        asyncio.create_task(run_strategy())
        logger.info("Background tasks for market data feeds and strategy started.")
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    return {
        "status": "ok",
        "message": "HFT Latency Arbitrage API is running",
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Get current system status including latest prices and performance metrics."""
    try:
        return {
            "status": "ok",
            "message": "HFT Latency Arbitrage System is running",
            "latest_prices": get_latest_prices(),
            "trades_count": len(get_trades_log()),
            "cumulative_pnl": get_cumulative_pnl(),
            "portfolio_value": get_portfolio_value(),
            "positions": get_positions(),
            "last_confidence": last_confidence
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades():
    """Get the list of executed trades."""
    try:
        return get_trades_log()
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 