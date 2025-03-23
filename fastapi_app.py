from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import asyncio
from market_data_feed import run_feeds, get_latest_prices
from strategy_engine import run_strategy, get_cached_signals
from trade_simulator import get_trades_log, get_cumulative_pnl, get_positions, get_portfolio_value

app = FastAPI(title="Quant Momentum Strategy API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Start background tasks
    asyncio.create_task(run_feeds())
    asyncio.create_task(run_strategy())

@app.get("/status")
async def get_status():
    return {
        "status": "ok",
        "message": "HFT Strategy API is running",
        "latest_prices": get_latest_prices(),
        "trades_count": len(get_trades_log()),
        "cumulative_pnl": get_cumulative_pnl(),
        "portfolio_value": get_portfolio_value(),
        "positions": get_positions(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/trades")
async def get_trades():
    return get_trades_log()

@app.get("/signals")
async def get_signals():
    return get_cached_signals()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 