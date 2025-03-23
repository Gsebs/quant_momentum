from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import asyncio
from market_data_feed import run_feeds
from strategy_engine import run_strategy
from trade_simulator import get_trades_log, get_cumulative_pnl

app = FastAPI(title="Quant Momentum Strategy API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
latest_prices = {}
last_confidence = 0.0

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
        "timestamp": datetime.now().isoformat()
    }

@app.get("/trades")
async def get_trades():
    return get_trades_log()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 