from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from .services.market_data import MarketDataService
from .services.trading import TradingService
from .services.database import DatabaseService
from .services.websocket import WebSocketManager
from .models.database import init_db, get_db
from .config.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HFT Latency Arbitrage Trading System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
market_data_service = MarketDataService()
trading_service = TradingService()
websocket_manager = WebSocketManager(DatabaseService(next(get_db())))

@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_db()
    
    # Start market data service
    asyncio.create_task(market_data_service.start())
    
    # Start trading service
    asyncio.create_task(trading_service.start())

@app.on_event("shutdown")
async def shutdown_event():
    # Stop services
    await market_data_service.stop()
    await trading_service.stop()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        websocket_manager.disconnect(websocket)

# API endpoints
@app.get("/status")
async def get_status(db: Session = Depends(get_db)):
    db_service = DatabaseService(db)
    status = db_service.get_latest_system_status()
    if not status:
        raise HTTPException(status_code=404, detail="System status not found")
    return status

@app.get("/opportunities")
async def get_opportunities(
    skip: int = 0,
    limit: int = 100,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    db_service = DatabaseService(db)
    opportunities = db_service.get_opportunities(skip, limit, symbol, status)
    return opportunities

@app.get("/trades")
async def get_trades(
    skip: int = 0,
    limit: int = 100,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    db_service = DatabaseService(db)
    trades = db_service.get_trades(skip, limit, symbol, status)
    return trades

@app.get("/performance")
async def get_performance(
    timeframe: str = "1h",
    db: Session = Depends(get_db)
):
    db_service = DatabaseService(db)
    
    # Calculate time range based on timeframe
    end_time = datetime.utcnow()
    if timeframe == "1h":
        start_time = end_time - timedelta(hours=1)
    elif timeframe == "1d":
        start_time = end_time - timedelta(days=1)
    elif timeframe == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif timeframe == "1m":
        start_time = end_time - timedelta(days=30)
    else:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    performance = db_service.get_performance_history(start_time, end_time)
    stats = db_service.get_trading_stats(start_time, end_time)
    
    return {
        "performance": performance,
        "stats": stats
    }

@app.post("/settings")
async def update_settings(settings_data: dict, db: Session = Depends(get_db)):
    # Update trading service settings
    await trading_service.update_settings(settings_data)
    
    # Update market data service settings
    await market_data_service.update_settings(settings_data)
    
    return {"message": "Settings updated successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "market_data": market_data_service.is_running(),
            "trading": trading_service.is_running()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 