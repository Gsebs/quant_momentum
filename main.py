import asyncio
import logging
import logging.config
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import json
from datetime import datetime

from backend.core.trading_engine import TradingEngine
from backend.config.config import get_config, validate_config
from backend.services.market_data import MarketDataService
from backend.services.trading import TradingService

# Initialize FastAPI app
app = FastAPI(title="HFT Latency Arbitrage Trading System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = get_config()
logging.config.dictConfig(config['logging'])
logger = logging.getLogger(__name__)

trading_engine = None
market_data_service = None
trading_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global trading_engine, market_data_service, trading_service
    
    try:
        # Validate configuration
        if not validate_config():
            raise ValueError("Invalid configuration")

        # Initialize services
        market_data_service = MarketDataService(config['exchanges'])
        trading_service = TradingService(config['exchanges'])
        
        # Initialize trading engine
        trading_engine = TradingEngine(config)
        await trading_engine.initialize()
        
        logger.info("Trading system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global trading_engine
    
    try:
        if trading_engine:
            await trading_engine.stop()
        logger.info("Trading system stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping trading system: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "HFT Latency Arbitrage Trading System",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not trading_engine:
        return {"status": "initializing"}
    
    status = trading_engine.get_status()
    return {
        "status": "healthy" if trading_engine.is_running else "stopped",
        "timestamp": datetime.now().isoformat(),
        **status
    }

@app.post("/start")
async def start_trading():
    """Start the trading engine"""
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    if trading_engine.is_running:
        raise HTTPException(status_code=400, detail="Trading system already running")
    
    try:
        await trading_engine.start()
        return {"status": "started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_trading():
    """Stop the trading engine"""
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    if not trading_engine.is_running:
        raise HTTPException(status_code=400, detail="Trading system not running")
    
    try:
        await trading_engine.stop()
        return {"status": "stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/opportunities")
async def get_opportunities():
    """Get current arbitrage opportunities"""
    if not market_data_service:
        raise HTTPException(status_code=503, detail="Market data service not initialized")
    
    opportunities = []
    for symbol in config['trading_pairs']:
        opportunities.extend(
            market_data_service.find_arbitrage_opportunities(
                symbol,
                config['trading_params']['min_profit_threshold']
            )
        )
    
    return {"opportunities": opportunities}

@app.get("/trades")
async def get_trades():
    """Get recent trades"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    return {"trades": trading_service.get_trade_history()}

@app.get("/performance")
async def get_performance():
    """Get trading performance metrics"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    return trading_service.get_performance_metrics()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            await websocket.send_text(json.dumps({
                "type": "message",
                "content": f"Received: {data}",
                "timestamp": datetime.now().isoformat()
            }))
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 