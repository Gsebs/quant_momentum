"""
FastAPI backend for HFT system monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import json

app = FastAPI(title="HFT System Monitor")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SystemState:
    """Global system state singleton."""
    def __init__(self):
        self.market_data = {}
        self.trades = []
        self.metrics = {
            'total_profit': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_latency_ms': 0.0,
            'last_update': None
        }
        self.alerts = []
        
state = SystemState()

class Trade(BaseModel):
    """Trade model."""
    timestamp: datetime
    symbol: str
    side: str
    price: float
    quantity: float
    profit: float
    latency_ms: float

class Alert(BaseModel):
    """Alert model."""
    timestamp: datetime
    level: str
    message: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "running"}

@app.get("/status")
async def get_status():
    """Get current system status."""
    return {
        "status": "running",
        "last_update": state.metrics['last_update'],
        "total_profit": state.metrics['total_profit'],
        "win_rate": state.metrics['win_rate'],
        "total_trades": state.metrics['total_trades'],
        "avg_latency_ms": state.metrics['avg_latency_ms']
    }

@app.get("/market-data")
async def get_market_data():
    """Get current market data."""
    return state.market_data

@app.get("/trades")
async def get_trades(limit: int = 100):
    """Get recent trades."""
    return state.trades[-limit:]

@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return state.metrics

@app.get("/alerts")
async def get_alerts(limit: int = 100):
    """Get system alerts."""
    return state.alerts[-limit:]

@app.post("/update-state")
async def update_state(
    market_data: Optional[Dict] = None,
    trades: Optional[List[Trade]] = None,
    metrics: Optional[Dict] = None,
    alerts: Optional[List[Alert]] = None
):
    """Update system state."""
    if market_data is not None:
        state.market_data.update(market_data)
    
    if trades is not None:
        state.trades.extend(trades)
        # Keep only last 1000 trades
        state.trades = state.trades[-1000:]
    
    if metrics is not None:
        state.metrics.update(metrics)
        state.metrics['last_update'] = datetime.now()
    
    if alerts is not None:
        state.alerts.extend(alerts)
        # Keep only last 1000 alerts
        state.alerts = state.alerts[-1000:]
    
    return {"status": "updated"} 