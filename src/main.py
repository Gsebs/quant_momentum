import asyncio
import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .market_making import MarketMakingStrategy

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize strategy
strategy = MarketMakingStrategy()

@app.get("/")
async def root():
    return {"status": "running", "message": "Market Making Strategy API"}

@app.post("/start")
async def start_strategy():
    try:
        asyncio.create_task(strategy.run())
        return {"status": "success", "message": "Strategy started"}
    except Exception as e:
        logger.error(f"Error starting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    try:
        order_book = await strategy.get_order_book()
        if order_book:
            bid_vwap, ask_vwap, spread = await strategy.calculate_optimal_spread(order_book)
            return {
                "status": "success",
                "bid_vwap": bid_vwap,
                "ask_vwap": ask_vwap,
                "spread": spread
            }
        return {"status": "error", "message": "Could not fetch order book"}
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 