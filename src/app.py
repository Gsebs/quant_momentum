from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis cache with error handling
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0
    )
    redis_client.ping()  # Test the connection
    logger.info("Successfully connected to Redis")
except (redis.ConnectionError, redis.ResponseError) as e:
    logger.warning(f"Failed to connect to Redis: {e}")
    redis_client = None

app = FastAPI(title="HFT Strategy API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "ok", "message": "HFT Strategy API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status
    }

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port) 