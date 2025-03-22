from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory cache
cache = {}

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="HFT Strategy API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    return {
        "status": "healthy",
        "cache": "in-memory"
    }

@app.get("/api/metrics")
@limiter.limit("1000/hour")
async def get_metrics(request: Request):
    """Get system metrics."""
    return {
        "status": "ok",
        "cache_size": len(cache)
    }

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port) 