import os
import logging
import redis
from typing import Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Get Redis URL from environment variable
REDIS_URL = os.getenv('REDIS_URL')

if not REDIS_URL:
    logger.error("REDIS_URL environment variable not set")
    redis_client = None
else:
    try:
        redis_client = redis.from_url(REDIS_URL)
        logger.info("Successfully connected to Redis")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {str(e)}")
        redis_client = None

def set_cache(key: str, value: Any, expiry: int = 3600) -> bool:
    """Set a value in Redis cache."""
    if not redis_client:
        return False
    try:
        redis_client.setex(key, expiry, str(value))
        return True
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False

def get_cache(key: str) -> Optional[str]:
    """Get a value from Redis cache."""
    if not redis_client:
        return None
    try:
        value = redis_client.get(key)
        return value.decode('utf-8') if value else None
    except Exception as e:
        logger.error(f"Error getting cache: {str(e)}")
        return None 