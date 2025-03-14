import os
import redis
import requests_cache
from datetime import timedelta
from functools import wraps
import json
import logging

logger = logging.getLogger(__name__)

# Initialize Redis connection
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL)

# Initialize requests-cache for Yahoo Finance API calls
requests_cache.install_cache(
    'yfinance_cache',
    backend='memory',
    expire_after=timedelta(hours=6)
)

def cache_key(*args, **kwargs):
    """Generate a cache key from function arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

def redis_cache(expire_time=3600):
    """Redis caching decorator with expiration time in seconds."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            try:
                # Try to get cached result
                cached_result = redis_client.get(key)
                if cached_result:
                    logger.info(f"Cache hit for {key}")
                    return json.loads(cached_result)
                
                # If not in cache, execute function and cache result
                result = func(*args, **kwargs)
                redis_client.setex(key, expire_time, json.dumps(result))
                logger.info(f"Cache miss for {key}, stored new result")
                return result
                
            except redis.RedisError as e:
                logger.warning(f"Redis error: {str(e)}, falling back to direct execution")
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Unexpected error in cache layer: {str(e)}")
                raise
                
        return wrapper
    return decorator

def clear_cache():
    """Clear all cached data."""
    try:
        redis_client.flushall()
        requests_cache.clear()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise 