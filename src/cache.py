import os
import redis
import requests_cache
from datetime import timedelta
from functools import wraps
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

# Configure Redis with better connection handling and pooling
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_retry = Retry(
    ExponentialBackoff(cap=1.0, base=0.5),  # Increased retry parameters
    5  # Increased max retries
)

def get_redis_client():
    try:
        client = redis.from_url(
            redis_url,
            ssl_cert_reqs=None,
            decode_responses=True,  # Always decode responses
            socket_timeout=15,
            socket_connect_timeout=15,
            retry_on_timeout=True,
            retry=redis_retry,
            max_connections=50,
            health_check_interval=15
        )
        # Test connection
        client.ping()
        return client
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to Redis: {str(e)}")
        return None

redis_client = get_redis_client()

# Initialize requests-cache for Yahoo Finance API calls
requests_cache.install_cache(
    'yfinance_cache',
    backend='memory',
    expire_after=timedelta(hours=6)
)

def serialize_value(value: Any) -> str:
    """Serialize value for Redis storage."""
    try:
        if isinstance(value, pd.DataFrame):
            return json.dumps({
                'type': 'dataframe',
                'data': value.to_dict(orient='split')
            })
        elif isinstance(value, np.ndarray):
            return json.dumps({
                'type': 'ndarray',
                'data': value.tolist()
            })
        elif isinstance(value, (dict, list)):
            return json.dumps({
                'type': 'json',
                'data': value
            })
        else:
            return json.dumps({
                'type': 'primitive',
                'data': value
            })
    except Exception as e:
        logger.error(f"Error serializing value: {str(e)}")
        return json.dumps({'type': 'error', 'data': str(e)})

def deserialize_value(value_str: str) -> Any:
    """Deserialize value from Redis storage."""
    try:
        if not isinstance(value_str, str):
            logger.error(f"Expected string value, got {type(value_str)}")
            return None
            
        value_dict = json.loads(value_str)
        value_type = value_dict.get('type')
        value_data = value_dict.get('data')

        if value_type == 'dataframe':
            return pd.DataFrame(**value_data)
        elif value_type == 'ndarray':
            return np.array(value_data)
        elif value_type == 'json':
            return value_data
        elif value_type == 'error':
            logger.error(f"Deserialized error value: {value_data}")
            return None
        else:
            return value_data
    except Exception as e:
        logger.error(f"Error deserializing value: {str(e)}")
        return None

def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

def redis_cache(expire_time: int = 3600):
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
                    result = deserialize_value(cached_result)
                    if result is not None:
                        return result
                    logger.warning(f"Could not deserialize cached value for {key}")
                
                # If not in cache or deserialization failed, execute function
                result = func(*args, **kwargs)
                
                # Cache the result
                try:
                    serialized_result = serialize_value(result)
                    redis_client.setex(key, expire_time, serialized_result)
                    logger.info(f"Cache miss for {key}, stored new result")
                except Exception as e:
                    logger.warning(f"Failed to cache result for {key}: {str(e)}")
                
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
    """Clear all cached data"""
    try:
        redis_client.flushall()
        requests_cache.clear()
        logger.info("Cache cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False 