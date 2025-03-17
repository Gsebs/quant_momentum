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
import urllib.parse
import time

logger = logging.getLogger(__name__)

# Configure Redis with better connection handling and pooling
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Parse Redis URL to handle authentication properly
def parse_redis_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
        return {
            'host': parsed.hostname or 'localhost',
            'port': int(parsed.port or 6379),
            'username': parsed.username,
            'password': parsed.password,
            'db': int(parsed.path.lstrip('/') or 0)
        }
    except Exception as e:
        logger.error(f"Error parsing Redis URL: {str(e)}")
        return None

def get_redis_client():
    try:
        redis_config = parse_redis_url(redis_url)
        if not redis_config:
            logger.error("Failed to parse Redis URL")
            return None

        # Configure retry strategy
        retry_strategy = Retry(
            ExponentialBackoff(cap=1.0, base=0.5),
            5
        )

        # Create Redis client with robust configuration
        client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            username=redis_config['username'],
            password=redis_config['password'],
            db=redis_config['db'],
            ssl=True,  # Always use SSL for Heroku Redis
            ssl_cert_reqs=None,  # Don't verify SSL cert
            decode_responses=True,
            socket_timeout=30,  # Increased timeouts
            socket_connect_timeout=30,
            socket_keepalive=True,
            retry_on_timeout=True,
            retry=retry_strategy,
            max_connections=20,  # Reduced max connections
            health_check_interval=30
        )

        # Test connection with retry
        for attempt in range(3):
            try:
                client.ping()
                logger.info("Successfully connected to Redis")
                return client
            except redis.ConnectionError as e:
                if attempt == 2:  # Last attempt
                    raise
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                time.sleep(1)  # Wait before retry
                
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to Redis: {str(e)}")
        return None

# Initialize Redis client
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

def get_redis():
    """Get Redis client, attempting to reconnect if necessary"""
    global redis_client
    try:
        if redis_client is None or not redis_client.ping():
            redis_client = get_redis_client()
        return redis_client
    except Exception as e:
        logger.error(f"Error getting Redis client: {str(e)}")
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
            client = get_redis()
            if client is None:
                logger.warning(f"Redis unavailable, falling back to direct execution")
                return func(*args, **kwargs)
                
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            try:
                # Try to get cached result
                cached_result = client.get(key)
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
                    client.setex(key, expire_time, serialized_result)
                    logger.info(f"Cache miss for {key}, stored new result")
                except Exception as e:
                    logger.warning(f"Failed to cache result for {key}: {str(e)}")
                
                return result
                
            except redis.RedisError as e:
                logger.warning(f"Redis error: {str(e)}, falling back to direct execution")
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Unexpected error in cache layer: {str(e)}")
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

def clear_cache():
    """Clear all cached data"""
    try:
        client = get_redis()
        if client:
            client.flushall()
            requests_cache.clear()
            logger.info("Cache cleared successfully")
            return True
        else:
            logger.error("Redis client not available")
            return False
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False 