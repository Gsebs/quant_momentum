import os
<<<<<<< HEAD
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
=======
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

# Global cache instance
cache = None

def init_cache():
    """Initialize the Redis cache."""
    global cache
    if cache is not None:
        return cache
        
    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        logger.error("REDIS_URL environment variable not set")
        return None
        
    try:
        cache = RedisCache(redis.from_url(redis_url, decode_responses=True))
        return cache
    except Exception as e:
        logger.error(f"Error initializing Redis cache: {e}")
        return None

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
    """Get Redis client with retries"""
    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        logger.error("REDIS_URL environment variable not set")
        return None
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()  # Test connection
            return client
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("Max Redis connection retries reached")
                return None

_redis_client = None

def get_redis() -> Optional[redis.Redis]:
    """Get or create Redis client singleton"""
    global _redis_client
    if _redis_client is None:
        _redis_client = get_redis_client()
    return _redis_client

def set_cache(key: str, value: Any, expiry: int = 300) -> bool:
    """Set cache with retries and error handling"""
    client = get_redis()
    if not client:
        return False

    try:
        # Convert value to JSON string
        json_value = json.dumps(value)
        client.set(key, json_value, ex=expiry)
        return True
    except Exception as e:
        logger.error(f"Error setting cache for key {key}: {str(e)}")
        return False

def get_cache(key: str) -> Optional[Any]:
    """Get cache with retries and error handling"""
    client = get_redis()
    if not client:
        return None

    try:
        value = client.get(key)
        if value is None:
            return None
        return json.loads(value)
    except Exception as e:
        logger.error(f"Error getting cache for key {key}: {str(e)}")
        return None

def clear_cache(key: str = None) -> bool:
    """Clear cache with retries and error handling"""
    client = get_redis()
    if not client:
        return False

    try:
        if key:
            client.delete(key)
        else:
            client.flushall()
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False

# Initialize Redis client
redis_client = get_redis()

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

class RedisCache:
    def __init__(self, redis_client: redis.Redis):
        """Initialize Redis cache"""
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour default TTL
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL"""
        try:
            # Convert value to JSON string
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
                
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.set(key, value)
            
            # Set TTL if provided
            if ttl is not None:
                pipe.expire(key, ttl)
            elif self.default_ttl > 0:
                pipe.expire(key, self.default_ttl)
                
            pipe.execute()
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error setting {key}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
            
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache"""
        try:
            value = self.redis.get(key)
            if value is None:
                return default
                
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.decode('utf-8')
                
        except redis.RedisError as e:
            logger.error(f"Redis error getting {key}: {str(e)}")
            return default
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return default
            
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache"""
        try:
            return bool(self.redis.delete(key))
        except redis.RedisError as e:
            logger.error(f"Redis error deleting {key}: {str(e)}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache"""
        try:
            return bool(self.redis.exists(key))
        except redis.RedisError as e:
            logger.error(f"Redis error checking existence of {key}: {str(e)}")
            return False
            
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in the cache"""
        try:
            return self.redis.incr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis error incrementing {key}: {str(e)}")
            return None
            
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key"""
        try:
            return bool(self.redis.expire(key, ttl))
        except redis.RedisError as e:
            logger.error(f"Redis error setting expiration for {key}: {str(e)}")
            return False
            
    async def get_market_state(self, symbol: str) -> Optional[Dict]:
        """Get current market state for a symbol"""
        try:
            state = await self.get(f'market_state:{symbol}')
            if not state:
                return None
                
            # Add TTL information
            ttl = self.redis.ttl(f'market_state:{symbol}')
            if ttl > 0:
                state['ttl'] = ttl
                
            return state
            
        except Exception as e:
            logger.error(f"Error getting market state for {symbol}: {str(e)}")
            return None
            
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list:
        """Get recent trades for a symbol"""
        try:
            # Get trade keys sorted by timestamp
            pattern = f'trade:*'
            trade_keys = self.redis.keys(pattern)
            trades = []
            
            for key in sorted(trade_keys, reverse=True)[:limit]:
                trade = await self.get(key)
                if trade and trade.get('symbol') == symbol:
                    trades.append(trade)
                    
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {str(e)}")
            return []
            
    async def cleanup_old_data(self, max_age: int = 86400) -> int:
        """Clean up old data from cache"""
        try:
            deleted = 0
            cutoff = datetime.now() - timedelta(seconds=max_age)
            
            # Scan all keys
            for key in self.redis.scan_iter("*"):
                try:
                    # Check if key has TTL
                    ttl = self.redis.ttl(key)
                    if ttl < 0:  # No TTL set
                        # Try to parse timestamp from key
                        try:
                            key_str = key.decode('utf-8')
                            if ':' in key_str:
                                timestamp = float(key_str.split(':')[1])
                                if datetime.fromtimestamp(timestamp) < cutoff:
                                    self.redis.delete(key)
                                    deleted += 1
                        except ValueError:
                            continue
                            
                except redis.RedisError:
                    continue
                    
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return 0
            
    async def get_statistics(self) -> Dict:
        """Get cache statistics"""
        try:
            info = self.redis.info()
            return {
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_keys': len(self.redis.keys('*')),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return {} 
>>>>>>> heroku/main
