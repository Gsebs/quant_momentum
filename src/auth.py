"""
Authentication Module for Quantitative HFT Algorithm.
"""

import os
import logging
from functools import wraps
from flask import request, jsonify, current_app
import jwt
from datetime import datetime, timedelta
import hashlib
import secrets
from typing import Dict, Optional, Callable, Tuple, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get secret key from environment or generate one
SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))

# Default admin users
DEFAULT_ADMIN_USERS = {'admin', 'root'}

def generate_token(username: str) -> str:
    """Generate a JWT token"""
    try:
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(days=1),
            'iat': datetime.utcnow(),
            'is_admin': username in DEFAULT_ADMIN_USERS
        }
        return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    except Exception as e:
        logger.error(f"Error generating token: {str(e)}")
        return None

def verify_token(token: str) -> Optional[Dict]:
    """Verify a JWT token"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return None

def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs) -> Tuple[Union[Dict, Any], int]:
        # Skip auth in test environment
        if os.getenv('TESTING') == 'true':
            return f(*args, **kwargs)
            
        # Handle case where there's no request context
        if not hasattr(request, 'headers'):
            return {'error': 'No request context'}, 401
            
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return {'error': 'No authorization header'}, 401
            
        try:
            token_type, token = auth_header.split()
            if token_type.lower() != 'bearer':
                return {'error': 'Invalid token type'}, 401
                
            payload = verify_token(token)
            if not payload:
                return {'error': 'Invalid token'}, 401
                
            # Add user info to request context
            request.user = payload
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return {'error': 'Authentication failed'}, 401
            
    return decorated

def require_admin(f: Callable) -> Callable:
    """Decorator to require admin privileges for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs) -> Tuple[Union[Dict, Any], int]:
        # Skip admin check in test environment
        if os.getenv('TESTING') == 'true':
            return f(*args, **kwargs)
            
        # First check authentication
        auth_result = require_auth(lambda: None)()
        if isinstance(auth_result, tuple) and auth_result[1] != 200:
            return auth_result
            
        # Check admin status
        if not hasattr(request, 'user') or not is_admin(request.user.get('username')):
            return {'error': 'Admin privileges required'}, 403
            
        return f(*args, **kwargs)
            
    return decorated

def is_admin(username: str) -> bool:
    """Check if a user has admin privileges."""
    # Skip admin check in test environment
    if os.getenv('TESTING') == 'true':
        return False  # Return False in test environment unless explicitly testing admin
        
    # Check against default admin users
    if username in DEFAULT_ADMIN_USERS:
        return True
        
    # Check against configured admin users
    if hasattr(current_app, 'config'):
        admin_users = current_app.config.get('admin_users', set())
        return username in admin_users
        
    return False

class AuthManager:
    """Manages authentication and authorization for the API."""
    
    def __init__(self, config: Dict):
        """Initialize the auth manager with configuration."""
        self.config = config
        self.secret_key = config.get('jwt_secret', SECRET_KEY)
        self.token_expiry = timedelta(hours=config.get('token_expiry_hours', 24))
        self.refresh_token_expiry = timedelta(days=config.get('refresh_token_expiry_days', 7))
        self.admin_users = set(config.get('admin_users', [])) | DEFAULT_ADMIN_USERS
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def generate_token(self, username: str, is_refresh: bool = False) -> str:
        """Generate a JWT token for a user."""
        try:
            expiry = datetime.utcnow() + (self.refresh_token_expiry if is_refresh else self.token_expiry)
            payload = {
                'username': username,
                'exp': expiry,
                'iat': datetime.utcnow(),
                'type': 'refresh' if is_refresh else 'access',
                'is_admin': username in self.admin_users
            }
            
            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm='HS256'
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Error generating token: {str(e)}")
            raise
            
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
            
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {str(e)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error verifying token: {str(e)}")
            return None
            
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate a new access token using a refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                return None
                
            return self.generate_token(payload['username'])
            
        except Exception as e:
            self.logger.error(f"Error refreshing access token: {str(e)}")
            return None
            
    def is_admin(self, username: str) -> bool:
        """Check if a user has admin privileges."""
        return username in self.admin_users

# Initialize auth manager with default configuration
auth_manager = AuthManager({
    'jwt_secret': os.getenv('JWT_SECRET', SECRET_KEY),
    'token_expiry_hours': 24,
    'refresh_token_expiry_days': 7,
    'admin_users': list(DEFAULT_ADMIN_USERS)
})

# ... rest of the existing code ... 