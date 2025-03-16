import os
import sys
import logging
from time import sleep
from redis.exceptions import ConnectionError, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def initialize_with_retry(max_retries=3, retry_delay=5):
    """Initialize application with retry logic"""
    for attempt in range(max_retries):
        try:
            # Import the Flask app
            from app import app
            logger.info("Successfully imported app")
            
            # Initialize data and Redis connection
            from app import initialize_data, initialize_portfolio_state
            from src.cache import redis_client
            
            if redis_client is None:
                raise ConnectionError("Failed to establish Redis connection")
                
            initialize_data()
            initialize_portfolio_state()
            logger.info("Successfully initialized data and Redis connection")
            return app
            
        except (ConnectionError, TimeoutError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            else:
                logger.error(f"Failed to initialize after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error in WSGI initialization: {str(e)}")
            raise

# Initialize application with retry logic
app = initialize_with_retry()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 