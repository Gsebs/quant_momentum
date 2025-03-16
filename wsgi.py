import os
import sys
import logging

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

try:
    # Import the Flask app
    from app import app
    logger.info("Successfully imported app")
    
    # Initialize data and Redis connection
    from app import initialize_data, initialize_portfolio_state
    initialize_data()
    initialize_portfolio_state()
    logger.info("Successfully initialized data and Redis connection")
    
except Exception as e:
    logger.error(f"Error in WSGI initialization: {str(e)}")
    raise

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 