import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from app import app
    logger.info("Successfully imported app")
except Exception as e:
    logger.error(f"Error importing app: {str(e)}")
    raise

if __name__ == "__main__":
    app.run() 