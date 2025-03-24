import asyncio
import logging
import numpy as np
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared state and simulator
from market_data_feed import latest_prices, reliable_tickers
from trade_simulator import execute_trade

# Global state for tracking predictions
last_confidence = 0.0  # store last prediction confidence (0 to 1)
last_prediction_time = None  # timestamp of last prediction

# Strategy parameters
THRESHOLD = 0.005  # 0.5% price difference threshold
MIN_CONFIDENCE = 0.5  # minimum confidence to execute trade
MIN_TIME_BETWEEN_TRADES = 1.0  # minimum seconds between trades
MAX_POSITION_SIZE = 1.0  # maximum position size in base currency

# Try to import an ML model (for example, using XGBoost)
model = None
try:
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("model.xgb")
    logger.info("ML model loaded successfully")
except Exception as e:
    model = None
    logger.warning(f"ML model not loaded, will use dummy predictions: {e}")

class LatencyArbitrageStrategy:
    def __init__(self):
        self.price_history: Dict[str, list] = {
            "binance": [],
            "coinbase": []
        }
        self.max_history_size = 100  # keep last 100 price points
        self.last_trade_time = 0
        self.positions: Dict[str, float] = {
            "binance": 0.0,
            "coinbase": 0.0
        }

    def update_price_history(self, exchange: str, price: float):
        """Update price history for an exchange."""
        self.price_history[exchange].append(price)
        if len(self.price_history[exchange]) > self.max_history_size:
            self.price_history[exchange].pop(0)

    def calculate_volatility(self, exchange: str) -> float:
        """Calculate price volatility for an exchange."""
        prices = self.price_history[exchange]
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def calculate_spread(self) -> Optional[float]:
        """Calculate current spread between exchanges."""
        if not latest_prices["binance"] or not latest_prices["coinbase"]:
            return None
        return latest_prices["coinbase"] - latest_prices["binance"]

    def calculate_execution_probability(self, spread_pct: float, volatility: float) -> float:
        """Calculate probability of successful trade execution."""
        if model:
            try:
                # Prepare features for model
                features = np.array([[
                    spread_pct,
                    volatility,
                    len(self.price_history["binance"]),
                    len(self.price_history["coinbase"])
                ]])
                dmatrix = xgb.DMatrix(features)
                return float(model.predict(dmatrix)[0])
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                return 0.0
        else:
            # Dummy confidence: if spread is significantly large, set high confidence
            return min(1.0, max(0.0, abs(spread_pct) * 100))

    async def should_execute_trade(self, spread_pct: float, volatility: float) -> bool:
        """Determine if we should execute a trade based on current conditions."""
        global last_confidence, last_prediction_time

        # Check minimum time between trades
        current_time = datetime.now().timestamp()
        if current_time - self.last_trade_time < MIN_TIME_BETWEEN_TRADES:
            return False

        # Calculate execution probability
        confidence = self.calculate_execution_probability(spread_pct, volatility)
        last_confidence = confidence
        last_prediction_time = current_time

        # Check if confidence meets threshold
        if confidence < MIN_CONFIDENCE:
            return False

        # Check position limits
        if any(abs(pos) >= MAX_POSITION_SIZE for pos in self.positions.values()):
            return False

        return True

    async def execute_arbitrage_trade(self, buy_exchange: str, sell_exchange: str):
        """Execute an arbitrage trade between exchanges."""
        try:
            # Get current prices
            buy_price = latest_prices[buy_exchange]
            sell_price = latest_prices[sell_exchange]
            
            if not buy_price or not sell_price:
                logger.error("Missing prices for trade execution")
                return

            # Execute trade
            profit = await execute_trade(
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price
            )

            # Update positions
            self.positions[buy_exchange] += 1.0
            self.positions[sell_exchange] -= 1.0
            self.last_trade_time = datetime.now().timestamp()

            logger.info(f"Executed arbitrage trade: {profit:.2f} profit")
            return profit

        except Exception as e:
            logger.error(f"Error executing arbitrage trade: {e}")
            return None

# Create global strategy instance
strategy = LatencyArbitrageStrategy()

async def run_strategy():
    """Continuously check for arbitrage opportunities and execute trades."""
    while True:
        try:
            # Update price history
            if latest_prices["binance"]:
                strategy.update_price_history("binance", latest_prices["binance"])
            if latest_prices["coinbase"]:
                strategy.update_price_history("coinbase", latest_prices["coinbase"])

            # Calculate spread and volatility
            spread = strategy.calculate_spread()
            if spread is not None:
                spread_pct = spread / latest_prices["binance"]
                volatility = strategy.calculate_volatility("binance")

                # Check if we should trade
                if await strategy.should_execute_trade(spread_pct, volatility):
                    if spread > 0:
                        # Coinbase price higher: buy on Binance, sell on Coinbase
                        await strategy.execute_arbitrage_trade("binance", "coinbase")
                    else:
                        # Binance price higher: buy on Coinbase, sell on Binance
                        await strategy.execute_arbitrage_trade("coinbase", "binance")

            # Short sleep to yield control
            await asyncio.sleep(0.1)  # 100ms between checks

        except Exception as e:
            logger.error(f"Error in strategy loop: {e}")
            await asyncio.sleep(1)  # Wait longer on error

def get_last_confidence() -> float:
    """Get the last prediction confidence."""
    return last_confidence

def get_last_prediction_time() -> Optional[datetime]:
    """Get the timestamp of the last prediction."""
    if last_prediction_time:
        return datetime.fromtimestamp(last_prediction_time)
    return None 