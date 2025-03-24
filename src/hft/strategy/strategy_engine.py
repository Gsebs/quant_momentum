import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import xgboost as xgb
from ..data_feed.market_data import market_data
from ..simulation.trade_simulator import execute_trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HFTStrategy:
    def __init__(self, 
                 threshold: float = 0.005,  # 0.5% price difference threshold
                 min_confidence: float = 0.7,  # Minimum ML confidence to execute trade
                 max_position_size: float = 1.0,  # Maximum position size in BTC
                 latency_ms: int = 50,  # Expected latency in milliseconds
                 slippage_pct: float = 0.0005):  # 0.05% slippage
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.latency_ms = latency_ms
        self.slippage_pct = slippage_pct
        self.running = False
        self.last_confidence = 0.0
        self.model = self._load_model()
        self.trade_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized HFT Strategy with threshold={threshold}, min_confidence={min_confidence}")

    def _load_model(self) -> Optional[xgb.Booster]:
        """Load the trained XGBoost model for arbitrage prediction."""
        try:
            model = xgb.Booster()
            model.load_model("model.xgb")
            logger.info("Successfully loaded ML model")
            return model
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}. Using dummy predictions.")
            return None

    def _calculate_features(self, binance_data: Dict[str, Any], coinbase_data: Dict[str, Any]) -> np.ndarray:
        """Calculate features for the ML model."""
        if not binance_data or not coinbase_data:
            return np.array([])

        price_b = binance_data.get("price", 0)
        price_c = coinbase_data.get("price", 0)
        
        if price_b == 0 or price_c == 0:
            return np.array([])

        # Calculate price difference and percentage
        diff = price_c - price_b
        diff_pct = diff / price_b if price_b != 0 else 0

        # Calculate time difference between updates
        time_b = binance_data.get("last_update", 0)
        time_c = coinbase_data.get("last_update", 0)
        time_diff = abs(time_b - time_c) / 1000  # Convert to seconds

        # Calculate volatility (price change over time)
        volatility = abs(diff_pct) / time_diff if time_diff > 0 else 0

        # Features array: [price_b, price_c, diff_pct, time_diff, volatility]
        features = np.array([price_b, price_c, diff_pct, time_diff, volatility])
        return features

    def _predict_confidence(self, features: np.ndarray) -> float:
        """Use ML model to predict confidence in the arbitrage opportunity."""
        if not self.model or len(features) == 0:
            # Dummy prediction based on price difference
            return min(1.0, max(0.0, abs(features[2]) * 100))  # features[2] is diff_pct

        try:
            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            pred = self.model.predict(dmatrix)
            return float(pred[0])
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.0

    async def _execute_arbitrage(self, 
                               buy_exchange: str,
                               sell_exchange: str,
                               buy_price: float,
                               sell_price: float,
                               confidence: float) -> None:
        """Execute an arbitrage trade with the given parameters."""
        try:
            # Calculate position size based on confidence
            position_size = self.max_position_size * confidence

            # Execute the trade
            profit = await execute_trade(
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=position_size
            )

            # Record trade
            trade_record = {
                "timestamp": datetime.now().timestamp(),
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "position_size": position_size,
                "profit": profit,
                "confidence": confidence
            }
            self.trade_history.append(trade_record)
            logger.info(f"Executed arbitrage trade: {trade_record}")

        except Exception as e:
            logger.error(f"Error executing arbitrage trade: {e}")

    async def run(self):
        """Main strategy loop that monitors prices and executes trades."""
        self.running = True
        logger.info("Starting HFT strategy")

        while self.running:
            try:
                binance_data = market_data.get("binance", {})
                coinbase_data = market_data.get("coinbase", {})

                if binance_data and coinbase_data:
                    # Calculate features for ML model
                    features = self._calculate_features(binance_data, coinbase_data)
                    
                    if len(features) > 0:
                        # Get ML prediction
                        confidence = self._predict_confidence(features)
                        self.last_confidence = confidence

                        # Check if price difference exceeds threshold
                        diff_pct = features[2]  # Price difference percentage
                        if abs(diff_pct) > self.threshold and confidence >= self.min_confidence:
                            price_b = features[0]  # Binance price
                            price_c = features[1]  # Coinbase price

                            # Determine trade direction
                            if diff_pct > 0:
                                # Coinbase price higher: buy on Binance, sell on Coinbase
                                await self._execute_arbitrage(
                                    buy_exchange="binance",
                                    sell_exchange="coinbase",
                                    buy_price=price_b,
                                    sell_price=price_c,
                                    confidence=confidence
                                )
                            else:
                                # Binance price higher: buy on Coinbase, sell on Binance
                                await self._execute_arbitrage(
                                    buy_exchange="coinbase",
                                    sell_exchange="binance",
                                    buy_price=price_c,
                                    sell_price=price_b,
                                    confidence=confidence
                                )

                # Short sleep to yield control
                await asyncio.sleep(0.001)  # 1ms sleep

            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(1)  # Longer sleep on error

    async def stop(self):
        """Stop the strategy."""
        self.running = False
        logger.info("Stopping HFT strategy")

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get the list of executed trades."""
        return self.trade_history

    def get_last_confidence(self) -> float:
        """Get the last ML model confidence value."""
        return self.last_confidence 