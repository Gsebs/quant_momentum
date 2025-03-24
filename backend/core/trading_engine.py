import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from ..services.market_data import MarketDataService
from ..services.trading import TradingService

class TradingEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.market_data_service = MarketDataService(
            exchanges=config.get('exchanges', ['binance', 'coinbase'])
        )
        self.trading_service = TradingService(
            exchanges=config.get('exchanges', ['binance', 'coinbase'])
        )
        self.is_running = False
        self.trading_pairs = config.get('trading_pairs', ['BTC/USDT', 'ETH/USDT'])
        self.min_profit_threshold = config.get('min_profit_threshold', 0.1)
        self.opportunity_queue = asyncio.Queue()

    async def initialize(self):
        """Initialize the trading engine and its components"""
        try:
            # Initialize market data service
            await self.market_data_service.initialize()
            
            # Initialize trading service with API keys
            await self.trading_service.initialize(self.config.get('api_keys', {}))
            
            self.logger.info("Trading engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise

    async def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return

        self.is_running = True
        self.logger.info("Starting trading engine")

        try:
            # Start market data feed
            market_data_task = asyncio.create_task(
                self.market_data_service.start_price_feed(self.trading_pairs)
            )

            # Start opportunity detection
            opportunity_detection_task = asyncio.create_task(
                self._detect_opportunities()
            )

            # Start opportunity processing
            opportunity_processing_task = asyncio.create_task(
                self._process_opportunities()
            )

            # Wait for all tasks
            await asyncio.gather(
                market_data_task,
                opportunity_detection_task,
                opportunity_processing_task
            )

        except Exception as e:
            self.logger.error(f"Error in trading engine: {str(e)}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            self.logger.warning("Trading engine is not running")
            return

        self.is_running = False
        self.logger.info("Stopping trading engine")

        try:
            # Close services
            await self.market_data_service.close()
            await self.trading_service.close()
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {str(e)}")
            raise

    async def _detect_opportunities(self):
        """Continuously detect arbitrage opportunities"""
        while self.is_running:
            try:
                for symbol in self.trading_pairs:
                    opportunities = self.market_data_service.find_arbitrage_opportunities(
                        symbol,
                        self.min_profit_threshold
                    )

                    for opportunity in opportunities:
                        # Add amount based on available balance and risk limits
                        opportunity['amount'] = self._calculate_position_size(opportunity)
                        
                        if opportunity['amount'] > 0:
                            await self.opportunity_queue.put(opportunity)
                            self.logger.info(f"New opportunity detected: {opportunity}")

                await asyncio.sleep(0.1)  # Adjust frequency as needed

            except Exception as e:
                self.logger.error(f"Error detecting opportunities: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

    async def _process_opportunities(self):
        """Process detected opportunities"""
        while self.is_running:
            try:
                opportunity = await self.opportunity_queue.get()
                
                # Execute arbitrage trade
                trade_result = await self.trading_service.execute_arbitrage(opportunity)
                
                if trade_result:
                    self.logger.info(f"Executed arbitrage trade: {trade_result}")
                else:
                    self.logger.warning(f"Failed to execute arbitrage trade for opportunity: {opportunity}")

                self.opportunity_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing opportunity: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

    def _calculate_position_size(self, opportunity: Dict) -> float:
        """Calculate the position size based on available balance and risk limits"""
        try:
            # Get current balance from trading service
            current_balance = self.trading_service.current_balance
            
            # Calculate maximum position size based on risk limits
            max_position = min(
                self.trading_service.risk_limits['max_position_size'],
                current_balance / opportunity['buy_price']
            )
            
            # Calculate expected profit
            expected_profit = (opportunity['sell_price'] - opportunity['buy_price']) * max_position
            
            # Adjust position size if expected profit is too small
            if expected_profit < self.trading_service.risk_limits['min_profit_threshold']:
                return 0.0
            
            return max_position

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def get_status(self) -> Dict:
        """Get the current status of the trading engine"""
        return {
            'is_running': self.is_running,
            'trading_pairs': self.trading_pairs,
            'opportunities_in_queue': self.opportunity_queue.qsize(),
            'performance_metrics': self.trading_service.get_performance_metrics()
        } 