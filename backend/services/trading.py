import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel
from decimal import Decimal

class OrderConfig(BaseModel):
    symbol: str
    side: str
    amount: float
    price: float
    exchange: str
    order_type: str = 'limit'

class TradeResult(BaseModel):
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    exchange: str
    timestamp: datetime
    status: str
    profit_loss: Optional[float] = None

class TradingService:
    def __init__(self, exchanges: List[str] = None):
        self.exchanges = exchanges or ['binance', 'coinbase']
        self.exchange_instances: Dict[str, ccxt.Exchange] = {}
        self.active_trades: Dict[str, TradeResult] = {}
        self.logger = logging.getLogger(__name__)
        self.risk_limits = {
            'max_position_size': 1.0,  # Maximum position size in BTC
            'max_daily_loss': 1000.0,  # Maximum daily loss in USD
            'max_drawdown': 0.1,       # Maximum drawdown (10%)
            'min_profit_threshold': 0.1 # Minimum profit threshold in USD
        }
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0

    async def initialize(self, api_keys: Dict[str, Dict[str, str]]):
        """Initialize exchange connections with API keys"""
        for exchange_id in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': api_keys.get(exchange_id, {}).get('api_key'),
                    'secret': api_keys.get(exchange_id, {}).get('secret'),
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                self.exchange_instances[exchange_id] = exchange
                await exchange.load_markets()
                self.logger.info(f"Initialized {exchange_id} exchange")
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_id}: {str(e)}")

    async def execute_arbitrage(self, opportunity: Dict) -> Optional[TradeResult]:
        """Execute arbitrage trade based on opportunity"""
        try:
            # Check risk limits
            if not self._check_risk_limits(opportunity):
                self.logger.warning("Risk limits exceeded, skipping trade")
                return None

            # Execute buy order
            buy_order = await self._execute_order(OrderConfig(
                symbol=opportunity['symbol'],
                side='buy',
                amount=opportunity['amount'],
                price=opportunity['buy_price'],
                exchange=opportunity['buy_exchange']
            ))

            if not buy_order:
                return None

            # Execute sell order
            sell_order = await self._execute_order(OrderConfig(
                symbol=opportunity['symbol'],
                side='sell',
                amount=opportunity['amount'],
                price=opportunity['sell_price'],
                exchange=opportunity['sell_exchange']
            ))

            if not sell_order:
                # Handle failed sell order (implement recovery strategy)
                await self._handle_failed_sell(buy_order)
                return None

            # Calculate profit/loss
            profit_loss = (sell_order.price - buy_order.price) * buy_order.amount
            self.daily_pnl += profit_loss
            self._update_balance(profit_loss)

            trade_result = TradeResult(
                order_id=f"{buy_order.order_id}_{sell_order.order_id}",
                symbol=opportunity['symbol'],
                side='arbitrage',
                amount=buy_order.amount,
                price=buy_order.price,
                exchange=f"{buy_order.exchange}_{sell_order.exchange}",
                timestamp=datetime.now(),
                status='completed',
                profit_loss=profit_loss
            )

            self.active_trades[trade_result.order_id] = trade_result
            return trade_result

        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            return None

    async def _execute_order(self, order_config: OrderConfig) -> Optional[TradeResult]:
        """Execute a single order"""
        try:
            exchange = self.exchange_instances[order_config.exchange]
            order = await exchange.create_order(
                symbol=order_config.symbol,
                type=order_config.order_type,
                side=order_config.side,
                amount=order_config.amount,
                price=order_config.price
            )

            return TradeResult(
                order_id=order['id'],
                symbol=order_config.symbol,
                side=order_config.side,
                amount=order['amount'],
                price=order['price'],
                exchange=order_config.exchange,
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                status=order['status']
            )

        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None

    def _check_risk_limits(self, opportunity: Dict) -> bool:
        """Check if the trade complies with risk limits"""
        # Check position size
        if opportunity['amount'] > self.risk_limits['max_position_size']:
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.risk_limits['max_daily_loss']:
            return False

        # Check drawdown
        if self.current_balance < self.peak_balance * (1 - self.risk_limits['max_drawdown']):
            return False

        # Check minimum profit threshold
        expected_profit = (opportunity['sell_price'] - opportunity['buy_price']) * opportunity['amount']
        if expected_profit < self.risk_limits['min_profit_threshold']:
            return False

        return True

    def _update_balance(self, profit_loss: float):
        """Update balance and peak balance"""
        self.current_balance += profit_loss
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

    async def _handle_failed_sell(self, buy_order: TradeResult):
        """Handle failed sell order (implement recovery strategy)"""
        # Implement recovery strategy (e.g., market sell, cancel buy order)
        pass

    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchange_instances.values():
            await exchange.close()

    def get_trade_history(self) -> List[TradeResult]:
        """Get history of all trades"""
        return list(self.active_trades.values())

    def get_performance_metrics(self) -> Dict:
        """Get trading performance metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            'total_trades': len(self.active_trades),
            'winning_trades': len([t for t in self.active_trades.values() if t.profit_loss and t.profit_loss > 0]),
            'losing_trades': len([t for t in self.active_trades.values() if t.profit_loss and t.profit_loss < 0])
        } 