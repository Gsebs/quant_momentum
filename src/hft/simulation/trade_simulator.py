import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradeSimulator:
    def __init__(self,
                 latency_ms: int = 50,  # Expected latency in milliseconds
                 slippage_pct: float = 0.0005,  # 0.05% slippage
                 fee_pct: float = 0.001,  # 0.1% fee per trade
                 initial_balance: float = 10000.0):  # Initial balance in USD
        self.latency_ms = latency_ms
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        self.balance = initial_balance
        self.trades: List[Dict[str, Any]] = []
        self.cumulative_pnl = 0.0
        logger.info(f"Initialized Trade Simulator with latency={latency_ms}ms, slippage={slippage_pct*100}%")

    async def execute_trade(self,
                          buy_exchange: str,
                          sell_exchange: str,
                          buy_price: float,
                          sell_price: float,
                          quantity: float) -> float:
        """
        Simulate executing a trade with latency, slippage, and fees.
        Returns the profit/loss from the trade.
        """
        try:
            # Simulate network/exchange latency
            await asyncio.sleep(self.latency_ms / 1000.0)  # Convert ms to seconds

            # Apply slippage
            exec_buy_price = buy_price * (1 + self.slippage_pct)  # Price increases when buying
            exec_sell_price = sell_price * (1 - self.slippage_pct)  # Price decreases when selling

            # Calculate fees
            buy_fee = exec_buy_price * self.fee_pct * quantity
            sell_fee = exec_sell_price * self.fee_pct * quantity
            total_fees = buy_fee + sell_fee

            # Calculate profit/loss
            gross_profit = (exec_sell_price - exec_buy_price) * quantity
            net_profit = gross_profit - total_fees

            # Update balance and cumulative PnL
            self.balance += net_profit
            self.cumulative_pnl += net_profit

            # Record trade
            trade_record = {
                "timestamp": datetime.now().timestamp(),
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": exec_buy_price,
                "sell_price": exec_sell_price,
                "quantity": quantity,
                "gross_profit": gross_profit,
                "fees": total_fees,
                "net_profit": net_profit,
                "balance": self.balance
            }
            self.trades.append(trade_record)

            logger.info(f"Executed trade: Buy {quantity} BTC on {buy_exchange} at ${exec_buy_price:,.2f}, "
                       f"Sell on {sell_exchange} at ${exec_sell_price:,.2f}, "
                       f"Net Profit: ${net_profit:,.2f}")

            return net_profit

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return 0.0

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get the list of executed trades."""
        return self.trades

    def get_cumulative_pnl(self) -> float:
        """Get the cumulative profit/loss."""
        return self.cumulative_pnl

    def get_balance(self) -> float:
        """Get the current balance."""
        return self.balance

    def get_trade_stats(self) -> Dict[str, Any]:
        """Get statistics about executed trades."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }

        profits = [t["net_profit"] for t in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) * 100,
            "avg_profit": sum(profits) / len(self.trades),
            "max_profit": max(profits),
            "max_loss": min(profits),
            "avg_win": sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(losing_trades) / len(losing_trades) if losing_trades else 0
        } 