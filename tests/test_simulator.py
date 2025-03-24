import pytest
import asyncio
from datetime import datetime
from src.hft.simulation.trade_simulator import TradeSimulator

@pytest.fixture
def simulator():
    """Create a test instance of the trade simulator."""
    return TradeSimulator(
        latency_ms=100,
        slippage_pct=0.1,
        fee_pct=0.1,
        initial_balance=10000.0
    )

def test_simulator_initialization(simulator):
    """Test trade simulator initialization."""
    assert simulator.latency_ms == 100
    assert simulator.slippage_pct == 0.1
    assert simulator.fee_pct == 0.1
    assert simulator.initial_balance == 10000.0
    assert simulator.balance == 10000.0
    assert simulator.cumulative_pnl == 0.0
    assert simulator.trades == []

@pytest.mark.asyncio
async def test_trade_execution(simulator):
    """Test trade execution with the simulator."""
    # Execute a profitable trade
    trade_result = await simulator.execute_trade(
        buy_exchange="binance",
        sell_exchange="coinbase",
        buy_price=50000.0,
        sell_price=50100.0,
        quantity=1.0
    )
    
    assert trade_result["success"] is True
    assert trade_result["buy_price"] == 50000.0
    assert trade_result["sell_price"] == 50100.0
    assert trade_result["quantity"] == 1.0
    assert trade_result["fees"] > 0
    assert trade_result["net_profit"] > 0
    assert len(simulator.trades) == 1
    
    # Verify balance and PnL updates
    assert simulator.balance > simulator.initial_balance
    assert simulator.cumulative_pnl > 0

@pytest.mark.asyncio
async def test_trade_execution_with_loss(simulator):
    """Test trade execution with a loss."""
    # Execute a losing trade
    trade_result = await simulator.execute_trade(
        buy_exchange="coinbase",
        sell_exchange="binance",
        buy_price=50100.0,
        sell_price=50000.0,
        quantity=1.0
    )
    
    assert trade_result["success"] is True
    assert trade_result["net_profit"] < 0
    assert simulator.balance < simulator.initial_balance
    assert simulator.cumulative_pnl < 0

def test_trade_stats(simulator):
    """Test trade statistics calculation."""
    # Add some test trades
    simulator.trades = [
        {
            "timestamp": datetime.now().timestamp(),
            "buy_exchange": "binance",
            "sell_exchange": "coinbase",
            "buy_price": 50000.0,
            "sell_price": 50100.0,
            "quantity": 1.0,
            "fees": 100.0,
            "net_profit": 900.0
        },
        {
            "timestamp": datetime.now().timestamp(),
            "buy_exchange": "coinbase",
            "sell_exchange": "binance",
            "buy_price": 50100.0,
            "sell_price": 50000.0,
            "quantity": 1.0,
            "fees": 100.0,
            "net_profit": -900.0
        }
    ]
    
    stats = simulator.get_trade_stats()
    
    assert stats["total_trades"] == 2
    assert stats["winning_trades"] == 1
    assert stats["losing_trades"] == 1
    assert stats["win_rate"] == 0.5
    assert stats["avg_profit_loss"] == 0.0
    assert stats["max_profit"] == 900.0
    assert stats["max_loss"] == -900.0

@pytest.mark.asyncio
async def test_latency_simulation(simulator):
    """Test latency simulation in trade execution."""
    start_time = datetime.now()
    
    await simulator.execute_trade(
        buy_exchange="binance",
        sell_exchange="coinbase",
        buy_price=50000.0,
        sell_price=50100.0,
        quantity=1.0
    )
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1000
    
    # Allow for some timing variance
    assert execution_time >= simulator.latency_ms
    assert execution_time <= simulator.latency_ms * 1.5

def test_slippage_calculation(simulator):
    """Test slippage calculation in trade execution."""
    # Test buy slippage
    buy_price = 50000.0
    effective_buy = simulator.calculate_effective_price(buy_price, "buy")
    assert effective_buy > buy_price  # Buy price should be higher due to slippage
    
    # Test sell slippage
    sell_price = 50100.0
    effective_sell = simulator.calculate_effective_price(sell_price, "sell")
    assert effective_sell < sell_price  # Sell price should be lower due to slippage

def test_fee_calculation(simulator):
    """Test fee calculation in trade execution."""
    price = 50000.0
    quantity = 1.0
    
    # Calculate fees
    buy_fee = simulator.calculate_fees(price, quantity)
    assert buy_fee == price * quantity * (simulator.fee_pct / 100)
    
    # Test total cost including fees
    total_cost = simulator.calculate_total_cost(price, quantity)
    assert total_cost == (price * quantity) + buy_fee 