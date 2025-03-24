import pytest
import asyncio
from datetime import datetime
from src.hft.strategy.strategy_engine import HFTStrategy
from src.hft.simulation.trade_simulator import TradeSimulator

@pytest.fixture
def strategy():
    """Create a test instance of the HFT strategy."""
    return HFTStrategy(
        threshold=0.5,
        min_confidence=0.7,
        max_position_size=1.0,
        latency_ms=100,
        slippage_pct=0.1
    )

@pytest.fixture
def simulator():
    """Create a test instance of the trade simulator."""
    return TradeSimulator(
        latency_ms=100,
        slippage_pct=0.1,
        fee_pct=0.1,
        initial_balance=10000.0
    )

def test_strategy_initialization(strategy):
    """Test strategy initialization with correct parameters."""
    assert strategy.threshold == 0.5
    assert strategy.min_confidence == 0.7
    assert strategy.max_position_size == 1.0
    assert strategy.latency_ms == 100
    assert strategy.slippage_pct == 0.1
    assert strategy.trades == []
    assert strategy.last_confidence == 0.0

def test_feature_calculation(strategy):
    """Test feature calculation from market data."""
    market_data = {
        "binance": {"price": 50000.0, "timestamp": datetime.now().timestamp()},
        "coinbase": {"price": 50100.0, "timestamp": datetime.now().timestamp()}
    }
    
    features = strategy.calculate_features(market_data)
    
    assert len(features) == 5  # We expect 5 features
    assert features[0] == 0.2  # Price difference percentage
    assert features[1] == 50000.0  # Binance price
    assert features[2] == 50100.0  # Coinbase price
    assert features[3] == 100.0  # Price difference
    assert features[4] == 0.0  # Time difference (simulated)

@pytest.mark.asyncio
async def test_trade_execution(simulator):
    """Test trade execution with the simulator."""
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
async def test_strategy_loop(strategy):
    """Test the main strategy loop."""
    # Mock market data
    market_data = {
        "binance": {"price": 50000.0, "timestamp": datetime.now().timestamp()},
        "coinbase": {"price": 50100.0, "timestamp": datetime.now().timestamp()}
    }
    
    # Start the strategy loop
    task = asyncio.create_task(strategy.run())
    
    # Let it run for a short time
    await asyncio.sleep(0.1)
    
    # Stop the strategy
    await strategy.stop()
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Verify the strategy stopped cleanly
    assert not strategy.running 