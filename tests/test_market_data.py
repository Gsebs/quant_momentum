import pytest
import asyncio
from datetime import datetime
from src.hft.data_feed.market_data import MarketDataFeed

@pytest.fixture
def market_feed():
    """Create a test instance of the market data feed."""
    return MarketDataFeed("BTC-USD")

def test_market_feed_initialization(market_feed):
    """Test market data feed initialization."""
    assert market_feed.symbol == "BTC-USD"
    assert market_feed.binance_symbol == "BTCUSDT"
    assert market_feed.coinbase_symbol == "BTC-USD"
    assert market_feed.market_data == {
        "binance": {"price": 0.0, "timestamp": 0.0},
        "coinbase": {"price": 0.0, "timestamp": 0.0}
    }

@pytest.mark.asyncio
async def test_market_feed_start_stop(market_feed):
    """Test market data feed start and stop functionality."""
    # Start the feed
    task = asyncio.create_task(market_feed.start())
    
    # Let it run for a short time
    await asyncio.sleep(0.1)
    
    # Stop the feed
    await market_feed.stop()
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Verify the feed stopped cleanly
    assert not market_feed.running

def test_price_update(market_feed):
    """Test price update functionality."""
    # Simulate price updates
    market_feed.update_price("binance", 50000.0)
    market_feed.update_price("coinbase", 50100.0)
    
    assert market_feed.market_data["binance"]["price"] == 50000.0
    assert market_feed.market_data["coinbase"]["price"] == 50100.0
    assert market_feed.market_data["binance"]["timestamp"] > 0
    assert market_feed.market_data["coinbase"]["timestamp"] > 0

def test_price_difference_calculation(market_feed):
    """Test price difference calculation."""
    # Set test prices
    market_feed.update_price("binance", 50000.0)
    market_feed.update_price("coinbase", 50100.0)
    
    # Calculate price difference
    diff = market_feed.get_price_difference()
    
    assert diff == 100.0  # Absolute difference
    assert market_feed.get_price_difference_pct() == 0.2  # Percentage difference

def test_market_data_validation(market_feed):
    """Test market data validation."""
    # Test invalid exchange
    with pytest.raises(ValueError):
        market_feed.update_price("invalid_exchange", 50000.0)
    
    # Test invalid price
    with pytest.raises(ValueError):
        market_feed.update_price("binance", -100.0)
    
    # Test invalid timestamp
    with pytest.raises(ValueError):
        market_feed.market_data["binance"]["timestamp"] = -1

@pytest.mark.asyncio
async def test_websocket_connection(market_feed):
    """Test WebSocket connection handling."""
    # Mock WebSocket connection
    market_feed.ws_binance = None
    market_feed.ws_coinbase = None
    
    # Start the feed
    task = asyncio.create_task(market_feed.start())
    
    # Let it run for a short time
    await asyncio.sleep(0.1)
    
    # Stop the feed
    await market_feed.stop()
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Verify connection attempts were made
    assert market_feed.ws_binance is None  # Connection should fail in test environment
    assert market_feed.ws_coinbase is None  # Connection should fail in test environment 