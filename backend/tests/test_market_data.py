import pytest
import asyncio
from datetime import datetime
from ..services.market_data import MarketDataService
from ..config.config import settings

@pytest.fixture
async def market_data_service():
    service = MarketDataService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_market_data_service_initialization(market_data_service):
    assert market_data_service.is_running()
    assert len(market_data_service.exchanges) == len(settings.EXCHANGES)

@pytest.mark.asyncio
async def test_price_fetching(market_data_service):
    # Wait for initial price data
    await asyncio.sleep(2)
    
    for symbol in settings.TRADING_PAIRS:
        for exchange in settings.EXCHANGES:
            price = market_data_service.get_latest_price(exchange, symbol)
            assert price is not None
            assert isinstance(price, float)
            assert price > 0

@pytest.mark.asyncio
async def test_opportunity_detection(market_data_service):
    # Wait for initial price data
    await asyncio.sleep(2)
    
    opportunities = market_data_service.get_opportunities()
    assert isinstance(opportunities, list)
    
    if opportunities:
        opportunity = opportunities[0]
        assert 'symbol' in opportunity
        assert 'buy_exchange' in opportunity
        assert 'sell_exchange' in opportunity
        assert 'price_difference' in opportunity
        assert 'estimated_profit' in opportunity
        assert 'confidence' in opportunity

@pytest.mark.asyncio
async def test_error_handling(market_data_service):
    # Test invalid symbol
    price = market_data_service.get_latest_price('binance', 'INVALID/PAIR')
    assert price is None
    
    # Test invalid exchange
    price = market_data_service.get_latest_price('invalid_exchange', 'BTC/USDT')
    assert price is None

@pytest.mark.asyncio
async def test_connection_recovery(market_data_service):
    # Simulate connection loss
    market_data_service.exchanges['binance'].close()
    
    # Wait for reconnection
    await asyncio.sleep(5)
    
    # Check if reconnected
    assert market_data_service.exchanges['binance'].is_connected()

@pytest.mark.asyncio
async def test_opportunity_filtering(market_data_service):
    # Wait for initial price data
    await asyncio.sleep(2)
    
    # Test minimum profit threshold
    opportunities = market_data_service.get_opportunities()
    for opp in opportunities:
        assert opp['estimated_profit'] >= settings.MIN_PROFIT_THRESHOLD
    
    # Test minimum confidence threshold
    for opp in opportunities:
        assert opp['confidence'] >= settings.MIN_CONFIDENCE_THRESHOLD

@pytest.mark.asyncio
async def test_price_update_frequency(market_data_service):
    # Wait for initial price data
    await asyncio.sleep(2)
    
    # Get initial price
    initial_price = market_data_service.get_latest_price('binance', 'BTC/USDT')
    
    # Wait for price update
    await asyncio.sleep(2)
    
    # Get updated price
    updated_price = market_data_service.get_latest_price('binance', 'BTC/USDT')
    
    # Prices should be different
    assert initial_price != updated_price 