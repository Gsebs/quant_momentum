import pytest
import asyncio
from datetime import datetime
from ..services.trading import TradingService
from ..services.market_data import MarketDataService
from ..config.config import settings

@pytest.fixture
async def market_data_service():
    service = MarketDataService()
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
async def trading_service(market_data_service):
    service = TradingService(market_data_service)
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_trading_service_initialization(trading_service):
    assert trading_service.is_running()
    assert trading_service.market_data_service is not None

@pytest.mark.asyncio
async def test_opportunity_evaluation(trading_service):
    # Create a test opportunity
    opportunity = {
        'symbol': 'BTC/USDT',
        'buy_exchange': 'binance',
        'sell_exchange': 'coinbase',
        'buy_price': 50000.0,
        'sell_price': 50100.0,
        'price_difference': 100.0,
        'estimated_profit': 0.1,
        'confidence': 0.9
    }
    
    # Evaluate the opportunity
    should_execute = trading_service.evaluate_opportunity(opportunity)
    assert isinstance(should_execute, bool)

@pytest.mark.asyncio
async def test_risk_management(trading_service):
    # Test position size limits
    position_size = trading_service.calculate_position_size('BTC/USDT', 50000.0)
    assert position_size <= settings.MAX_POSITION_SIZE
    
    # Test daily loss limit
    assert trading_service.check_daily_loss_limit(100.0)  # Should be within limit
    assert not trading_service.check_daily_loss_limit(10000.0)  # Should exceed limit

@pytest.mark.asyncio
async def test_order_execution(trading_service):
    # Create a test order
    order = {
        'symbol': 'BTC/USDT',
        'exchange': 'binance',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0
    }
    
    # Test order execution
    result = await trading_service.execute_order(order)
    assert result is not None
    assert 'status' in result
    assert 'order_id' in result

@pytest.mark.asyncio
async def test_error_handling(trading_service):
    # Test invalid order
    invalid_order = {
        'symbol': 'INVALID/PAIR',
        'exchange': 'binance',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0
    }
    
    result = await trading_service.execute_order(invalid_order)
    assert result['status'] == 'error'
    assert 'error_message' in result

@pytest.mark.asyncio
async def test_trade_history(trading_service):
    # Execute a test trade
    order = {
        'symbol': 'BTC/USDT',
        'exchange': 'binance',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0
    }
    
    await trading_service.execute_order(order)
    
    # Check trade history
    trades = trading_service.get_trade_history()
    assert len(trades) > 0
    assert trades[-1]['symbol'] == order['symbol']
    assert trades[-1]['exchange'] == order['exchange']

@pytest.mark.asyncio
async def test_performance_metrics(trading_service):
    # Execute some test trades
    orders = [
        {
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000.0
        },
        {
            'symbol': 'BTC/USDT',
            'exchange': 'coinbase',
            'side': 'sell',
            'amount': 0.001,
            'price': 50100.0
        }
    ]
    
    for order in orders:
        await trading_service.execute_order(order)
    
    # Check performance metrics
    metrics = trading_service.get_performance_metrics()
    assert 'total_trades' in metrics
    assert 'winning_trades' in metrics
    assert 'total_profit_loss' in metrics
    assert 'win_rate' in metrics

@pytest.mark.asyncio
async def test_connection_recovery(trading_service):
    # Simulate connection loss
    trading_service.market_data_service.exchanges['binance'].close()
    
    # Wait for reconnection
    await asyncio.sleep(5)
    
    # Check if reconnected
    assert trading_service.market_data_service.exchanges['binance'].is_connected() 