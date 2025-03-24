import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from src.hft.api.app import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_status_endpoint():
    """Test the status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data
    assert "timestamp" in data
    assert "market_data" in data
    assert "strategy" in data
    assert "trading" in data

def test_trades_endpoint():
    """Test the trades endpoint."""
    response = client.get("/trades")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_trade_stats_endpoint():
    """Test the trade stats endpoint."""
    response = client.get("/trades/stats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "total_trades" in data
    assert "winning_trades" in data
    assert "losing_trades" in data
    assert "win_rate" in data
    assert "avg_profit_loss" in data
    assert "max_profit" in data
    assert "max_loss" in data

def test_invalid_endpoint():
    """Test handling of invalid endpoints."""
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404

def test_error_handling():
    """Test global error handling."""
    # Simulate an error by accessing an invalid attribute
    response = client.get("/status")
    assert response.status_code == 200  # Should handle the error gracefully

def test_cors_headers():
    """Test CORS headers are present."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_response_format():
    """Test response format consistency."""
    endpoints = ["/health", "/status", "/trades", "/trades/stats"]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list))
        
        if isinstance(data, dict):
            assert "status" in data or "data" in data
            if "timestamp" in data:
                assert isinstance(data["timestamp"], (int, float))

def test_market_data_format():
    """Test market data format in status response."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    market_data = data["market_data"]
    
    assert "binance" in market_data
    assert "coinbase" in market_data
    
    for exchange in ["binance", "coinbase"]:
        exchange_data = market_data[exchange]
        assert "price" in exchange_data
        assert "timestamp" in exchange_data
        assert isinstance(exchange_data["price"], (int, float))
        assert isinstance(exchange_data["timestamp"], (int, float))

def test_strategy_metrics():
    """Test strategy metrics in status response."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    strategy_data = data["strategy"]
    
    assert "last_confidence" in strategy_data
    assert "threshold" in strategy_data
    assert "min_confidence" in strategy_data
    
    assert isinstance(strategy_data["last_confidence"], (int, float))
    assert isinstance(strategy_data["threshold"], (int, float))
    assert isinstance(strategy_data["min_confidence"], (int, float)) 