import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

class PriceData(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    exchange: str
    volume: float
    bid: float
    ask: float

class MarketDataService:
    def __init__(self, exchanges: List[str] = None):
        self.exchanges = exchanges or ['binance', 'coinbase']
        self.exchange_instances: Dict[str, ccxt.Exchange] = {}
        self.price_data: Dict[str, Dict[str, PriceData]] = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize exchange connections"""
        for exchange_id in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                self.exchange_instances[exchange_id] = exchange
                self.price_data[exchange_id] = {}
                await exchange.load_markets()
                self.logger.info(f"Initialized {exchange_id} exchange")
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_id}: {str(e)}")

    async def fetch_ticker(self, exchange_id: str, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data for a specific symbol from an exchange"""
        try:
            exchange = self.exchange_instances[exchange_id]
            ticker = await exchange.fetch_ticker(symbol)
            
            price_data = PriceData(
                symbol=symbol,
                price=ticker['last'],
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                exchange=exchange_id,
                volume=ticker['quoteVolume'],
                bid=ticker['bid'],
                ask=ticker['ask']
            )
            
            self.price_data[exchange_id][symbol] = price_data
            return price_data
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} from {exchange_id}: {str(e)}")
            return None

    async def start_price_feed(self, symbols: List[str]):
        """Start continuous price feed for specified symbols"""
        while True:
            for exchange_id in self.exchange_instances:
                for symbol in symbols:
                    await self.fetch_ticker(exchange_id, symbol)
            await asyncio.sleep(1)  # Adjust frequency as needed

    def get_price_data(self, exchange_id: str, symbol: str) -> Optional[PriceData]:
        """Get the latest price data for a symbol from an exchange"""
        return self.price_data.get(exchange_id, {}).get(symbol)

    def get_all_prices(self, symbol: str) -> Dict[str, PriceData]:
        """Get the latest price data for a symbol from all exchanges"""
        return {
            exchange_id: data[symbol]
            for exchange_id, data in self.price_data.items()
            if symbol in data
        }

    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchange_instances.values():
            await exchange.close()

    def find_arbitrage_opportunities(self, symbol: str, min_profit_threshold: float = 0.1) -> List[Dict]:
        """Find arbitrage opportunities across exchanges"""
        opportunities = []
        all_prices = self.get_all_prices(symbol)
        
        for buy_exchange, buy_data in all_prices.items():
            for sell_exchange, sell_data in all_prices.items():
                if buy_exchange != sell_exchange:
                    price_difference = sell_data.bid - buy_data.ask
                    if price_difference > min_profit_threshold:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'buy_price': buy_data.ask,
                            'sell_price': sell_data.bid,
                            'price_difference': price_difference,
                            'timestamp': datetime.now()
                        })
        
        return opportunities 