import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from .ml_model import MLPredictor
from .execution_simulator import ExecutionSimulator
from .latency_arbitrage import LatencyArbitrageStrategy

logger = logging.getLogger(__name__)

@dataclass
class MarketEvent:
    timestamp: float
    exchange: str
    symbol: str
    price: float
    volume: float
    side: str
    orderbook: Dict

@dataclass
class BacktestTrade:
    timestamp: float
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    volume: float
    fees: float
    latency_ms: float
    profit: float
    success: bool

class Backtester:
    def __init__(self,
                 data_path: str,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 base_latency_ms: float = 5.0,
                 fee_rate: float = 0.001,
                 min_profit_threshold: float = 0.001,
                 max_position: float = 1.0):
        
        self.data_path = Path(data_path)
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.base_latency_ms = base_latency_ms
        self.fee_rate = fee_rate
        self.min_profit_threshold = min_profit_threshold
        self.max_position = max_position
        
        # Components
        self.ml_model = MLPredictor()
        self.execution_simulator = ExecutionSimulator(
            base_latency_ms=base_latency_ms,
            fee_rate=fee_rate
        )
        self.strategy = LatencyArbitrageStrategy(
            symbols=symbols,
            fee_rate=fee_rate,
            min_profit_threshold=min_profit_threshold,
            max_position=max_position
        )
        
        # State
        self.market_state: Dict[str, Dict] = defaultdict(dict)
        self.equity_curve: List[Tuple[float, float]] = []
        self.trades: List[BacktestTrade] = []
        self.current_capital = initial_capital
        
        # Performance metrics
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
    async def load_historical_data(self) -> List[MarketEvent]:
        """Load and preprocess historical market data."""
        events = []
        
        try:
            # Load data for each symbol and exchange
            for symbol in self.symbols:
                # Load Binance data
                binance_file = self.data_path / f"binance_{symbol.lower()}_trades.csv"
                if binance_file.exists():
                    df_binance = pd.read_csv(binance_file)
                    for _, row in df_binance.iterrows():
                        events.append(MarketEvent(
                            timestamp=row['timestamp_ms'] / 1000.0,
                            exchange='binance',
                            symbol=symbol,
                            price=row['price'],
                            volume=row['volume'],
                            side=row['side'],
                            orderbook=json.loads(row['orderbook'])
                        ))
                
                # Load Coinbase data
                coinbase_file = self.data_path / f"coinbase_{symbol.lower()}_trades.csv"
                if coinbase_file.exists():
                    df_coinbase = pd.read_csv(coinbase_file)
                    for _, row in df_coinbase.iterrows():
                        events.append(MarketEvent(
                            timestamp=row['timestamp_ms'] / 1000.0,
                            exchange='coinbase',
                            symbol=symbol,
                            price=row['price'],
                            volume=row['volume'],
                            side=row['side'],
                            orderbook=json.loads(row['orderbook'])
                        ))
            
            # Sort events by timestamp
            events.sort(key=lambda x: x.timestamp)
            logger.info(f"Loaded {len(events)} market events")
            
            return events
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise
    
    def _update_market_state(self, event: MarketEvent):
        """Update internal market state with new event."""
        key = f"{event.exchange}_{event.symbol}"
        self.market_state[key] = {
            'timestamp': event.timestamp,
            'price': event.price,
            'volume': event.volume,
            'orderbook': event.orderbook
        }
    
    def _check_arbitrage_opportunity(self, symbol: str) -> Optional[Dict]:
        """Check for arbitrage opportunity between exchanges."""
        binance_key = f"binance_{symbol}"
        coinbase_key = f"coinbase_{symbol}"
        
        if binance_key not in self.market_state or coinbase_key not in self.market_state:
            return None
        
        binance_data = self.market_state[binance_key]
        coinbase_data = self.market_state[coinbase_key]
        
        # Calculate price gap
        price_gap = (binance_data['price'] - coinbase_data['price']) / coinbase_data['price']
        
        # If gap exceeds threshold, return opportunity details
        if abs(price_gap) > self.min_profit_threshold:
            return {
                'timestamp': max(binance_data['timestamp'], coinbase_data['timestamp']),
                'symbol': symbol,
                'buy_exchange': 'coinbase' if price_gap > 0 else 'binance',
                'sell_exchange': 'binance' if price_gap > 0 else 'coinbase',
                'buy_price': coinbase_data['price'] if price_gap > 0 else binance_data['price'],
                'sell_price': binance_data['price'] if price_gap > 0 else coinbase_data['price'],
                'gap_bps': price_gap * 10000
            }
        
        return None
    
    async def _simulate_execution(self, opportunity: Dict) -> Optional[BacktestTrade]:
        """Simulate trade execution with latency and slippage."""
        try:
            # Get market state for execution
            buy_key = f"{opportunity['buy_exchange']}_{opportunity['symbol']}"
            sell_key = f"{opportunity['sell_exchange']}_{opportunity['symbol']}"
            
            # Simulate execution on both exchanges
            buy_result = await self.execution_simulator.simulate_execution(
                exchange=opportunity['buy_exchange'],
                symbol=opportunity['symbol'],
                side='buy',
                size=self.max_position,
                price=opportunity['buy_price']
            )
            
            sell_result = await self.execution_simulator.simulate_execution(
                exchange=opportunity['sell_exchange'],
                symbol=opportunity['symbol'],
                side='sell',
                size=self.max_position,
                price=opportunity['sell_price']
            )
            
            if not buy_result or not sell_result:
                return None
            
            # Calculate actual profit
            volume = min(self.max_position, buy_result['size'], sell_result['size'])
            total_fees = buy_result['fees'] + sell_result['fees']
            actual_profit = (sell_result['executed_price'] - buy_result['executed_price']) * volume - total_fees
            
            return BacktestTrade(
                timestamp=opportunity['timestamp'],
                buy_exchange=opportunity['buy_exchange'],
                sell_exchange=opportunity['sell_exchange'],
                symbol=opportunity['symbol'],
                buy_price=buy_result['executed_price'],
                sell_price=sell_result['executed_price'],
                volume=volume,
                fees=total_fees,
                latency_ms=max(buy_result['latency_ms'], sell_result['latency_ms']),
                profit=actual_profit,
                success=actual_profit > 0
            )
            
        except Exception as e:
            logger.error(f"Error simulating execution: {str(e)}")
            return None
    
    def _update_metrics(self, trade: BacktestTrade):
        """Update performance metrics after a trade."""
        self.total_trades += 1
        if trade.success:
            self.successful_trades += 1
        
        self.total_profit += trade.profit
        self.total_fees += trade.fees
        self.current_capital += trade.profit
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Update equity curve
        self.equity_curve.append((trade.timestamp, self.current_capital))
    
    async def run_backtest(self):
        """Run the backtest simulation."""
        try:
            # Load historical data
            events = await self.load_historical_data()
            
            # Initialize ML model
            await self.ml_model.load_model()
            
            # Process events in chronological order
            for event in events:
                # Update market state
                self._update_market_state(event)
                
                # Check for arbitrage opportunity
                opportunity = self._check_arbitrage_opportunity(event.symbol)
                if not opportunity:
                    continue
                
                # Get ML model prediction
                pred_prob = await self.ml_model.predict({
                    'symbol': event.symbol,
                    'timestamp': event.timestamp,
                    'orderbook': event.orderbook,
                    'price': event.price,
                    'volume': event.volume
                })
                
                # If model confirms opportunity, simulate execution
                if pred_prob > 0.6:  # Confidence threshold
                    trade = await self._simulate_execution(opportunity)
                    if trade:
                        self._update_metrics(trade)
                        self.trades.append(trade)
            
            # Calculate final metrics
            self._calculate_final_metrics()
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics."""
        if not self.trades:
            logger.warning("No trades executed in backtest")
            return
        
        # Convert trades to DataFrame for analysis
        df_trades = pd.DataFrame([vars(t) for t in self.trades])
        
        # Calculate metrics
        total_days = (df_trades['timestamp'].max() - df_trades['timestamp'].min()) / (24 * 3600)
        avg_daily_profit = self.total_profit / max(1, total_days)
        
        # Calculate Sharpe Ratio (assuming daily)
        daily_returns = df_trades.groupby(
            pd.to_datetime(df_trades['timestamp'], unit='s').dt.date
        )['profit'].sum() / self.initial_capital
        
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) \
            if len(daily_returns) > 1 else 0
        
        self.metrics = {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate': self.successful_trades / max(1, self.total_trades),
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'profit_factor': abs(self.total_profit / self.total_fees) if self.total_fees else 0,
            'return_on_capital': (self.current_capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_daily_profit': avg_daily_profit,
            'avg_trade_profit': self.total_profit / max(1, self.total_trades),
            'avg_latency_ms': df_trades['latency_ms'].mean()
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        if not self.equity_curve:
            logger.warning("No data to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot equity curve
        times, equity = zip(*self.equity_curve)
        ax1.plot(times, equity, label='Portfolio Value')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        
        # Plot trade profits
        if self.trades:
            profits = [t.profit for t in self.trades]
            ax2.hist(profits, bins=50, alpha=0.75)
            ax2.set_title('Trade Profit Distribution')
            ax2.set_xlabel('Profit')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, output_dir: str):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save trades to CSV
        trades_df = pd.DataFrame([vars(t) for t in self.trades])
        trades_df.to_csv(output_path / 'trades.csv', index=False)
        
        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.to_csv(output_path / 'equity_curve.csv', index=False)
        
        # Save metrics
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save plots
        self.plot_results(save_path=str(output_path / 'results.png'))
        
        logger.info(f"Results saved to {output_dir}") 