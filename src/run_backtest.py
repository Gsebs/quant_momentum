#!/usr/bin/env python3
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json

from .backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_backtest(args):
    """Run backtest with specified parameters."""
    try:
        # Create backtester instance
        backtester = Backtester(
            data_path=args.data_path,
            symbols=args.symbols.split(','),
            initial_capital=args.initial_capital,
            base_latency_ms=args.latency_ms,
            fee_rate=args.fee_rate,
            min_profit_threshold=args.min_profit_bps / 10000,
            max_position=args.max_position
        )
        
        # Run backtest
        logger.info("Starting backtest...")
        await backtester.run_backtest()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"backtest_{timestamp}"
        
        # Save results
        backtester.save_results(output_dir)
        
        # Print summary
        print("\nBacktest Results Summary:")
        print("=" * 50)
        metrics = backtester.metrics
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:,.2f}")
        print(f"Return on Capital: {metrics['return_on_capital']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Average Trade Profit: ${metrics['avg_trade_profit']:,.2f}")
        print(f"Average Latency: {metrics['avg_latency_ms']:.2f}ms")
        print("=" * 50)
        
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run HFT strategy backtest')
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to historical data directory'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC-USD',
        help='Comma-separated list of trading symbols'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital in USD'
    )
    
    parser.add_argument(
        '--latency-ms',
        type=float,
        default=5.0,
        help='Base latency in milliseconds'
    )
    
    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.001,
        help='Trading fee rate (e.g., 0.001 for 0.1%)'
    )
    
    parser.add_argument(
        '--min-profit-bps',
        type=float,
        default=10.0,
        help='Minimum profit threshold in basis points'
    )
    
    parser.add_argument(
        '--max-position',
        type=float,
        default=1.0,
        help='Maximum position size in base currency'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run backtest
    asyncio.run(run_backtest(args))

if __name__ == '__main__':
    main() 