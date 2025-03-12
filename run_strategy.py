"""
This is the main script that ties everything together! It's pretty cool because it:
1. Downloads all the data we need
2. Finds the best stocks to buy using our momentum strategy
3. Tests if our strategy actually works by backtesting
4. Shows us how much money we would have made

The best part is you just run this one script and it does everything automatically.
"""

import logging
from src.strategy import MomentumStrategy
from src.backtest import run_backtest_from_recommendations
from datetime import datetime, timedelta

# Set up logging so we can see what's happening as it runs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    This is where all the magic happens! The function:
    1. Sets up our momentum strategy
    2. Runs it to find the best stocks to buy
    3. Tests how well it would have worked in the past year
    4. Shows us all the important stats (returns, risks, etc.)
    
    I made it super easy to use - just run this and it does everything for you.
    """
    try:
        # First, let's set up our strategy
        strategy = MomentumStrategy()
        
        # Run it and get our stock picks
        # This part downloads data, calculates momentum, and ranks stocks
        signals = strategy.run()
        
        # Save our picks so we can look at them later
        logger.info("Saved our picks to data/momentum_signals.xlsx")
        
        # Now for the fun part - testing if our strategy actually works!
        logger.info("Testing our strategy...")
        
        # We'll test it over the last year
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Run the backtest with our top 10 picks
        # This simulates trading these stocks over the past year
        result = run_backtest_from_recommendations(
            recommendations_file='data/momentum_signals.xlsx',
            start_date=start_date,
            end_date=end_date,
            top_n=10  # We'll trade the top 10 stocks
        )
        
        # Show how well we did - all the important numbers
        logger.info(f"Here's how we did:")
        logger.info(f"Started with: ${result.metrics['initial_capital']:,.2f}")
        logger.info(f"Ended with: ${result.metrics['final_value']:,.2f}")
        logger.info(f"Total return: {result.metrics['total_return']:.4f}")
        logger.info(f"Yearly return: {result.metrics['annualized_return']:.4f}")
        logger.info(f"Biggest drop: {result.metrics['max_drawdown']:.4f}")
        logger.info(f"How much we traded: {result.metrics['avg_turnover']:.4f}")
        logger.info(f"Number of trades: {result.metrics['num_trades']}")
        
    except Exception as e:
        logger.error(f"Oops, something went wrong: {str(e)}")
        raise

if __name__ == "__main__":
    main() 