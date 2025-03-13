from datetime import datetime
from src.backtest import run_backtest_from_recommendations

if __name__ == "__main__":
    print("Starting backtest...")
    
    # Run backtest using the recommendations file
    result = run_backtest_from_recommendations(
        recommendations_file="data/momentum_signals.xlsx",
        start_date="2023-01-01",  # Using last year of data
        end_date=datetime.now().strftime("%Y-%m-%d"),
        top_n=10,
        initial_capital=100000
    )
    
    # Print backtest results
    print("\nBacktest Results:")
    metrics = result.summary()
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Average Turnover: {metrics.get('avg_turnover', 0):.2%}")
    print(f"Number of Trades: {metrics.get('num_trades', 0)}")
    
    print("\nPerformance plot has been saved to data/reports/performance_plot.png") 