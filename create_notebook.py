import nbformat as nbf
from src import config
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
cells = [
    nbf.v4.new_markdown_cell("""# Quantitative Momentum Strategy
This notebook implements a quantitative momentum strategy that:
1. Calculates momentum signals for S&P 500 stocks
2. Ranks stocks based on momentum
3. Selects top stocks for the portfolio
4. Runs a backtest to evaluate performance"""),
    
    nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.strategy import MomentumStrategy
from src.backtest import backtest_strategy
from src import config
from datetime import datetime, timedelta

# Set up the strategy
strategy = MomentumStrategy()

# Run strategy to generate signals
signals = strategy.run()

# Get tickers from signals
tickers = signals.index.tolist()

# Print first few tickers
print("First 5 tickers:", tickers[:5])

# Run backtest
backtest = backtest_strategy(
    tickers=tickers[:config.TOP_N_STOCKS],
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    initial_capital=config.INITIAL_CAPITAL
)"""),
    
    nbf.v4.new_code_cell("""# Display performance metrics
metrics = backtest.metrics
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.index.name = 'Metric'
print("Performance Metrics:")
print(metrics_df)

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(backtest.portfolio_values, label='Portfolio Value')
plt.title('Strategy Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
notebook_path = os.path.join('notebooks', '002_quantitative_momentum_strategy.ipynb')
with open(notebook_path, 'w') as f:
    nbf.write(nb, f) 