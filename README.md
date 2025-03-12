# Quantitative Momentum Trading Strategy

A sophisticated Python implementation of a quantitative momentum trading strategy targeting S&P 500 stocks. This project combines traditional momentum indicators with machine learning to identify and capitalize on market trends.

## Features

- **Multi-timeframe Momentum Analysis**
  - 1-month, 3-month, 6-month, and 12-month return calculations
  - Weighted momentum scoring system
  - Volatility-adjusted returns

- **Technical Indicators**
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Volatility measures
  - Trend strength indicators

- **Advanced Analytics**
  - Machine learning signal enhancement
  - Risk-adjusted position sizing
  - Sector exposure analysis
  - Comprehensive backtesting system

- **Performance Analysis**
  - Equity curve visualization
  - Risk metrics calculation
  - Trade analysis reporting
  - Performance attribution

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quant_momentum.git
   cd quant_momentum
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data notebooks
   ```

## Usage

1. Configure strategy parameters in `src/config.py`

2. Run the strategy:
   ```bash
   python create_notebook.py
   jupyter notebook notebooks/002_quantitative_momentum_strategy.ipynb
   ```

3. View results in the `data` directory:
   - `momentum_signals.xlsx`: Detailed signal analysis
   - `momentum_report.xlsx`: Strategy performance report

## Project Structure

```
quant_momentum/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── create_notebook.py        # Notebook generation script
├── src/                     # Source code
│   ├── __init__.py
│   ├── strategy.py          # Core momentum strategy
│   ├── backtest.py         # Backtesting engine
│   ├── data.py             # Data acquisition
│   ├── config.py           # Configuration parameters
│   ├── ml_model.py         # ML signal enhancement
│   ├── indicators.py       # Technical indicators
│   └── risk.py             # Risk management
├── data/                   # Data storage (created at runtime)
└── notebooks/              # Jupyter notebooks
    └── 002_quantitative_momentum_strategy.ipynb
```

## Strategy Details

### Signal Generation
- Calculates momentum across multiple timeframes
- Incorporates technical indicators for trend confirmation
- Adjusts signals based on volatility and market conditions

### Position Sizing
- Risk-adjusted position sizing based on momentum strength
- Volatility scaling for risk management
- Sector exposure limits

### Risk Management
- Stop-loss implementation
- Position size limits
- Sector diversification rules
- Volatility-based position adjustment

## Performance Metrics

The strategy tracks various performance metrics:
- Total and annualized returns
- Sharpe and Sortino ratios
- Maximum drawdown
- Win/loss ratio
- Average trade profitability
- Portfolio turnover

## Data Sources

- Stock price data: Yahoo Finance
- S&P 500 constituents: Wikipedia
- Market data: Various public APIs

## Disclaimer

This software is for educational and research purposes only. It is not intended to be investment advice. Trading stocks carries significant risks, and you should carefully consider your financial condition before making any investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
