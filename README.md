# Momentum Trading Strategy

This is a Python-based momentum trading strategy that I built to find and trade stocks with strong momentum. It uses both traditional technical indicators and machine learning to make better trading decisions.

## Features

- **Data Collection**: Automatically downloads S&P 500 stock data using yfinance
- **Momentum Indicators**: Calculates various momentum metrics (RSI, MACD, returns over different timeframes)
- **Risk Management**: Includes position sizing, sector exposure limits, and volatility adjustments
- **Machine Learning**: Enhances trading signals using ML models
- **Backtesting**: Tests the strategy on historical data with realistic transaction costs
- **Performance Analysis**: Calculates key metrics like Sharpe ratio, drawdowns, and returns

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd quant_momentum
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Just run the main script:
```bash
python run_strategy.py
```

This will:
1. Download stock data
2. Calculate momentum signals
3. Generate trade recommendations
4. Run a backtest
5. Show performance metrics

## Project Structure

- `src/strategy.py`: Core momentum strategy implementation
- `src/backtest.py`: Backtesting engine
- `src/risk.py`: Risk management functions
- `src/data.py`: Data collection utilities
- `run_strategy.py`: Main script that ties everything together

## Results

The strategy looks at multiple factors:
- Returns over different timeframes (1M, 3M, 6M, 12M)
- Technical indicators (RSI, MACD)
- Volume trends
- Volatility
- Sector exposure

It then ranks stocks based on these factors and picks the best ones to trade.

## Contributing

Feel free to open issues or submit pull requests if you have ideas for improvements!
