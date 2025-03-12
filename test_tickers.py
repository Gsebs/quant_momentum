from src.data import get_sp500_tickers
import logging

logging.basicConfig(level=logging.INFO)

tickers = get_sp500_tickers()
print(f"Number of tickers: {len(tickers)}")
print("First 10 tickers:", tickers[:10] if tickers else "No tickers found") 