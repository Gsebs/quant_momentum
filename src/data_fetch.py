"""
Functions for fetching and processing stock market data
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Optional, Dict
from . import config

def get_sp500_tickers() -> List[str]:
    """
    Returns a list of S&P 500 tickers.
    First tries to read from local CSV file, if not available,
    attempts to fetch from Wikipedia, and finally falls back to yfinance.
    """
    # Try reading from local CSV first
    csv_path = os.path.join(config.DATA_DIR, "sp500.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "Ticker" in df.columns:
                return df["Ticker"].tolist()
        except Exception as e:
            print(f"Error reading sp500.csv: {str(e)}")
    
    # Try fetching from Wikipedia
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Save to CSV for future use
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)
        sp500_table.to_csv(csv_path, index=False)
        print(f"Saved S&P 500 tickers to {csv_path}")
        
        return tickers
    except Exception as e:
        print(f"Error fetching from Wikipedia: {str(e)}")
    
    # Fallback to yfinance
    try:
        sp500 = yf.Ticker(config.MARKET_INDEX)
        return sp500.info.get('components', [])
    except Exception as e:
        print(f"Error fetching from yfinance: {str(e)}")
        return []

def fetch_historical_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame with historical price data
    """
    start = start_date or config.START_DATE
    end = end_date or config.END_DATE
    
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False
        )
        if data.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return pd.DataFrame()

def fetch_data_for_tickers(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        Dictionary mapping tickers to their historical data
    """
    data_dict = {}
    total = len(tickers)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"Fetching data for {ticker}... ({i}/{total})")
        data = fetch_historical_data(ticker, start_date, end_date, interval)
        if not data.empty:
            data_dict[ticker] = data
            
            # Save individual ticker data
            save_stock_data(data, ticker)
    
    return data_dict

def save_stock_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Save stock data to CSV file
    
    Args:
        df: DataFrame with stock data
        symbol: Stock ticker symbol
    """
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    filename = os.path.join(config.DATA_DIR, f"{symbol}_historical.csv")
    df.to_csv(filename)
    print(f"Saved data for {symbol} to {filename}")

def fetch_and_save_all_stocks() -> None:
    """
    Download and save historical data for all S&P 500 stocks
    """
    tickers = get_sp500_tickers()
    if not tickers:
        print("Error: Could not retrieve S&P 500 tickers")
        return
    
    print(f"Found {len(tickers)} S&P 500 tickers")
    data_dict = fetch_data_for_tickers(tickers)
    print(f"Successfully downloaded data for {len(data_dict)} stocks")

if __name__ == "__main__":
    fetch_and_save_all_stocks() 