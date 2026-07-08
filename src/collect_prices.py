"""
Pull daily stock prices and saves as a parquet file in data/raw/prices
"""
import yfinance as yf
from src.config import TICKERS, START_DATE, END_DATE, RAW_PRICES_DIR

for ticker in TICKERS:
    stock_df = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', auto_adjust=True)
    stock_df.to_parquet(RAW_PRICES_DIR / f'{ticker}.parquet')
    print(f'{ticker}: {len(stock_df)} rows saved')