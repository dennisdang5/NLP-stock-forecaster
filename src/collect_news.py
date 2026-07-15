import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from src.config import TICKERS, START_DATE, END_DATE, RAW_NEWS_DIR

"""
Pulls EODHD news month by month such that one file per ticker-month
"""

load_dotenv()
API_KEY = os.getenv('EODHD_API_KEY')

# Per month file location
CHUNK_DIR = RAW_NEWS_DIR / 'chunks'
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Build (from, to, label) for each month in the config range
month_starts = pd.date_range(START_DATE, END_DATE, freq='MS')
WINDOWS = [(s.strftime('%Y-%m-%d'),
           (s + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d'),
            s.strftime('%Y-%m'))
           for s in month_starts]

"""
Get raw news for one ticker
"""
def fetch_news(ticker, date_from, date_to):
    response = requests.get(
        'https://eodhd.com/api/news',
        params= {
            's': f'{ticker}.US',
            'from':date_from,
            'to':date_to,
            'limit':1000,
            'offset':0,
            'api_token':API_KEY,
            'fmt':'json'
        },
        timeout= 30
    )
    response.raise_for_status()
    return response.json()

for ticker in TICKERS:
    for date_from, date_to, month_label in WINDOWS:
        chunk_path = CHUNK_DIR / f'{ticker}_{month_label}.parquet'

        # Skip anything that has been collected already
        if chunk_path.exists():
            continue

        articles = fetch_news(ticker, date_from, date_to)
        rows = []
        for article in articles:
            row = {
                'ticker': ticker,
                'published_at': pd.to_datetime(article['date'], utc=True),
                'headline': article['title'],
                'url': article.get('link', '')
            }
            rows.append(row)

        pd.DataFrame(rows).to_parquet(chunk_path)
        print(f'saved {ticker} {month_label}: {len(rows)} headlines')
        time.sleep(1.5)

print(f'\nDone with this run')