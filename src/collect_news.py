import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from src.config import TICKERS, START_DATE, END_DATE, RAW_NEWS_DIR

load_dotenv()
API_KEY = os.getenv('EODHD_API_KEY')

"""
Get raw news for one ticker
"""
def fetch_news(ticker):
    response = requests.get(
        'https://eodhd.com/api/news',
        params= {
            's': f'{ticker}.US',
            'from':START_DATE,
            'to':END_DATE,
            'limit':1000,
            'offset':0,
            'api_token':API_KEY,
            'fmt':'json'
        },
        timeout= 30
    )
    response.raise_for_status()
    return response.json()

all_news = [] # List of ALL tickers and their data
for ticker in TICKERS:
    articles = fetch_news(ticker)
    for article in articles:
        all_news.append({
            'ticker': ticker,
            'published_at': pd.to_datetime(article['date'], utc=True),
            'headline': article['title'],
            'url': article.get('link', '')
        })
    print(f'{ticker}: {len(articles)} headlines') # Number of headlines for that specific ticker
    time.sleep(1.5) # Prevent api throttling

df = pd.DataFrame(all_news)
df.to_parquet(RAW_NEWS_DIR / 'headlines.parquet')
print(f'\nTotal: {len(df)} headlines saved')
