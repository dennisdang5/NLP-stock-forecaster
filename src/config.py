"""
Central configuration where every file imports its settings from here
"""
from pathlib import Path

# Prediction targets
TICKERS = ['AAPL', 'MSFT', 'NVDA']
START_DATE = '2024-07-01'
END_DATE = '2025-07-01'

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PRICES_DIR = PROJECT_ROOT / 'data' / 'raw' / 'prices'
RAW_NEWS_DIR = PROJECT_ROOT / 'data' / 'raw' / 'news'
INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

# Directory check
for directory in (RAW_PRICES_DIR, RAW_NEWS_DIR, INTERIM_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

