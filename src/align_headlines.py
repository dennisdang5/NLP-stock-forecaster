import pandas as pd
from src.config import INTERIM_DIR, RAW_NEWS_DIR

"""
Attaches the correct up/down label to each headline based on the headlines next trading IFF the headline came out during the trading day.
Otherwise, the headline released before the trading day (9:30 ET) will be assigned a label for that current day
"""

# Load both datasets
news = pd.read_parquet(RAW_NEWS_DIR / 'headlines.parquet')
labels = pd.read_parquet(INTERIM_DIR / 'daily_labels.parquet')

# Convert timezone of UTC to Eastern time and establish market open time
news['published_eastern_time'] = news['published_at'].dt.tz_convert('America/New_York')
market_open = pd.Timestamp('9:30').time()

# Function to determine which day the timestamp news headline should predict
# Before the open -> predict today's sessions
# After the open -> predict the next day
def points_to(timestamp):
    if timestamp.time() < market_open:
        return timestamp.date()
    return (timestamp + pd.Timedelta(days=1)).date()
