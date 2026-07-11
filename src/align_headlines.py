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

news['target_date'] = pd.to_datetime(news['published_eastern_time'].apply(points_to))

# Match each headline's target date with the correct real trading day either at or after the date it points to and attaches the session's up/down label
labels = labels.copy()
labels['day'] = pd.to_datetime(labels['day'])

news = news.sort_values('target_date')
labels = labels.sort_values('day')

aligned = pd.merge_asof(
    news, labels,
    left_on='target_date', right_on='day',
    by='ticker',
    direction='forward' # find the next trading day >= target_date
)

# Drop headlines with no future session yet
before = len(aligned)
aligned = aligned.dropna(subset=['label'])
aligned['label'] = aligned['label'].astype(int)
dropped = before - len(aligned)

aligned.to_parquet(INTERIM_DIR / 'aligned_headlines.parquet')
print(f'Aligned {len(aligned)} headlines ({dropped} dropped as too recent)')