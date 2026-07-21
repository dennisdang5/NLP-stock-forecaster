"""
Collapses per headline features into one row per ticker day
"""
import pandas as pd
from src.config import PROCESSED_DIR

features = pd.read_parquet(PROCESSED_DIR / 'headlines_features.parquet')

# Group all the headlines that share the same ticker and trading day then summarize each group into a single row via mean
daily_result = features.groupby(['ticker', 'day']).agg(
    finbert_pos = ('finbert_pos', 'mean'),
    finbert_neg = ('finbert_neg', 'mean'),
    finbert_neu = ('finbert_neu', 'mean'),
    headline_count = ('headline', 'count'), # Number of headlines that day
    label = ('label', 'first') # Up/down outcome of that day
).reset_index()

daily_result.to_parquet(PROCESSED_DIR / 'daily_features.parquet')
