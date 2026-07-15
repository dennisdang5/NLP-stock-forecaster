"""
Combines all per month chunk files into the final headlines.parquet file
"""

import pandas as pd

from src.collect_news import CHUNK_DIR
from src.config import RAW_NEWS_DIR

CHUNK_DIR = RAW_NEWS_DIR / 'chunks'

chunks = []
for file in sorted(CHUNK_DIR.glob('*.parquet')):
    chunk = pd.read_parquet(file)
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Drop any duplicate headlines just in case
df = df.drop_duplicates(subset=['ticker', 'published_at', 'headline'])
df = df.sort_values(['ticker', 'published_at']).reset_index(drop=True)

df.to_parquet(RAW_NEWS_DIR / 'headlines.parquet')
print(f'Combined {len(chunks)} chunk files into {len(df)} headlines')