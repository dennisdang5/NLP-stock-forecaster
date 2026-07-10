import pandas as pd
from src.config import TICKERS, INTERIM_DIR, RAW_PRICES_DIR

"""
Labels each trading day up (1) or down (0)
"""

labeled_data = []
for ticker in TICKERS:
    df = pd.read_parquet(RAW_PRICES_DIR / f'{ticker}.parquet')
    df = df.reset_index() # Date becomes a column

    # Normalize the column names to lowercase and deals with tuple format from yfinance (column, ticker)
    new_columns = []
    for column in df.columns:
        if isinstance(column, tuple):
            clean_name = column[0]
        else:
            clean_name = column

        clean_name = str(clean_name).lower()
        new_columns.append(clean_name)
    df.columns = new_columns

    output = pd.DataFrame({
        'ticker': ticker,
        'day': pd.to_datetime(df['date']).dt.date,
        'open_to_close': (df['close'] - df['open']) / df['open']
    })
    output['label'] = (output['open_to_close'] >= 0).astype(int) # Compares every value in column 'open_to_close' to >= 0 if true it gets type cast to 1, false otherwise
    labeled_data.append(output)

labels = pd.concat(labeled_data, ignore_index=True)
labels.to_parquet(INTERIM_DIR / 'daily_labels.parquet')

print(labels.head())
print()
print(labels['label'].value_counts()) # Naive baseline for further comparison