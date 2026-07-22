"""
Trains a classifier on daily sentiment features with a chronological split in order to deal with lookahead leakage
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.config import PROCESSED_DIR

daily_features = pd.read_parquet(PROCESSED_DIR / 'daily_features.parquet')

# Sort by day for chronological split
daily_features = daily_features.sort_values('day').reset_index(drop=True)
features_columns = ['finbert_pos', 'finbert_neg', 'finbert_neu']
X = daily_features[features_columns]
y = daily_features['label']