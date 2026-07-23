"""
Walk forward evaluation of logistic classifier
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from src.config import PROCESSED_DIR

daily_features = pd.read_parquet(PROCESSED_DIR / 'daily_features.parquet')
daily_features = daily_features.sort_values('day').reset_index(drop=True)

features_columns = ['finbert_pos', 'finbert_neg', 'finbert_neu']
X = daily_features[features_columns]
y = daily_features['label']

# 5 rounds of expanding window train/test
splitter = TimeSeriesSplit(n_splits=5)

model_scores = []
baseline_scores = []

round_num = 1
for split in splitter.split(X):
    train_idx = split[0] # First item is training position
    test_idx = split[1] # Second item is test position

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    logistic_classifier = LogisticRegression(max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    logistic_prediction = logistic_classifier.predict(X_test)

    logistic_accuracy = accuracy_score(y_test, logistic_prediction)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    model_scores.append(logistic_accuracy)
    baseline_scores.append(baseline)

    print(f'Round {round_num}: train={len(train_idx):3d} test={len(test_idx):3d} '
          f'| model={logistic_accuracy:.3f} baseline={baseline:.3f}')

    round_num = round_num + 1

print(f'\nMean model accuracy: {np.mean(model_scores):.3f} '
      f'(std {np.std(model_scores):.3f})')
print(f'Mean baseline: {np.mean(baseline_scores):.3f}')