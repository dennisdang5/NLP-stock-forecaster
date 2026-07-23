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


# Train/test split 80/20
split = int(len(daily_features) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train logistic regression (model #1)
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)

# Evaluation
logistic_prediction = logistic_classifier.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_prediction)

# Baseline from guessing majority class in the test set
majority_baseline = max(y_test.mean(), 1 - y_test.mean())

print(f'\nModel accuracy: {accuracy_score:.3f}')
print(f'Majority baseline: {majority_baseline:.3f}')
print(f'\n{classification_report(y_test, logistic_prediction, zero_division=0)}')