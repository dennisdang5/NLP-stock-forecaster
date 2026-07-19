"""
Runs headlines through a frozen FinBert and saves sentiment features (pos, neg, neu)
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import INTERIM_DIR, PROCESSED_DIR

# Load frozen FinBert
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
model.eval() # Frozen

aligned_headlines = pd.read_parquet(INTERIM_DIR / 'aligned_headlines.parquet')
headlines = aligned_headlines['headline'].tolist()

def get_sentiment(texts, batch_size=32):
