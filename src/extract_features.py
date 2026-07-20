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

"""
Returns probability for each headline
"""
def get_sentiment(texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt') # Max length is 64 because during EDA headline max was 44
        with torch.no_grad(): # No gradients kept
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1) # Turns the scores into probabilities such that they sum up to 1
        all_probs.append(probs)
        if i % (batch_size * 10) == 0: # Print every 10 batches
            print(f'{min(i + batch_size, len(texts))} / {len(texts)}')
    return torch.cat(all_probs).numpy()

print(f' Running FinBERT on {len(headlines)} headlines')
features = get_sentiment(headlines)

# Attach features back to the df
aligned_headlines['finbert_pos'] = features[:, 0]
aligned_headlines['finbert_neg'] = features[:, 1]
aligned_headlines['finbert_neu'] = features[:, 2]

# Cache to disk
aligned_headlines.to_parquet(PROCESSED_DIR / 'headlines_features.parquet')