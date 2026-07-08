# Sentiment Analysis Project

Predicts whether a stock goes **up or down** the next day, based on the
sentiment of its news headlines. Uses **FinBERT** to read each headline,
then a simple classifier to make the up/down call.

**Stocks:** AAPL, MSFT, NVDA
**Time range:** Jul 2024 – Jul 2025
**Target:** next-day open-to-close return (up or down)

## How it works
raw data → align headlines to price moves → FinBERT features → classifier → evaluation
