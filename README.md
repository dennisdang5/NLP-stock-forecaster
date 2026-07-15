# Sentiment Analysis Project

Predicts whether a stock goes **up or down** the next day, based on the
sentiment of its news headlines. Uses **EODHD** to read each headline,
then a simple classifier to make the up/down call.

**Stocks:** AAPL, MSFT, NVDA
**Time range:** Jan 2026 – Jul 2026
**Target:** next-day open-to-close return (up or down)

## How it works
raw data → align headlines to price moves → FinBERT features → classifier → evaluation
