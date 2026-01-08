# Speech Forecasting and Pricing

Predicting mentions in public speech and identifying probabilistic mispricing in prediction markets.

---

## Overview

This project forecasts whether specific words or phrases will be mentioned in a future public speech (for example, a central bank statement). It assigns probabilities to those events using a statistical predictive model and compares them against market-implied probabilities from prediction markets.

The system is intentionally **conservative and robustness-focused**. Rather than maximizing raw backtest performance, it emphasizes stability, disciplined filtering, and risk control. Trades are selected only when multiple independent signals align, and parameters are chosen based on robustness to small changes rather than peak optimization.

At a high level, the project combines natural language processing, probabilistic modeling, and market pricing analysis to study where language expectations and market beliefs diverge in a controlled, systematic way.

---

## Repository Structure
- scrape-fomc.py
- extract-powell.py
- clean-text.py
- fetch-kalshi.py
- clean-kalshi.py
- build-features.py
- model-v1.py
- score-and-test.py
- sweep-params.py
- README.md

---

## Workflow

The files should be run in the order described above, with the exception of `score-and-test.py`, which is automatically executed by `sweep-params.py`.

The overall workflow is as follows:

1. Fetch transcripts from all Federal Reserve Chair Jerome Powell press conferences using publicly available data from `www.federalreserve.gov`
2. Clean the transcripts to retain only Powell’s spoken remarks, restricted to alphanumeric content
3. Fetch and clean market data from Kalshi
4. Build a feature set using only Powell’s speech data
5. Train a logistic regression model to predict word and phrase mentions
6. Apply a scoring system to identify suitable prediction targets and run walk-forward backtests
7. Perform hyperparameter tuning to identify a stable maximum across key performance metrics

The primary outputs are:

- `recommendations.csv`: a ranked list of potential trades for the upcoming event
- `trade_log.csv`: detailed backtest results for historical evaluation

---

## Requirements

- Python 3.9+
- Standard Python libraries
- `requests` (for market data fetching)

---

## Disclaimer

This project is for educational and demonstration purposes only.  
It does not constitute financial advice or a recommendation to trade.
