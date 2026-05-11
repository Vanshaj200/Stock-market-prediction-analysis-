# Stock Market Prediction System

ML-based prediction system for 6 NSE-listed Indian stocks using XGBoost, technical indicators, and news sentiment analysis (VADER).

## Target Stocks

| Symbol | Company |
|--------|---------|
| LT.NS | Larsen & Turbo |
| ADANIENT.NS | Adani Enterprises |
| COALINDIA.NS | Coal India |
| BBOX.NS | Black Box |
| TATACOMM.NS | Tata Communications |
| INFY.NS | Infosys |

## Architecture

```
Data Ingestion (yfinance + Google News RSS)
       ↓
Storage (data/prices/*.csv  |  data/news/*.json)
       ↓
Preprocessing (Technical Indicators: RSI, MACD, Bollinger Bands, etc.)
       ↓
NLP Pipeline (VADER Sentiment Analysis)
       ↓
ML Pipeline (XGBoost Binary Classifier: UP / DOWN)
       ↓
Streamlit Dashboard (Individual stock + Portfolio view)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (fetch data + train models)
python run_pipeline.py

# 3. Launch dashboard
streamlit run src/dashboard/app.py
```

## Pipeline Steps (run individually)

```bash
python run_pipeline.py --step 1   # Fetch stock prices
python run_pipeline.py --step 2   # Fetch news articles
python run_pipeline.py --step 3   # Calculate technical indicators
python run_pipeline.py --step 4   # Analyze news sentiment
python run_pipeline.py --step 5   # Train XGBoost models

# Target specific stocks
python run_pipeline.py --stocks LT.NS INFY.NS
```

## Project Structure

```
.
├── config/
│   └── config.yaml          # Stock list, model params
├── data/
│   ├── prices/              # OHLCV CSVs + processed CSVs
│   ├── news/raw/            # News JSON files
│   ├── news/processed/      # Sentiment CSV files
│   ├── fundamentals/        # SQLite (future)
│   └── models/              # Trained .pkl files + metadata
├── src/
│   ├── data_ingestion/      # yfinance + RSS fetchers
│   ├── preprocessing/       # Technical indicators, data cleaning
│   ├── nlp_pipeline/        # VADER sentiment analyzer
│   ├── ml_pipeline/         # Feature engineering, XGBoost trainer
│   ├── prediction/          # Inference engine
│   └── dashboard/           # Streamlit app
├── logs/                    # Pipeline run logs
├── run_pipeline.py          # Master pipeline entry point
└── requirements.txt
```

## Features (31 total)

**Price / Technical (27)**
- Returns: 1d, 5d, 10d, 20d
- Volatility: 20d rolling std
- RSI (14-period)
- MACD + signal + histogram
- Bollinger Bands: high, low, mid, width, %B
- Volume: change, MA-20, ratio
- Price position (52-week high/low)
- SMA 10/20/50 + close-vs-SMA ratios

**Sentiment (4)**
- sentiment_7d, sentiment_30d, sentiment_momentum, news_volume_7d

**Fundamental (4, placeholders)**
- pe_ratio, debt_to_equity, roe, eps_growth

## Key Design Decisions

- **No cloud storage** — all data in `data/` (CSV, JSON)
- **Free APIs only** — yfinance + Google News RSS
- **VADER not FinBERT** — zero download, no GPU required, suitable for batch
- **No `ta` library** — indicators implemented directly with pandas/numpy (avoids sgmllib3k build failure on Python 3.11)
- **Temporal train/val/test split** — 70/15/15, no data leakage

## Cron Schedule (optional)

```cron
# Daily at 6 PM after market close
0 18 * * 1-5 cd /path/to/project && python run_pipeline.py --step 1 --step 2 --step 3 --step 4

# Weekly model retraining (Sunday midnight)
0 0 * * 0 cd /path/to/project && python run_pipeline.py --step 5
```

## Disclaimer

Predictions are for educational purposes only. Not financial advice.
