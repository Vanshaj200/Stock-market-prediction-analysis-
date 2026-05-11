"""
Master pipeline: fetch data → preprocess → sentiment → train models.
Run this once to bootstrap, then daily for incremental updates.

Usage:
    python run_pipeline.py               # full pipeline, all stocks
    python run_pipeline.py --step 1      # only fetch prices
    python run_pipeline.py --stocks LT.NS INFY.NS  # specific stocks
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")

sys.path.insert(0, "src")

import yaml
import pandas as pd

from data_ingestion.fetch_stock_prices import StockPriceFetcher
from data_ingestion.fetch_news import NewsFetcher
from preprocessing.technical_indicators import TechnicalIndicators
from nlp_pipeline.sentiment_analyzer import SentimentAnalyzer
from ml_pipeline.feature_engineering import FeatureEngineer
from ml_pipeline.train_model import ModelTrainer


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_1_fetch_prices(config: dict, stocks: list | None = None):
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching Stock Prices")
    logger.info("=" * 60)

    fetcher = StockPriceFetcher()
    targets = stocks or [s["symbol"] for s in config["stocks"]]

    for symbol in targets:
        logger.info(f"  Updating {symbol}...")
        df = fetcher.update_latest(symbol)
        if df is not None:
            logger.info(f"  ✓ {symbol}: {len(df)} total rows")
        else:
            logger.warning(f"  ✗ {symbol}: no data returned")

    logger.info("✓ Step 1 complete\n")


def step_2_fetch_news(config: dict, stocks: list | None = None):
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching News Articles")
    logger.info("=" * 60)

    fetcher = NewsFetcher()
    stock_conf = {s["symbol"]: s for s in config["stocks"]}
    targets = stocks or list(stock_conf.keys())

    for symbol in targets:
        name = stock_conf.get(symbol, {}).get("name", symbol)
        query = stock_conf.get(symbol, {}).get("query", f"{name} stock India")
        logger.info(f"  Fetching news for {name}...")
        articles = fetcher.fetch_google_news_rss(query)
        fetcher.save_articles(articles, symbol)

    logger.info("✓ Step 2 complete\n")


def step_3_preprocess_prices(config: dict, stocks: list | None = None):
    logger.info("=" * 60)
    logger.info("STEP 3: Calculating Technical Indicators")
    logger.info("=" * 60)

    targets = stocks or [s["symbol"] for s in config["stocks"]]
    price_dir = Path("data/prices")

    for symbol in targets:
        safe = symbol.replace(".", "_")
        price_file = price_dir / f"{safe}_prices.csv"

        if not price_file.exists():
            logger.warning(f"  {symbol}: price file missing, skipping")
            continue

        try:
            df = pd.read_csv(price_file)
            df_proc = TechnicalIndicators.add_all_indicators(df)
            out_file = price_dir / f"{safe}_processed.csv"
            df_proc.to_csv(out_file, index=False)
            logger.info(f"  ✓ {symbol}: {len(df_proc.columns)} features, {len(df_proc)} rows")
        except Exception as e:
            logger.error(f"  ✗ {symbol}: {e}")

    logger.info("✓ Step 3 complete\n")


def step_4_analyze_sentiment(config: dict, stocks: list | None = None):
    logger.info("=" * 60)
    logger.info("STEP 4: Analyzing News Sentiment")
    logger.info("=" * 60)

    analyzer = SentimentAnalyzer()
    news_dir = Path("data/news/raw")
    out_dir = Path("data/news/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = stocks or [s["symbol"] for s in config["stocks"]]

    for symbol in targets:
        safe = symbol.replace(".", "_")
        json_files = sorted(news_dir.glob(f"{symbol}_*.json"))

        if not json_files:
            logger.warning(f"  {symbol}: no news files found, skipping")
            continue

        all_articles = []
        for jf in json_files:
            try:
                with open(jf) as f:
                    all_articles.extend(json.load(f))
            except Exception:
                pass

        if not all_articles:
            continue

        results = []
        for article in all_articles:
            text = article.get("content") or article.get("summary", "")
            if not text:
                continue
            sentiment = analyzer.analyze(text)
            # Extract YYYY-MM-DD from published string
            raw_date = article.get("published", "")[:10]
            results.append({
                "article_id": article.get("id", ""),
                "date": raw_date,
                "sentiment": sentiment["sentiment"],
                "sentiment_score": sentiment["score"],
                "confidence": sentiment["confidence"],
            })

        if results:
            sent_df = pd.DataFrame(results)
            out_file = out_dir / f"{safe}_sentiment.csv"
            sent_df.to_csv(out_file, index=False)
            logger.info(f"  ✓ {symbol}: {len(results)} articles analyzed")
        else:
            logger.warning(f"  {symbol}: no text content to analyze")

    logger.info("✓ Step 4 complete\n")


def step_5_train_models(config: dict, stocks: list | None = None):
    logger.info("=" * 60)
    logger.info("STEP 5: Training ML Models")
    logger.info("=" * 60)

    engineer = FeatureEngineer()
    targets = stocks or [s["symbol"] for s in config["stocks"]]

    for symbol in targets:
        logger.info(f"  Training model for {symbol}...")
        try:
            X, y, feature_names = engineer.combine_all_features(symbol)

            trainer = ModelTrainer()
            trainer.feature_names = feature_names

            X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_time_series(X, y)
            trainer.train(X_train, y_train, X_val, y_val)
            metrics = trainer.evaluate(X_test, y_test)
            trainer.save(symbol)

            logger.info(
                f"  ✓ {symbol}: accuracy={metrics['accuracy']:.3f} "
                f"f1={metrics['f1_score']:.3f}"
            )
        except FileNotFoundError as e:
            logger.warning(f"  ✗ {symbol}: {e}")
        except Exception as e:
            logger.error(f"  ✗ {symbol}: {e}", exc_info=True)

    logger.info("✓ Step 5 complete\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock Market Prediction Pipeline")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run only a specific step (1-5). Omit to run all steps.",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        help="Space-separated list of stock symbols to process (e.g. LT.NS INFY.NS)",
    )
    args = parser.parse_args()

    logger.info("🚀 Stock Prediction Pipeline Starting")
    logger.info(f"   Timestamp : {datetime.now()}")
    logger.info(f"   Log file  : {log_file}\n")

    config = load_config()
    stocks = args.stocks

    steps = {
        1: step_1_fetch_prices,
        2: step_2_fetch_news,
        3: step_3_preprocess_prices,
        4: step_4_analyze_sentiment,
        5: step_5_train_models,
    }

    run_steps = [args.step] if args.step else [1, 2, 3, 4, 5]

    try:
        for step_num in run_steps:
            steps[step_num](config, stocks)

        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("  streamlit run src/dashboard/app.py")
        logger.info("  python run_pipeline.py --step 1  (daily price update)")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 60)
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
