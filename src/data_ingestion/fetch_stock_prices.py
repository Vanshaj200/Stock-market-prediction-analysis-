"""
Fetch historical stock prices using yfinance.
Handles rate limiting, retries, and data validation.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class StockPriceFetcher:
    def __init__(self, data_dir="data/prices", max_retries=3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

    def fetch_historical(self, symbol, start_date, end_date=None, interval="1d"):
        """Fetch OHLCV data with retry logic and validation."""
        if end_date is None:
            end_date = datetime.now()

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching {symbol} (attempt {attempt + 1}/{self.max_retries})")

                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    logger.error(f"Missing columns for {symbol}: {missing_cols}")
                    return None

                df = df.reset_index()
                df.columns = [col.lower().replace(" ", "_") for col in df.columns]

                # Normalize date column (yfinance may return 'date' or 'datetime')
                if "date" not in df.columns and "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})

                # Strip timezone info if present
                if hasattr(df["date"], "dt") and df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
                df = df.dropna(subset=["close"])

                filename = self.data_dir / f"{symbol.replace('.', '_')}_prices.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(df)} rows for {symbol}")
                return df

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {symbol} after {self.max_retries} attempts")
                    return None

    def update_latest(self, symbol, history_days=730, full_history=False):
        """Update with latest data only (incremental fetch).

        Args:
            symbol: Stock ticker (e.g. "LT.NS")
            history_days: Days of history to fetch on first bootstrap (default 730)
            full_history: When True, fetch ALL data from 1995-01-01 and overwrite
                          any existing CSV. Bypasses incremental update logic.
        """
        filename = self.data_dir / f"{symbol.replace('.', '_')}_prices.csv"

        # Full history mode: always do a complete refetch (don't use incremental path)
        if full_history:
            logger.info(f"Full history mode: refetching {symbol} from 1995-01-01")
            return self.fetch_historical(symbol, "1995-01-01")

        if filename.exists():
            existing_df = pd.read_csv(filename)
            if existing_df.empty:
                start_date = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")
                return self.fetch_historical(symbol, start_date)

            last_date = pd.to_datetime(existing_df["date"].iloc[-1])
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

            new_df = self.fetch_historical(symbol, start_date)

            if new_df is not None and len(new_df) > 0:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["date"])
                combined_df.to_csv(filename, index=False)
                logger.info(f"Updated {symbol}: +{len(new_df)} rows")
                return combined_df
            return existing_df
        else:
            start_date = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")
            return self.fetch_historical(symbol, start_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = StockPriceFetcher()
    stocks = ["LT.NS", "COALINDIA.NS", "TATACOMM.NS", "ADANIENT.NS", "INFY.NS", "BBOX.NS"]
    for symbol in stocks:
        fetcher.update_latest(symbol)
        time.sleep(1)
