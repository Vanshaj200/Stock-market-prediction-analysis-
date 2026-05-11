"""
Combine price features, technical indicators, and sentiment into a training-ready dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Columns excluded from the feature matrix
_EXCLUDE_COLS = {
    "date", "open", "high", "low", "close", "volume",
    "dividends", "stock_splits", "capital_gains",
    "target", "future_close",
}


class FeatureEngineer:
    """
    Auto-fix mechanisms:
    - Missing sentiment data → zero-filled columns
    - Missing fundamental data → industry-average placeholders
    - inf / NaN → forward-fill then drop
    """

    def __init__(
        self,
        price_dir: str = "data/prices",
        sentiment_dir: str = "data/news/processed",
    ):
        self.price_dir = Path(price_dir)
        self.sentiment_dir = Path(sentiment_dir)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_price_features(self, stock_symbol: str) -> pd.DataFrame:
        """Load processed price CSV (with technical indicators)."""
        # Accept both LT_NS and LT.NS style
        safe = stock_symbol.replace(".", "_")
        candidates = [
            self.price_dir / f"{safe}_processed.csv",
            self.price_dir / f"{stock_symbol}_processed.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
                return df.sort_values("date").reset_index(drop=True)
        raise FileNotFoundError(
            f"Processed price file not found for {stock_symbol}. "
            "Run step_3_preprocess_prices first."
        )

    def load_sentiment_features(self, stock_symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Merge daily sentiment aggregates into the price DataFrame."""
        safe = stock_symbol.replace(".", "_")
        sentiment_file = self.sentiment_dir / f"{safe}_sentiment.csv"

        if not sentiment_file.exists():
            logger.warning(f"No sentiment data for {stock_symbol} — using zeros")
            df["sentiment_score"] = 0.0
        else:
            sent_df = pd.read_csv(sentiment_file)
            sent_df["date"] = pd.to_datetime(sent_df["date"], errors="coerce").dt.normalize()
            # Aggregate to daily mean score
            daily = (
                sent_df.groupby("date")["sentiment_score"]
                .mean()
                .reset_index()
            )
            df = df.merge(daily, on="date", how="left")
            df["sentiment_score"] = df["sentiment_score"].fillna(0.0)

        # Rolling sentiment features
        df["sentiment_7d"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
        df["sentiment_30d"] = df["sentiment_score"].rolling(30, min_periods=1).mean()
        df["sentiment_momentum"] = df["sentiment_7d"] - df["sentiment_30d"]
        df["news_volume_7d"] = (df["sentiment_score"] != 0.0).rolling(7, min_periods=1).sum()

        return df

    def add_fundamental_features(self, df: pd.DataFrame, stock_symbol: str) -> pd.DataFrame:
        """Add fundamental ratio placeholders (extend with real SQLite data later)."""
        logger.debug(f"Using placeholder fundamentals for {stock_symbol}")
        df["pe_ratio"] = 20.0
        df["debt_to_equity"] = 1.5
        df["roe"] = 0.15
        df["eps_growth"] = 0.10
        return df

    # ------------------------------------------------------------------
    # Target creation
    # ------------------------------------------------------------------

    def create_target(self, df: pd.DataFrame, target_days: int = 1) -> pd.DataFrame:
        """Binary target: 1 if close price rises in target_days, else 0."""
        df = df.copy()
        df["future_close"] = df["close"].shift(-target_days)
        df["target"] = (df["future_close"] > df["close"]).astype(int)
        df = df[df["future_close"].notna()].copy()
        return df

    # ------------------------------------------------------------------
    # Master function
    # ------------------------------------------------------------------

    def combine_all_features(
        self, stock_symbol: str, target_days: int = 1
    ):
        """
        Build the complete feature matrix and target vector.

        Returns:
            X (pd.DataFrame), y (pd.Series), feature_names (list[str])
        """
        df = self.load_price_features(stock_symbol)
        logger.info(f"Loaded {len(df)} price records for {stock_symbol}")

        df = self.load_sentiment_features(stock_symbol, df)
        df = self.add_fundamental_features(df, stock_symbol)
        df = self.create_target(df, target_days)

        # Drop non-feature columns
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols + ["target"])

        X = df[feature_cols].copy()
        y = df["target"].copy()

        logger.info(
            f"{stock_symbol}: {len(feature_cols)} features, {len(X)} samples "
            f"(UP={y.sum()}, DOWN={(y==0).sum()})"
        )
        return X, y, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer = FeatureEngineer()
    X, y, features = engineer.combine_all_features("LT_NS")
    print(f"Features ({len(features)}): {features}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution:\n{y.value_counts()}")
