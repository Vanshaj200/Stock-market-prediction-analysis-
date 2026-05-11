"""
Calculate technical indicators directly with pandas/numpy.
No ta library dependency — avoids the sgmllib3k build issue on Python 3.11.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Compute common technical indicators using pandas/numpy.

    Auto-fix mechanisms:
    - inf values replaced with NaN
    - NaN filled forward then backward
    - Rows with insufficient history dropped after all indicators added
    """

    @staticmethod
    def add_returns(df: pd.DataFrame, periods=(1, 5, 10, 20)) -> pd.DataFrame:
        for p in periods:
            df[f"returns_{p}d"] = df["close"].pct_change(p)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df[f"volatility_{window}d"] = df["close"].pct_change().rolling(window).std()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        try:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(window).mean()
            loss = (-delta.clip(upper=0)).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = (100 - 100 / (1 + rs)).clip(0, 100)
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            df["rsi"] = 50.0
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        try:
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
            df["macd_diff"] = df["macd"] - df["macd_signal"]
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            df["macd"] = 0.0
            df["macd_signal"] = 0.0
            df["macd_diff"] = 0.0
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        try:
            rolling = df["close"].rolling(window)
            mid = rolling.mean()
            std = rolling.std()
            df["bb_high"] = mid + num_std * std
            df["bb_low"] = mid - num_std * std
            df["bb_mid"] = mid
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"].replace(0, np.nan)
            df["bb_pct"] = (df["close"] - df["bb_low"]) / (
                df["bb_high"] - df["bb_low"]
            ).replace(0, np.nan)
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            df["bb_high"] = df["close"]
            df["bb_low"] = df["close"]
            df["bb_mid"] = df["close"]
            df["bb_width"] = 0.0
            df["bb_pct"] = 0.5
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = (df["volume"] / df["volume_ma_20"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], 1.0
        )
        return df

    @staticmethod
    def add_price_position(df: pd.DataFrame) -> pd.DataFrame:
        """Price position relative to recent high/low."""
        df["high_52w"] = df["close"].rolling(252, min_periods=1).max()
        df["low_52w"] = df["close"].rolling(252, min_periods=1).min()
        rng = (df["high_52w"] - df["low_52w"]).replace(0, np.nan)
        df["price_position"] = (df["close"] - df["low_52w"]) / rng
        return df

    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        for window in [10, 20, 50]:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"close_vs_sma{window}"] = df["close"] / df[f"sma_{window}"].replace(0, np.nan) - 1
        return df

    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = cls.add_returns(df)
        df = cls.add_volatility(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_volume_features(df)
        df = cls.add_price_position(df)
        df = cls.add_moving_averages(df)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows with insufficient history")

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("data/prices/LT_NS_prices.csv")
    out = TechnicalIndicators.add_all_indicators(df)
    print(f"Output shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    out.to_csv("data/prices/LT_NS_processed.csv", index=False)
