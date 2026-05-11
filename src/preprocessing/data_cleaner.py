"""
Data cleaning utilities shared across the pipeline.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    @staticmethod
    def standardize_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
        """Parse date column and strip timezone info."""
        df = df.copy()
        df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")
        if hasattr(df[col], "dt") and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
        df[col] = df[col].dt.normalize()
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, cols: list, n_std: float = 5.0) -> pd.DataFrame:
        """Clip values beyond n_std standard deviations per column."""
        df = df.copy()
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df[col] = df[col].clip(mean - n_std * std, mean + n_std * std)
        return df

    @staticmethod
    def validate_price_df(df: pd.DataFrame, symbol: str) -> bool:
        """Check that a price DataFrame has minimum required data."""
        if df is None or df.empty:
            logger.error(f"{symbol}: DataFrame is empty")
            return False
        required = {"close", "open", "high", "low", "volume"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"{symbol}: Missing columns {missing}")
            return False
        if len(df) < 50:
            logger.warning(f"{symbol}: Only {len(df)} rows — predictions may be unreliable")
        return True
