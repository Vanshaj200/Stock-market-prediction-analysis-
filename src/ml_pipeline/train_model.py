"""
Train XGBoost classifier for stock direction prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import xgboost as xgb
import joblib
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Auto-fix mechanisms:
    - Adjusts train/val split ratios if total samples < 300
    - Calculates scale_pos_weight automatically to handle class imbalance
    - Warns (does not crash) if validation accuracy < 0.55
    - Saves model, scaler, and metadata atomically
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: xgb.XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def split_time_series(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        """Temporal (non-shuffling) split to preserve time ordering."""
        n = len(X)
        if n < 300:
            logger.warning(f"Only {n} samples — using 60/20/20 split")
            train_ratio, val_ratio = 0.60, 0.20

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        splits = (
            X_arr[:train_end], X_arr[train_end:val_end], X_arr[val_end:],
            y_arr[:train_end], y_arr[train_end:val_end], y_arr[val_end:],
        )
        logger.info(
            f"Split — Train: {train_end}, Val: {val_end - train_end}, "
            f"Test: {n - val_end}"
        )
        return splits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train, y_train,
        X_val, y_val,
        params: dict | None = None,
    ) -> xgb.XGBClassifier:
        """Fit XGBoost with early stopping on validation loss."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Class imbalance correction
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        logger.info(f"Class balance — NEG:{neg} POS:{pos}  scale_pos_weight={scale_pos_weight:.2f}")

        default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "verbosity": 0,
        }
        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        logger.info(f"Training done — best iteration: {self.model.best_iteration}")

        val_pred = self.model.predict(X_val_s)
        val_acc = accuracy_score(y_val, val_pred)
        if val_acc < 0.55:
            logger.warning(
                f"Validation accuracy {val_acc:.3f} < 0.55 — consider more data or tuning"
            )
        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test) -> dict:
        """Compute metrics on held-out test set."""
        X_test_s = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_s)

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        logger.info("=" * 50)
        logger.info("TEST METRICS")
        logger.info("=" * 50)
        for k, v in self.metrics.items():
            logger.info(f"  {k.upper()}: {v:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
        return self.metrics

    def get_feature_importance(self, feature_names: list, top_n: int = 15) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, stock_symbol: str):
        """Persist model, scaler, and metadata to disk."""
        safe = stock_symbol.replace(".", "_")
        joblib.dump(self.model, self.model_dir / f"xgboost_{safe}.pkl")
        joblib.dump(self.scaler, self.model_dir / f"scaler_{safe}.pkl")

        metadata = {
            "stock_symbol": stock_symbol,
            "metrics": self.metrics,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names or [],
            "best_iteration": int(self.model.best_iteration),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(self.model_dir / f"{safe}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved for {stock_symbol}")

    def load(self, stock_symbol: str):
        """Load a previously saved model and scaler."""
        safe = stock_symbol.replace(".", "_")
        self.model = joblib.load(self.model_dir / f"xgboost_{safe}.pkl")
        self.scaler = joblib.load(self.model_dir / f"scaler_{safe}.pkl")

        meta_path = self.model_dir / f"{safe}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names")
            self.metrics = meta.get("metrics", {})

        logger.info(f"Model loaded for {stock_symbol}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    logging.basicConfig(level=logging.INFO)
    from ml_pipeline.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()
    X, y, feature_names = engineer.combine_all_features("LT_NS")

    trainer = ModelTrainer()
    trainer.feature_names = feature_names

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_time_series(X, y)
    trainer.train(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)

    print("\nTop features:")
    print(trainer.get_feature_importance(feature_names))

    trainer.save("LT_NS")
