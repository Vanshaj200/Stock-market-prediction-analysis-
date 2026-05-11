"""
Walk-forward validation and model evaluation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward (expanding window) cross-validation for time-series.
    Each fold trains on all data up to split point and tests on next window.
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 100):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict | None = None,
    ) -> dict:
        """
        Run walk-forward validation and return aggregated metrics.
        """
        if model_params is None:
            model_params = {
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "verbosity": 0,
            }

        n = len(X)
        fold_size = (n - self.min_train_size) // self.n_splits

        if fold_size < 10:
            logger.warning(
                f"Not enough data for {self.n_splits} folds — using single train/test split"
            )
            return self._single_split(X, y, model_params)

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)

        fold_metrics = []
        for fold in range(self.n_splits):
            train_end = self.min_train_size + fold * fold_size
            test_end = min(train_end + fold_size, n)

            X_tr, y_tr = X_arr[:train_end], y_arr[:train_end]
            X_te, y_te = X_arr[train_end:test_end], y_arr[train_end:test_end]

            if len(X_te) == 0:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = xgb.XGBClassifier(**model_params)
            model.fit(X_tr_s, y_tr, verbose=False)
            y_pred = model.predict(X_te_s)

            fold_metrics.append({
                "fold": fold + 1,
                "train_size": train_end,
                "test_size": len(X_te),
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
            })

            logger.info(
                f"Fold {fold+1}/{self.n_splits}: "
                f"acc={fold_metrics[-1]['accuracy']:.3f} "
                f"f1={fold_metrics[-1]['f1']:.3f}"
            )

        if not fold_metrics:
            return {}

        df_folds = pd.DataFrame(fold_metrics)
        summary = {
            "n_folds": len(df_folds),
            "mean_accuracy": float(df_folds["accuracy"].mean()),
            "std_accuracy": float(df_folds["accuracy"].std()),
            "mean_precision": float(df_folds["precision"].mean()),
            "mean_recall": float(df_folds["recall"].mean()),
            "mean_f1": float(df_folds["f1"].mean()),
            "fold_details": df_folds.to_dict(orient="records"),
        }
        logger.info(
            f"Walk-forward result: "
            f"accuracy={summary['mean_accuracy']:.3f}±{summary['std_accuracy']:.3f} "
            f"f1={summary['mean_f1']:.3f}"
        )
        return summary

    def _single_split(self, X, y, model_params: dict) -> dict:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
        split = int(len(X_arr) * 0.8)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_arr[:split])
        X_te_s = scaler.transform(X_arr[split:])
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_tr_s, y_arr[:split], verbose=False)
        y_pred = model.predict(X_te_s)
        return {
            "n_folds": 1,
            "mean_accuracy": float(accuracy_score(y_arr[split:], y_pred)),
            "mean_f1": float(f1_score(y_arr[split:], y_pred, zero_division=0)),
        }
