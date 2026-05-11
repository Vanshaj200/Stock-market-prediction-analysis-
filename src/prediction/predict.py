"""
Prediction engine: load trained models and generate stock direction forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Auto-fix mechanisms:
    - Returns NEUTRAL with confidence=0 if model file is missing
    - Validates feature count against training schema before prediction
    - Portfolio aggregation weighted by individual confidence scores
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self._models: dict = {}
        self._scalers: dict = {}
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _safe_symbol(self, symbol: str) -> str:
        return symbol.replace(".", "_")

    def load_model(self, stock_symbol: str) -> bool:
        """Load model and scaler for a stock. Returns True on success."""
        safe = self._safe_symbol(stock_symbol)
        if safe in self._models:
            return True

        model_path = self.model_dir / f"xgboost_{safe}.pkl"
        scaler_path = self.model_dir / f"scaler_{safe}.pkl"
        meta_path = self.model_dir / f"{safe}_metadata.json"

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False

        self._models[safe] = joblib.load(model_path)
        self._scalers[safe] = joblib.load(scaler_path)

        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata[safe] = json.load(f)
        else:
            self._metadata[safe] = {}

        logger.info(f"Loaded model for {stock_symbol}")
        return True

    # ------------------------------------------------------------------
    # Single stock prediction
    # ------------------------------------------------------------------

    def predict_stock(self, stock_symbol: str, features_df: pd.DataFrame) -> dict:
        """
        Predict next-day direction for a single stock.

        Args:
            stock_symbol: e.g. "LT_NS" or "LT.NS"
            features_df: DataFrame with one or more rows of feature values
                         (only the last row is used)

        Returns:
            dict with keys: stock_symbol, prediction, probability_up,
                            probability_down, confidence, timestamp
        """
        neutral = {
            "stock_symbol": stock_symbol,
            "prediction": "NEUTRAL",
            "probability_up": 0.5,
            "probability_down": 0.5,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        safe = self._safe_symbol(stock_symbol)
        if not self.load_model(stock_symbol):
            neutral["error"] = "Model not found"
            return neutral

        try:
            model = self._models[safe]
            scaler = self._scalers[safe]
            meta = self._metadata.get(safe, {})

            # Use most recent row
            row = features_df.tail(1)

            # Validate feature count
            expected_features = meta.get("feature_names", [])
            if expected_features:
                missing = set(expected_features) - set(row.columns)
                if missing:
                    logger.warning(f"Missing features for {stock_symbol}: {missing}")
                    row = row.reindex(columns=expected_features, fill_value=0.0)
                else:
                    row = row[expected_features]

            X_scaled = scaler.transform(row.values)
            prediction_int = int(model.predict(X_scaled)[0])
            probs = model.predict_proba(X_scaled)[0]

            return {
                "stock_symbol": stock_symbol,
                "prediction": "UP" if prediction_int == 1 else "DOWN",
                "probability_up": float(probs[1]),
                "probability_down": float(probs[0]),
                "confidence": float(max(probs)),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction failed for {stock_symbol}: {e}")
            neutral["error"] = str(e)
            return neutral

    # ------------------------------------------------------------------
    # Portfolio prediction
    # ------------------------------------------------------------------

    def predict_portfolio(
        self,
        stock_symbols: list,
        all_features: dict,
    ) -> dict:
        """
        Predict all stocks and aggregate into a portfolio view.

        Args:
            stock_symbols: list of stock identifiers
            all_features: dict mapping stock_symbol → features DataFrame

        Returns:
            Portfolio-level prediction dict
        """
        predictions = []
        for symbol in stock_symbols:
            if symbol not in all_features:
                logger.warning(f"No features for {symbol}, skipping")
                continue
            pred = self.predict_stock(symbol, all_features[symbol])
            predictions.append(pred)

        if not predictions:
            return {
                "portfolio": stock_symbols,
                "overall_prediction": "NEUTRAL",
                "aggregate_score": 0.0,
                "aggregate_confidence": 0.0,
                "individual_predictions": [],
                "timestamp": datetime.now().isoformat(),
            }

        total_score = 0.0
        total_weight = 0.0
        for pred in predictions:
            direction_map = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}
            score = direction_map.get(pred["prediction"], 0.0)
            weight = pred["confidence"]
            total_score += score * weight
            total_weight += weight

        portfolio_score = total_score / total_weight if total_weight > 0 else 0.0
        if portfolio_score > 0.05:
            overall = "UP"
        elif portfolio_score < -0.05:
            overall = "DOWN"
        else:
            overall = "NEUTRAL"

        return {
            "portfolio": stock_symbols,
            "overall_prediction": overall,
            "aggregate_score": float(portfolio_score),
            "aggregate_confidence": float(total_weight / len(predictions)),
            "individual_predictions": predictions,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    logging.basicConfig(level=logging.INFO)
    from ml_pipeline.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()
    predictor = StockPredictor()

    symbol = "LT_NS"
    X, y, features = engineer.combine_all_features(symbol)
    latest = X.tail(1)

    result = predictor.predict_stock(symbol, latest)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"P(UP)={result['probability_up']:.2%}  P(DOWN)={result['probability_down']:.2%}")
