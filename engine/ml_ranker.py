"""XGBoost Stacked Ensemble Ranker — meta-learner on pillar scores.

Trains a gradient-boosted model on evaluated paper trading outcomes.
Pillar scores are first-layer features; XGBoost learns non-linear
interactions that the linear weighted average misses.

Degrades gracefully: returns None when xgboost is not installed,
training data is insufficient, or the model fails to predict.
"""

import logging
import time

import config

logger = logging.getLogger(__name__)

# Feature columns extracted from signal_backtest
FEATURE_COLS = [
    "technical_score", "fundamental_score", "sentiment_score", "forecast_score",
    "momentum_score", "rsi", "adx", "bb_pct",
    "pe_ratio", "peg_ratio", "revenue_growth", "roe", "short_pct",
    "vix_percentile", "vol_20d",
    "return_10d_prior", "return_30d_prior", "return_90d_prior",
]

# In-memory model cache
_model_cache: dict = {
    "model": None,
    "medians": None,      # For NaN imputation
    "trained_at": 0.0,
    "n_samples": 0,
}

_RETRAIN_HOURS = 24.0


def _xgboost_available() -> bool:
    """Check if xgboost is importable."""
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def train_model(min_samples: int | None = None) -> bool:
    """Train/retrain XGBoost on signal_backtest. Returns True if model is ready.

    No-op if the model is already fresh (< _RETRAIN_HOURS old).
    """
    if min_samples is None:
        min_samples = getattr(config, "ML_RANKER_MIN_SAMPLES", 50)

    # Skip if model is fresh
    if (_model_cache["model"] is not None
            and (time.time() - _model_cache["trained_at"]) < _RETRAIN_HOURS * 3600):
        return True

    if not _xgboost_available():
        logger.debug("XGBoost not installed — ML ranker unavailable")
        return False

    try:
        import numpy as np
        import xgboost as xgb
        from engine.discovery_backtest import init_backtest_db, _connect

        init_backtest_db()
        cols = ", ".join(FEATURE_COLS)
        with _connect() as conn:
            rows = conn.execute(
                f"""SELECT {cols}, return_90d
                    FROM signal_backtest
                    WHERE evaluated_90d = 1
                      AND source = 'discovery'
                      AND return_90d IS NOT NULL""",
            ).fetchall()

        if len(rows) < min_samples:
            logger.info("ML ranker: only %d samples (need %d) — skipping", len(rows), min_samples)
            return False

        # Build feature matrix
        n_features = len(FEATURE_COLS)
        X = np.array([[row[i] for i in range(n_features)] for row in rows], dtype=np.float64)
        y = np.array([row[n_features] for row in rows], dtype=np.float64)

        # Compute medians for NaN imputation
        medians = np.nanmedian(X, axis=0)
        for col in range(n_features):
            mask = np.isnan(X[:, col])
            X[mask, col] = medians[col]

        # Replace any remaining NaN medians with 0
        medians = np.where(np.isnan(medians), 0.0, medians)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        _model_cache["model"] = model
        _model_cache["medians"] = medians
        _model_cache["trained_at"] = time.time()
        _model_cache["n_samples"] = len(rows)

        logger.info("ML ranker trained on %d samples (features: %d)", len(rows), n_features)
        return True

    except Exception as e:
        logger.warning("ML ranker training failed: %s", e)
        return False


def predict_alpha(features: dict) -> float | None:
    """Predict 90-day return from feature dict. Returns None if unavailable."""
    model = _model_cache.get("model")
    medians = _model_cache.get("medians")
    if model is None or medians is None:
        return None

    try:
        import numpy as np

        x = np.array([features.get(col, np.nan) for col in FEATURE_COLS], dtype=np.float64)
        # Impute NaN with training medians
        mask = np.isnan(x)
        x[mask] = medians[mask]

        pred = model.predict(x.reshape(1, -1))[0]
        return float(pred)
    except Exception as e:
        logger.debug("ML ranker prediction failed: %s", e)
        return None


def is_available() -> bool:
    """Check if a trained model exists and xgboost is installed."""
    return _model_cache.get("model") is not None and _xgboost_available()
