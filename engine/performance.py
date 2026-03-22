"""Out-of-sample forecast performance tracking.

Reads evaluated predictions from forecast_store.json and computes
accuracy metrics: hit rate (directional), average error, RMSE,
per-expert comparison, per-ticker breakdown, and rolling accuracy.
"""

import json
import math
from pathlib import Path

import config

# Expert names (must match forecasting.py)
EXPERT_NAMES = [
    "linear_regression", "mean_reversion", "momentum",
    "volatility_adjusted",
    "macro_vix", "macro_bonds", "macro_oil",
]


def _load_evaluated_predictions() -> list[dict]:
    """Load predictions that have been evaluated (actual price known)."""
    store_path = Path(__file__).parent.parent / config.FORECAST_STORE_FILE
    if not store_path.exists():
        return []

    try:
        with open(store_path, "r") as f:
            store = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    predictions = store.get("predictions", [])
    return [
        p for p in predictions
        if p.get("evaluated") and p.get("actual_price") is not None
    ]


def _direction_correct(current: float, predicted: float, actual: float) -> bool:
    """Check if the predicted direction (up/down) matches actual."""
    predicted_up = predicted >= current
    actual_up = actual >= current
    return predicted_up == actual_up


def get_forecast_performance() -> dict:
    """Compute comprehensive out-of-sample forecast performance metrics.

    Returns dict with:
        sufficient_data: bool — whether enough predictions exist (>=5)
        total_predictions: int
        hit_rate: float — fraction with correct direction (0-1)
        avg_error_pct: float — mean |predicted - actual| / actual * 100
        rmse: float — root mean square error
        expert_comparison: dict[str, {hit_rate, avg_error_pct, rmse}]
        per_ticker: dict[str, {hit_rate, avg_error_pct, count}]
        rolling_accuracy: list[{date, hit_rate, window}] — rolling 30-prediction window
    """
    preds = _load_evaluated_predictions()

    empty_result = {
        "sufficient_data": False,
        "total_predictions": len(preds),
        "hit_rate": 0.0,
        "avg_error_pct": 0.0,
        "rmse": 0.0,
        "expert_comparison": {},
        "per_ticker": {},
        "rolling_accuracy": [],
    }

    if len(preds) < 5:
        return empty_result

    # --- Ensemble metrics ---
    hits = 0
    errors_pct = []
    errors_abs = []

    for p in preds:
        current = p["current_price"]
        predicted = p["ensemble_prediction"]
        actual = p["actual_price"]

        if current is None or current <= 0:
            continue

        if _direction_correct(current, predicted, actual):
            hits += 1

        error = abs(predicted - actual)
        errors_abs.append(error)
        errors_pct.append(error / actual * 100 if actual > 0 else 0)

    n = len(preds)
    hit_rate = hits / n if n > 0 else 0.0
    avg_error_pct = sum(errors_pct) / len(errors_pct) if errors_pct else 0.0
    rmse = math.sqrt(sum(e ** 2 for e in errors_abs) / len(errors_abs)) if errors_abs else 0.0

    # --- Per-expert metrics ---
    expert_comparison = {}
    for expert_name in EXPERT_NAMES:
        e_hits = 0
        e_errors_pct = []
        e_errors_abs = []
        e_count = 0

        for p in preds:
            current = p["current_price"]
            actual = p["actual_price"]
            expert_pred = p.get("expert_predictions", {}).get(expert_name)

            if expert_pred is None or current is None or current <= 0:
                continue

            e_count += 1
            if _direction_correct(current, expert_pred, actual):
                e_hits += 1

            error = abs(expert_pred - actual)
            e_errors_abs.append(error)
            e_errors_pct.append(error / actual * 100 if actual > 0 else 0)

        if e_count >= 3:
            expert_comparison[expert_name] = {
                "hit_rate": round(e_hits / e_count, 4),
                "avg_error_pct": round(sum(e_errors_pct) / len(e_errors_pct), 2),
                "rmse": round(math.sqrt(sum(e ** 2 for e in e_errors_abs) / len(e_errors_abs)), 4),
                "count": e_count,
            }

    # --- Per-ticker metrics ---
    ticker_data: dict[str, dict] = {}
    for p in preds:
        ticker = p["ticker"]
        current = p["current_price"]
        predicted = p["ensemble_prediction"]
        actual = p["actual_price"]

        if current is None or current <= 0:
            continue

        if ticker not in ticker_data:
            ticker_data[ticker] = {"hits": 0, "errors_pct": [], "count": 0}

        ticker_data[ticker]["count"] += 1
        if _direction_correct(current, predicted, actual):
            ticker_data[ticker]["hits"] += 1
        error_pct = abs(predicted - actual) / actual * 100 if actual > 0 else 0
        ticker_data[ticker]["errors_pct"].append(error_pct)

    per_ticker = {}
    for ticker, data in ticker_data.items():
        if data["count"] >= 2:
            per_ticker[ticker] = {
                "hit_rate": round(data["hits"] / data["count"], 4),
                "avg_error_pct": round(sum(data["errors_pct"]) / len(data["errors_pct"]), 2),
                "count": data["count"],
            }

    # --- Rolling accuracy (30-prediction sliding window) ---
    rolling_accuracy = []
    window_size = 30

    # Sort by timestamp
    sorted_preds = sorted(preds, key=lambda p: p.get("timestamp", ""))

    if len(sorted_preds) >= window_size:
        for i in range(window_size, len(sorted_preds) + 1):
            window = sorted_preds[i - window_size:i]
            w_hits = 0
            for p in window:
                current = p["current_price"]
                predicted = p["ensemble_prediction"]
                actual = p["actual_price"]
                if current and current > 0 and _direction_correct(current, predicted, actual):
                    w_hits += 1

            rolling_accuracy.append({
                "date": window[-1].get("target_date", window[-1].get("timestamp", ""))[:10],
                "hit_rate": round(w_hits / window_size, 4),
                "window": window_size,
            })

    return {
        "sufficient_data": True,
        "total_predictions": n,
        "hit_rate": round(hit_rate, 4),
        "avg_error_pct": round(avg_error_pct, 2),
        "rmse": round(rmse, 4),
        "expert_comparison": expert_comparison,
        "per_ticker": per_ticker,
        "rolling_accuracy": rolling_accuracy,
    }
