"""Mixture of Experts (MoE) price forecasting engine with MAE tracking."""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config
from utils.data_fetch import get_macro_data, get_price_history


@dataclass
class ExpertForecast:
    name: str
    predicted_price: float
    confidence_low: float
    confidence_high: float


@dataclass
class EnsembleForecast:
    ticker: str
    horizon_days: int
    current_price: float
    predicted_price: float
    confidence_low: float
    confidence_high: float
    direction: str
    pct_change: float
    expert_forecasts: list[ExpertForecast]
    expert_weights: dict[str, float]
    ensemble_mae: float | None
    expert_maes: dict[str, float | None]
    timestamp: str


# ---------------------------------------------------------------------------
# Expert Models
# ---------------------------------------------------------------------------

def _expert_linear_regression(closes: np.ndarray, horizon: int) -> ExpertForecast:
    """Fit linear trend on recent data and extrapolate."""
    lookback = min(config.EXPERT_LR_LOOKBACK, len(closes))
    y = closes[-lookback:]
    X = np.arange(lookback).reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    pred = float(model.predict(np.array([[lookback - 1 + horizon]]))[0])

    residuals = y - model.predict(X).flatten()
    std_err = float(np.std(residuals))
    band = std_err * config.EXPERT_CONFIDENCE_Z

    return ExpertForecast(
        name="linear_regression",
        predicted_price=max(pred, 0.01),
        confidence_low=max(pred - band, 0.01),
        confidence_high=pred + band,
    )


def _expert_mean_reversion(closes: np.ndarray, horizon: int) -> ExpertForecast:
    """Predict partial reversion toward SMA-50."""
    current = closes[-1]
    sma_window = min(50, len(closes))
    sma = float(np.mean(closes[-sma_window:]))

    speed = config.EXPERT_REVERSION_SPEED * (horizon / 5)
    speed = min(speed, 1.0)
    pred = current + speed * (sma - current)

    deviations = closes[-sma_window:] - np.mean(closes[-sma_window:])
    std_dev = float(np.std(deviations))
    band = std_dev * config.EXPERT_CONFIDENCE_Z

    return ExpertForecast(
        name="mean_reversion",
        predicted_price=max(pred, 0.01),
        confidence_low=max(pred - band, 0.01),
        confidence_high=pred + band,
    )


def _expert_momentum(closes: np.ndarray, horizon: int) -> ExpertForecast:
    """Extrapolate recent rate of change."""
    current = closes[-1]
    window = min(config.EXPERT_MOMENTUM_WINDOW, len(closes) - 1)
    if window < 1:
        return ExpertForecast("momentum", current, current * 0.9, current * 1.1)

    past = closes[-(window + 1)]
    roc = (current - past) / past if past != 0 else 0
    pred = current * (1 + roc * horizon / window)

    # Confidence from volatility of rolling ROC
    if len(closes) > window + 1:
        rocs = np.diff(closes[-window - 5:]) / closes[-window - 5:-1]
        roc_std = float(np.std(rocs)) * math.sqrt(horizon)
        band = current * roc_std * config.EXPERT_CONFIDENCE_Z
    else:
        band = abs(pred - current) * 0.5

    return ExpertForecast(
        name="momentum",
        predicted_price=max(pred, 0.01),
        confidence_low=max(pred - band, 0.01),
        confidence_high=pred + band,
    )


def _expert_volatility(closes: np.ndarray, horizon: int) -> ExpertForecast:
    """ATR-based random walk — no directional bias, wide confidence bands."""
    current = closes[-1]

    # Calculate simple ATR proxy from close-to-close changes
    if len(closes) > 14:
        daily_ranges = np.abs(np.diff(closes[-15:]))
        atr = float(np.mean(daily_ranges))
    else:
        atr = float(np.std(closes)) * 0.5

    band = atr * math.sqrt(horizon) * config.EXPERT_ATR_CONFIDENCE_MULT

    return ExpertForecast(
        name="volatility_adjusted",
        predicted_price=current,
        confidence_low=max(current - band, 0.01),
        confidence_high=current + band,
    )


# ---------------------------------------------------------------------------
# Macro Expert Models
# ---------------------------------------------------------------------------

def _compute_correlation(stock_returns: np.ndarray, macro_returns: np.ndarray) -> float:
    """Compute Pearson correlation, return 0 if insufficient data."""
    min_len = min(len(stock_returns), len(macro_returns))
    if min_len < 20:
        return 0.0
    s = stock_returns[-min_len:]
    m = macro_returns[-min_len:]
    std_s, std_m = np.std(s), np.std(m)
    if std_s == 0 or std_m == 0:
        return 0.0
    return float(np.corrcoef(s, m)[0, 1])


def _macro_expert_generic(
    name: str,
    closes: np.ndarray,
    macro_closes: np.ndarray,
    horizon: int,
    invert: bool = False,
) -> ExpertForecast:
    """Generic macro expert: uses correlation between stock and macro indicator.

    If invert=True, a rising macro indicator is bearish for the stock (e.g. VIX, bond yields).
    """
    current = closes[-1]

    stock_returns = np.diff(closes) / closes[:-1]
    macro_returns = np.diff(macro_closes) / macro_closes[:-1]

    corr = _compute_correlation(stock_returns, macro_returns)

    # If correlation is too weak, predict current price (neutral)
    if abs(corr) < config.MACRO_CORRELATION_MIN:
        band = float(np.std(closes[-20:])) * config.EXPERT_CONFIDENCE_Z
        return ExpertForecast(name, current, max(current - band, 0.01), current + band)

    # Macro trend: rate of change over last 10 days
    macro_window = min(10, len(macro_closes) - 1)
    macro_roc = (macro_closes[-1] - macro_closes[-(macro_window + 1)]) / macro_closes[-(macro_window + 1)]

    # Direction: positive macro_roc + positive correlation = bullish
    # If inverted (VIX/bonds), flip the signal
    signal = macro_roc * abs(corr)
    if invert:
        signal = -signal

    # Scale signal to reasonable price move (cap at 3% per 5 days)
    move_pct = max(-0.03, min(0.03, signal)) * (horizon / 5)
    pred = current * (1 + move_pct)

    band = float(np.std(closes[-20:])) * config.EXPERT_CONFIDENCE_Z

    return ExpertForecast(
        name=name,
        predicted_price=max(pred, 0.01),
        confidence_low=max(pred - band, 0.01),
        confidence_high=pred + band,
    )


def _expert_macro_vix(
    closes: np.ndarray, horizon: int, macro_data: dict | None = None,
) -> ExpertForecast:
    """VIX expert: rising VIX = risk-off = bearish (inverted correlation)."""
    macro = macro_data if macro_data is not None else get_macro_data()
    vix_df = macro.get("vix")
    if vix_df is None or vix_df.empty:
        return ExpertForecast("macro_vix", closes[-1], closes[-1] * 0.95, closes[-1] * 1.05)
    return _macro_expert_generic("macro_vix", closes, vix_df["Close"].values.astype(float), horizon, invert=True)


def _expert_macro_bonds(
    closes: np.ndarray, horizon: int, macro_data: dict | None = None,
) -> ExpertForecast:
    """Bond yield expert: rising yields = higher discount rate = bearish for growth."""
    macro = macro_data if macro_data is not None else get_macro_data()
    bond_df = macro.get("bonds_10y")
    if bond_df is None or bond_df.empty:
        return ExpertForecast("macro_bonds", closes[-1], closes[-1] * 0.95, closes[-1] * 1.05)
    return _macro_expert_generic("macro_bonds", closes, bond_df["Close"].values.astype(float), horizon, invert=True)


def _expert_macro_oil(
    closes: np.ndarray, horizon: int, macro_data: dict | None = None,
) -> ExpertForecast:
    """Oil price expert: energy stocks benefit from rising oil, others may not."""
    macro = macro_data if macro_data is not None else get_macro_data()
    oil_df = macro.get("oil")
    if oil_df is None or oil_df.empty:
        return ExpertForecast("macro_oil", closes[-1], closes[-1] * 0.95, closes[-1] * 1.05)
    return _macro_expert_generic("macro_oil", closes, oil_df["Close"].values.astype(float), horizon, invert=False)


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

EXPERT_NAMES = [
    "linear_regression", "mean_reversion", "momentum",
    "volatility_adjusted",
    "macro_vix", "macro_bonds", "macro_oil",
]


def compute_expert_weights(
    expert_maes: dict[str, list[float]],
    current_price: float,
) -> dict[str, float]:
    """Weight experts inversely proportional to their rolling MAE."""
    n = len(EXPERT_NAMES)
    equal_weight = 1.0 / n
    epsilon = 0.01 * current_price

    raw_weights = {}
    for name in EXPERT_NAMES:
        maes = expert_maes.get(name, [])
        if len(maes) < config.FORECAST_MIN_HISTORY:
            raw_weights[name] = equal_weight
        else:
            recent = maes[-config.FORECAST_ROLLING_WINDOW:]
            avg_mae = sum(recent) / len(recent)
            raw_weights[name] = 1.0 / (avg_mae + epsilon)

    total = sum(raw_weights.values())
    return {name: w / total for name, w in raw_weights.items()}


# ---------------------------------------------------------------------------
# Forecast Store (persistence for MAE tracking)
# ---------------------------------------------------------------------------

def _store_path() -> Path:
    return Path(__file__).parent.parent / config.FORECAST_STORE_FILE


def _load_store() -> dict:
    path = _store_path()
    if path.exists():
        with open(path, "r") as f:
            store = json.load(f)
        # Migrate old-style plain ticker keys to {ticker}_{horizon} format
        maes = store.get("rolling_maes", {})
        keys_to_migrate = [k for k in maes if "_" not in k and not k.startswith("__")]
        for old_key in keys_to_migrate:
            new_key = f"{old_key}_{config.FORECAST_HORIZON_DAYS}"
            if new_key not in maes:
                maes[new_key] = maes[old_key]
            del maes[old_key]
        return store
    return {"predictions": [], "rolling_maes": {}}


def _save_store(store: dict):
    path = _store_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2, default=str)
    tmp.replace(path)  # Atomic rename — prevents corruption from partial writes


def _evaluate_past_predictions(store: dict) -> dict:
    """Evaluate predictions whose target date has passed. Updates MAEs in place."""
    today = pd.Timestamp.now().normalize()

    for pred in store["predictions"]:
        if pred.get("evaluated"):
            continue
        target = pd.Timestamp(pred["target_date"])
        if target > today:
            continue

        ticker = pred["ticker"]
        df = get_price_history(ticker)
        if df.empty:
            continue

        # Find actual price closest to target date
        mask = df.index <= target
        if not mask.any():
            continue
        actual = float(df.loc[mask, "Close"].iloc[-1])
        pred["actual_price"] = actual
        pred["evaluated"] = True

        # Initialize MAE storage keyed by {ticker}_{horizon}
        horizon = pred.get("horizon_days", config.FORECAST_HORIZON_DAYS)
        mae_key = f"{ticker}_{horizon}"
        if mae_key not in store["rolling_maes"]:
            store["rolling_maes"][mae_key] = {n: [] for n in EXPERT_NAMES}
            store["rolling_maes"][mae_key]["ensemble"] = []

        ticker_maes = store["rolling_maes"][mae_key]

        # Compute absolute errors per expert
        for name in EXPERT_NAMES:
            expert_pred = pred["expert_predictions"].get(name)
            if expert_pred is not None:
                ae = abs(expert_pred - actual)
                ticker_maes.setdefault(name, []).append(ae)
                # Trim to rolling window
                ticker_maes[name] = ticker_maes[name][-config.FORECAST_ROLLING_WINDOW:]

        # Ensemble error
        ensemble_pred = pred.get("ensemble_prediction")
        if ensemble_pred is not None:
            ticker_maes.setdefault("ensemble", []).append(abs(ensemble_pred - actual))
            ticker_maes["ensemble"] = ticker_maes["ensemble"][-config.FORECAST_ROLLING_WINDOW:]

    # Trim old evaluated predictions (keep last 500)
    store["predictions"] = store["predictions"][-500:]
    return store


# ---------------------------------------------------------------------------
# Backtest Warmup — pre-train weights from historical data
# ---------------------------------------------------------------------------

BACKTEST_DAYS = 250  # Number of historical trading days to simulate (~12 months)


def _run_experts_on_slice(
    closes: np.ndarray,
    dates: pd.DatetimeIndex,
    horizon: int,
    macro_data: dict | None = None,
) -> dict[str, float]:
    """Run all experts on a historical slice and return their predicted prices.

    Args:
        macro_data: Pre-sliced macro DataFrames keyed by indicator name.
                    When provided, macro experts use point-in-time data
                    instead of live data (fixes look-ahead bias in backtests).
    """
    current = float(closes[-1])
    results = {}

    for fn, name in [
        (_expert_linear_regression, "linear_regression"),
        (_expert_mean_reversion, "mean_reversion"),
        (_expert_momentum, "momentum"),
        (_expert_volatility, "volatility_adjusted"),
    ]:
        try:
            ef = fn(closes, horizon)
            results[name] = ef.predicted_price
        except Exception:
            results[name] = current

    # Macro experts — use sliced macro_data when in backtest to avoid look-ahead bias
    for fn, name in [
        (_expert_macro_vix, "macro_vix"),
        (_expert_macro_bonds, "macro_bonds"),
        (_expert_macro_oil, "macro_oil"),
    ]:
        try:
            ef = fn(closes, horizon, macro_data=macro_data)
            results[name] = ef.predicted_price
        except Exception:
            results[name] = current

    return results


def warmup_backtest(ticker: str, store: dict, df: pd.DataFrame, horizon: int) -> dict:
    """Backtest the last BACKTEST_DAYS days of predictions to pre-train expert weights.

    For each historical trading day (that has a known outcome), simulate what
    each expert would have predicted, compare to actual, and populate the
    rolling MAE lists. Only runs once per ticker — if the ticker already has
    MAE data, this is a no-op.

    Macro experts receive point-in-time sliced data to avoid look-ahead bias.
    """
    # Skip if this ticker+horizon already has backtest data
    mae_key = f"{ticker}_{horizon}"
    if mae_key in store.get("rolling_maes", {}):
        existing = store["rolling_maes"][mae_key]
        if any(len(v) >= BACKTEST_DAYS for v in existing.values()):
            return store

    closes_all = df["Close"].values.astype(float)
    dates_all = df.index

    if len(closes_all) < 200 + horizon:
        return store  # Need 200 days for SMA-200 warm-up + horizon for actuals

    # Preload full macro data once — will be sliced per backtest date
    full_macro = get_macro_data()

    # Initialize MAE storage keyed by {ticker}_{horizon}
    if mae_key not in store["rolling_maes"]:
        store["rolling_maes"][mae_key] = {n: [] for n in EXPERT_NAMES}
        store["rolling_maes"][mae_key]["ensemble"] = []

    ticker_maes = store["rolling_maes"][mae_key]

    # Walk through the last BACKTEST_DAYS points where we know the outcome
    # We need at least `horizon` days after the prediction point for the actual
    start_idx = max(200, len(closes_all) - BACKTEST_DAYS - horizon)
    end_idx = len(closes_all) - horizon

    for i in range(start_idx, end_idx):
        # Data available up to day i (exclusive of future)
        closes_slice = closes_all[:i + 1]
        dates_slice = dates_all[:i + 1]
        backtest_date = dates_all[i]

        # Slice macro data to point-in-time (no look-ahead bias)
        sliced_macro = {}
        for key, macro_df in full_macro.items():
            if macro_df is not None and not macro_df.empty:
                sliced_macro[key] = macro_df.loc[:backtest_date]

        # Actual price `horizon` trading days later
        actual = float(closes_all[i + horizon])

        # Run each expert on the historical slice with point-in-time macro data
        expert_preds = _run_experts_on_slice(
            closes_slice, dates_slice, horizon, macro_data=sliced_macro,
        )

        # Compute absolute errors
        for name in EXPERT_NAMES:
            pred_val = expert_preds.get(name)
            if pred_val is not None:
                ae = abs(pred_val - actual)
                ticker_maes.setdefault(name, []).append(ae)

        # Ensemble (equal-weight since no prior MAE data during backtest)
        ensemble_pred = sum(expert_preds.values()) / len(expert_preds)
        ticker_maes.setdefault("ensemble", []).append(abs(ensemble_pred - actual))

    # Trim all to rolling window
    for key in ticker_maes:
        ticker_maes[key] = ticker_maes[key][-config.FORECAST_ROLLING_WINDOW:]

    return store


# ---------------------------------------------------------------------------
# Main Forecast Function
# ---------------------------------------------------------------------------

def forecast(ticker: str, horizon_days: int | None = None) -> EnsembleForecast:
    """Run all experts, compute gating weights, produce ensemble forecast."""
    if horizon_days is None:
        horizon_days = config.FORECAST_HORIZON_DAYS

    df = get_price_history(ticker)
    if df.empty or len(df) < 20:
        raise ValueError(f"Insufficient price data for {ticker}")

    closes = df["Close"].values.astype(float)
    dates = df.index
    current_price = float(closes[-1])

    # Run all seven experts (4 price-based + 3 macro)
    expert_results = [
        _expert_linear_regression(closes, horizon_days),
        _expert_mean_reversion(closes, horizon_days),
        _expert_momentum(closes, horizon_days),
        _expert_volatility(closes, horizon_days),
        _expert_macro_vix(closes, horizon_days),
        _expert_macro_bonds(closes, horizon_days),
        _expert_macro_oil(closes, horizon_days),
    ]

    # Load store, backtest warmup if needed, evaluate past predictions
    store = _load_store()
    store = warmup_backtest(ticker, store, df, horizon_days)
    store = _evaluate_past_predictions(store)

    mae_key = f"{ticker}_{horizon_days}"
    ticker_maes = store.get("rolling_maes", {}).get(mae_key, {})
    weights = compute_expert_weights(ticker_maes, current_price)

    # Weighted ensemble
    predicted = sum(
        e.predicted_price * weights[e.name] for e in expert_results
    )
    conf_low = sum(
        e.confidence_low * weights[e.name] for e in expert_results
    )
    conf_high = sum(
        e.confidence_high * weights[e.name] for e in expert_results
    )

    direction = "UP" if predicted >= current_price else "DOWN"
    pct_change = ((predicted - current_price) / current_price) * 100

    # Ensemble MAE (if enough history)
    ensemble_mae_list = ticker_maes.get("ensemble", [])
    ensemble_mae = (
        sum(ensemble_mae_list) / len(ensemble_mae_list)
        if len(ensemble_mae_list) >= config.FORECAST_MIN_HISTORY
        else None
    )

    # Per-expert MAEs
    expert_maes_out = {}
    for name in EXPERT_NAMES:
        mae_list = ticker_maes.get(name, [])
        if len(mae_list) >= config.FORECAST_MIN_HISTORY:
            expert_maes_out[name] = round(sum(mae_list) / len(mae_list), 4)
        else:
            expert_maes_out[name] = None

    now_str = datetime.now(timezone.utc).isoformat()
    target_date = (pd.Timestamp.now() + pd.tseries.offsets.BDay(horizon_days)).strftime("%Y-%m-%d")

    # Record prediction for future evaluation
    store["predictions"].append({
        "ticker": ticker,
        "timestamp": now_str,
        "horizon_days": horizon_days,
        "target_date": target_date,
        "current_price": current_price,
        "expert_predictions": {e.name: round(e.predicted_price, 4) for e in expert_results},
        "ensemble_prediction": round(predicted, 4),
        "expert_weights": {k: round(v, 4) for k, v in weights.items()},
        "actual_price": None,
        "evaluated": False,
    })

    _save_store(store)

    return EnsembleForecast(
        ticker=ticker,
        horizon_days=horizon_days,
        current_price=current_price,
        predicted_price=round(predicted, 4),
        confidence_low=round(conf_low, 4),
        confidence_high=round(conf_high, 4),
        direction=direction,
        pct_change=round(pct_change, 2),
        expert_forecasts=expert_results,
        expert_weights={k: round(v, 4) for k, v in weights.items()},
        ensemble_mae=round(ensemble_mae, 4) if ensemble_mae is not None else None,
        expert_maes=expert_maes_out,
        timestamp=now_str,
    )


# ---------------------------------------------------------------------------
# Dual-Horizon Forecast
# ---------------------------------------------------------------------------

def forecast_dual_horizon(ticker: str) -> dict:
    """Run short-horizon and long-horizon forecasts for a ticker.

    Short horizon (5-day): captures technical/momentum signals.
    Long horizon (63-day): captures fundamental/macro signals.

    Returns dict with:
        "short": EnsembleForecast at FORECAST_HORIZON_DAYS (5)
        "long": EnsembleForecast at FORECAST_HORIZON_LONG (63)
    """
    horizon_short = config.FORECAST_HORIZON_DAYS
    horizon_long = getattr(config, "FORECAST_HORIZON_LONG", 63)

    fc_short = forecast(ticker, horizon_days=horizon_short)

    try:
        fc_long = forecast(ticker, horizon_days=horizon_long)
    except Exception:
        # If long forecast fails (insufficient data), duplicate short
        fc_long = None

    return {"short": fc_short, "long": fc_long}
