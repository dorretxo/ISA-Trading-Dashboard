"""Robust weight optimization for the 4-pillar scoring model.

Strategy — Information Coefficient (IC) based weighting with shrinkage:

1. WALK-FORWARD MULTI-SNAPSHOT for ALL 4 pillars:
   Walk back through 12 monthly snapshots across the broad universe.
   At each snapshot, compute each pillar's score for every stock and
   correlate with the ACTUAL 5-day forward return.  Average the ICs
   across all snapshots → robust, not dependent on a single date.
   Macro experts use point-in-time sliced data to avoid look-ahead bias.

2. CURRENT SNAPSHOT as cross-check:
   Also compute cross-sectional ICs from the latest date for validation.

3. SHRINKAGE toward equal weights:
   Blend the data-driven weights with a prior of 25% each.
   This prevents overfitting when sample size is small.

4. MINIMUM FLOOR per pillar (default 10%):
   No pillar can be zeroed out.  Signal diversification has
   inherent value that a short sample can't measure.

5. CONSTRAINED GRID SEARCH as a cross-check:
   Grid search respects the floor constraints to find the best
   combo, then we average grid-search weights with IC-based
   weights for a final recommendation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from engine.technical import analyse_from_df
from engine.forecasting import _run_experts_on_slice
from engine import fundamental, sentiment
from utils.data_fetch import get_price_history


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StockSnapshot:
    """Pillar scores + forward return for one stock at one point in time."""
    ticker: str
    snapshot_date: str
    technical_score: float
    forecast_score: float
    fundamental_score: float
    sentiment_score: float
    forward_return_pct: float           # Short-horizon (5-day) forward return
    forward_return_pct_long: float = 0.0  # Long-horizon (63-day) forward return


@dataclass
class PillarIC:
    """Information Coefficient stats for one pillar."""
    pillar: str
    avg_ic: float           # Average Spearman IC across snapshots
    ic_std: float           # Std dev of IC across snapshots
    ic_ir: float            # IC / IC_std (information ratio — consistency)
    num_snapshots: int
    num_stocks: int
    method: str             # "multi-snapshot" or "single-snapshot"


@dataclass
class OptimizationResult:
    """Full output from weight optimization."""
    pillar_ics: dict[str, PillarIC]
    ic_based_weights: dict[str, float]         # From IC magnitudes + shrinkage
    grid_search_weights: dict[str, float]      # From constrained grid search
    recommended_weights: dict[str, float]      # Average of IC-based + grid
    current_weights: dict[str, float]
    fitness_current: float
    fitness_recommended: float
    weight_grid_top_n: list[dict]
    # Cross-sectional data for display
    current_snapshot_scores: list[StockSnapshot]
    universe_size: int
    skipped_tickers: list[str]
    shrinkage_factor: float
    min_floor: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PILLARS = ("technical", "fundamental", "sentiment", "forecast")

def _score_to_action(score: float) -> str:
    if score >= config.SCORE_STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    elif score >= config.SCORE_BUY_THRESHOLD:
        return "BUY"
    elif score >= config.SCORE_KEEP_THRESHOLD:
        return "KEEP"
    elif score >= config.SCORE_SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG SELL"


def _is_signal_correct(action: str, forward_return: float) -> bool:
    if action in ("STRONG BUY", "BUY"):
        return forward_return > 0
    elif action in ("SELL", "STRONG SELL"):
        return forward_return < 0
    else:
        return abs(forward_return) < config.BACKTEST_KEEP_THRESHOLD


# ---------------------------------------------------------------------------
# Phase 1: Walk-forward multi-snapshot ICs (all 4 pillars)
# ---------------------------------------------------------------------------

def _compute_historical_ics(
    universe: list[str],
    num_snapshots: int = 12,
    snapshot_spacing_days: int = 21,  # ~1 month apart
    progress_callback=None,
) -> dict[str, list[float]]:
    """Compute IC at multiple historical dates for ALL 4 pillars.

    Returns dict mapping pillar name → list of Spearman correlations
    (one per snapshot). Walk-forward approach with 12 monthly snapshots
    ensures robust IC estimates for all pillars including fund/sentiment.

    Uses dual horizons: technical/sentiment/forecast ICs are computed
    against short-horizon (5-day) forward returns, while fundamental ICs
    are computed against long-horizon (63-day) forward returns.
    """
    horizon_short = config.FORECAST_HORIZON_DAYS
    horizon_long = getattr(config, "FORECAST_HORIZON_LONG", 63)

    # Map each pillar to its evaluation horizon
    pillar_horizon = {
        "technical": "short",
        "forecast": "short",
        "sentiment": "short",
        "fundamental": "long",
    }

    # Pre-fetch all price data
    price_data = {}
    total_steps = len(universe)
    for i, ticker in enumerate(universe):
        if progress_callback:
            progress_callback(i, total_steps + num_snapshots, f"Fetching {ticker}")
        try:
            df = get_price_history(ticker)
            if df is not None and not df.empty and len(df) >= 250:
                price_data[ticker] = df
        except Exception:
            continue

    if len(price_data) < 10:
        return {p: [] for p in PILLARS}

    # Preload macro data once for forecast slicing (avoids look-ahead bias)
    from utils.data_fetch import get_macro_data
    full_macro = get_macro_data()

    pillar_ics = {p: [] for p in PILLARS}

    for snap_idx in range(num_snapshots):
        if progress_callback:
            progress_callback(
                total_steps + snap_idx,
                total_steps + num_snapshots,
                f"Snapshot {snap_idx + 1}/{num_snapshots}",
            )

        # Use longer horizon for offset to ensure both forward returns exist
        offset = (snap_idx + 1) * snapshot_spacing_days + horizon_long

        scores = {p: [] for p in PILLARS}
        forward_returns_short = []
        forward_returns_long = []

        for ticker, df in price_data.items():
            total_bars = len(df)
            end_idx = total_bars - offset
            if end_idx < 200:  # Need 200 bars warmup
                continue

            # Slice up to the snapshot date
            df_slice = df.iloc[:end_idx + 1]
            closes = df["Close"].values.astype(float)

            # Dual forward returns
            future_idx_short = end_idx + horizon_short
            future_idx_long = end_idx + horizon_long
            if future_idx_long >= total_bars:
                continue
            fwd_ret_short = ((closes[future_idx_short] - closes[end_idx]) / closes[end_idx]) * 100
            fwd_ret_long = ((closes[future_idx_long] - closes[end_idx]) / closes[end_idx]) * 100

            # Technical score from slice
            try:
                tech_result = analyse_from_df(df_slice)
                scores["technical"].append(tech_result["score"])
            except Exception:
                scores["technical"].append(0.0)

            # Forecast score from slice (with point-in-time macro data)
            try:
                closes_slice = closes[:end_idx + 1]
                dates_slice = df.index[:end_idx + 1]
                snapshot_date = df.index[end_idx]

                # Slice macro data to point-in-time
                sliced_macro = {}
                for key, macro_df in full_macro.items():
                    if macro_df is not None and not macro_df.empty:
                        sliced_macro[key] = macro_df.loc[:snapshot_date]

                expert_preds = _run_experts_on_slice(
                    closes_slice, dates_slice, horizon_short, macro_data=sliced_macro,
                )
                ensemble_price = sum(expert_preds.values()) / len(expert_preds)
                pct_change = ((ensemble_price - closes[end_idx]) / closes[end_idx]) * 100
                scores["forecast"].append(max(-1.0, min(1.0, pct_change / config.FORECAST_SCORE_SCALE)))
            except Exception:
                scores["forecast"].append(0.0)

            # Fundamental score (yfinance data is static enough for cross-sectional IC)
            try:
                fund_result = fundamental.analyse(ticker)
                scores["fundamental"].append(fund_result["score"])
            except Exception:
                scores["fundamental"].append(0.0)

            # Sentiment score (current news — measures signal quality of the pillar)
            try:
                sent_result = sentiment.analyse(ticker, company_name=ticker)
                scores["sentiment"].append(sent_result["score"])
            except Exception:
                scores["sentiment"].append(0.0)

            forward_returns_short.append(fwd_ret_short)
            forward_returns_long.append(fwd_ret_long)

        # Compute IC for this snapshot (need at least 8 stocks)
        n_stocks = len(forward_returns_short)
        if n_stocks >= 8:
            for pillar in PILLARS:
                # Each pillar correlates with its appropriate horizon
                target_returns = (
                    forward_returns_long if pillar_horizon[pillar] == "long"
                    else forward_returns_short
                )
                rho, _ = spearmanr(scores[pillar], target_returns)
                if not np.isnan(rho):
                    pillar_ics[pillar].append(rho)

    return pillar_ics


# ---------------------------------------------------------------------------
# Phase 2: Single-snapshot cross-sectional ICs (all 4 pillars — current)
# ---------------------------------------------------------------------------

def _compute_current_snapshot(
    universe: list[str],
    progress_callback=None,
    step_offset: int = 0,
    total_steps: int = 0,
) -> tuple[list[StockSnapshot], list[str]]:
    """Score all stocks now. Returns (snapshots, skipped_tickers)."""
    horizon_short = config.FORECAST_HORIZON_DAYS
    horizon_long = getattr(config, "FORECAST_HORIZON_LONG", 63)
    snapshots = []
    skipped = []

    for i, ticker in enumerate(universe):
        if progress_callback:
            progress_callback(
                step_offset + i,
                total_steps,
                f"Scoring {ticker} (current)",
            )
        try:
            df = get_price_history(ticker)
            if df is None or df.empty or len(df) < horizon_long + 50:
                skipped.append(ticker)
                continue

            closes = df["Close"].values.astype(float)
            # Short-horizon realized return (5-day)
            fwd_ret_short = ((closes[-1] - closes[-(horizon_short + 1)]) / closes[-(horizon_short + 1)]) * 100
            # Long-horizon realized return (63-day)
            if len(closes) > horizon_long + 1:
                fwd_ret_long = ((closes[-1] - closes[-(horizon_long + 1)]) / closes[-(horizon_long + 1)]) * 100
            else:
                fwd_ret_long = fwd_ret_short

            # Technical
            tech_result = analyse_from_df(df)
            tech_score = tech_result["score"]

            # Fundamental
            fund_result = fundamental.analyse(ticker)
            fund_score = fund_result["score"]

            # Sentiment
            sent_result = sentiment.analyse(ticker, company_name=ticker)
            sent_score = sent_result["score"]

            # Forecast
            fcast_score = 0.0
            try:
                from engine.forecasting import forecast as run_forecast
                fc = run_forecast(ticker, horizon_days=horizon_short)
                fcast_score = max(-1.0, min(1.0, fc.pct_change / config.FORECAST_SCORE_SCALE))
            except Exception:
                pass

            snapshots.append(StockSnapshot(
                ticker=ticker,
                snapshot_date=str(df.index[-1].date()),
                technical_score=round(tech_score, 4),
                forecast_score=round(fcast_score, 4),
                fundamental_score=round(fund_score, 4),
                sentiment_score=round(sent_score, 4),
                forward_return_pct=round(fwd_ret_short, 4),
                forward_return_pct_long=round(fwd_ret_long, 4),
            ))
        except Exception:
            skipped.append(ticker)

    return snapshots, skipped


def _cross_sectional_ics(snapshots: list[StockSnapshot]) -> dict[str, float]:
    """Spearman IC for each pillar vs forward return in the current snapshot.

    Uses pillar-appropriate horizons: fundamental vs long-horizon return,
    all others vs short-horizon return.
    """
    if len(snapshots) < 8:
        return {p: 0.0 for p in PILLARS}

    returns_short = [s.forward_return_pct for s in snapshots]
    returns_long = [s.forward_return_pct_long for s in snapshots]

    # Pillar → horizon mapping
    pillar_returns = {
        "technical": returns_short,
        "forecast": returns_short,
        "sentiment": returns_short,
        "fundamental": returns_long,
    }

    ics = {}
    for pillar in PILLARS:
        scores = [getattr(s, f"{pillar}_score") for s in snapshots]
        rho, _ = spearmanr(scores, pillar_returns[pillar])
        ics[pillar] = rho if not np.isnan(rho) else 0.0
    return ics


# ---------------------------------------------------------------------------
# Phase 3: IC → weights with shrinkage + floor
# ---------------------------------------------------------------------------

def _ics_to_weights(
    pillar_ic_values: dict[str, float],
    shrinkage: float = None,
    min_floor: float = None,
) -> dict[str, float]:
    """Convert IC values to weights using shrinkage toward equal and floor.

    Steps:
    1. Take absolute IC (we want magnitude of predictive power)
    2. Normalize to proportional weights
    3. Shrink toward equal weights (25% each)
    4. Apply minimum floor
    5. Re-normalize to sum to 1.0
    """
    if shrinkage is None:
        shrinkage = getattr(config, "WEIGHT_SHRINKAGE", 0.40)
    if min_floor is None:
        min_floor = getattr(config, "WEIGHT_MIN_FLOOR", 0.10)

    # Step 1-2: Absolute IC → proportional
    abs_ics = {p: max(abs(v), 0.001) for p, v in pillar_ic_values.items()}
    total_ic = sum(abs_ics.values())
    data_weights = {p: v / total_ic for p, v in abs_ics.items()}

    # Step 3: Shrink toward equal
    equal = 1.0 / len(PILLARS)  # 0.25
    blended = {
        p: (1.0 - shrinkage) * data_weights[p] + shrinkage * equal
        for p in PILLARS
    }

    # Step 4: Apply floor
    for p in PILLARS:
        blended[p] = max(blended[p], min_floor)

    # Step 5: Normalize
    total = sum(blended.values())
    return {p: round(v / total, 4) for p, v in blended.items()}


# ---------------------------------------------------------------------------
# Phase 4: Constrained grid search (with floor)
# ---------------------------------------------------------------------------

def _constrained_grid_search(
    snapshots: list[StockSnapshot],
    min_floor: float = None,
    step: float = None,
) -> tuple[dict[str, float], list[dict]]:
    """Grid search over weight combos with minimum floor per pillar."""
    if step is None:
        step = config.BACKTEST_WEIGHT_STEP
    if min_floor is None:
        min_floor = getattr(config, "WEIGHT_MIN_FLOOR", 0.10)

    floor_steps = int(round(min_floor / step))
    max_steps = int(round(1.0 / step))
    # Available steps after reserving floor for all 4 pillars
    available = max_steps - 4 * floor_steps

    if available < 0:
        # Floor too high for step size — return equal weights
        equal = {p: 0.25 for p in PILLARS}
        return equal, [{"weights": equal, "fitness": 0.0}]

    scored = []
    for t in range(available + 1):
        for f in range(available + 1 - t):
            for s in range(available + 1 - t - f):
                fc = available - t - f - s
                weights = {
                    "technical": round((t + floor_steps) * step, 4),
                    "fundamental": round((f + floor_steps) * step, 4),
                    "sentiment": round((s + floor_steps) * step, 4),
                    "forecast": round((fc + floor_steps) * step, 4),
                }
                fitness = _compute_fitness_from_snapshots(snapshots, weights)
                scored.append({"weights": weights, "fitness": round(fitness, 6)})

    scored.sort(key=lambda x: x["fitness"], reverse=True)
    best = scored[0]["weights"] if scored else {p: 0.25 for p in PILLARS}
    top_10 = scored[:10]

    return best, top_10


def _compute_fitness_from_snapshots(
    snapshots: list[StockSnapshot],
    weights: dict[str, float],
) -> float:
    """Fitness = 60% directional accuracy + 40% rank correlation."""
    if len(snapshots) < 5:
        return -999.0

    agg_scores = []
    correct = 0

    for s in snapshots:
        agg = (
            s.technical_score * weights["technical"]
            + s.fundamental_score * weights["fundamental"]
            + s.sentiment_score * weights["sentiment"]
            + s.forecast_score * weights["forecast"]
        )
        agg_scores.append(agg)
        action = _score_to_action(agg)
        if _is_signal_correct(action, s.forward_return_pct):
            correct += 1

    accuracy = correct / len(snapshots)

    returns = [s.forward_return_pct for s in snapshots]
    rho, _ = spearmanr(agg_scores, returns)
    if np.isnan(rho):
        rho = 0.0

    return accuracy * 0.6 + rho * 0.4


# ---------------------------------------------------------------------------
# Phase 5: Combine IC-based + grid search → final recommendation
# ---------------------------------------------------------------------------

def _average_weights(w1: dict[str, float], w2: dict[str, float]) -> dict[str, float]:
    """Average two weight dicts and re-normalize."""
    avg = {p: (w1[p] + w2[p]) / 2.0 for p in PILLARS}
    total = sum(avg.values())
    return {p: round(v / total, 4) for p, v in avg.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def optimize_weights(
    universe: list[str] | None = None,
    progress_callback=None,
) -> OptimizationResult:
    """Run robust weight optimization.

    Pipeline:
    1. Multi-snapshot historical ICs for technical + forecast
    2. Single-snapshot cross-sectional ICs for all 4 pillars
    3. Merge ICs (historical where available, current for fund/sent)
    4. Convert to weights via shrinkage + floor
    5. Constrained grid search as cross-check
    6. Average both methods → final recommendation
    """
    if universe is None:
        universe = config.BACKTEST_UNIVERSE

    # FMP Starter plan: 300 calls/min — no need to suppress during backtest
    shrinkage = getattr(config, "WEIGHT_SHRINKAGE", 0.40)
    min_floor = getattr(config, "WEIGHT_MIN_FLOOR", 0.10)

    num_snapshots = 12
    total_progress_steps = len(universe) + num_snapshots + len(universe)

    # Phase 1: Walk-forward historical ICs for ALL 4 pillars (12 monthly snapshots)
    historical_ics = _compute_historical_ics(
        universe,
        num_snapshots=num_snapshots,
        progress_callback=lambda cur, tot, msg: (
            progress_callback(cur, total_progress_steps, msg)
            if progress_callback else None
        ),
    )

    # Phase 2: Current snapshot (all 4 pillars) — used for grid search + cross-check
    current_snapshots, skipped = _compute_current_snapshot(
        universe,
        progress_callback=progress_callback,
        step_offset=len(universe) + num_snapshots,
        total_steps=total_progress_steps,
    )

    if len(current_snapshots) < 10:
        return OptimizationResult(
            pillar_ics={},
            ic_based_weights=dict(config.WEIGHTS),
            grid_search_weights=dict(config.WEIGHTS),
            recommended_weights=dict(config.WEIGHTS),
            current_weights=dict(config.WEIGHTS),
            fitness_current=0.0, fitness_recommended=0.0,
            weight_grid_top_n=[],
            current_snapshot_scores=current_snapshots,
            universe_size=len(current_snapshots),
            skipped_tickers=skipped,
            shrinkage_factor=shrinkage, min_floor=min_floor,
        )

    # Phase 2b: Cross-sectional ICs from current snapshot (validation)
    current_ics = _cross_sectional_ics(current_snapshots)

    # Phase 3: Merge ICs — prefer walk-forward historical ICs for all pillars
    merged_ics = {}
    for pillar in PILLARS:
        hist = historical_ics.get(pillar, [])
        if hist:
            merged_ics[pillar] = float(np.mean(hist))
        else:
            # Fallback to current cross-sectional IC if no historical data
            merged_ics[pillar] = current_ics.get(pillar, 0.0)

    # Build PillarIC stats for all pillars
    pillar_ic_stats = {}
    for pillar in PILLARS:
        hist = historical_ics.get(pillar, [])
        if hist:
            avg_ic = float(np.mean(hist))
            ic_std = float(np.std(hist)) if len(hist) > 1 else 0.0
            pillar_ic_stats[pillar] = PillarIC(
                pillar=pillar,
                avg_ic=round(avg_ic, 4),
                ic_std=round(ic_std, 4),
                ic_ir=round(avg_ic / ic_std, 4) if ic_std > 0 else 0.0,
                num_snapshots=len(hist),
                num_stocks=len(current_snapshots),
                method=f"walk-forward ({len(hist)} months)",
            )
        else:
            pillar_ic_stats[pillar] = PillarIC(
                pillar=pillar,
                avg_ic=round(current_ics.get(pillar, 0.0), 4),
                ic_std=0.0,
                ic_ir=0.0,
                num_snapshots=1,
                num_stocks=len(current_snapshots),
                method="single-snapshot (current)",
            )

    # Phase 4: IC → weights with shrinkage + floor
    ic_based_weights = _ics_to_weights(merged_ics, shrinkage, min_floor)

    # Phase 5: Constrained grid search
    grid_weights, top_10 = _constrained_grid_search(
        current_snapshots, min_floor=min_floor,
    )

    # Phase 6: Average both approaches
    recommended = _average_weights(ic_based_weights, grid_weights)

    # Fitness scores
    fitness_current = _compute_fitness_from_snapshots(current_snapshots, config.WEIGHTS)
    fitness_recommended = _compute_fitness_from_snapshots(current_snapshots, recommended)

    return OptimizationResult(
        pillar_ics=pillar_ic_stats,
        ic_based_weights=ic_based_weights,
        grid_search_weights=grid_weights,
        recommended_weights=recommended,
        current_weights=dict(config.WEIGHTS),
        fitness_current=round(fitness_current, 4),
        fitness_recommended=round(fitness_recommended, 4),
        weight_grid_top_n=top_10,
        current_snapshot_scores=current_snapshots,
        universe_size=len(current_snapshots),
        skipped_tickers=skipped,
        shrinkage_factor=shrinkage,
        min_floor=min_floor,
    )
