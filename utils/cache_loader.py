"""Cache-first data loader for the Streamlit dashboard.

Reads pre-computed artifacts from orchestrator_state.json so the dashboard
opens instantly.  Falls back to live computation when cache is missing.

Public API:
    load_dashboard_data(force_refresh=False) -> DashboardData
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config
from utils.data_fetch import load_portfolio

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_STATE_PATH = _ROOT / config.ORCHESTRATOR_STATE_FILE


# ---------------------------------------------------------------------------
# Data structure returned to app.py
# ---------------------------------------------------------------------------

@dataclass
class ArtifactStatus:
    """Lightweight status for a single cached artifact."""
    last_updated: str | None = None   # ISO timestamp
    status: str = "unknown"           # "ok" | "stale" | "error" | "missing" | "unknown"
    error: str | None = None          # last error message, if any


@dataclass
class DashboardData:
    """Everything app.py needs to render, loaded from cache or computed live."""
    holdings: list[dict]
    results: list[dict]
    risk_data: dict
    position_weights: list[dict]
    vix_regime: dict | None

    # Timestamps (ISO strings or None)
    portfolio_timestamp: str | None = None
    discovery_timestamp: str | None = None
    optimizer_timestamp: str | None = None
    exit_signals_timestamp: str | None = None

    # Per-artifact status
    portfolio_status: ArtifactStatus = field(default_factory=ArtifactStatus)
    optimizer_status: ArtifactStatus = field(default_factory=ArtifactStatus)
    discovery_status: ArtifactStatus = field(default_factory=ArtifactStatus)
    exit_status: ArtifactStatus = field(default_factory=ArtifactStatus)

    # Cached optimizer (dict or None)
    cached_optimizer: dict | None = None

    # Cached exit signals (list of dicts or None)
    cached_exit_signals: list[dict] | None = None

    # Cached discovery (list of dicts)
    cached_discovery: list[dict] = field(default_factory=list)

    # Whether data came from cache or live
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Freshness helper
# ---------------------------------------------------------------------------

def format_freshness(iso_timestamp: str | None) -> str:
    """Convert an ISO timestamp to a human-readable freshness string."""
    if not iso_timestamp:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        delta = datetime.now() - dt
        minutes = int(delta.total_seconds() / 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except (ValueError, TypeError):
        return "unknown"


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _load_state() -> dict | None:
    """Load orchestrator state file, return None if missing/corrupt."""
    if not _STATE_PATH.exists():
        return None
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("State file unreadable: %s", e)
        return None


def _load_from_cache(holdings: list[dict]) -> DashboardData | None:
    """Try to load dashboard data from cached state. Returns None if unavailable."""
    state = _load_state()
    if not state:
        return None

    cached_portfolio = state.get("cached_portfolio")
    if not cached_portfolio or not cached_portfolio.get("results"):
        return None

    results = cached_portfolio["results"]
    risk_data = cached_portfolio.get("risk_data", {})
    position_weights = cached_portfolio.get("position_weights", [])
    vix_regime = cached_portfolio.get("vix_regime")

    # Restore correlation matrix if present (must be a nested list of floats)
    corr = cached_portfolio.get("correlation_matrix")
    if corr is not None and isinstance(corr, list):
        try:
            import numpy as np
            risk_data["correlation_matrix"] = np.array(corr, dtype=float)
        except (ValueError, TypeError):
            risk_data["correlation_matrix"] = None
    else:
        risk_data.setdefault("correlation_matrix", None)

    # Cached optimizer
    cached_opt = state.get("cached_optimizer")
    opt_ts = cached_opt.get("timestamp") if cached_opt else None

    # Cached exit signals
    cached_exit = state.get("cached_exit_signals")
    exit_ts = cached_exit.get("timestamp") if cached_exit else None
    exit_sigs = cached_exit.get("signals", []) if cached_exit else None

    # Cached discovery
    cached_disc = state.get("cached_discovery", [])

    return DashboardData(
        holdings=holdings,
        results=results,
        risk_data=risk_data,
        position_weights=position_weights,
        vix_regime=vix_regime,
        portfolio_timestamp=cached_portfolio.get("timestamp"),
        discovery_timestamp=state.get("last_discovery_run"),
        optimizer_timestamp=opt_ts,
        exit_signals_timestamp=exit_ts,
        cached_optimizer=cached_opt,
        cached_exit_signals=exit_sigs,
        cached_discovery=cached_disc,
        from_cache=True,
    )


def _compute_live(holdings: list[dict]) -> DashboardData:
    """Run full live analysis (slow path — used when cache is missing or Refresh clicked)."""
    from engine.scoring import analyse_portfolio

    results, risk_data, position_weights = analyse_portfolio(holdings)
    now = datetime.now().isoformat()

    # Get VIX regime
    vix_regime = None
    try:
        from engine.regime import get_vix_regime
        vix_regime = get_vix_regime()
    except Exception:
        pass

    # Run optimizer
    cached_opt = None
    try:
        from engine.portfolio_optimizer import optimize_portfolio
        alloc = optimize_portfolio(results, holdings, risk_data, position_weights, vix_regime)
        cached_opt = {
            "holdings": [
                {
                    "ticker": h.ticker, "name": h.name,
                    "current_weight": h.current_weight,
                    "optimal_weight": h.optimal_weight,
                    "rebalance_delta": h.rebalance_delta,
                    "expected_return": h.expected_return,
                    "volatility": h.volatility,
                    "sharpe_contribution": h.sharpe_contribution,
                    "sector": h.sector, "currency": h.currency,
                    "action": h.action,
                    "aggregate_score": h.aggregate_score,
                    "fx_cost_if_rebalanced": h.fx_cost_if_rebalanced,
                }
                for h in alloc.holdings
            ],
            "portfolio_expected_return": alloc.portfolio_expected_return,
            "portfolio_volatility": alloc.portfolio_volatility,
            "portfolio_sharpe": alloc.portfolio_sharpe,
            "risk_free_rate": alloc.risk_free_rate,
            "method": alloc.method,
            "sector_weights": alloc.sector_weights,
            "fx_exposure": alloc.fx_exposure,
            "turnover": alloc.turnover,
            "rebalance_trades": alloc.rebalance_trades,
            "warnings": alloc.warnings,
            "timestamp": now,
        }
    except Exception as e:
        logger.warning("Live optimizer failed: %s", e)

    # Run exit engine
    exit_sigs = None
    try:
        from engine.exit_engine import assess_exits
        exits = assess_exits(results, holdings)
        exit_sigs = [
            {"ticker": e.ticker, "name": e.name, "signal_type": e.signal_type,
             "severity": e.severity, "message": e.message,
             "current_score": e.current_score, "current_price": e.current_price}
            for e in exits
        ]
    except Exception as e:
        logger.warning("Live exit engine failed: %s", e)

    # Load cached discovery from state
    state = _load_state()
    cached_disc = state.get("cached_discovery", []) if state else []
    disc_ts = state.get("last_discovery_run") if state else None

    # Persist to cache for next load
    try:
        from utils.state_manager import load_state as _ls, save_state as _ss
        st = _ls()
        st["cached_portfolio"] = {
            "results": results,
            "risk_data": {k: v for k, v in risk_data.items() if k != "correlation_matrix"},
            "position_weights": position_weights,
            "vix_regime": vix_regime,
            "timestamp": now,
        }
        if risk_data.get("correlation_matrix") is not None:
            try:
                cm = risk_data["correlation_matrix"]
                if hasattr(cm, "values"):  # pandas DataFrame
                    st["cached_portfolio"]["correlation_matrix"] = cm.values.tolist()
                elif hasattr(cm, "tolist"):  # numpy array
                    st["cached_portfolio"]["correlation_matrix"] = cm.tolist()
            except Exception:
                pass
        if cached_opt:
            st["cached_optimizer"] = cached_opt
        if exit_sigs is not None:
            st["cached_exit_signals"] = {"signals": exit_sigs, "timestamp": now}
        st["last_portfolio_run"] = now
        _ss(st)
    except Exception as e:
        logger.warning("Failed to persist live results to cache: %s", e)

    return DashboardData(
        holdings=holdings,
        results=results,
        risk_data=risk_data,
        position_weights=position_weights,
        vix_regime=vix_regime,
        portfolio_timestamp=now,
        discovery_timestamp=disc_ts,
        optimizer_timestamp=now if cached_opt else None,
        exit_signals_timestamp=now if exit_sigs else None,
        cached_optimizer=cached_opt,
        cached_exit_signals=exit_sigs,
        cached_discovery=cached_disc,
        from_cache=False,
    )


def load_dashboard_data(force_refresh: bool = False) -> DashboardData:
    """Main entry point: load from cache or compute live.

    Args:
        force_refresh: if True, always recompute live (used by Refresh button)
    """
    holdings = load_portfolio()

    if not force_refresh:
        cached = _load_from_cache(holdings)
        if cached:
            logger.info("Dashboard loaded from cache (portfolio: %s)",
                        format_freshness(cached.portfolio_timestamp))
            return cached

    logger.info("Running live analysis (force_refresh=%s)", force_refresh)
    return _compute_live(holdings)
