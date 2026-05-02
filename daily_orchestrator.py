#!/usr/bin/env python3
"""ISA Portfolio Autonomous Orchestrator (v4.0).

Headless script designed to run via Windows Task Scheduler or cron.
Evaluates the portfolio, optionally runs the Global Discovery Engine,
and emails an actionable brief ONLY when strict quantitative hurdles are met.

Usage:
    python daily_orchestrator.py                     # Full run (portfolio + weekly discovery)
    python daily_orchestrator.py --dry-run            # Same but no email sent
    python daily_orchestrator.py --portfolio-only     # Skip discovery entirely
    python daily_orchestrator.py --force-discovery    # Force discovery even if not scheduled
    python daily_orchestrator.py --dry-run --portfolio-only  # Quick test
"""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from utils.data_fetch import load_portfolio
from utils.state_manager import (
    get_cached_discovery,
    is_on_cooldown,
    load_state,
    prune_expired_cooldowns,
    save_state,
    set_cooldown,
    should_run_discovery,
)
from utils.email_sender import build_alert_email, send_email
from engine.paper_trading import init_db as _init_paper_db, log_signal as _log_paper_signal, resolve_pending_signals
from engine.discovery_backtest import record_discovery_picks, record_portfolio_signals, evaluate_matured_signals
from engine.exit_engine import reconcile_actions_with_exits, exit_signal_to_dict

# ---------------------------------------------------------------------------
# Logging setup — console + file
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


def _setup_logging(dry_run: bool = False) -> None:
    """Configure root logger to output to console and a rotating log."""
    level = logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=[
        logging.StreamHandler(sys.stdout),
    ])
    # Suppress noisy third-party loggers
    for name in ("urllib3", "yfinance", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("orchestrator")


def _paper_trading_enabled_for_run(dry_run: bool) -> bool:
    """Return True when this run may write to the paper-trading ledger."""
    if not getattr(config, "PAPER_TRADING_ENABLED", False):
        return False
    return (not dry_run) or bool(getattr(config, "PAPER_TRADING_LOG_DRY_RUN", False))


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorRunLock:
    """Exclusive lock file for a single orchestrator process."""
    path: Path
    token: str
    metadata: dict
    reclaimed: dict | None = None
    released: bool = False

    def release(self) -> None:
        """Release the lock if we still own it."""
        if self.released:
            return
        try:
            current = _read_lock_metadata(self.path)
            if current and current.get("token") not in (None, self.token):
                logger.warning(
                    "Lock ownership changed while running; leaving %s in place.",
                    self.path,
                )
                return
            self.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to release orchestrator lock %s: %s", self.path, e)
        finally:
            self.released = True


class OrchestratorAlreadyRunning(RuntimeError):
    """Raised when another orchestrator invocation is already active."""

    def __init__(self, metadata: dict | None = None):
        super().__init__("another orchestrator run is already active")
        self.metadata = metadata or {}


def _read_lock_metadata(path: Path) -> dict | None:
    """Best-effort JSON loader for the lock file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _lock_age_seconds(path: Path, metadata: dict | None) -> float:
    """Estimate how old the lock is, preferring its started_at timestamp."""
    if metadata:
        started_at = metadata.get("started_at")
        if started_at:
            try:
                started = datetime.fromisoformat(str(started_at))
                return max((datetime.now() - started).total_seconds(), 0.0)
            except ValueError:
                pass
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except OSError:
        return 0.0


def _is_lock_stale(path: Path, metadata: dict | None, stale_seconds: int) -> bool:
    """Return True when the existing lock is old enough to reclaim safely."""
    return _lock_age_seconds(path, metadata) >= max(int(stale_seconds), 1)


def _acquire_orchestrator_lock(
    *,
    dry_run: bool,
    force_discovery: bool,
    portfolio_only: bool,
    path: Path | None = None,
    stale_seconds: int | None = None,
) -> OrchestratorRunLock:
    """Acquire the single-instance orchestrator lock or raise if busy."""
    lock_path = path or (ROOT / getattr(config, "ORCHESTRATOR_LOCK_FILE", "feature_cache/orchestrator.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if stale_seconds is None:
        max_runtime = int(getattr(config, "ORCHESTRATOR_MAX_RUNTIME", 3600))
        stale_seconds = int(
            getattr(config, "ORCHESTRATOR_LOCK_STALE_SECONDS", max_runtime + 1800)
        )

    token = f"{os.getpid()}-{time.time_ns()}"
    metadata = {
        "pid": os.getpid(),
        "token": token,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": str(ROOT),
        "argv": list(sys.argv),
        "dry_run": bool(dry_run),
        "force_discovery": bool(force_discovery),
        "portfolio_only": bool(portfolio_only),
    }
    reclaimed: dict | None = None

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, default=str)
                f.flush()
                os.fsync(f.fileno())
            return OrchestratorRunLock(
                path=lock_path,
                token=token,
                metadata=metadata,
                reclaimed=reclaimed,
            )
        except FileExistsError:
            existing = _read_lock_metadata(lock_path)
            if not _is_lock_stale(lock_path, existing, stale_seconds):
                raise OrchestratorAlreadyRunning(existing)

            logger.warning(
                "Stale orchestrator lock detected at %s (age %.0fs >= %ss); reclaiming.",
                lock_path,
                _lock_age_seconds(lock_path, existing),
                stale_seconds,
            )
            try:
                lock_path.unlink()
                reclaimed = existing
            except FileNotFoundError:
                continue
            except OSError as e:
                raise OrchestratorAlreadyRunning({
                    "path": str(lock_path),
                    "error": f"stale_lock_reclaim_failed: {e}",
                }) from e


# ---------------------------------------------------------------------------
# Decision log — append-only JSONL file
# ---------------------------------------------------------------------------

def _log_decision(event: str, details: dict | None = None) -> None:
    """Append a structured decision record to the JSONL log file."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "details": details or {},
    }
    log_path = ROOT / config.ORCHESTRATOR_LOG_FILE
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to write decision log: %s", e)


# ---------------------------------------------------------------------------
# Timeout wrapper (thread-based, Windows-compatible)
# ---------------------------------------------------------------------------

def _run_with_timeout(func, args=(), timeout_seconds: int = 300):
    """Run func(*args) with a timeout.  Raises TimeoutError on expiry.

    Uses threading.Thread.join(timeout) because Windows lacks signal.SIGALRM.
    Logs a heartbeat every 60s while waiting so operators can see the process
    is alive even when the inner function is silent.
    """
    result_box = [None]
    error_box = [None]
    _start = time.time()

    def _target():
        try:
            result_box[0] = func(*args)
        except Exception as e:
            error_box[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    # Poll with heartbeat instead of blocking join — gives visibility
    _HEARTBEAT_INTERVAL = 60  # seconds
    remaining = timeout_seconds
    while remaining > 0 and thread.is_alive():
        wait = min(_HEARTBEAT_INTERVAL, remaining)
        thread.join(timeout=wait)
        remaining -= wait
        if thread.is_alive() and remaining > 0:
            elapsed = time.time() - _start
            logger.info("[heartbeat] %s running for %.0fs (timeout in %.0fs)",
                        func.__name__, elapsed, remaining)

    if thread.is_alive():
        elapsed = time.time() - _start
        logger.error("[timeout] %s exceeded %ds timeout (ran %.0fs). "
                     "Thread is orphaned — it will finish naturally but results are discarded.",
                     func.__name__, timeout_seconds, elapsed)
        _log_decision("timeout", {
            "function": func.__name__,
            "timeout_s": timeout_seconds,
            "elapsed_s": round(elapsed, 1),
        })
        raise TimeoutError(f"{func.__name__} exceeded {timeout_seconds}s timeout")
    if error_box[0]:
        raise error_box[0]
    return result_box[0]


# ---------------------------------------------------------------------------
# Portfolio analysis step
# ---------------------------------------------------------------------------

def _run_portfolio_analysis(holdings: list[dict]) -> tuple:
    """Run analyse_portfolio and return (results, risk_data, position_weights)."""
    from engine.scoring import analyse_portfolio
    return analyse_portfolio(holdings)


def _get_regime() -> dict:
    """Get VIX regime, returning neutral defaults on failure."""
    try:
        from engine.regime import get_vix_regime
        return get_vix_regime()
    except Exception as e:
        logger.warning("VIX regime detection failed: %s", e)
        return {"vix_level": 0.0, "vix_percentile": 50.0, "regime_label": "NEUTRAL"}


# ---------------------------------------------------------------------------
# Discovery step
# ---------------------------------------------------------------------------

def _run_discovery_pipeline(holdings: list[dict], risk_data: dict):
    """Run the global discovery engine. Returns DiscoveryResult."""
    from engine.discovery import run_discovery
    return run_discovery(holdings, risk_data)


def _reconstitute_dynamic_universe() -> dict | None:
    """Refresh the dynamic universe supplement before discovery.

    Pulls fresh ETF-holdings candidates through the validation gates so that
    new entrants are admitted only if they pass liquidity/mcap floors. Stale
    tickers are pruned after N consecutive misses. Failures are non-fatal —
    discovery will fall back to the existing supplement and direct ETF inject.
    """
    try:
        from utils.global_universe import (
            decompose_etf_holdings,
            reconstitute_universe,
        )
    except Exception as e:
        logger.debug("Universe reconstitution modules unavailable: %s", e)
    return None


def _dynamic_universe_cache_fresh() -> bool:
    """Return True when dynamic universe validation was refreshed recently."""
    try:
        ttl_hours = float(getattr(config, "DISCOVERY_RECONSTITUTE_TTL_HOURS", 24))
        if ttl_hours <= 0:
            return False
        path = ROOT / "feature_cache" / "universe_dynamic.json"
        if not path.exists():
            return False
        import json
        from datetime import datetime

        payload = json.loads(path.read_text(encoding="utf-8"))
        refreshed_at = payload.get("refreshed_at")
        if not refreshed_at:
            return False
        refreshed = datetime.fromisoformat(refreshed_at)
        age_hours = (datetime.now() - refreshed).total_seconds() / 3600.0
        if age_hours < ttl_hours:
            logger.info(
                "Dynamic universe reconstitution skipped: cache fresh %.1fh < %.1fh",
                age_hours,
                ttl_hours,
            )
            return True
    except Exception as exc:
        logger.debug("Dynamic universe freshness check failed: %s", exc)
    return False

    extras: list[str] = []
    if getattr(config, "DISCOVERY_USE_ETF_DECOMPOSITION", True):
        try:
            extras = decompose_etf_holdings() or []
        except Exception as e:
            logger.warning("ETF decomposition failed pre-reconstitution: %s", e)
            extras = []

    # Benchmark-driven entrants (S&P 500, FTSE 100/250, CAC, DAX, IBEX, ...).
    # Cached 30d; per-benchmark isolation means one failed scrape does not
    # block the others. Graduating through reconstitute_universe() means
    # each constituent still has to clear the liquidity/mcap floors.
    if getattr(config, "DISCOVERY_USE_BENCHMARK_REBUILD", True):
        try:
            from utils.benchmark_constituents import get_benchmark_tickers
            ttl_days = int(getattr(config, "DISCOVERY_BENCHMARK_TTL_DAYS", 30))
            bench = get_benchmark_tickers(ttl_days=ttl_days) or []
            before = len(extras)
            seen = set(extras)
            for t in bench:
                if t not in seen:
                    extras.append(t)
                    seen.add(t)
            logger.info(
                "Benchmark constituents: %d unique tickers, %d net-new",
                len(bench),
                len(extras) - before,
            )
        except Exception as e:
            logger.warning("Benchmark constituents fetch failed: %s", e)

    try:
        summary = reconstitute_universe(extra_tickers=extras)
        logger.info(
            "Dynamic universe reconstituted: %d validated, %d pruned "
            "(ETF extras fed in: %d)",
            summary.get("validated", 0),
            summary.get("pruned", 0),
            len(extras),
        )
        return summary
    except Exception as e:
        logger.warning("reconstitute_universe() failed: %s", e)
        return None


def save_discovery_results(disc_result, state: dict) -> int:
    """Persist discovery results to state and record picks for backtest.

    Extracted so both the background orchestrator and the UI button can call it.
    Returns the number of backtest picks recorded.
    """
    n_candidates = len(disc_result.candidates)

    state["cached_discovery"] = [
        {
            "ticker": c.ticker,
            "name": c.name,
            "exchange": c.exchange,
            "country": c.country,
            "sector": c.sector,
            "industry": c.industry,
            "market_cap": c.market_cap,
            "currency": c.currency,
            "aggregate_score": c.aggregate_score,
            "technical_score": c.technical_score,
            "fundamental_score": c.fundamental_score,
            "sentiment_score": c.sentiment_score,
            "forecast_score": c.forecast_score,
            "action": c.action,
            "why": c.why,
            "fx_penalty_applied": c.fx_penalty_applied,
            "fx_penalty_pct": c.fx_penalty_pct,
            "max_correlation": c.max_correlation,
            "correlated_with": c.correlated_with,
            "sector_weight_if_added": c.sector_weight_if_added,
            "portfolio_fit_score": c.portfolio_fit_score,
            "momentum_score": c.momentum_score,
            "return_90d": c.return_90d,
            "return_30d": c.return_30d,
            "volume_ratio": c.volume_ratio,
            "expected_return_90d": c.expected_return_90d,
            "analyst_target": getattr(c, "analyst_target", None),
            "analyst_upside": getattr(c, "analyst_upside", None),
            "num_analysts": getattr(c, "num_analysts", None),
            "insider_buys": getattr(c, "insider_buys", 0),
            "insider_sells": getattr(c, "insider_sells", 0),
            "insider_net": getattr(c, "insider_net", ""),
            "beta_90d": getattr(c, "beta_90d", None),
            "debt_to_equity": getattr(c, "debt_to_equity", None),
            "entry_stance": getattr(c, "entry_stance", "Ready"),
            "ticker_identity_warning": getattr(c, "ticker_identity_warning", None),
            "parabolic_penalty": c.parabolic_penalty,
            "is_parabolic": c.is_parabolic,
            "earnings_near": c.earnings_near,
            "earnings_imminent": c.earnings_imminent,
            "earnings_days": c.earnings_days,
            "cap_tier": c.cap_tier,
            "confidence_discount": c.confidence_discount,
            "max_weight_scale": c.max_weight_scale,
            "post_earnings_recent": c.post_earnings_recent,
            "post_earnings_days": c.post_earnings_days,
            "earnings_miss": c.earnings_miss,
            "earnings_miss_pct": c.earnings_miss_pct,
            "near_52w_high": c.near_52w_high,
            "pct_from_52w_high": c.pct_from_52w_high,
            "entry_lens": getattr(c, "entry_lens", "momentum"),
            "entry_price": getattr(c, "entry_price", None),
            "entry_method": getattr(c, "entry_method", ""),
            "entry_zone_low": getattr(c, "entry_zone_low", None),
            "entry_zone_high": getattr(c, "entry_zone_high", None),
            "fill_probability": getattr(c, "fill_probability", None),
            "stop_loss": getattr(c, "stop_loss", None),
            "stop_method": getattr(c, "stop_method", ""),
            "stop_distance_pct": getattr(c, "stop_distance_pct", None),
            "take_profit": getattr(c, "take_profit", None),
            "target_method": getattr(c, "target_method", ""),
            "position_size_shares": getattr(c, "position_size_shares", 0),
            "position_weight": getattr(c, "position_weight", 0),
            "risk_amount": getattr(c, "risk_amount", 0),
            "r_r_ratio": getattr(c, "r_r_ratio", None),
            "sizing_method": getattr(c, "sizing_method", ""),
            "kelly_cap_fraction": getattr(c, "kelly_cap_fraction", None),
            "support_levels": getattr(c, "support_levels", {}),
            "regime_info": getattr(c, "regime_info", {}),
            "regime": getattr(c, "regime", None),
            # Dividend safety
            "dividend_yield": getattr(c, "dividend_yield", None),
            "payout_ratio": getattr(c, "payout_ratio", None),
            "ex_dividend_date": getattr(c, "ex_dividend_date", None),
            "ex_dividend_days": getattr(c, "ex_dividend_days", None),
            "five_year_avg_yield": getattr(c, "five_year_avg_yield", None),
            # Balance sheet strength
            "balance_sheet_grade": getattr(c, "balance_sheet_grade", None),
            "net_debt_ebitda": getattr(c, "net_debt_ebitda", None),
            "current_ratio": getattr(c, "current_ratio", None),
            "cash_to_debt": getattr(c, "cash_to_debt", None),
            # Governance red flag
            "governance_flag": getattr(c, "governance_flag", False),
            "governance_reasons": getattr(c, "governance_reasons", []),
            # Asymmetric / binary outcome flag
            "asymmetric_risk_flag": getattr(c, "asymmetric_risk_flag", False),
            "asymmetric_risk_reason": getattr(c, "asymmetric_risk_reason", None),
            # Enterprise factor bundle (roadmap items #1, #2)
            "enterprise_value": getattr(c, "enterprise_value", None),
            "ev_ebit": getattr(c, "ev_ebit", None),
            "ev_ebitda": getattr(c, "ev_ebitda", None),
            "ebit_yield": getattr(c, "ebit_yield", None),
            "ev_ebit_score": getattr(c, "ev_ebit_score", None),
            "gpa": getattr(c, "gpa", None),
            "gpa_score": getattr(c, "gpa_score", None),
            "f_score": getattr(c, "f_score", None),
            "f_score_gate": getattr(c, "f_score_gate", False),
            "f_score_score": getattr(c, "f_score_score", None),
            "f_score_coverage": getattr(c, "f_score_coverage", 0.0),
            # Day-1 quality / self-learning diagnostics
            "institutional_prior_score": getattr(c, "institutional_prior_score", None),
            "institutional_prior_percentile": getattr(c, "institutional_prior_percentile", None),
            "institutional_prior_confidence": getattr(c, "institutional_prior_confidence", None),
            "institutional_prior_rank": getattr(c, "institutional_prior_rank", None),
            "strong_buy_eligible": getattr(c, "strong_buy_eligible", None),
            "ready_contract_status": getattr(c, "ready_contract_status", None),
            "ready_contract_reasons": getattr(c, "ready_contract_reasons", None),
            "meta_label_proba": getattr(c, "meta_success_prob", None),
            "ml_alpha_raw": getattr(c, "ml_alpha_raw", None),
            "ml_shadow_score": getattr(c, "ml_shadow_score", None),
            "sleeve_composite": getattr(c, "sleeve_composite", None),
            "sleeve_momentum": getattr(c, "sleeve_momentum", None),
            "sleeve_quality": getattr(c, "sleeve_quality", None),
            "sleeve_value": getattr(c, "sleeve_value", None),
            "sleeve_low_risk": getattr(c, "sleeve_low_risk", None),
            "sleeve_pead": getattr(c, "sleeve_pead", None),
            "sleeve_ready": getattr(c, "sleeve_ready", None),
            # Split rank components (roadmap item #5)
            "selection_rank": getattr(c, "selection_rank", None),
            "timing_rank": getattr(c, "timing_rank", None),
            "portfolio_fit_rank": getattr(c, "portfolio_fit_rank", None),
            "final_rank": c.final_rank,
        }
        for c in disc_result.candidates
    ]
    state["cached_discovery_meta"] = {
        "screened_count": disc_result.screened_count,
        "after_momentum_screen": disc_result.after_momentum_screen,
        "after_quick_filter": disc_result.after_quick_filter,
        "after_corr_filter": disc_result.after_corr_filter,
        "after_quick_rank": disc_result.after_quick_rank,
        "fully_scored": disc_result.fully_scored,
        "run_time_seconds": disc_result.run_time_seconds,
        "fx_penalties_applied": disc_result.fx_penalties_applied,
    }
    state["last_discovery_run"] = datetime.now().isoformat()

    save_state(state)
    logger.info("Discovery results saved to state (%d candidates).", n_candidates)

    # Record picks for backtest tracking
    n_recorded = 0
    try:
        n_recorded = record_discovery_picks(disc_result.candidates)
        logger.info("Recorded %d discovery picks for backtest.", n_recorded)
    except Exception as e:
        logger.warning("Failed to record discovery picks: %s", e)

    return n_recorded


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def _evaluate_swaps(
    results: list[dict],
    cached_candidates: list[dict],
    state: dict,
) -> list[dict]:
    """Evaluate discovery candidates against the weakest holdings (multi-swap).

    Returns list of swap recommendations that pass ALL hurdles:
    1. candidate.aggregate_score - target.aggregate_score >= HURDLE_RATE
    2. candidate.portfolio_fit_score >= PORTFOLIO_FIT_MIN
    3. candidate is not on cooldown

    Multi-swap: evaluates up to MAX_SWAPS_PER_RUN weakest holdings as swap targets.
    Each candidate can only be recommended once, and each holding can only be
    swapped out once per run.
    """
    if not results or not cached_candidates:
        return []

    hurdle = getattr(config, "HURDLE_RATE", 0.20)
    fit_min = getattr(config, "PORTFOLIO_FIT_MIN", 0.50)
    max_swaps = getattr(config, "DISCOVERY_MAX_SWAPS_PER_RUN", 3)
    swap_threshold = getattr(config, "SWAP_CANDIDATE_THRESHOLD", -0.10)

    # Find swap-eligible holdings — prioritise exit-flagged holdings first.
    # Exit reconciliation sets _exit_override=True and _exit_posterior on holdings
    # downgraded by stop/momentum/decay signals. These should be swapped before
    # merely low-scoring holdings, since the exit engine has identified active risk.
    exit_flagged = [
        r for r in results
        if r.get("_exit_override") and r.get("final_action", r.get("action")) in ("SELL", "STRONG SELL")
    ]
    # Sort exit-flagged by posterior score ascending (worst risk first)
    exit_flagged.sort(key=lambda r: r.get("_exit_posterior", r.get("aggregate_score", 0)))

    # Then add score-based candidates (not already in exit list)
    exit_tickers = {r["ticker"] for r in exit_flagged}
    sorted_by_score = sorted(results, key=lambda r: r.get("aggregate_score", 0))
    score_eligible = [
        r for r in sorted_by_score
        if r.get("aggregate_score", 0) < swap_threshold and r["ticker"] not in exit_tickers
    ]

    # Combine: exit-flagged first, then score-based
    swap_eligible = exit_flagged + score_eligible

    # Even if none qualify, always consider the single weakest
    if not swap_eligible and sorted_by_score:
        swap_eligible = [sorted_by_score[0]]

    # Sort candidates by final discovery decision rank descending (best first)
    sorted_candidates = sorted(
        cached_candidates,
        key=lambda c: (c.get("final_rank", c.get("aggregate_score", 0)), c.get("aggregate_score", 0)),
        reverse=True,
    )

    swap_recs = []
    used_candidates = set()
    used_holdings = set()

    for target in swap_eligible:
        if len(swap_recs) >= max_swaps:
            break

        target_ticker = target["ticker"]
        target_score = target.get("_exit_posterior", target.get("aggregate_score", 0))

        if target_ticker in used_holdings:
            continue

        for cand in sorted_candidates:
            cand_ticker = cand.get("ticker", "")
            if cand_ticker in used_candidates:
                continue

            cand_score = cand.get("final_rank", cand.get("aggregate_score", 0))
            cand_fit = cand.get("portfolio_fit_score", 0)
            cand_action = cand.get("action", "NEUTRAL")
            delta = cand_score - target_score

            passes_hurdle = delta >= hurdle
            passes_fit = cand_fit >= fit_min
            passes_action = cand_action in ("BUY", "STRONG BUY")
            on_cooldown = is_on_cooldown(state, cand_ticker)
            recommended = passes_action and passes_hurdle and passes_fit and not on_cooldown

            # Log every evaluation for audit
            _log_decision("swap_eval", {
                "candidate": cand_ticker,
                "candidate_score": round(cand_score, 3),
                "candidate_action": cand_action,
                "candidate_fit": round(cand_fit, 3),
                "target": target_ticker,
                "target_score": round(target_score, 3),
                "delta": round(delta, 3),
                "passes_action": passes_action,
                "passes_hurdle": passes_hurdle,
                "passes_fit": passes_fit,
                "on_cooldown": on_cooldown,
                "recommended": recommended,
            })

            if recommended:
                set_cooldown(state, cand_ticker)
                used_candidates.add(cand_ticker)
                used_holdings.add(target_ticker)
                swap_recs.append({
                    "candidate": cand,
                    "weakest_ticker": target_ticker,
                    "weakest_score": round(target_score, 3),
                    "score_delta": round(delta, 3),
                })
                break  # Move to next target holding

    return swap_recs


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_orchestrator(
    dry_run: bool = False,
    force_discovery: bool = False,
    portfolio_only: bool = False,
) -> dict:
    """Main orchestration entry point.

    Returns a summary dict for logging / testing.
    """
    start = time.time()
    max_runtime = getattr(config, "ORCHESTRATOR_MAX_RUNTIME", 3600)
    summary = {
        "dry_run": dry_run,
        "portfolio_ran": False,
        "discovery_ran": False,
        "alerts": [],
        "swap_recs": [],
        "email_sent": False,
        "error": None,
    }

    def _check_timeout(stage: str):
        """Raise if total runtime exceeds max. Ensures process always completes."""
        elapsed = time.time() - start
        if elapsed > max_runtime:
            raise TimeoutError(f"Orchestrator timeout after {elapsed:.0f}s in {stage} "
                               f"(limit: {max_runtime}s). Partial results will be used.")

    _log_decision("run_start", {
        "dry_run": dry_run,
        "force_discovery": force_discovery,
        "portfolio_only": portfolio_only,
    })

    # --- Paper trading: resolve yesterday's pending signals (T+1 fills) ---
    if _paper_trading_enabled_for_run(dry_run):
        try:
            _init_paper_db()
            n_resolved = resolve_pending_signals()
            if n_resolved:
                logger.info("Paper trading: resolved %d pending signals.", n_resolved)
        except Exception as e:
            logger.warning("Paper trading fill resolution failed: %s", e)
    elif dry_run and getattr(config, "PAPER_TRADING_ENABLED", False):
        logger.info("[DRY RUN] Paper trading ledger writes skipped.")

    # --- Discovery backtest: evaluate picks that are 90+ days old ---
    try:
        n_evaluated = evaluate_matured_signals()
        if n_evaluated:
            logger.info("Signal backtest: evaluated %d matured signal-horizon pairs.", n_evaluated)
    except Exception as e:
        logger.warning("Discovery backtest evaluation failed: %s", e)

    # --- Load state and portfolio ---
    state = load_state()
    prune_expired_cooldowns(state)

    if getattr(config, "ML_DRIFT_MONITOR_ENABLED", True):
        try:
            from utils.drift_monitor import get_drift_status

            drift = get_drift_status(write=True)
            drift_payload = dict(drift.__dict__)
            previous = state.get("drift_status") or {}
            consecutive = int(previous.get("consecutive_alerts", 0) or 0)
            if drift.drift_detected:
                consecutive += 1
            else:
                consecutive = 0
            drift_payload["consecutive_alerts"] = consecutive
            state["drift_status"] = drift_payload
            summary["drift_status"] = drift_payload
            if drift.drift_detected:
                _log_decision("drift_alert", drift_payload)
                logger.warning("Drift monitor alert: %s", drift_payload)
            if consecutive >= 2:
                runtime_overrides = state.setdefault("runtime_overrides", {})
                runtime_overrides["ML_RANKER_MODE"] = "shadow"
                runtime_overrides["ML_RANKER_SHADOW_ONLY"] = True
                runtime_overrides["percentile_action_tightening"] = 0.5
                _log_decision("drift_debounce_confirmed", drift_payload)
                logger.warning("Drift confirmed on consecutive runs; ML override set to shadow.")

            # Threshold-learner hook: persist drift-active flag so the next
            # discovery run picks the conservative profile (Russo & Van Roy
            # 2014 — Thompson sampling with safety-first override).
            if getattr(config, "THRESHOLD_LEARNER_ENABLED", True):
                try:
                    from engine.threshold_learner import load_state as _load_tl_state
                    from engine.threshold_learner import save_state as _save_tl_state
                    tl_state = _load_tl_state()
                    tl_state.drift_active = bool(consecutive >= 2)
                    _save_tl_state(tl_state)
                except Exception as exc:
                    logger.debug("Threshold-learner drift sync failed: %s", exc)
        except Exception as e:
            logger.warning("Drift monitor failed: %s", e)

    try:
        holdings = load_portfolio()
    except Exception as e:
        logger.error("Failed to load portfolio: %s", e)
        summary["error"] = f"portfolio_load_failed: {e}"
        _log_decision("error", {"stage": "load_portfolio", "error": str(e)})
        return summary

    if not holdings:
        logger.warning("Portfolio is empty — nothing to analyse.")
        summary["error"] = "empty_portfolio"
        return summary

    # --- Step 1: Portfolio analysis ---
    logger.info("Running portfolio analysis on %d holdings...", len(holdings))
    try:
        results, risk_data, position_weights = _run_with_timeout(
            _run_portfolio_analysis,
            args=(holdings,),
            timeout_seconds=getattr(config, "PORTFOLIO_ANALYSIS_TIMEOUT", 300),
        )
        summary["portfolio_ran"] = True
        state["last_portfolio_run"] = datetime.now().isoformat()
        _log_decision("portfolio_done", {
            "n_holdings": len(results),
            "risk_score": risk_data.get("risk_score", 0),
        })
    except TimeoutError:
        logger.error("Portfolio analysis timed out. Aborting run.")
        summary["error"] = "portfolio_timeout"
        _log_decision("portfolio_timeout")
        save_state(state)
        return summary
    except Exception as e:
        logger.error("Portfolio analysis failed: %s", e)
        summary["error"] = f"portfolio_failed: {e}"
        _log_decision("error", {"stage": "portfolio_analysis", "error": str(e)})
        save_state(state)
        return summary

    _check_timeout("portfolio_analysis")

    # --- Step 2: VIX regime ---
    vix_regime = _get_regime()
    logger.info("VIX regime: %s (level %.1f, percentile %.0f%%)",
                vix_regime["regime_label"], vix_regime["vix_level"],
                vix_regime["vix_percentile"])

    # --- Step 2b: Portfolio optimisation ---
    portfolio_alloc = None
    try:
        from engine.portfolio_optimizer import optimize_portfolio as _opt_portfolio
        portfolio_alloc = _opt_portfolio(results, holdings, risk_data, position_weights, vix_regime)
        logger.info("Portfolio optimiser: expected return %.1f%%, vol %.1f%%, Sharpe %.2f, turnover %.1f%%",
                     portfolio_alloc.portfolio_expected_return * 100,
                     portfolio_alloc.portfolio_volatility * 100,
                     portfolio_alloc.portfolio_sharpe,
                     portfolio_alloc.turnover * 100)
        if portfolio_alloc.rebalance_trades:
            for t in portfolio_alloc.rebalance_trades:
                logger.info("  Rebalance: %s %s %+.1f%% (£%.0f)",
                             t["direction"], t["ticker"], t["delta_pct"], t["trade_value"])
        _log_decision("portfolio_optimised", {
            "expected_return": portfolio_alloc.portfolio_expected_return,
            "volatility": portfolio_alloc.portfolio_volatility,
            "sharpe": portfolio_alloc.portfolio_sharpe,
            "turnover": portfolio_alloc.turnover,
            "n_trades": len(portfolio_alloc.rebalance_trades),
            "method": portfolio_alloc.method,
        })
    except Exception as e:
        logger.warning("Portfolio optimisation failed: %s", e)

    # --- Step 2c: Record portfolio signals for comprehensive backtest ---
    try:
        # Get adaptive weights currently in use (for point-in-time logging)
        _active_weights = None
        try:
            from engine.discovery_backtest import get_adaptive_weights as _get_aw
            _active_weights = _get_aw(source="portfolio", horizon="90d")
        except Exception:
            pass
        if _active_weights is None:
            _active_weights = dict(config.WEIGHTS)

        n_bt = record_portfolio_signals(
            results, position_weights, vix_regime,
            optimizer_alloc=portfolio_alloc,
            pillar_weights=_active_weights,
        )
        if n_bt:
            logger.info("Recorded %d portfolio signals for backtest.", n_bt)
    except Exception as e:
        logger.warning("Failed to record portfolio signals for backtest: %s", e)

    # --- Step 3: Exit intelligence and final risk-adjusted alerts ---
    summary["exit_signals"] = []
    try:
        from engine.exit_engine import assess_exits
        exit_signals = assess_exits(results, holdings)
        for es in exit_signals:
            logger.info("EXIT SIGNAL [%s] %s — %s: %s",
                         es.severity.upper(), es.ticker, es.signal_type, es.message)
            _log_decision("exit_signal", {
                "ticker": es.ticker,
                "type": es.signal_type,
                "severity": es.severity,
                "message": es.message,
                "score": es.current_score,
            })
        reconcile_actions_with_exits(results, exit_signals)
        result_map = {r["ticker"]: r for r in results}
        summary["exit_signals"] = [
            exit_signal_to_dict(e, result_map.get(e.ticker))
            for e in exit_signals
        ]
        for es in exit_signals:
            r = result_map.get(es.ticker)
            if r and r.get("_exit_override"):
                logger.info(
                    "EXIT RECONCILE: %s %s -> %s "
                    "(prior=%.3f, exit_score=%.3f, penalty=%.3f, posterior=%.3f)",
                    es.ticker, r.get("base_action"), r.get("final_action"),
                    r.get("aggregate_score", 0) or 0,
                    es.exit_score,
                    r.get("_exit_penalty", 0) or 0,
                    r.get("_exit_posterior", 0) or 0,
                )
    except Exception as e:
        logger.warning("Exit intelligence failed: %s", e)

    # Final alerts should use the post-risk-adjustment action, not the alpha prior.
    alerts = [r for r in results if r.get("final_action", r.get("action")) in ("SELL", "STRONG SELL")]
    summary["alerts"] = [
        {
            "ticker": a["ticker"],
            "action": a.get("final_action", a.get("action")),
            "base_action": a.get("base_action", a.get("action")),
            "prior_score": a.get("aggregate_score"),
            "posterior_score": a.get("_exit_posterior"),
            "exit_score": a.get("exit_score"),
            "exit_penalty": a.get("_exit_penalty"),
            "current_price": a.get("current_price"),
            "structural_stop_loss": a.get("structural_stop_loss", a.get("stop_loss")),
            "trailing_exit_stop": a.get("trailing_exit_stop"),
        }
        for a in alerts
    ]
    for a in alerts:
        logger.info("FINAL ALERT: %s â€” %s (base: %s, prior=%.3f, posterior=%s)",
                    a["ticker"],
                    a.get("final_action", a.get("action")),
                    a.get("base_action", a.get("action")),
                    a.get("aggregate_score", 0) or 0,
                    a.get("_exit_posterior", "n/a"))
        _log_decision("alert_found", {
            "ticker": a["ticker"],
            "action": a.get("final_action", a.get("action")),
            "base_action": a.get("base_action", a.get("action")),
            "score": a.get("aggregate_score", 0),
            "posterior_score": a.get("_exit_posterior"),
        })

    # --- Paper trading: log SELL signals ---
    if _paper_trading_enabled_for_run(dry_run):
        for a in alerts:
            try:
                _log_paper_signal(
                    ticker=a["ticker"],
                    side="SELL",
                    source="portfolio_alert",
                    signal_price=a.get("current_price", 0),
                    quantity=a.get("quantity"),
                    score=a.get("aggregate_score"),
                    action=a.get("final_action", a.get("action")),
                )
            except Exception as e:
                logger.warning("Paper trading signal log failed for %s: %s", a["ticker"], e)

    _check_timeout("pre_discovery")

    # --- Step 4: Discovery (weekly or forced) ---
    if not portfolio_only:
        run_disc = should_run_discovery(state) or force_discovery
        if run_disc:
            # Refresh the dynamic universe supplement before assembly so
            # ETF-graduated tickers pass liquidity/mcap validation (rather
            # than being injected directly with thin metadata).
            if getattr(config, "DISCOVERY_RECONSTITUTE_UNIVERSE_ENABLED", True):
                if not _dynamic_universe_cache_fresh():
                    _reconstitute_dynamic_universe()
            else:
                logger.info("Dynamic universe reconstitution skipped by config")
            logger.info("Running Global Discovery Engine v4 (multi-lens, 240 deep-scored, may take 30-60 min)...")
            timeout = getattr(config, "DISCOVERY_TIMEOUT", 900)
            try:
                disc_result = _run_with_timeout(
                    _run_discovery_pipeline,
                    args=(holdings, risk_data),
                    timeout_seconds=timeout,
                )
                if disc_result.error:
                    logger.warning("Discovery completed with error: %s", disc_result.error)
                    _log_decision("discovery_error", {"error": disc_result.error})
                else:
                    n_candidates = len(disc_result.candidates)
                    logger.info("Discovery complete: %d candidates in %.1fs",
                                n_candidates, disc_result.run_time_seconds)

                    # Persist results + record backtest picks (shared with UI button)
                    save_discovery_results(disc_result, state)
                    summary["discovery_ran"] = True

                    _log_decision("discovery_done", {
                        "screened": disc_result.screened_count,
                        "after_momentum": disc_result.after_momentum_screen,
                        "after_filter": disc_result.after_quick_filter,
                        "after_corr": disc_result.after_corr_filter,
                        "after_rank": disc_result.after_quick_rank,
                        "fully_scored": disc_result.fully_scored,
                        "final_candidates": n_candidates,
                        "runtime_s": disc_result.run_time_seconds,
                    })
            except TimeoutError:
                logger.warning("Discovery timed out after %ds. Using cached results.", timeout)
                _log_decision("discovery_timeout", {"timeout_s": timeout})
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error("Discovery failed: %s\n%s", e, tb)
                _log_decision("discovery_error", {
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": tb[-2000:],
                })
        else:
            logger.info("Discovery not scheduled (last run: %s). Using cached results.",
                        state.get("last_discovery_run", "never"))

    _skip_post_discovery = False
    try:
        _check_timeout("post_discovery")
    except TimeoutError:
        logger.warning("Timeout after discovery — saving state with results so far. "
                       "Skipping swap evaluation; will proceed to email/summary.")
        save_state(state)
        _skip_post_discovery = True

    # --- Step 5: Evaluate swap opportunities ---
    if _skip_post_discovery:
        swap_recs = []
        logger.info("Skipping swap evaluation due to timeout.")
    else:
        cached = get_cached_discovery(state) if not portfolio_only else []
        swap_recs = _evaluate_swaps(results, cached, state)
    summary["swap_recs"] = [
        {"candidate": s["candidate"]["ticker"], "delta": s["score_delta"]}
        for s in swap_recs
    ]

    # Persist cooldowns set during swap evaluation
    if swap_recs:
        save_state(state)
        for s in swap_recs:
            logger.info("SWAP: Sell %s (%.3f) -> Buy %s (%.3f), delta +%.3f",
                        s["weakest_ticker"], s["weakest_score"],
                        s["candidate"]["ticker"], s["candidate"]["aggregate_score"],
                        s["score_delta"])

        # --- Paper trading: log both legs of each swap ---
        if _paper_trading_enabled_for_run(dry_run):
            for s in swap_recs:
                cand = s["candidate"]
                # SELL leg — the weak holding being swapped out
                weak_result = next((r for r in results if r["ticker"] == s["weakest_ticker"]), None)
                try:
                    _log_paper_signal(
                        ticker=s["weakest_ticker"],
                        side="SELL",
                        source="discovery_swap",
                        signal_price=weak_result["current_price"] if weak_result else 0,
                        quantity=weak_result.get("quantity") if weak_result else None,
                        score=s["weakest_score"],
                        action="SELL",
                        swap_from=None,
                    )
                except Exception as e:
                    logger.warning("Paper signal (swap SELL) failed for %s: %s", s["weakest_ticker"], e)

                # BUY leg — the discovery candidate (fetch live price for signal)
                try:
                    from utils.data_fetch import get_current_price
                    cand_price = get_current_price(cand["ticker"]) or 0
                    _log_paper_signal(
                        ticker=cand["ticker"],
                        side="BUY",
                        source="discovery_swap",
                        signal_price=cand_price,
                        quantity=weak_result.get("quantity") if weak_result else None,
                        score=cand.get("aggregate_score"),
                        action=cand.get("action", "BUY"),
                        swap_from=s["weakest_ticker"],
                    )
                except Exception as e:
                    logger.warning("Paper signal (swap BUY) failed for %s: %s", cand["ticker"], e)

    # --- Step 6: Build and send email ---
    # Always send when discovery ran (user wants confirmation of completion).
    # Otherwise send only if there are alerts or swap recommendations.
    discovery_ran = summary.get("discovery_ran", False)
    has_exit_signals = bool(summary.get("exit_signals"))
    has_triggers = bool(alerts) or bool(swap_recs) or discovery_ran or has_exit_signals

    if has_triggers:
        logger.info("Building alert email (alerts=%d, swaps=%d)...", len(alerts), len(swap_recs))
        subject, html = build_alert_email(
            results, risk_data, position_weights,
            vix_regime, alerts, swap_recs,
            dry_run=dry_run,
            optimizer_alloc=portfolio_alloc,
            discovery_candidates=get_cached_discovery(state),
            exit_signals=summary.get("exit_signals"),
            discovery_meta=state.get("cached_discovery_meta"),
            artifact_timestamps={
                "portfolio": (state.get("cached_portfolio") or {}).get("timestamp") or state.get("last_portfolio_run"),
                "discovery": state.get("last_discovery_run"),
                "optimizer": (state.get("cached_optimizer") or {}).get("timestamp"),
                "exit": ((state.get("cached_exit_signals") or {}).get("timestamp")),
            },
            discovery_ran=discovery_ran,
        )

        success = send_email(subject, html, dry_run=dry_run)
        summary["email_sent"] = success

        if dry_run:
            _log_decision("email_dry_run", {
                "subject": subject,
                "html_length": len(html),
                "alerts": len(alerts),
                "swaps": len(swap_recs),
            })
            logger.info("[DRY RUN] Email would be: %s", subject)
        elif success:
            state["last_email_sent"] = datetime.now().isoformat()
            _log_decision("email_sent", {"subject": subject})
        else:
            _log_decision("email_failed", {"subject": subject})
            logger.error("Email send failed — check SMTP credentials and network.")
    else:
        logger.info("No alerts or swap opportunities. Exiting silently.")
        _log_decision("no_action", {
            "n_alerts": 0,
            "n_swaps": 0,
            "weakest_ticker": (sorted(results, key=lambda r: r.get("aggregate_score", 0))[0]["ticker"]
                               if results else None),
            "weakest_score": (sorted(results, key=lambda r: r.get("aggregate_score", 0))[0].get("aggregate_score", 0)
                              if results else None),
        })

    # --- Cache all artifacts for instant dashboard load ---
    try:
        state["cached_portfolio"] = {
            "results": results,
            "risk_data": {k: v for k, v in risk_data.items() if k != "correlation_matrix"},
            "position_weights": position_weights,
            "vix_regime": vix_regime,
            "timestamp": datetime.now().isoformat(),
        }
        # Correlation matrix as nested list (numpy → JSON)
        if risk_data.get("correlation_matrix") is not None:
            try:
                cm = risk_data["correlation_matrix"]
                if hasattr(cm, "values"):  # pandas DataFrame
                    state["cached_portfolio"]["correlation_matrix"] = cm.values.tolist()
                elif hasattr(cm, "tolist"):  # numpy array
                    state["cached_portfolio"]["correlation_matrix"] = cm.tolist()
            except Exception:
                pass

        # Portfolio optimizer
        if portfolio_alloc:
            state["cached_optimizer"] = {
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
                    for h in portfolio_alloc.holdings
                ],
                "portfolio_expected_return": portfolio_alloc.portfolio_expected_return,
                "portfolio_volatility": portfolio_alloc.portfolio_volatility,
                "portfolio_sharpe": portfolio_alloc.portfolio_sharpe,
                "risk_free_rate": portfolio_alloc.risk_free_rate,
                "method": portfolio_alloc.method,
                "sector_weights": portfolio_alloc.sector_weights,
                "fx_exposure": portfolio_alloc.fx_exposure,
                "turnover": portfolio_alloc.turnover,
                "rebalance_trades": portfolio_alloc.rebalance_trades,
                "warnings": portfolio_alloc.warnings,
                "timestamp": datetime.now().isoformat(),
            }

        # Exit signals
        if summary.get("exit_signals"):
            state["cached_exit_signals"] = {
                "signals": summary["exit_signals"],
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("Cached all artifacts for dashboard instant load.")
    except Exception as e:
        logger.warning("Failed to cache artifacts: %s", e)
    finally:
        # Always save state — ensures discovery results, cooldowns, and
        # portfolio cache are persisted even if artifact caching partially fails
        save_state(state)

    # Self-monitoring: analyze recent telemetry and emit recommendations.
    # Reads the decision log just written above; never mutates config.
    if getattr(config, "AUTO_TUNE_ENABLED", True):
        try:
            from engine import auto_tune
            report = auto_tune.run(
                window_days=int(getattr(config, "AUTO_TUNE_WINDOW_DAYS", 30)),
            )
            summary_line = auto_tune.summarize(report)
            logger.info(summary_line)
            _log_decision("auto_tune_done", {
                "n_recommendations": len(report.recommendations),
                "params": [r.param for r in report.recommendations],
                "summary": summary_line,
            })
        except Exception as e:
            logger.warning("auto_tune run failed: %s", e)

    elapsed = round(time.time() - start, 1)
    _log_decision("run_complete", {"elapsed_s": elapsed, "summary": summary})
    logger.info("Orchestrator finished in %.1fs", elapsed)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ISA Portfolio Autonomous Orchestrator (v4.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python daily_orchestrator.py --dry-run --portfolio-only   Quick test, no email
  python daily_orchestrator.py --dry-run --force-discovery  Test full pipeline
  python daily_orchestrator.py                              Production run
""",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log decisions and build email but do not actually send it",
    )
    parser.add_argument(
        "--force-discovery", action="store_true",
        help="Run discovery even if not scheduled (overrides weekly frequency)",
    )
    parser.add_argument(
        "--portfolio-only", action="store_true",
        help="Skip discovery entirely — only evaluate current holdings",
    )
    args = parser.parse_args()

    _setup_logging(dry_run=args.dry_run)

    if args.dry_run:
        logger.info("=== DRY RUN MODE — no emails will be sent ===")

    lock = None
    try:
        lock = _acquire_orchestrator_lock(
            dry_run=args.dry_run,
            force_discovery=args.force_discovery,
            portfolio_only=args.portfolio_only,
        )
        if lock.reclaimed:
            _log_decision("lock_reclaimed", {
                "path": str(lock.path),
                "previous": lock.reclaimed,
            })
        logger.info("Acquired orchestrator lock: %s", lock.path)

        summary = run_orchestrator(
            dry_run=args.dry_run,
            force_discovery=args.force_discovery,
            portfolio_only=args.portfolio_only,
        )
    except OrchestratorAlreadyRunning as e:
        md = e.metadata or {}
        logger.warning(
            "Another orchestrator run is already active (pid=%s, started=%s, host=%s). Exiting without running.",
            md.get("pid", "unknown"),
            md.get("started_at", "unknown"),
            md.get("hostname", "unknown"),
        )
        _log_decision("run_skipped_lock", {
            "active_run": md,
            "argv": list(sys.argv),
        })
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("Orchestrator interrupted by user (SIGINT/Ctrl+C)")
        _log_decision("interrupted", {"reason": "KeyboardInterrupt"})
        sys.exit(130)
    except Exception as e:
        # Catch-all: no pipeline run should ever crash silently
        import traceback
        tb = traceback.format_exc()
        logger.critical("FATAL: Orchestrator crashed with unhandled exception:\n%s", tb)
        _log_decision("fatal_crash", {
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": tb[-2000:],  # Last 2000 chars of traceback
        })
        sys.exit(2)
    finally:
        if lock is not None:
            lock.release()

    # Exit code: 0 = success, 1 = error
    sys.exit(0 if summary.get("error") is None else 1)


if __name__ == "__main__":
    main()
