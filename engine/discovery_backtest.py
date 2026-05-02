"""Comprehensive Signal Backtest — Track ALL recommendations and adapt ALL models.

Captures every signal the system generates (portfolio + discovery + swaps),
with every metric available at signal time. Evaluates actual outcomes at
5d, 10d, 30d, 60d, 90d horizons. Feeds back into:
- Main portfolio scoring weights (config.WEIGHTS)
- Discovery final ranking weights
- Action threshold calibration
- Forecast accuracy tracking
- Stop-loss / take-profit hit rates

Data stored in paper_trading.db for durability.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

import config
from engine.factors import compute_factor_scores_from_result

logger = logging.getLogger(__name__)

from engine.paper_trading import _connect


def _json_or_none(value) -> str | None:
    """Serialize a small payload to JSON for storage, returning None on empty/invalid."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)) and not value:
        return None
    if isinstance(value, dict) and not value:
        return None
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BACKTEST_SCHEMA = """
-- Every signal the system produces (portfolio analysis + discovery + swaps)
CREATE TABLE IF NOT EXISTS signal_backtest (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date            TEXT    NOT NULL,
    ticker              TEXT    NOT NULL,
    name                TEXT,
    source              TEXT    NOT NULL,       -- portfolio | discovery | swap_sell | swap_buy
    signal_price        REAL    NOT NULL,

    -- Action & aggregate
    action              TEXT,                   -- STRONG BUY / BUY / KEEP / SELL / STRONG SELL
    aggregate_score     REAL,
    final_rank          REAL,                   -- portfolio-aware discovery rank

    -- 4 pillar scores
    technical_score     REAL,
    fundamental_score   REAL,
    sentiment_score     REAL,
    forecast_score      REAL,

    -- Key sub-metrics (for granular analysis)
    rsi                 REAL,
    adx                 REAL,
    bb_pct              REAL,
    pe_ratio            REAL,
    peg_ratio           REAL,
    revenue_growth      REAL,
    roe                 REAL,
    short_pct           REAL,
    news_score          REAL,
    momentum_score      REAL,
    quality_factor_score REAL,
    qmj_factor_score    REAL,
    value_factor_score   REAL,
    momentum_factor_score REAL,
    volatility_factor_score REAL,
    bab_factor_score    REAL,
    turnover_cost_score REAL,
    quality_score_fundamental REAL,
    gross_profitability REAL,
    fcf_to_assets       REAL,
    earnings_stability  REAL,
    eps_growth_variance_5y REAL,
    factor_mom_momentum_tilt REAL,
    factor_mom_value_tilt REAL,
    factor_mom_quality_tilt REAL,
    factor_mom_momentum_ret_1m REAL,
    factor_mom_momentum_ret_3m REAL,
    factor_mom_value_ret_1m REAL,
    factor_mom_value_ret_3m REAL,
    factor_mom_quality_ret_1m REAL,
    factor_mom_quality_ret_3m REAL,
    network_momentum    REAL,
    qa_sentiment_score  REAL,
    institutional_prior_score REAL,
    institutional_prior_percentile REAL,
    institutional_prior_confidence REAL,
    institutional_prior_components TEXT,
    ready_contract_status TEXT,
    ready_contract_reasons TEXT,
    strong_buy_eligible INTEGER,

    -- Forecast specifics
    forecast_price_5d   REAL,
    forecast_price_63d  REAL,

    -- Stop-loss / take-profit at signal time
    stop_loss           REAL,
    take_profit         REAL,

    -- Point-in-time raw features (added v2 — full feature store)
    macd_signal         REAL,                   -- MACD - signal line
    sma_50              REAL,
    sma_200             REAL,
    stoch_k             REAL,
    williams_r          REAL,
    obv_trend           TEXT,                   -- rising / falling / flat
    atr                 REAL,
    profit_margin       REAL,
    fcf_yield           REAL,
    debt_equity         REAL,
    inst_ownership      REAL,
    insider_net         TEXT,                   -- net_buy / net_sell / neutral
    analyst_upside      REAL,                   -- % upside to consensus target
    analyst_rec         TEXT,                   -- buy / hold / sell
    reddit_score        REAL,
    fmp_news_score      REAL,
    earnings_proximity  INTEGER,               -- days to next earnings
    return_10d_prior    REAL,                   -- 10-day momentum at signal time
    return_30d_prior    REAL,                   -- 30-day momentum at signal time
    return_90d_prior    REAL,                   -- 90-day momentum at signal time
    vol_20d             REAL,                   -- 20-day realized vol at signal time
    vix_percentile      REAL,
    tnx_level           REAL,                   -- 10Y yield at signal time
    optimal_weight      REAL,                   -- from portfolio optimizer
    pillar_weights_json TEXT,                   -- weights used (JSON: {"tech":0.3,...})
    entry_lens          TEXT,                   -- momentum / value / quality
    entry_price         REAL,                   -- planned limit entry
    entry_method        TEXT,                   -- selected entry rule
    fill_probability    REAL,                   -- estimated fill probability
    planned_position_weight REAL,               -- planned weight from strategy layer
    planned_risk_amount REAL,                   -- max loss at stop in base currency
    position_sizing_method TEXT,                -- stop budget / Kelly-capped
    kelly_cap_fraction  REAL,                   -- empirical half-Kelly cap used
    r_r_ratio           REAL,                   -- target / stop reward-to-risk

    -- Context
    sector              TEXT,
    exchange            TEXT,
    regime              TEXT,                   -- BULL / NEUTRAL / BEAR
    vix_level           REAL,
    portfolio_weight    REAL,                   -- current weight at signal time

    -- Multi-horizon evaluation (filled later)
    price_5d            REAL,
    price_10d           REAL,
    price_30d           REAL,
    price_60d           REAL,
    price_90d           REAL,
    return_5d           REAL,
    return_10d          REAL,
    return_30d          REAL,
    return_60d          REAL,
    return_90d          REAL,
    spy_return_90d      REAL,
    beat_market         INTEGER,

    -- Stop/target hit tracking
    stop_hit            INTEGER,               -- 1 if price touched stop_loss within 90d
    stop_hit_day        INTEGER,               -- day # when stop was hit
    target_hit          INTEGER,               -- 1 if price touched take_profit within 90d
    target_hit_day      INTEGER,               -- day # when target was hit

    -- Forecast accuracy
    actual_price_5d     REAL,
    actual_price_63d    REAL,
    forecast_error_5d   REAL,                  -- absolute % error
    forecast_error_63d  REAL,

    -- Action correctness (was the action right?)
    action_correct      INTEGER,               -- 1 if action aligned with actual outcome

    evaluated_5d        INTEGER NOT NULL DEFAULT 0,
    evaluated_10d       INTEGER NOT NULL DEFAULT 0,
    evaluated_30d       INTEGER NOT NULL DEFAULT 0,
    evaluated_60d       INTEGER NOT NULL DEFAULT 0,
    evaluated_90d       INTEGER NOT NULL DEFAULT 0
);

-- Aggregated pillar effectiveness (updated after evaluations)
CREATE TABLE IF NOT EXISTS pillar_effectiveness (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at              TEXT    NOT NULL,
    source                  TEXT    NOT NULL,   -- portfolio | discovery | all
    pillar                  TEXT    NOT NULL,
    horizon                 TEXT    NOT NULL,   -- 30d | 60d | 90d
    regime                  TEXT,
    information_coefficient REAL,
    hit_rate                REAL,
    avg_return_high         REAL,
    avg_return_low          REAL,
    sample_size             INTEGER
);

-- Action calibration (how accurate are action labels?)
CREATE TABLE IF NOT EXISTS action_calibration (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at      TEXT    NOT NULL,
    action          TEXT    NOT NULL,           -- STRONG BUY / BUY / KEEP / SELL / STRONG SELL
    source          TEXT    NOT NULL,
    avg_return_90d  REAL,
    hit_rate        REAL,                       -- % where direction was correct
    sample_size     INTEGER
);

-- Regime effectiveness
CREATE TABLE IF NOT EXISTS regime_effectiveness (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at      TEXT    NOT NULL,
    regime          TEXT    NOT NULL,
    avg_return_90d  REAL,
    best_pillar     TEXT,                       -- which pillar had highest IC in this regime
    sample_size     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_sb_ticker ON signal_backtest(ticker);
CREATE INDEX IF NOT EXISTS idx_sb_source ON signal_backtest(source);
CREATE INDEX IF NOT EXISTS idx_sb_run_date ON signal_backtest(run_date);
CREATE INDEX IF NOT EXISTS idx_sb_evaluated ON signal_backtest(evaluated_90d);

-- Keep legacy tables for backward compatibility
CREATE TABLE IF NOT EXISTS discovery_picks AS SELECT * FROM signal_backtest WHERE 0;
CREATE TABLE IF NOT EXISTS discovery_pillar_stats AS SELECT * FROM pillar_effectiveness WHERE 0;
"""


_FEATURE_STORE_COLUMNS = [
    ("final_rank", "REAL"),
    ("price_5d", "REAL"), ("price_10d", "REAL"),
    ("return_5d", "REAL"), ("return_10d", "REAL"),
    ("evaluated_5d", "INTEGER NOT NULL DEFAULT 0"),
    ("evaluated_10d", "INTEGER NOT NULL DEFAULT 0"),
    ("macd_signal", "REAL"), ("sma_50", "REAL"), ("sma_200", "REAL"),
    ("stoch_k", "REAL"), ("williams_r", "REAL"), ("obv_trend", "TEXT"),
    ("atr", "REAL"), ("profit_margin", "REAL"), ("fcf_yield", "REAL"),
    ("debt_equity", "REAL"), ("inst_ownership", "REAL"), ("insider_net", "TEXT"),
    ("analyst_upside", "REAL"), ("analyst_rec", "TEXT"),
    ("reddit_score", "REAL"), ("fmp_news_score", "REAL"),
    ("quality_factor_score", "REAL"), ("value_factor_score", "REAL"),
    ("momentum_factor_score", "REAL"), ("volatility_factor_score", "REAL"),
    ("qmj_factor_score", "REAL"), ("bab_factor_score", "REAL"),
    ("turnover_cost_score", "REAL"),
    ("quality_score_fundamental", "REAL"), ("gross_profitability", "REAL"),
    ("fcf_to_assets", "REAL"), ("earnings_stability", "REAL"),
    ("eps_growth_variance_5y", "REAL"),
    ("factor_mom_momentum_tilt", "REAL"), ("factor_mom_value_tilt", "REAL"),
    ("factor_mom_quality_tilt", "REAL"),
    ("factor_mom_momentum_ret_1m", "REAL"), ("factor_mom_momentum_ret_3m", "REAL"),
    ("factor_mom_value_ret_1m", "REAL"), ("factor_mom_value_ret_3m", "REAL"),
    ("factor_mom_quality_ret_1m", "REAL"), ("factor_mom_quality_ret_3m", "REAL"),
    ("network_momentum", "REAL"), ("qa_sentiment_score", "REAL"),
    ("earnings_proximity", "INTEGER"),
    ("return_10d_prior", "REAL"), ("return_30d_prior", "REAL"),
    ("return_90d_prior", "REAL"), ("vol_20d", "REAL"),
    ("downside_vol_60d", "REAL"), ("max_dd_252d", "REAL"),
    ("vix_percentile", "REAL"), ("tnx_level", "REAL"),
    ("optimal_weight", "REAL"), ("pillar_weights_json", "TEXT"),
    ("entry_lens", "TEXT"), ("entry_price", "REAL"), ("entry_method", "TEXT"),
    ("fill_probability", "REAL"), ("planned_position_weight", "REAL"),
    ("planned_risk_amount", "REAL"), ("position_sizing_method", "TEXT"),
    ("kelly_cap_fraction", "REAL"), ("r_r_ratio", "REAL"),
    # PEAD factor (Martineau 2022)
    ("pead_factor_score", "REAL"), ("sue_score", "REAL"),
    ("revision_momentum_3m", "REAL"),
    # Enterprise factors (Piotroski 2000, Novy-Marx 2013, Greenblatt 2006)
    # Persisted at signal time for bucketed hit-rate diagnostics.
    ("f_score", "INTEGER"), ("f_score_coverage", "REAL"),
    ("ev_ebit", "REAL"), ("ev_ebit_score", "REAL"),
    ("gpa", "REAL"), ("gpa_score", "REAL"),
    # Late-cycle / stretch diagnostic: signal_price / sma_200 - 1
    ("price_vs_sma200_stretch", "REAL"),
    # Gate-v2 shadow telemetry and active trap safeguard outcome
    ("gate_v2_status", "TEXT"), ("gate_v2_reasons", "TEXT"),
    ("trap_safeguard_triggered", "INTEGER"), ("trap_safeguard_reason", "TEXT"),
    # Institutional prior and Ready Strong Buy contract
    ("institutional_prior_score", "REAL"),
    ("institutional_prior_percentile", "REAL"),
    ("institutional_prior_confidence", "REAL"),
    ("institutional_prior_components", "TEXT"),
    ("ready_contract_status", "TEXT"),
    ("ready_contract_reasons", "TEXT"),
    ("strong_buy_eligible", "INTEGER"),
    # Triple-barrier and meta-label fields (Lopez de Prado 2018)
    ("tb_label", "INTEGER"),
    ("tb_return", "REAL"),
    ("tb_days", "INTEGER"),
    ("tb_hit", "TEXT"),
    ("tb_horizon", "INTEGER"),
    ("tb_updated_at", "TEXT"),
    ("meta_success", "INTEGER"),
    ("meta_prob", "REAL"),
    # Cache-only sleeve composites used by Stage 5a.
    ("sleeve_momentum", "REAL"),
    ("sleeve_quality", "REAL"),
    ("sleeve_value", "REAL"),
    ("sleeve_low_risk", "REAL"),
    ("sleeve_pead", "REAL"),
    ("sleeve_ready", "REAL"),
    # Synthetic PIT replay provenance
    ("replay_version", "TEXT"),
    ("replay_asof", "TEXT"),
    # PIT-safe factor backfill provenance
    ("factors_backfilled", "INTEGER"),
    ("factors_source", "TEXT"),
    ("factors_backfilled_at", "TEXT"),
    ("factors_unavailable", "INTEGER"),
    ("factors_unavailable_reason", "TEXT"),
    # Action-gate telemetry (Tier 1-4 hard gates)
    ("altman_z", "REAL"),
    ("altman_zone", "TEXT"),
    ("beneish_m", "REAL"),
    ("action_gate_ceiling", "TEXT"),
    ("action_gate_flags_json", "TEXT"),
    ("threshold_profile", "TEXT"),
]


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL single-line comments (-- ...) from a statement."""
    return "\n".join(
        line for line in sql.splitlines()
        if not line.strip().startswith("--")
    ).strip()


def init_backtest_db():
    """Create tables if they don't exist, and migrate new columns."""
    with _connect() as conn:
        # Create tables one by one to handle "already exists" gracefully
        for raw_stmt in _BACKTEST_SCHEMA.split(";"):
            stmt = _strip_sql_comments(raw_stmt)
            if stmt:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # Table/index already exists

        # Migrate: add new feature store columns to existing table
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
        for col_name, col_type in _FEATURE_STORE_COLUMNS:
            if col_name not in existing:
                try:
                    conn.execute(f"ALTER TABLE signal_backtest ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        pillar_existing = {row[1] for row in conn.execute("PRAGMA table_info(pillar_effectiveness)").fetchall()}
        if "regime" not in pillar_existing:
            try:
                conn.execute("ALTER TABLE pillar_effectiveness ADD COLUMN regime TEXT")
            except sqlite3.OperationalError:
                pass


# ---------------------------------------------------------------------------
# Record signals — called by the orchestrator after every analysis
# ---------------------------------------------------------------------------

def _compute_prior_momentum(ticker: str) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute 10d, 30d, 90d prior returns and 20d vol at signal time."""
    try:
        data = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
        if data is None or len(data) < 20:
            return None, None, None, None
        closes = data["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        ret_10d = float(closes.iloc[-1] / closes.iloc[-min(10, len(closes))] - 1) * 100 if len(closes) >= 10 else None
        ret_30d = float(closes.iloc[-1] / closes.iloc[-min(30, len(closes))] - 1) * 100 if len(closes) >= 30 else None
        ret_90d = float(closes.iloc[-1] / closes.iloc[-min(90, len(closes))] - 1) * 100 if len(closes) >= 90 else None
        vol_20d = float(closes.pct_change().tail(20).std() * np.sqrt(252) * 100) if len(closes) >= 20 else None
        return ret_10d, ret_30d, ret_90d, vol_20d
    except Exception:
        return None, None, None, None


def record_portfolio_signals(
    results: list[dict],
    position_weights: list[dict] | None = None,
    regime: dict | None = None,
    optimizer_alloc=None,
    pillar_weights: dict | None = None,
) -> int:
    """Record all portfolio holding signals with full point-in-time features.

    Called after every portfolio analysis run. Captures every raw input
    available at decision time so future models can train on what was
    truly known, not just the final scores.
    """
    init_backtest_db()
    now = datetime.now().isoformat(timespec="seconds")
    today = now[:10]
    pw_map = {pw["ticker"]: pw.get("current_weight", 0) for pw in (position_weights or [])}
    regime_label = regime.get("regime_label", "NEUTRAL") if regime else "NEUTRAL"
    vix = regime.get("vix_level", 0) if regime else 0
    vix_pct = regime.get("vix_percentile") if regime else None

    # Optimal weights from portfolio optimizer
    opt_map = {}
    if optimizer_alloc:
        try:
            opt_map = {h.ticker: h.optimal_weight for h in optimizer_alloc.holdings}
        except Exception:
            pass

    # TNX (10Y yield) for macro context
    tnx_level = None
    try:
        tnx = yf.download("^TNX", period="5d", progress=False, auto_adjust=True)
        if tnx is not None and not tnx.empty:
            _close = tnx["Close"]
            if isinstance(_close, pd.DataFrame):
                _close = _close.iloc[:, 0]
            tnx_level = float(_close.iloc[-1])
    except Exception:
        pass

    weights_json = json.dumps(pillar_weights) if pillar_weights else None
    count = 0

    with _connect() as conn:
        for r in results:
            ticker = r["ticker"]

            # Skip duplicates (same ticker + same day)
            existing = conn.execute(
                "SELECT id FROM signal_backtest WHERE ticker=? AND source='portfolio' AND run_date LIKE ?",
                (ticker, today + "%"),
            ).fetchone()
            if existing:
                continue

            # Compute prior momentum & vol
            ret_10d, ret_30d, ret_90d, vol_20d = _compute_prior_momentum(ticker)
            factor_snapshot = dict(r)
            factor_snapshot.update({
                "return_10d_prior": ret_10d,
                "return_30d_prior": ret_30d,
                "return_90d_prior": ret_90d,
                "vol_20d": vol_20d,
                "beta_90d": r.get("beta_90d") or r.get("beta"),
                "avg_dollar_volume": r.get("avg_dollar_volume"),
                "market_cap": r.get("market_cap"),
            })
            factor_scores = compute_factor_scores_from_result(factor_snapshot)

            payload = {
                "run_date": now,
                "ticker": ticker,
                "name": r.get("name", ""),
                "source": "portfolio",
                "signal_price": r.get("current_price", 0),
                "action": r.get("action"),
                "aggregate_score": r.get("aggregate_score"),
                "technical_score": r.get("technical_score"),
                "fundamental_score": r.get("fundamental_score"),
                "sentiment_score": r.get("sentiment_score"),
                "forecast_score": r.get("forecast_score"),
                "rsi": r.get("rsi"),
                "adx": r.get("adx"),
                "bb_pct": r.get("bb_pct"),
                "pe_ratio": r.get("pe_ratio"),
                "peg_ratio": r.get("peg_ratio"),
                "revenue_growth": r.get("revenue_growth"),
                "roe": r.get("roe"),
                "short_pct": r.get("short_pct"),
                "news_score": r.get("news_score"),
                "momentum_score": r.get("momentum_score"),
                "quality_factor_score": factor_scores.get("quality_factor_score"),
                "qmj_factor_score": factor_scores.get("qmj_factor_score"),
                "value_factor_score": factor_scores.get("value_factor_score"),
                "momentum_factor_score": factor_scores.get("momentum_factor_score"),
                "volatility_factor_score": factor_scores.get("volatility_factor_score"),
                "bab_factor_score": factor_scores.get("bab_factor_score"),
                "turnover_cost_score": factor_scores.get("turnover_cost_score"),
                "pead_factor_score": factor_scores.get("pead_factor_score"),
                "sue_score": factor_scores.get("sue_score"),
                "revision_momentum_3m": factor_scores.get("revision_momentum_3m"),
                "quality_score_fundamental": r.get("quality_score_fundamental"),
                "gross_profitability": r.get("gross_profitability"),
                "fcf_to_assets": r.get("fcf_to_assets"),
                "earnings_stability": r.get("earnings_stability"),
                "eps_growth_variance_5y": r.get("eps_growth_variance_5y"),
                "factor_mom_momentum_tilt": None,
                "factor_mom_value_tilt": None,
                "factor_mom_quality_tilt": None,
                "factor_mom_momentum_ret_1m": None,
                "factor_mom_momentum_ret_3m": None,
                "factor_mom_value_ret_1m": None,
                "factor_mom_value_ret_3m": None,
                "factor_mom_quality_ret_1m": None,
                "factor_mom_quality_ret_3m": None,
                "network_momentum": None,
                "qa_sentiment_score": r.get("qa_sentiment_score"),
                "forecast_price_5d": r.get("forecast_price"),
                "forecast_price_63d": r.get("forecast_price_long"),
                "stop_loss": r.get("stop_loss"),
                "take_profit": r.get("take_profit"),
                "macd_signal": r.get("macd_signal"),
                "sma_50": r.get("sma_50"),
                "sma_200": r.get("sma_200"),
                "stoch_k": r.get("stoch_k"),
                "williams_r": r.get("williams_r"),
                "obv_trend": r.get("obv_trend"),
                "atr": r.get("atr"),
                "profit_margin": r.get("profit_margin"),
                "fcf_yield": r.get("fcf_yield"),
                "debt_equity": r.get("debt_to_equity"),
                "inst_ownership": r.get("inst_ownership"),
                "insider_net": r.get("insider_net"),
                "analyst_upside": r.get("analyst_upside"),
                "analyst_rec": r.get("analyst_rec"),
                "reddit_score": r.get("reddit_score"),
                "fmp_news_score": r.get("fmp_news_score"),
                "earnings_proximity": r.get("earnings_proximity_days"),
                "return_10d_prior": ret_10d,
                "return_30d_prior": ret_30d,
                "return_90d_prior": ret_90d,
                "vol_20d": vol_20d,
                "vix_percentile": vix_pct,
                "tnx_level": tnx_level,
                "optimal_weight": opt_map.get(ticker),
                "pillar_weights_json": weights_json,
                "entry_lens": r.get("entry_lens"),
                "entry_price": r.get("entry_price"),
                "entry_method": r.get("entry_method"),
                "fill_probability": r.get("fill_probability"),
                "planned_position_weight": r.get("position_weight"),
                "planned_risk_amount": r.get("risk_amount"),
                "position_sizing_method": r.get("sizing_method"),
                "kelly_cap_fraction": r.get("kelly_cap_fraction"),
                "r_r_ratio": r.get("r_r_ratio"),
                "sector": r.get("sector"),
                "exchange": r.get("exchange"),
                "regime": regime_label,
                "vix_level": vix,
                "portfolio_weight": pw_map.get(ticker, 0),
            }
            columns = ", ".join(payload.keys())
            placeholders = ", ".join("?" for _ in payload)
            conn.execute(
                f"INSERT INTO signal_backtest ({columns}) VALUES ({placeholders})",
                tuple(payload.values()),
            )
            count += 1

    logger.info("Recorded %d portfolio signals for backtest (with feature store).", count)
    return count


def record_stage5b_panel(candidates: list[dict]) -> int:
    """Record ALL Stage 5b candidates for ML training (not just final picks).

    This fixes survivorship bias in ML training: by recording the full panel
    of candidates that reached the fundamentals-enriched stage, the ML model
    can learn from stocks that were cut for portfolio-shape reasons (sector cap,
    correlation, FX) rather than only training on funnel survivors.

    Candidates are dict-based (Stage 5b output, before ScoredCandidate conversion).
    Source is 'discovery_panel' to distinguish from final 'discovery' picks.

    Uses batched inserts and explicit transaction control to prevent SQLite locking
    issues (especially on cloud-synced directories).
    """
    import time as _time
    _t0 = _time.time()

    init_backtest_db()
    now = datetime.now().isoformat(timespec="seconds")
    today = now[:10]
    count = 0
    _skipped_dedup = 0
    _skipped_price = 0
    _BATCH_SIZE = 50

    try:
        with _connect() as conn:
            # Pre-fetch existing tickers for today in one query (avoid N+1)
            existing_tickers = set()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM signal_backtest "
                    "WHERE source='discovery_panel' AND run_date LIKE ?",
                    (today + "%",),
                ).fetchall()
                existing_tickers = {r[0] for r in rows}
            except Exception:
                pass

            batch = []
            for c in candidates:
                ticker = c.get("symbol", "") or c.get("ticker", "")
                if not ticker:
                    continue

                # Dedup: skip if already recorded today for this source
                if ticker in existing_tickers:
                    _skipped_dedup += 1
                    continue

                # Use last price from feature store (already cached, no API call)
                signal_price = c.get("_last_price", 0)
                if not signal_price or signal_price <= 0:
                    _skipped_price += 1
                    continue

                # Extract available features from Stage 5b enrichment
                factor_snapshot = {
                    "quality_score_fundamental": c.get("_quality_score_fundamental"),
                    "gross_profitability": c.get("_gross_profitability"),
                    "fcf_to_assets": c.get("_fcf_to_assets"),
                    "earnings_stability": c.get("_earnings_stability"),
                    "eps_growth_variance_5y": c.get("_eps_growth_variance_5y"),
                    "pe_ratio": c.get("_pe_ratio"),
                    "peg_ratio": c.get("_peg_ratio"),
                    "revenue_growth": c.get("_revenue_growth"),
                    "roe": c.get("_roe"),
                    "short_pct": c.get("_short_pct"),
                    "momentum_score": c.get("_momentum_score"),
                    "return_10d_prior": c.get("_ret_10d"),
                    "return_30d_prior": c.get("_ret_30d"),
                    "return_90d_prior": c.get("_ret_90d"),
                    "vol_20d": c.get("_vol_20d"),
                    "beta_90d": c.get("_beta"),
                    "avg_dollar_volume": c.get("_avg_dollar_volume"),
                    "market_cap": c.get("_market_cap"),
                }
                try:
                    factor_scores = compute_factor_scores_from_result(factor_snapshot)
                except Exception:
                    factor_scores = {}
                sleeves = c.get("_sleeve_breakdown") or {}

                payload = {
                    "run_date": now,
                    "ticker": ticker,
                    "name": c.get("name", ""),
                    "source": "discovery_panel",
                    "signal_price": signal_price,
                    "action": "",
                    "aggregate_score": c.get("_quick_score"),
                    "final_rank": None,
                    "technical_score": None,
                    "fundamental_score": c.get("_fundamental_bonus"),
                    "sentiment_score": None,
                    "forecast_score": None,
                    "momentum_score": c.get("_momentum_score"),
                    "quality_factor_score": factor_scores.get("quality_factor_score"),
                    "qmj_factor_score": factor_scores.get("qmj_factor_score"),
                    "value_factor_score": factor_scores.get("value_factor_score"),
                    "momentum_factor_score": factor_scores.get("momentum_factor_score"),
                    "volatility_factor_score": factor_scores.get("volatility_factor_score"),
                    "bab_factor_score": factor_scores.get("bab_factor_score"),
                    "turnover_cost_score": factor_scores.get("turnover_cost_score"),
                    "pead_factor_score": factor_scores.get("pead_factor_score"),
                    "sue_score": factor_scores.get("sue_score"),
                    "revision_momentum_3m": factor_scores.get("revision_momentum_3m"),
                    "sleeve_momentum": sleeves.get("momentum"),
                    "sleeve_quality": sleeves.get("quality"),
                    "sleeve_value": sleeves.get("value"),
                    "sleeve_low_risk": sleeves.get("low_risk"),
                    "sleeve_pead": sleeves.get("pead"),
                    "sleeve_ready": sleeves.get("ready"),
                    "quality_score_fundamental": c.get("_quality_score_fundamental"),
                    "gross_profitability": c.get("_gross_profitability"),
                    "fcf_to_assets": c.get("_fcf_to_assets"),
                    "earnings_stability": c.get("_earnings_stability"),
                    "eps_growth_variance_5y": c.get("_eps_growth_variance_5y"),
                    "pe_ratio": c.get("_pe_ratio"),
                    "peg_ratio": c.get("_peg_ratio"),
                    "revenue_growth": c.get("_revenue_growth"),
                    "roe": c.get("_roe"),
                    "short_pct": c.get("_short_pct"),
                    "return_10d_prior": c.get("_ret_10d"),
                    "return_30d_prior": c.get("_ret_30d"),
                    "return_90d_prior": c.get("_ret_90d"),
                    "vol_20d": c.get("_vol_20d"),
                    "sector": c.get("sector", ""),
                    "exchange": c.get("exchange", ""),
                    "entry_lens": c.get("_entry_lens", ""),
                }
                batch.append(tuple(payload.values()))

                # Flush batch every N rows to avoid holding locks too long
                if len(batch) >= _BATCH_SIZE:
                    columns = ", ".join(payload.keys())
                    placeholders = ", ".join("?" for _ in payload)
                    conn.executemany(
                        f"INSERT INTO signal_backtest ({columns}) VALUES ({placeholders})",
                        batch,
                    )
                    count += len(batch)
                    batch = []

            # Flush remaining
            if batch:
                columns = ", ".join(payload.keys())
                placeholders = ", ".join("?" for _ in payload)
                conn.executemany(
                    f"INSERT INTO signal_backtest ({columns}) VALUES ({placeholders})",
                    batch,
                )
                count += len(batch)

        _elapsed = _time.time() - _t0
        logger.info("Recorded %d Stage 5b panel candidates in %.1fs "
                     "(skipped: %d dedup, %d no-price).",
                     count, _elapsed, _skipped_dedup, _skipped_price)
    except Exception as e:
        _elapsed = _time.time() - _t0
        logger.warning("Stage 5b panel recording failed after %.1fs (%d inserted so far): %s",
                        _elapsed, count, e)
    return count


def record_discovery_picks(candidates: list) -> int:
    """Record discovery candidates for backtesting."""
    init_backtest_db()
    now = datetime.now().isoformat(timespec="seconds")
    today = now[:10]
    count = 0

    with _connect() as conn:
        for c in candidates:
            if hasattr(c, "ticker"):
                ticker, name = c.ticker, c.name
                agg = c.aggregate_score
                tech, fund, sent, fcast = c.technical_score, c.fundamental_score, c.sentiment_score, c.forecast_score
                mom, rank = c.momentum_score, c.final_rank
                action, sector, exchange = c.action, c.sector, c.exchange
                entry_lens = getattr(c, "entry_lens", "momentum")
                entry_price = getattr(c, "entry_price", None)
                entry_method = getattr(c, "entry_method", "")
                fill_probability = getattr(c, "fill_probability", None)
                planned_position_weight = getattr(c, "position_weight", None)
                planned_risk_amount = getattr(c, "risk_amount", None)
                position_sizing_method = getattr(c, "sizing_method", "")
                kelly_cap_fraction = getattr(c, "kelly_cap_fraction", None)
                rr_ratio = getattr(c, "r_r_ratio", None)
                stop_loss = getattr(c, "stop_loss", None)
                take_profit = getattr(c, "take_profit", None)
                quality_factor_score = getattr(c, "quality_score_fundamental", None)
                gross_profitability = getattr(c, "gross_profitability", None)
                fcf_to_assets = getattr(c, "fcf_to_assets", None)
                earnings_stability = getattr(c, "earnings_stability", None)
                eps_growth_variance_5y = getattr(c, "eps_growth_variance_5y", None)
                qmj_factor_score = getattr(c, "qmj_factor_score", None)
                pead_factor_score = getattr(c, "pead_factor_score", None)
                sue_score = getattr(c, "sue_score", None)
                revision_momentum_3m = getattr(c, "revision_momentum_3m", None)
                bab_factor_score = getattr(c, "bab_factor_score", None)
                turnover_cost_score = getattr(c, "turnover_cost_score", None)
                factor_tilt = getattr(c, "factor_momentum_tilt", {}) or {}
                network_momentum = getattr(c, "network_momentum", None)
                qa_sentiment_score = getattr(c, "qa_sentiment_score", None)
                ret_10d = getattr(c, "return_10d", None)
                ret_30d = getattr(c, "return_30d", None)
                ret_90d = getattr(c, "return_90d", None)
                vol_20d = getattr(c, "vol_20d", None)
                pe_ratio = getattr(c, "pe_ratio", None)
                peg_ratio = getattr(c, "peg_ratio", None)
                revenue_growth = getattr(c, "revenue_growth", None)
                roe = getattr(c, "roe", None)
                short_pct = getattr(c, "short_pct", None)
                institutional_prior_score = getattr(c, "institutional_prior_score", None)
                institutional_prior_percentile = getattr(c, "institutional_prior_percentile", None)
                institutional_prior_confidence = getattr(c, "institutional_prior_confidence", None)
                institutional_prior_components = getattr(c, "institutional_prior_components", None)
                ready_contract_status = getattr(c, "ready_contract_status", None)
                ready_contract_reasons = getattr(c, "ready_contract_reasons", None)
                strong_buy_eligible = getattr(c, "strong_buy_eligible", None)
                meta_prob = getattr(c, "meta_success_prob", getattr(c, "meta_prob", None))
                regime_value = getattr(c, "regime", None)
                sleeve_momentum = getattr(c, "sleeve_momentum", None)
                sleeve_quality = getattr(c, "sleeve_quality", None)
                sleeve_value = getattr(c, "sleeve_value", None)
                sleeve_low_risk = getattr(c, "sleeve_low_risk", None)
                sleeve_pead = getattr(c, "sleeve_pead", None)
                sleeve_ready = getattr(c, "sleeve_ready", None)
            else:
                ticker = c.get("ticker", "")
                name = c.get("name", "")
                agg = c.get("aggregate_score", 0)
                tech = c.get("technical_score", 0)
                fund = c.get("fundamental_score", 0)
                sent = c.get("sentiment_score", 0)
                fcast = c.get("forecast_score", 0)
                mom = c.get("momentum_score", 0)
                rank = c.get("final_rank", 0)
                action = c.get("action", "")
                sector = c.get("sector", "")
                exchange = c.get("exchange", "")
                entry_lens = c.get("entry_lens", "momentum")
                entry_price = c.get("entry_price")
                entry_method = c.get("entry_method", "")
                fill_probability = c.get("fill_probability")
                planned_position_weight = c.get("position_weight")
                planned_risk_amount = c.get("risk_amount")
                position_sizing_method = c.get("sizing_method", "")
                kelly_cap_fraction = c.get("kelly_cap_fraction")
                rr_ratio = c.get("r_r_ratio")
                stop_loss = c.get("stop_loss")
                take_profit = c.get("take_profit")
                quality_factor_score = c.get("quality_score_fundamental")
                gross_profitability = c.get("gross_profitability")
                fcf_to_assets = c.get("fcf_to_assets")
                earnings_stability = c.get("earnings_stability")
                eps_growth_variance_5y = c.get("eps_growth_variance_5y")
                qmj_factor_score = c.get("qmj_factor_score")
                pead_factor_score = c.get("pead_factor_score")
                sue_score = c.get("sue_score")
                revision_momentum_3m = c.get("revision_momentum_3m")
                bab_factor_score = c.get("bab_factor_score")
                turnover_cost_score = c.get("turnover_cost_score")
                factor_tilt = c.get("factor_momentum_tilt", {}) or {}
                network_momentum = c.get("network_momentum")
                qa_sentiment_score = c.get("qa_sentiment_score")
                ret_10d = c.get("return_10d")
                ret_30d = c.get("return_30d")
                ret_90d = c.get("return_90d")
                vol_20d = c.get("vol_20d")
                pe_ratio = c.get("pe_ratio")
                peg_ratio = c.get("peg_ratio")
                revenue_growth = c.get("revenue_growth")
                roe = c.get("roe")
                short_pct = c.get("short_pct")
                institutional_prior_score = c.get("institutional_prior_score")
                institutional_prior_percentile = c.get("institutional_prior_percentile")
                institutional_prior_confidence = c.get("institutional_prior_confidence")
                institutional_prior_components = c.get("institutional_prior_components")
                ready_contract_status = c.get("ready_contract_status")
                ready_contract_reasons = c.get("ready_contract_reasons")
                strong_buy_eligible = c.get("strong_buy_eligible")
                meta_prob = c.get("meta_success_prob", c.get("meta_prob"))
                regime_value = c.get("regime")
                sleeve_breakdown = c.get("_sleeve_breakdown") or {}
                sleeve_momentum = c.get("sleeve_momentum", sleeve_breakdown.get("momentum"))
                sleeve_quality = c.get("sleeve_quality", sleeve_breakdown.get("quality"))
                sleeve_value = c.get("sleeve_value", sleeve_breakdown.get("value"))
                sleeve_low_risk = c.get("sleeve_low_risk", sleeve_breakdown.get("low_risk"))
                sleeve_pead = c.get("sleeve_pead", sleeve_breakdown.get("pead"))
                sleeve_ready = c.get("sleeve_ready", sleeve_breakdown.get("ready", sleeve_breakdown.get("ready_to_buy")))

            # Enterprise factors — computed by engine.fundamental.analyse() and
            # carried through on the candidate dict.  Dict access only (dataclass
            # path above is the legacy portfolio flow and doesn't carry these).
            if isinstance(c, dict):
                f_score_val = c.get("f_score")
                f_score_coverage = c.get("f_score_coverage")
                ev_ebit_val = c.get("ev_ebit")
                ev_ebit_score_val = c.get("ev_ebit_score")
                gpa_val = c.get("gpa")
                gpa_score_val = c.get("gpa_score")
                sma_200_val = c.get("sma_200")
                stretch_val = c.get("price_vs_sma200_stretch")
                gate_v2_status = c.get("gate_v2_status")
                gate_v2_reasons = c.get("gate_v2_reasons")
                trap_safeguard_triggered = c.get("trap_safeguard_triggered")
                trap_safeguard_reason = c.get("trap_safeguard_reason")
            else:
                f_score_val = getattr(c, "f_score", None)
                f_score_coverage = getattr(c, "f_score_coverage", None)
                ev_ebit_val = getattr(c, "ev_ebit", None)
                ev_ebit_score_val = getattr(c, "ev_ebit_score", None)
                gpa_val = getattr(c, "gpa", None)
                gpa_score_val = getattr(c, "gpa_score", None)
                sma_200_val = getattr(c, "sma_200", None)
                stretch_val = getattr(c, "price_vs_sma200_stretch", None)
                gate_v2_status = getattr(c, "gate_v2_status", None)
                gate_v2_reasons = getattr(c, "gate_v2_reasons", None)
                trap_safeguard_triggered = getattr(c, "trap_safeguard_triggered", None)
                trap_safeguard_reason = getattr(c, "trap_safeguard_reason", None)

            if isinstance(gate_v2_reasons, (list, tuple)):
                gate_v2_reasons_payload = json.dumps(list(gate_v2_reasons))
            elif gate_v2_reasons is None:
                gate_v2_reasons_payload = None
            else:
                gate_v2_reasons_payload = str(gate_v2_reasons)
            if isinstance(institutional_prior_components, dict):
                institutional_prior_components_payload = json.dumps(institutional_prior_components)
            elif institutional_prior_components is None:
                institutional_prior_components_payload = None
            else:
                institutional_prior_components_payload = str(institutional_prior_components)
            if isinstance(ready_contract_reasons, (list, tuple)):
                ready_contract_reasons_payload = json.dumps(list(ready_contract_reasons))
            elif ready_contract_reasons is None:
                ready_contract_reasons_payload = None
            else:
                ready_contract_reasons_payload = str(ready_contract_reasons)
            try:
                meta_prob_value = float(meta_prob) if meta_prob is not None else None
            except (TypeError, ValueError):
                meta_prob_value = None

            # Signal price: prefer the price the scoring engine saw at decision time.
            # Falling back to a fresh download only if the candidate lacks one, which
            # preserves return-IC integrity (the engine's signal-time anchor is what
            # later evaluations must compare against).
            import math
            signal_price = 0
            cand_price = c.get("current_price") if isinstance(c, dict) else None
            if cand_price is not None:
                try:
                    px = float(cand_price)
                    if math.isfinite(px) and px > 0:
                        signal_price = px
                except (TypeError, ValueError):
                    pass
            if signal_price <= 0:
                try:
                    data = yf.download(ticker, period="5d", progress=False, auto_adjust=True, timeout=30)
                    if data is not None and not data.empty:
                        raw = float(data["Close"].iloc[-1].item() if hasattr(data["Close"].iloc[-1], "item") else data["Close"].iloc[-1])
                        signal_price = raw if not math.isnan(raw) else 0
                except Exception:
                    signal_price = 0

            if signal_price <= 0:
                continue

            existing = conn.execute(
                "SELECT id FROM signal_backtest WHERE ticker=? AND source='discovery' AND run_date LIKE ?",
                (ticker, today + "%"),
            ).fetchone()
            if existing:
                continue

            factor_snapshot = {
                "quality_score_fundamental": quality_factor_score,
                "gross_profitability": gross_profitability,
                "fcf_to_assets": fcf_to_assets,
                "earnings_stability": earnings_stability,
                "eps_growth_variance_5y": eps_growth_variance_5y,
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "revenue_growth": revenue_growth,
                "roe": roe,
                "short_pct": short_pct,
                "momentum_score": mom,
                "return_10d_prior": ret_10d,
                "return_30d_prior": ret_30d,
                "return_90d_prior": ret_90d,
                "vol_20d": vol_20d,
                "beta_90d": getattr(c, "beta_90d", None) if hasattr(c, "ticker") else c.get("beta_90d", c.get("_beta")),
                "avg_dollar_volume": getattr(c, "avg_dollar_volume", None) if hasattr(c, "ticker") else c.get("avg_dollar_volume", c.get("_avg_dollar_volume")),
                "market_cap": getattr(c, "market_cap", None) if hasattr(c, "ticker") else c.get("market_cap", c.get("_market_cap")),
                "qmj_factor_score": qmj_factor_score,
                "pead_factor_score": pead_factor_score,
                "sue_score": sue_score,
                "revision_momentum_3m": revision_momentum_3m,
                "bab_factor_score": bab_factor_score,
                "turnover_cost_score": turnover_cost_score,
            }
            factor_scores = compute_factor_scores_from_result(factor_snapshot)

            payload = {
                "run_date": now,
                "ticker": ticker,
                "name": name,
                "source": "discovery",
                "regime": regime_value,
                "signal_price": signal_price,
                "action": action,
                "aggregate_score": agg,
                "final_rank": rank,
                "technical_score": tech,
                "fundamental_score": fund,
                "sentiment_score": sent,
                "forecast_score": fcast,
                "momentum_score": mom,
                "quality_factor_score": factor_scores.get("quality_factor_score"),
                "qmj_factor_score": qmj_factor_score if qmj_factor_score is not None else factor_scores.get("qmj_factor_score"),
                "value_factor_score": factor_scores.get("value_factor_score"),
                "momentum_factor_score": factor_scores.get("momentum_factor_score"),
                "volatility_factor_score": factor_scores.get("volatility_factor_score"),
                "bab_factor_score": bab_factor_score if bab_factor_score is not None else factor_scores.get("bab_factor_score"),
                "turnover_cost_score": turnover_cost_score if turnover_cost_score is not None else factor_scores.get("turnover_cost_score"),
                "quality_score_fundamental": quality_factor_score,
                "gross_profitability": gross_profitability,
                "fcf_to_assets": fcf_to_assets,
                "earnings_stability": earnings_stability,
                "eps_growth_variance_5y": eps_growth_variance_5y,
                "factor_mom_momentum_tilt": factor_tilt.get("factor_mom_momentum_tilt"),
                "factor_mom_value_tilt": factor_tilt.get("factor_mom_value_tilt"),
                "factor_mom_quality_tilt": factor_tilt.get("factor_mom_quality_tilt"),
                "factor_mom_momentum_ret_1m": factor_tilt.get("factor_mom_momentum_ret_1m"),
                "factor_mom_momentum_ret_3m": factor_tilt.get("factor_mom_momentum_ret_3m"),
                "factor_mom_value_ret_1m": factor_tilt.get("factor_mom_value_ret_1m"),
                "factor_mom_value_ret_3m": factor_tilt.get("factor_mom_value_ret_3m"),
                "factor_mom_quality_ret_1m": factor_tilt.get("factor_mom_quality_ret_1m"),
                "factor_mom_quality_ret_3m": factor_tilt.get("factor_mom_quality_ret_3m"),
                "network_momentum": network_momentum,
                "qa_sentiment_score": qa_sentiment_score,
                "institutional_prior_score": institutional_prior_score,
                "institutional_prior_percentile": institutional_prior_percentile,
                "institutional_prior_confidence": institutional_prior_confidence,
                "institutional_prior_components": institutional_prior_components_payload,
                "ready_contract_status": ready_contract_status,
                "ready_contract_reasons": ready_contract_reasons_payload,
                "strong_buy_eligible": None if strong_buy_eligible is None else (1 if strong_buy_eligible else 0),
                "meta_prob": meta_prob_value,
                "meta_success": None if meta_prob_value is None else (1 if meta_prob_value >= getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60) else 0),
                "sleeve_momentum": sleeve_momentum,
                "sleeve_quality": sleeve_quality,
                "sleeve_value": sleeve_value,
                "sleeve_low_risk": sleeve_low_risk,
                "sleeve_pead": sleeve_pead,
                "sleeve_ready": sleeve_ready,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "pead_factor_score": pead_factor_score if pead_factor_score is not None else factor_scores.get("pead_factor_score"),
                "sue_score": sue_score if sue_score is not None else factor_scores.get("sue_score"),
                "revision_momentum_3m": revision_momentum_3m if revision_momentum_3m is not None else factor_scores.get("revision_momentum_3m"),
                "sector": sector,
                "exchange": exchange,
                "entry_lens": entry_lens,
                "entry_price": entry_price,
                "entry_method": entry_method,
                "fill_probability": fill_probability,
                "planned_position_weight": planned_position_weight,
                "planned_risk_amount": planned_risk_amount,
                "position_sizing_method": position_sizing_method,
                "kelly_cap_fraction": kelly_cap_fraction,
                "r_r_ratio": rr_ratio,
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "revenue_growth": revenue_growth,
                "roe": roe,
                "short_pct": short_pct,
                "return_10d_prior": ret_10d,
                "return_30d_prior": ret_30d,
                "return_90d_prior": ret_90d,
                "vol_20d": vol_20d,
                # Enterprise factor snapshot
                "f_score": f_score_val,
                "f_score_coverage": f_score_coverage,
                "ev_ebit": ev_ebit_val,
                "ev_ebit_score": ev_ebit_score_val,
                "gpa": gpa_val,
                "gpa_score": gpa_score_val,
                "sma_200": sma_200_val,
                # Stretch = price / sma_200 - 1; None when either input is missing/invalid
                "price_vs_sma200_stretch": (
                    stretch_val
                    if stretch_val is not None
                    else (signal_price / sma_200_val - 1.0)
                    if (sma_200_val is not None and isinstance(sma_200_val, (int, float))
                        and sma_200_val > 0 and signal_price > 0)
                    else None
                ),
                "gate_v2_status": gate_v2_status,
                "gate_v2_reasons": gate_v2_reasons_payload,
                "trap_safeguard_triggered": 1 if trap_safeguard_triggered else 0,
                "trap_safeguard_reason": trap_safeguard_reason,
                # Action-gate telemetry (Tier 1-4 hard gates) — used by the
                # threshold learner for counterfactual outcome attribution.
                "altman_z": getattr(c, "altman_z", None) if hasattr(c, "altman_z") else c.get("altman_z"),
                "altman_zone": (
                    getattr(c, "altman_zone", None) if hasattr(c, "altman_zone")
                    else c.get("altman_zone")
                ),
                "beneish_m": getattr(c, "beneish_m", None) if hasattr(c, "beneish_m") else c.get("beneish_m"),
                "action_gate_ceiling": (
                    getattr(c, "action_gate_ceiling", None) if hasattr(c, "action_gate_ceiling")
                    else c.get("action_gate_ceiling")
                ),
                "action_gate_flags_json": _json_or_none(
                    getattr(c, "action_gate_flags", None) if hasattr(c, "action_gate_flags")
                    else c.get("action_gate_flags")
                ),
                "threshold_profile": (
                    getattr(c, "threshold_profile", None) if hasattr(c, "threshold_profile")
                    else c.get("threshold_profile")
                ),
            }
            columns = ", ".join(payload.keys())
            placeholders = ", ".join("?" for _ in payload)
            conn.execute(
                f"INSERT INTO signal_backtest ({columns}) VALUES ({placeholders})",
                tuple(payload.values()),
            )
            count += 1

    logger.info("Recorded %d discovery picks for backtest.", count)
    return count


# ---------------------------------------------------------------------------
# Evaluate matured signals — multi-horizon
# ---------------------------------------------------------------------------

_HORIZON_SPECS = [
    (5, "price_5d", "return_5d", "evaluated_5d"),
    (10, "price_10d", "return_10d", "evaluated_10d"),
    (30, "price_30d", "return_30d", "evaluated_30d"),
    (60, "price_60d", "return_60d", "evaluated_60d"),
    (90, "price_90d", "return_90d", "evaluated_90d"),
]


def _selected_horizon_specs(horizons: Iterable[int] | None = None) -> list[tuple[int, str, str, str]]:
    if horizons is None:
        return list(_HORIZON_SPECS)
    wanted = {int(h) for h in horizons}
    return [spec for spec in _HORIZON_SPECS if spec[0] in wanted]


def count_mature_signal_horizon_pairs(horizons: Iterable[int] | None = None) -> dict[int, int]:
    """Count matured, unevaluated signal-horizon pairs without network calls."""
    init_backtest_db()
    counts: dict[int, int] = {}
    with _connect() as conn:
        for horizon, _col_price, _col_return, col_flag in _selected_horizon_specs(horizons):
            cutoff = (datetime.now() - timedelta(days=horizon)).isoformat()
            counts[horizon] = int(conn.execute(
                f"SELECT COUNT(*) FROM signal_backtest WHERE {col_flag} = 0 AND run_date < ?",
                (cutoff,),
            ).fetchone()[0])
    return counts


def _fetch_price_at_offset(
    ticker: str,
    run_date: str,
    offset_days: int,
    *,
    use_open: bool = False,
) -> float | None:
    """Fetch the first available trading-bar price strictly at or after
    run_date + offset_days. Returns Open when ``use_open`` is True (entry
    anchor convention: fills happen on the next-bar open, not same-bar close).

    Prior behaviour looked backwards 3 days into the offset window, which
    allowed pre-signal bars to be used as the "entry" price — a subtle
    look-ahead for offset=0 and a stale-bar bias for later horizons.
    """
    try:
        base = datetime.strptime(run_date[:10], "%Y-%m-%d")
        start = (base + timedelta(days=max(0, offset_days))).strftime("%Y-%m-%d")
        end = (base + timedelta(days=offset_days + 10)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            col = "Open" if use_open else "Close"
            if col not in data.columns:
                col = "Close"
            series = data[col]
            if hasattr(series, "iloc"):
                val = float(series.iloc[0])
                if val == val and val > 0:  # NaN-safe
                    return val
    except Exception:
        pass
    return None


def _coerce_price_frame_index(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    idx = pd.to_datetime(frame.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            idx = idx.tz_convert(None)
        except (TypeError, AttributeError):
            pass
    frame.index = idx
    frame = frame[~frame.index.isna()]
    return frame.sort_index()


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, *, single_ticker: bool = False) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(str(v) for v in raw.columns.get_level_values(0))
            level1 = set(str(v) for v in raw.columns.get_level_values(1))
            if ticker in level0:
                frame = raw[ticker]
            elif ticker in level1:
                frame = raw.xs(ticker, axis=1, level=1)
            else:
                return None
        elif single_ticker:
            frame = raw
        else:
            return None
        frame = frame.dropna(how="all")
        return _coerce_price_frame_index(frame)
    except Exception:
        return None


def _download_price_frames(
    tickers: Sequence[str],
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Download OHLC frames in one yfinance request, with per-ticker fallback."""
    clean = sorted({str(t).upper() for t in tickers if t})
    if not clean:
        return {}
    try:
        from utils.price_store import download_price_history
        cached = download_price_history(clean, start=start, end=end, batch_size=getattr(config, "PRICE_CACHE_BATCH_SIZE", 75))
        frames = {ticker: frame for ticker, frame in cached.items() if frame is not None and not frame.empty}
        if len(frames) == len(clean):
            return frames
    except Exception as exc:
        logger.debug("Price cache path unavailable for batched evaluator: %s", exc)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    frames: dict[str, pd.DataFrame] = frames if "frames" in locals() else {}
    try:
        raw = yf.download(
            clean if len(clean) > 1 else clean[0],
            start=start_s,
            end=end_s,
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="ticker",
        )
        for ticker in clean:
            frame = _extract_ticker_frame(raw, ticker, single_ticker=(len(clean) == 1))
            if frame is not None and not frame.empty:
                frames[ticker] = frame
    except Exception as exc:
        logger.debug("Batched yfinance download failed for %d tickers: %s", len(clean), exc)

    missing = [ticker for ticker in clean if ticker not in frames]
    for ticker in missing:
        try:
            raw = yf.download(
                ticker,
                start=start_s,
                end=end_s,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            frame = _extract_ticker_frame(raw, ticker, single_ticker=True)
            if frame is not None and not frame.empty:
                frames[ticker] = frame
        except Exception:
            continue
    return frames


def _first_bar_value(
    frame: pd.DataFrame | None,
    run_date: str,
    offset_days: int,
    *,
    column: str = "Close",
    search_days: int = 10,
) -> float | None:
    if frame is None or frame.empty:
        return None
    try:
        base = datetime.strptime(run_date[:10], "%Y-%m-%d")
        start = pd.Timestamp(base + timedelta(days=max(0, offset_days)))
        end = start + pd.Timedelta(days=search_days)
        if column not in frame.columns:
            column = "Close"
        if column not in frame.columns:
            return None
        window = frame[(frame.index >= start) & (frame.index < end)]
        if window.empty:
            return None
        series = window[column]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        clean = series.dropna()
        if clean.empty:
            return None
        val = float(clean.iloc[0])
        return val if val == val and val > 0 else None
    except Exception:
        return None


def _check_stop_target_hits_from_frame(
    frame: pd.DataFrame | None,
    run_date: str,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[bool, int | None, bool, int | None]:
    if frame is None or frame.empty:
        return False, None, False, None
    try:
        base = datetime.strptime(run_date[:10], "%Y-%m-%d")
        start = pd.Timestamp(base + timedelta(days=1))
        end = pd.Timestamp(base + timedelta(days=95))
        window = frame[(frame.index >= start) & (frame.index < end)]
        if window.empty or "Low" not in window.columns or "High" not in window.columns:
            return False, None, False, None

        low_series = window["Low"]
        high_series = window["High"]
        if isinstance(low_series, pd.DataFrame):
            low_series = low_series.iloc[:, 0]
        if isinstance(high_series, pd.DataFrame):
            high_series = high_series.iloc[:, 0]

        stop_hit, stop_day = False, None
        target_hit, target_day = False, None
        for day_idx, (low, high) in enumerate(zip(low_series, high_series), start=1):
            try:
                low_f = float(low)
                high_f = float(high)
            except (TypeError, ValueError):
                continue
            if stop_loss and not stop_hit and low_f <= stop_loss:
                stop_hit = True
                stop_day = day_idx
            if take_profit and not target_hit and high_f >= take_profit:
                target_hit = True
                target_day = day_idx
            if stop_hit and target_hit:
                break
        return stop_hit, stop_day, target_hit, target_day
    except Exception:
        return False, None, False, None


def _check_stop_target_hits(
    ticker: str, run_date: str, stop_loss: float | None, take_profit: float | None,
) -> tuple[bool, int | None, bool, int | None]:
    """Check if stop-loss or take-profit was hit within 90 days of run_date.

    Returns (stop_hit, stop_day, target_hit, target_day).
    """
    try:
        base = datetime.strptime(run_date[:10], "%Y-%m-%d")
        start = (base + timedelta(days=1)).strftime("%Y-%m-%d")
        end = (base + timedelta(days=95)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return False, None, False, None

        stop_hit, stop_day = False, None
        target_hit, target_day = False, None

        for day_idx in range(len(data)):
            low = float(data["Low"].iloc[day_idx])
            high = float(data["High"].iloc[day_idx])

            if stop_loss and not stop_hit and low <= stop_loss:
                stop_hit = True
                stop_day = day_idx + 1

            if take_profit and not target_hit and high >= take_profit:
                target_hit = True
                target_day = day_idx + 1

            if stop_hit and target_hit:
                break

        return stop_hit, stop_day, target_hit, target_day
    except Exception:
        return False, None, False, None


def evaluate_matured_signals_batched(
    horizons: Iterable[int] | None = None,
    *,
    batch_size: int | None = None,
    dry_run: bool = False,
) -> int:
    """Evaluate matured signals using batched yfinance price windows.

    This is the production-safe warm-start path: it only labels signals that
    already exist in ``signal_backtest`` and therefore uses the signal-time
    features the app actually recorded. No historical fundamentals are
    reconstructed here, so there is no look-ahead channel.
    """
    init_backtest_db()
    specs = _selected_horizon_specs(horizons)
    if not specs:
        return 0

    pending_by_horizon: dict[int, list] = {}
    with _connect() as conn:
        for horizon, _col_price, _col_return, col_flag in specs:
            cutoff = (datetime.now() - timedelta(days=horizon)).isoformat()
            pending_by_horizon[horizon] = conn.execute(
                f"SELECT * FROM signal_backtest WHERE {col_flag} = 0 AND run_date < ?",
                (cutoff,),
            ).fetchall()

    assignments: dict[str, list[tuple[int, str, str, str, object]]] = defaultdict(list)
    for horizon, col_price, col_return, col_flag in specs:
        for sig in pending_by_horizon.get(horizon, []):
            ticker = str(sig["ticker"] or "").upper()
            if ticker:
                assignments[ticker].append((horizon, col_price, col_return, col_flag, sig))

    pending_total = sum(len(v) for v in assignments.values())
    if dry_run:
        logger.info("Batched evaluator dry run: %d signal-horizon pairs pending.", pending_total)
        return pending_total
    if pending_total == 0:
        return 0

    batch_size = int(batch_size or getattr(config, "SIGNAL_BACKTEST_BATCH_SIZE", 75))
    batch_size = max(1, batch_size)
    tickers = sorted(assignments)

    spy_frame = None
    if 90 in {spec[0] for spec in specs} and pending_by_horizon.get(90):
        all_90 = pending_by_horizon[90]
        min_90 = min(datetime.strptime(sig["run_date"][:10], "%Y-%m-%d") for sig in all_90)
        max_90 = max(datetime.strptime(sig["run_date"][:10], "%Y-%m-%d") for sig in all_90)
        spy_frame = _download_price_frames(["SPY"], min_90, max_90 + timedelta(days=105)).get("SPY")

    evaluated_total = 0
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        batch_assignments = [item for ticker in batch for item in assignments[ticker]]
        min_base = min(datetime.strptime(item[4]["run_date"][:10], "%Y-%m-%d") for item in batch_assignments)
        max_needed = max(
            datetime.strptime(item[4]["run_date"][:10], "%Y-%m-%d")
            + timedelta(days=(105 if item[0] == 90 else item[0] + 10))
            for item in batch_assignments
        )
        frames = _download_price_frames(batch, min_base, max_needed + timedelta(days=1))

        with _connect() as conn:
            for ticker in batch:
                frame = frames.get(ticker)
                if frame is None or frame.empty:
                    continue
                for horizon, col_price, col_return, col_flag, sig in assignments[ticker]:
                    price = _first_bar_value(frame, sig["run_date"], horizon, column="Close")
                    if price is None:
                        continue
                    signal_price = sig["signal_price"]
                    if not signal_price:
                        continue
                    ret = (price - signal_price) / signal_price * 100
                    updates = {col_price: price, col_return: round(ret, 2), col_flag: 1}

                    if horizon == 5 and sig["forecast_price_5d"]:
                        err_5d = abs(price - sig["forecast_price_5d"]) / signal_price * 100
                        updates["actual_price_5d"] = price
                        updates["forecast_error_5d"] = round(err_5d, 2)

                    if horizon == int(getattr(config, "TRIPLE_BARRIER_HORIZON_DAYS", 30)):
                        try:
                            from engine.labeling import triple_barrier_from_frame
                            tb = triple_barrier_from_frame(
                                frame,
                                sig["run_date"],
                                float(signal_price),
                                atr=sig["atr"],
                                stop_loss=sig["stop_loss"],
                                take_profit=sig["take_profit"],
                                horizon_days=horizon,
                            )
                            if tb is not None:
                                updates["tb_label"] = tb.label
                                updates["tb_return"] = round(tb.return_pct, 4)
                                updates["tb_days"] = tb.days
                                updates["tb_hit"] = tb.hit
                                updates["tb_horizon"] = horizon
                                updates["tb_updated_at"] = datetime.now().isoformat()
                        except Exception as exc:
                            logger.debug("Triple-barrier label failed for %s id=%s: %s", ticker, sig["id"], exc)

                    if horizon == 90:
                        spy_price_start = _first_bar_value(spy_frame, sig["run_date"], 1, column="Open")
                        spy_price_end = _first_bar_value(spy_frame, sig["run_date"], 90, column="Close")
                        spy_ret = (
                            (spy_price_end - spy_price_start) / spy_price_start * 100
                            if spy_price_start and spy_price_end else 0
                        )
                        updates["spy_return_90d"] = round(spy_ret, 2)
                        updates["beat_market"] = 1 if ret > spy_ret else 0

                        action = sig["action"]
                        if action in ("STRONG BUY", "BUY"):
                            updates["action_correct"] = 1 if ret > 0 else 0
                        elif action in ("SELL", "STRONG SELL"):
                            updates["action_correct"] = 1 if ret < 0 else 0
                        elif action == "KEEP":
                            updates["action_correct"] = 1 if abs(ret) < 15 else 0

                        if sig["forecast_price_63d"]:
                            actual_63d = _first_bar_value(frame, sig["run_date"], 63, column="Close")
                            if actual_63d:
                                err_63d = abs(actual_63d - sig["forecast_price_63d"]) / signal_price * 100
                                updates["actual_price_63d"] = actual_63d
                                updates["forecast_error_63d"] = round(err_63d, 2)

                        if sig["stop_loss"] or sig["take_profit"]:
                            s_hit, s_day, t_hit, t_day = _check_stop_target_hits_from_frame(
                                frame, sig["run_date"], sig["stop_loss"], sig["take_profit"],
                            )
                            updates["stop_hit"] = 1 if s_hit else 0
                            updates["stop_hit_day"] = s_day
                            updates["target_hit"] = 1 if t_hit else 0
                            updates["target_hit_day"] = t_day

                    set_clause = ", ".join(f"{k}=?" for k in updates)
                    values = list(updates.values()) + [sig["id"]]
                    conn.execute(f"UPDATE signal_backtest SET {set_clause} WHERE id=?", values)
                    evaluated_total += 1

        logger.info(
            "Batched evaluator: processed tickers %d-%d/%d; cumulative evaluated=%d",
            batch_start + 1,
            min(batch_start + batch_size, len(tickers)),
            len(tickers),
            evaluated_total,
        )

    if evaluated_total > 0:
        _recompute_all_stats()

    logger.info("Batched evaluator labelled %d signal-horizon pairs.", evaluated_total)
    return evaluated_total


def evaluate_matured_signals(horizon_days: int = 90) -> int:
    """Evaluate signals that have matured at the given horizon.

    Runs multi-horizon evaluation: 5d, 10d, 30d, 60d, and 90d checks.
    """
    if getattr(config, "SIGNAL_BACKTEST_BATCHED_EVALUATOR_ENABLED", True):
        horizons = getattr(config, "SIGNAL_BACKTEST_BATCH_HORIZONS", [5, 10, 30, 60, 90])
        return evaluate_matured_signals_batched(horizons=horizons)

    init_backtest_db()
    evaluated_total = 0

    for horizon, col_price, col_return, col_flag in [
        (5, "price_5d", "return_5d", "evaluated_5d"),
        (10, "price_10d", "return_10d", "evaluated_10d"),
        (30, "price_30d", "return_30d", "evaluated_30d"),
        (60, "price_60d", "return_60d", "evaluated_60d"),
        (90, "price_90d", "return_90d", "evaluated_90d"),
    ]:
        cutoff = (datetime.now() - timedelta(days=horizon)).isoformat()

        with _connect() as conn:
            pending = conn.execute(
                f"SELECT * FROM signal_backtest WHERE {col_flag} = 0 AND run_date < ?",
                (cutoff,),
            ).fetchall()

        if not pending:
            continue

        spy_cache = {}

        for sig in pending:
            ticker = sig["ticker"]
            run_date = sig["run_date"]

            price = _fetch_price_at_offset(ticker, run_date, horizon)
            if price is None:
                continue

            signal_price = sig["signal_price"]
            ret = (price - signal_price) / signal_price * 100

            updates = {col_price: price, col_return: round(ret, 2), col_flag: 1}

            if horizon == 5 and sig["forecast_price_5d"]:
                err_5d = abs(price - sig["forecast_price_5d"]) / signal_price * 100
                updates["actual_price_5d"] = price
                updates["forecast_error_5d"] = round(err_5d, 2)

            # On 90d evaluation, also do full analysis
            if horizon == 90:
                # SPY benchmark
                rd = run_date[:10]
                if rd not in spy_cache:
                    # T+1 open anchor for entry; T+90 close for horizon exit.
                    spy_price_start = _fetch_price_at_offset("SPY", run_date, 1, use_open=True)
                    spy_price_end = _fetch_price_at_offset("SPY", run_date, 90)
                    if spy_price_start and spy_price_end:
                        spy_cache[rd] = (spy_price_end - spy_price_start) / spy_price_start * 100
                    else:
                        spy_cache[rd] = 0
                spy_ret = spy_cache.get(rd, 0)
                updates["spy_return_90d"] = round(spy_ret, 2)
                updates["beat_market"] = 1 if ret > spy_ret else 0

                # Action correctness
                action = sig["action"]
                if action in ("STRONG BUY", "BUY"):
                    updates["action_correct"] = 1 if ret > 0 else 0
                elif action in ("SELL", "STRONG SELL"):
                    updates["action_correct"] = 1 if ret < 0 else 0
                elif action == "KEEP":
                    updates["action_correct"] = 1 if abs(ret) < 15 else 0

                # Forecast accuracy (63d; 5d is evaluated on the 5d horizon)
                if sig["forecast_price_63d"]:
                    actual_63d = _fetch_price_at_offset(ticker, run_date, 63)
                    if actual_63d:
                        err_63d = abs(actual_63d - sig["forecast_price_63d"]) / signal_price * 100
                        updates["actual_price_63d"] = actual_63d
                        updates["forecast_error_63d"] = round(err_63d, 2)

                # Stop/target hit analysis
                if sig["stop_loss"] or sig["take_profit"]:
                    s_hit, s_day, t_hit, t_day = _check_stop_target_hits(
                        ticker, run_date, sig["stop_loss"], sig["take_profit"],
                    )
                    updates["stop_hit"] = 1 if s_hit else 0
                    updates["stop_hit_day"] = s_day
                    updates["target_hit"] = 1 if t_hit else 0
                    updates["target_hit_day"] = t_day

            # Write updates
            set_clause = ", ".join(f"{k}=?" for k in updates)
            values = list(updates.values()) + [sig["id"]]
            with _connect() as conn:
                conn.execute(f"UPDATE signal_backtest SET {set_clause} WHERE id=?", values)

            evaluated_total += 1

    if evaluated_total > 0:
        _recompute_all_stats()

    logger.info("Evaluated %d signal-horizon pairs.", evaluated_total)
    return evaluated_total


# ---------------------------------------------------------------------------
# Recompute all effectiveness stats
# ---------------------------------------------------------------------------

def _recompute_all_stats():
    """Recompute pillar effectiveness, action calibration, and regime stats."""
    init_backtest_db()
    with _connect() as conn:
        all_signals = conn.execute(
            """SELECT * FROM signal_backtest
               WHERE evaluated_5d = 1 OR evaluated_10d = 1
                  OR evaluated_30d = 1 OR evaluated_60d = 1
                  OR evaluated_90d = 1"""
        ).fetchall()

    if len(all_signals) < 5:
        return

    now = datetime.now().isoformat(timespec="seconds")

    # --- Pillar effectiveness per source and horizon ---
    pillars = [
        "technical_score", "fundamental_score", "sentiment_score", "forecast_score",
        "sleeve_momentum", "sleeve_quality", "sleeve_value",
        "sleeve_low_risk", "sleeve_pead", "sleeve_ready",
    ]

    with _connect() as conn:
        conn.execute("DELETE FROM pillar_effectiveness")

        def _insert_pillar_effectiveness(signals, source: str, horizon_col: str, horizon_label: str, regime_label: str | None) -> None:
            if len(signals) < 5:
                return
            for pillar in pillars:
                scores = np.array([s[pillar] or 0 for s in signals])
                returns = np.array([s[horizon_col] or 0 for s in signals])

                valid = ~(np.isnan(scores) | np.isnan(returns))
                scores, returns = scores[valid], returns[valid]

                if len(scores) < 5 or np.std(scores) == 0:
                    continue

                median_score = np.median(scores)
                high = scores >= median_score
                low = scores < median_score

                avg_high = float(np.mean(returns[high])) if high.any() else 0
                avg_low = float(np.mean(returns[low])) if low.any() else 0
                hit_rate = float(np.mean(returns[high] > 0)) if high.any() else 0

                try:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(scores, returns)
                    ic = float(ic) if not np.isnan(ic) else 0
                except (ImportError, ValueError):
                    rank_s = np.argsort(np.argsort(scores))
                    rank_r = np.argsort(np.argsort(returns))
                    ic = float(np.corrcoef(rank_s, rank_r)[0, 1])
                    if np.isnan(ic):
                        ic = 0

                conn.execute(
                    """INSERT INTO pillar_effectiveness
                       (updated_at, source, pillar, horizon, regime, information_coefficient,
                        hit_rate, avg_return_high, avg_return_low, sample_size)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (now, source, pillar.replace("_score", ""), horizon_label, regime_label,
                     round(ic, 4), round(hit_rate, 3),
                     round(avg_high, 2), round(avg_low, 2), len(signals)),
                )

        for source in ["portfolio", "discovery", "all"]:
            if source == "all":
                signals = all_signals
            else:
                signals = [s for s in all_signals if s["source"] == source]

            if len(signals) < 5:
                continue

            for horizon_col, horizon_label, flag_col in [
                ("return_5d", "5d", "evaluated_5d"),
                ("return_10d", "10d", "evaluated_10d"),
                ("return_30d", "30d", "evaluated_30d"),
                ("return_60d", "60d", "evaluated_60d"),
                ("return_90d", "90d", "evaluated_90d"),
            ]:
                horizon_signals = [
                    s for s in signals
                    if s[flag_col] and s[horizon_col] is not None
                ]
                if len(horizon_signals) < 5:
                    continue
                _insert_pillar_effectiveness(horizon_signals, source, horizon_col, horizon_label, None)
                if getattr(config, "BAYESIAN_REGIME_CONDITIONAL", True):
                    for regime_label in sorted({str(s["regime"] or "").upper() for s in horizon_signals if s["regime"]}):
                        regime_signals = [
                            s for s in horizon_signals
                            if str(s["regime"] or "").upper() == regime_label
                        ]
                        _insert_pillar_effectiveness(horizon_signals if regime_label == "" else regime_signals, source, horizon_col, horizon_label, regime_label or None)

    # --- Action calibration ---
    # Prefer triple-barrier labels when available (PIT replay warm-start),
    # otherwise fall back to fixed 90d realised returns.  This prevents the
    # action thresholds from waiting months for live 90d labels.
    use_tb = bool(getattr(config, "ACTION_CALIBRATION_USE_TRIPLE_BARRIER", True))
    min_action_n = int(getattr(config, "ACTION_CALIBRATION_MIN_SAMPLE_SIZE", 20))
    signals_90d = [s for s in all_signals if s["evaluated_90d"] and s["return_90d"] is not None]
    calibration_signals = []
    for s in all_signals:
        if use_tb and s["tb_label"] is not None:
            calibration_signals.append(s)
        elif s["evaluated_90d"] and s["return_90d"] is not None:
            calibration_signals.append(s)

    def _calibration_return(s) -> float:
        if use_tb and s["tb_label"] is not None:
            return float(s["tb_return"] or 0)
        return float(s["return_90d"] or 0)

    def _calibration_hit(action: str, s) -> bool:
        if use_tb and s["tb_label"] is not None:
            label = int(s["tb_label"])
            if action in ("STRONG BUY", "BUY"):
                return label > 0
            if action in ("SELL", "STRONG SELL", "AVOID"):
                return label < 0
            return label == 0
        r = float(s["return_90d"] or 0)
        if action in ("STRONG BUY", "BUY"):
            return r > 0
        if action in ("SELL", "STRONG SELL", "AVOID"):
            return r < 0
        return abs(r) < 15

    with _connect() as conn:
        conn.execute("DELETE FROM action_calibration")

        for source in ["portfolio", "discovery", "all"]:
            if source == "all":
                signals = calibration_signals
            elif source == "discovery":
                signals = [
                    s for s in calibration_signals
                    if s["source"] == "discovery" or str(s["source"] or "").startswith("replay")
                ]
            else:
                signals = [s for s in calibration_signals if s["source"] == source]
            if not signals:
                continue

            actions = set(s["action"] for s in signals if s["action"])
            for action in actions:
                action_signals = [s for s in signals if s["action"] == action]
                if len(action_signals) < min_action_n:
                    continue

                returns = [_calibration_return(s) for s in action_signals]
                avg_ret = sum(returns) / len(returns)
                hits = sum(1 for s in action_signals if _calibration_hit(action, s))

                conn.execute(
                    """INSERT INTO action_calibration
                       (updated_at, action, source, avg_return_90d, hit_rate, sample_size)
                       VALUES (?,?,?,?,?,?)""",
                    (now, action, source, round(avg_ret, 2),
                     round(hits / len(returns), 3), len(action_signals)),
                )

    # --- Regime effectiveness ---
    with _connect() as conn:
        conn.execute("DELETE FROM regime_effectiveness")

        regime_signals = [
            s for s in all_signals
            if s["regime"] and (
                (s["evaluated_90d"] and s["return_90d"] is not None)
                or (use_tb and s["tb_label"] is not None)
            )
        ]

        def _regime_return(s) -> float:
            if use_tb and s["tb_label"] is not None:
                return float(s["tb_return"] or 0)
            return float(s["return_90d"] or 0)

        regimes = set(s["regime"] for s in regime_signals if s["regime"])
        for regime in regimes:
            reg_signals = [s for s in regime_signals if s["regime"] == regime]
            if len(reg_signals) < 3:
                continue

            returns = [_regime_return(s) for s in reg_signals]
            avg_ret = sum(returns) / len(returns)

            # Find best pillar for this regime
            best_pillar, best_ic = None, -1
            for pillar in pillars:
                scores = np.array([s[pillar] or 0 for s in reg_signals])
                rets = np.array([_regime_return(s) for s in reg_signals])
                if len(scores) >= 5 and np.std(scores) > 0:
                    try:
                        from scipy.stats import spearmanr
                        ic, _ = spearmanr(scores, rets)
                        if not np.isnan(ic) and ic > best_ic:
                            best_ic = ic
                            best_pillar = pillar.replace("_score", "")
                    except Exception:
                        pass

            conn.execute(
                """INSERT INTO regime_effectiveness
                   (updated_at, regime, avg_return_90d, best_pillar, sample_size)
                   VALUES (?,?,?,?,?)""",
                (now, regime, round(avg_ret, 2), best_pillar, len(reg_signals)),
            )

    logger.info("Recomputed all backtest stats (%d signals).", len(all_signals))


def backfill_missing_regimes_from_vix(
    *,
    source_prefix: str | None = "replay",
    recompute: bool = True,
) -> int:
    """Backfill signal-time VIX regimes for labelled historical rows.

    Replay rows often have clean labels but no regime context. This derives a
    one-year rolling VIX percentile using data available on or before each
    signal date, then writes BULL/NEUTRAL/BEAR labels in date batches.
    """
    init_backtest_db()
    where = "regime IS NULL"
    params: list = []
    if source_prefix:
        where += " AND source LIKE ?"
        params.append(f"{source_prefix}%")

    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT substr(run_date, 1, 10) AS run_day
                FROM signal_backtest
                WHERE {where}
                ORDER BY run_day""",
            params,
        ).fetchall()

    run_days = [r["run_day"] for r in rows if r["run_day"]]
    if not run_days:
        return 0

    try:
        start = (pd.to_datetime(min(run_days)) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(max(run_days)) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix_df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True, timeout=30)
        if vix_df is None or vix_df.empty:
            logger.warning("Regime backfill skipped: no VIX history returned")
            return 0
        closes = vix_df["Close"].dropna()
        if hasattr(closes, "columns"):
            closes = closes.iloc[:, 0]
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        closes = closes.sort_index()
    except Exception as exc:
        logger.warning("Regime backfill skipped: VIX download failed: %s", exc)
        return 0

    bull_threshold = float(getattr(config, "VIX_PERCENTILE_BULL", 25))
    bear_threshold = float(getattr(config, "VIX_PERCENTILE_BEAR", 75))
    updates: list[tuple[str, float, float, str]] = []
    for run_day in run_days:
        ts = pd.to_datetime(run_day)
        hist = closes.loc[closes.index <= ts].tail(252)
        if len(hist) < 60:
            continue
        vix_level = float(hist.iloc[-1])
        pct = float(np.sum(hist.values.astype(float) <= vix_level) / len(hist) * 100.0)
        if pct < bull_threshold:
            regime = "BULL"
        elif pct > bear_threshold:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"
        updates.append((regime, round(vix_level, 2), round(pct, 1), run_day))

    if not updates:
        return 0

    update_where = "substr(run_date, 1, 10) = ? AND regime IS NULL"
    if source_prefix:
        update_where += " AND source LIKE ?"

    with _connect() as conn:
        total = 0
        for regime, vix_level, pct, run_day in updates:
            values = [regime, vix_level, pct, run_day]
            if source_prefix:
                values.append(f"{source_prefix}%")
            cur = conn.execute(
                f"""UPDATE signal_backtest
                    SET regime = ?, vix_level = ?, vix_percentile = ?
                    WHERE {update_where}""",
                values,
            )
            total += cur.rowcount or 0

    if recompute and total:
        _recompute_all_stats()
    logger.info("Backfilled VIX regimes on %d signal rows.", total)
    return total


# ---------------------------------------------------------------------------
# Adaptive weights — feeds back into main scoring and discovery
# ---------------------------------------------------------------------------

def _query_pillar_ic(source: str, horizon: str) -> tuple[list, int]:
    """Query pillar_effectiveness for a given source/horizon.

    Returns (rows, min_sample_size).
    """
    with _connect() as conn:
        rows = conn.execute(
            """SELECT pillar, information_coefficient, sample_size
               FROM pillar_effectiveness
               WHERE source=? AND horizon=?""",
            (source, horizon),
        ).fetchall()
    min_samples = min((int(r["sample_size"] or 0) for r in rows), default=0)
    return rows, min_samples


def _estimate_signal_halflife(source: str = "all") -> dict[str, float]:
    """Estimate signal half-life per pillar from autocorrelation of rolling IC.

    Half-life = -log(2) / log(acf_lag_1).
    Pillars with longer half-lives (fundamental ~40-60d) get a weight boost
    for the 30-90 day holding horizon vs short-lived pillars (technical ~5-15d).

    Ledoit & Wolf (2004), Grinold & Kahn (2000) Chapter 14.
    """
    import math

    half_lives: dict[str, float] = {}
    pillars = ["technical", "fundamental", "sentiment", "forecast"]

    try:
        with _connect() as conn:
            for pillar in pillars:
                # Get rolling 7-day IC values over time
                col = f"{pillar}_score"
                rows = conn.execute(
                    f"""SELECT run_date, {col}, return_90d
                        FROM (
                            SELECT run_date, {col}, return_90d
                            FROM signal_backtest
                            WHERE evaluated_90d = 1
                              AND {col} IS NOT NULL
                              AND return_90d IS NOT NULL
                              AND source NOT LIKE 'replay%'
                              AND (source = ? OR ? = 'all')
                            ORDER BY date(run_date) DESC
                            LIMIT 2000
                        )
                        ORDER BY date(run_date)""",
                    (source, source),
                ).fetchall()

                if len(rows) < 40:
                    half_lives[pillar] = 30.0  # Default 30 days
                    continue

                # Compute rolling rank IC in 21-day windows
                import numpy as np
                scores = np.array([r[1] for r in rows], dtype=np.float64)
                returns = np.array([r[2] for r in rows], dtype=np.float64)

                window = 21
                rolling_ics = []
                for i in range(window, len(scores)):
                    s_window = scores[i - window:i]
                    r_window = returns[i - window:i]
                    if np.std(s_window) < 1e-8 or np.std(r_window) < 1e-8:
                        continue
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(s_window, r_window)
                    if not np.isnan(ic):
                        rolling_ics.append(ic)

                if len(rolling_ics) < 10:
                    half_lives[pillar] = 30.0
                    continue

                # Lag-1 autocorrelation of rolling IC
                ic_arr = np.array(rolling_ics)
                if np.std(ic_arr) < 1e-8:
                    half_lives[pillar] = 30.0
                    continue

                acf_1 = np.corrcoef(ic_arr[:-1], ic_arr[1:])[0, 1]
                if not np.isfinite(acf_1) or acf_1 <= 0:
                    half_lives[pillar] = 5.0  # Very short-lived signal
                else:
                    half_lives[pillar] = max(1.0, -math.log(2) / math.log(acf_1))

        logger.info("Signal half-lives: %s", {k: f"{v:.1f}d" for k, v in half_lives.items()})
    except Exception as e:
        logger.debug("Signal half-life estimation failed: %s", e)
        half_lives = {p: 30.0 for p in pillars}

    return half_lives


def _ic_rows_to_weights(rows, source: str, horizon: str, min_samples: int) -> dict[str, float] | None:
    """Convert pillar IC rows to shrunk, normalized weights.

    Enhanced with signal half-life weighting (Grinold & Kahn 2000):
    pillars with longer half-lives receive a boost for the 30-90 day horizon.
    """
    pillars = ["technical", "fundamental", "sentiment", "forecast"]
    raw = {pillar: 0.05 for pillar in pillars}
    for r in rows:
        # Floor at 0.05 to never fully zero out a pillar
        pillar = r["pillar"]
        if pillar not in raw:
            continue
        ic = max(r["information_coefficient"], 0.05)
        raw[pillar] = ic

    total = sum(raw.values())
    if total <= 0:
        return None

    weights = {k: round(v / total, 4) for k, v in raw.items()}

    # Signal half-life boost: pillars with longer persistence get upweighted
    # for the 30-90 day holding period
    half_lives = _estimate_signal_halflife(source)
    target_horizon_days = 63.0  # 90 calendar days ≈ 63 trading days
    for pillar in weights:
        hl = half_lives.get(pillar, 30.0)
        # Boost = fraction of signal remaining at target horizon
        # signal_remaining = 0.5^(horizon/half_life)
        import math
        signal_remaining = 0.5 ** (target_horizon_days / max(hl, 1.0))
        # Scale: 0.0 → 0.5x weight, 1.0 → 1.5x weight
        hl_multiplier = 0.5 + signal_remaining
        weights[pillar] = weights[pillar] * hl_multiplier

    # Re-normalize after half-life adjustment
    total_hl = sum(weights.values())
    if total_hl > 0:
        weights = {k: v / total_hl for k, v in weights.items()}

    # Apply Ledoit-Wolf shrinkage toward equal weights (Ledoit & Wolf 2004)
    # Shrinkage intensity adapts to sample size: more data → less shrinkage
    base_shrinkage = getattr(config, "WEIGHT_SHRINKAGE", 0.40)
    n_optimal = 200  # Optimal sample count for minimal shrinkage
    sample_adjusted_shrinkage = min(1.0, base_shrinkage * (n_optimal / max(min_samples, 1)))
    shrinkage = min(0.80, sample_adjusted_shrinkage)  # Cap at 80% shrinkage

    n = len(weights)
    equal = 1.0 / n
    shrunk = {k: round(shrinkage * equal + (1 - shrinkage) * v, 4) for k, v in weights.items()}

    # Normalize
    total_s = sum(shrunk.values())
    shrunk = {k: round(v / total_s, 4) for k, v in shrunk.items()}

    logger.info("Adaptive weights (%s/%s, n=%d, shrinkage=%.2f): %s",
                source, horizon, min_samples, shrinkage, shrunk)
    return shrunk


def get_adaptive_weights(source: str = "all", horizon: str = "90d") -> dict[str, float] | None:
    """Return IC-adjusted weights for scoring pillars.

    Used by both main scoring engine and discovery to adapt weights
    based on what actually predicts returns.

    Horizon fallback chain: tries the requested horizon first, then
    falls back to shorter horizons to reduce cold-start from ~3 months
    (90d only) down to 5-10d timing labels. Also broadens source if the
    requested source has too few samples.

    Args:
        source: 'portfolio', 'discovery', or 'all'
        horizon: '30d', '60d', or '90d' (preferred horizon)

    Returns dict like {"technical": 0.35, "fundamental": 0.20, ...} or None.
    """
    init_backtest_db()

    # Prefer longer horizons (more stable IC) but accept shorter during cold-start
    horizons = {
        "90d": ["90d", "60d", "30d", "10d", "5d"],
        "60d": ["60d", "30d", "10d", "5d"],
        "30d": ["30d", "10d", "5d"],
        "10d": ["10d", "5d"],
        "5d": ["5d"],
    }
    fallback_chain = horizons.get(horizon, ["90d", "60d", "30d", "10d", "5d"])

    # If specific source has too few samples, broaden to 'all'
    source_chain = [source] if source == "all" else [source, "all"]

    for hz in fallback_chain:
        for src in source_chain:
            rows, min_samples = _query_pillar_ic(src, hz)
            if not rows or min_samples < 20:
                continue

            result = _ic_rows_to_weights(rows, src, hz, min_samples)
            if result is not None:
                if hz != horizon or src != source:
                    logger.info(
                        "Adaptive weights: fell back from %s/%s to %s/%s (cold-start)",
                        source, horizon, src, hz,
                    )
                return result

    return None


# Backward compatible alias
def get_adaptive_discovery_weights() -> dict[str, float] | None:
    """Return adaptive weights specifically for discovery ranking."""
    return get_adaptive_weights(source="discovery", horizon="90d")


def get_kelly_fractions(source: str = "all") -> dict[str, float]:
    """Compute half-Kelly fractions per action tier from backtest data.

    Returns dict like {"STRONG BUY": 0.12, "BUY": 0.08, "KEEP": 0.05}
    where values are half-Kelly position size caps (as fractions of portfolio).
    Returns empty dict if insufficient data.
    """
    init_backtest_db()
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT action,
                          COUNT(*) as n,
                          AVG(CASE WHEN return_90d > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                          AVG(CASE WHEN return_90d > 0 THEN return_90d ELSE NULL END) as avg_win,
                          AVG(CASE WHEN return_90d <= 0 THEN ABS(return_90d) ELSE NULL END) as avg_loss
                   FROM signal_backtest
                   WHERE evaluated_90d = 1
                     AND return_90d IS NOT NULL
                     AND (source = ? OR ? = 'all')
                     AND action IN ('STRONG BUY', 'BUY', 'KEEP')
                   GROUP BY action
                   HAVING COUNT(*) >= 20""",
                (source, source),
            ).fetchall()
    except Exception as e:
        logger.warning("Kelly fraction query failed: %s", e)
        return {}

    result = {}
    for row in rows:
        action = row[0]
        win_rate = row[2]
        avg_win = row[3]
        avg_loss = row[4]

        if avg_loss is None or avg_loss < 0.01 or avg_win is None:
            continue

        b = avg_win / avg_loss  # win/loss ratio
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b  # Full Kelly

        if kelly <= 0:
            continue  # Negative Kelly = don't bet

        half_kelly = kelly / 2.0
        result[action] = min(half_kelly, 0.25)

    if result:
        logger.info("Kelly fractions (%s): %s", source, result)
    return result


# ---------------------------------------------------------------------------
# Query helpers for UI
# ---------------------------------------------------------------------------

def get_pick_performance(limit: int = 100) -> list[dict]:
    """Return evaluated signals with actual returns."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ticker, name, run_date, source, signal_price, action,
                      aggregate_score, return_30d, return_60d, return_90d,
                      spy_return_90d, beat_market, action_correct,
                      forecast_error_5d, forecast_error_63d,
                      stop_hit, stop_hit_day, target_hit, target_hit_day,
                      entry_method, entry_price, fill_probability,
                      position_sizing_method, planned_position_weight,
                      planned_risk_amount, r_r_ratio,
                      regime, sector
               FROM signal_backtest WHERE evaluated_90d = 1
               ORDER BY run_date DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pillar_stats(source: str = "all", horizon: str = "90d") -> list[dict]:
    """Return pillar effectiveness stats."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM pillar_effectiveness
               WHERE source=? AND horizon=?
               ORDER BY information_coefficient DESC""",
            (source, horizon),
        ).fetchall()
    return [dict(r) for r in rows]


def get_action_calibration(source: str = "all") -> list[dict]:
    """Return action accuracy stats."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM action_calibration WHERE source=? ORDER BY action",
            (source,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_regime_stats() -> list[dict]:
    """Return regime effectiveness stats."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM regime_effectiveness ORDER BY regime"
        ).fetchall()
    return [dict(r) for r in rows]


def get_stop_target_stats() -> dict:
    """Aggregate stop-loss and take-profit hit statistics."""
    init_backtest_db()
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN stop_loss IS NOT NULL THEN 1 ELSE 0 END) as with_stops,
                SUM(stop_hit) as stops_hit,
                AVG(CASE WHEN stop_hit=1 THEN stop_hit_day END) as avg_stop_day,
                SUM(target_hit) as targets_hit,
                AVG(CASE WHEN target_hit=1 THEN target_hit_day END) as avg_target_day,
                AVG(forecast_error_5d) as avg_forecast_err_5d,
                AVG(forecast_error_63d) as avg_forecast_err_63d
            FROM signal_backtest WHERE evaluated_90d = 1
        """).fetchone()
    return dict(row) if row else {}


def get_entry_plan_stats() -> list[dict]:
    """Return evaluated entry-plan statistics by entry method."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                COALESCE(entry_method, 'unknown') as entry_method,
                COUNT(*) as signals,
                AVG(CASE
                        WHEN entry_price IS NOT NULL AND price_30d IS NOT NULL AND entry_price > 0
                        THEN (price_30d - entry_price) / entry_price
                    END) as avg_return_from_entry_30d,
                AVG(CASE
                        WHEN target_hit = 1
                             AND (
                                 stop_hit = 0
                                 OR stop_hit IS NULL
                                 OR (
                                     target_hit_day IS NOT NULL
                                     AND stop_hit_day IS NOT NULL
                                     AND target_hit_day < stop_hit_day
                                 )
                             )
                        THEN 1.0 ELSE 0.0
                    END) as plan_hit_rate,
                AVG(CASE
                        WHEN entry_price IS NOT NULL
                             AND stop_loss IS NOT NULL
                             AND price_90d IS NOT NULL
                             AND entry_price > stop_loss
                        THEN (price_90d - entry_price) / (entry_price - stop_loss)
                    END) as avg_realized_r_multiple
            FROM signal_backtest
            WHERE source = 'discovery'
              AND evaluated_90d = 1
              AND entry_price IS NOT NULL
            GROUP BY COALESCE(entry_method, 'unknown')
            ORDER BY signals DESC, entry_method
        """).fetchall()
    return [dict(r) for r in rows]


def get_forecast_accuracy() -> list[dict]:
    """Return forecast accuracy breakdown by ticker."""
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT ticker,
                   COUNT(*) as signals,
                   AVG(forecast_error_5d) as avg_err_5d,
                   AVG(forecast_error_63d) as avg_err_63d,
                   AVG(return_90d) as avg_return
            FROM signal_backtest
            WHERE evaluated_90d = 1 AND forecast_price_5d IS NOT NULL
            GROUP BY ticker
            ORDER BY avg_err_5d
        """).fetchall()
    return [dict(r) for r in rows]


def get_pending_picks_count() -> int:
    """Return count of signals awaiting 90d evaluation."""
    init_backtest_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM signal_backtest WHERE evaluated_90d = 0"
        ).fetchone()
    return row["cnt"] if row else 0


def _parse_horizon_arg(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Signal backtest evaluation utilities")
    parser.add_argument("--count-pending", action="store_true", help="Print matured unevaluated counts")
    parser.add_argument("--batched-evaluate", action="store_true", help="Run batched yfinance evaluator")
    parser.add_argument("--horizons", default="5,10", help="Comma-separated horizons, e.g. 5,10,30")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Count only; do not download or update")
    args = parser.parse_args()

    horizons = _parse_horizon_arg(args.horizons)
    if args.count_pending or args.dry_run:
        print(json.dumps(count_mature_signal_horizon_pairs(horizons), indent=2, sort_keys=True))
    if args.batched_evaluate:
        n = evaluate_matured_signals_batched(
            horizons=horizons,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        print(f"evaluated_signal_horizon_pairs={n}")


if __name__ == "__main__":
    _main()
