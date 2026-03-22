"""Comprehensive Signal Backtest — Track ALL recommendations and adapt ALL models.

Captures every signal the system generates (portfolio + discovery + swaps),
with every metric available at signal time. Evaluates actual outcomes at
30d, 60d, 90d horizons. Feeds back into:
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
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

import config

logger = logging.getLogger(__name__)

from engine.paper_trading import _connect

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

    -- Context
    sector              TEXT,
    exchange            TEXT,
    regime              TEXT,                   -- BULL / NEUTRAL / BEAR
    vix_level           REAL,
    portfolio_weight    REAL,                   -- current weight at signal time

    -- Multi-horizon evaluation (filled later)
    price_30d           REAL,
    price_60d           REAL,
    price_90d           REAL,
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
    ("macd_signal", "REAL"), ("sma_50", "REAL"), ("sma_200", "REAL"),
    ("stoch_k", "REAL"), ("williams_r", "REAL"), ("obv_trend", "TEXT"),
    ("atr", "REAL"), ("profit_margin", "REAL"), ("fcf_yield", "REAL"),
    ("debt_equity", "REAL"), ("inst_ownership", "REAL"), ("insider_net", "TEXT"),
    ("analyst_upside", "REAL"), ("analyst_rec", "TEXT"),
    ("reddit_score", "REAL"), ("fmp_news_score", "REAL"),
    ("earnings_proximity", "INTEGER"),
    ("return_10d_prior", "REAL"), ("return_30d_prior", "REAL"),
    ("return_90d_prior", "REAL"), ("vol_20d", "REAL"),
    ("vix_percentile", "REAL"), ("tnx_level", "REAL"),
    ("optimal_weight", "REAL"), ("pillar_weights_json", "TEXT"),
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
        ret_10d = float(closes.iloc[-1] / closes.iloc[-min(10, len(closes))] - 1) * 100 if len(closes) >= 10 else None
        ret_30d = float(closes.iloc[-1] / closes.iloc[-min(30, len(closes))] - 1) * 100 if len(closes) >= 30 else None
        ret_90d = float(closes.iloc[-1] / closes.iloc[-min(90, len(closes))] - 1) * 100 if len(closes) >= 90 else None
        vol_20d = float(data["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100) if len(closes) >= 20 else None
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
            tnx_level = float(tnx["Close"].iloc[-1])
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

            conn.execute(
                """INSERT INTO signal_backtest
                   (run_date, ticker, name, source, signal_price,
                    action, aggregate_score,
                    technical_score, fundamental_score, sentiment_score, forecast_score,
                    rsi, adx, bb_pct, pe_ratio, peg_ratio, revenue_growth, roe, short_pct,
                    news_score, momentum_score,
                    forecast_price_5d, forecast_price_63d,
                    stop_loss, take_profit,
                    macd_signal, sma_50, sma_200, stoch_k, williams_r, obv_trend, atr,
                    profit_margin, fcf_yield, debt_equity, inst_ownership, insider_net,
                    analyst_upside, analyst_rec, reddit_score, fmp_news_score,
                    earnings_proximity,
                    return_10d_prior, return_30d_prior, return_90d_prior, vol_20d,
                    vix_percentile, tnx_level, optimal_weight, pillar_weights_json,
                    sector, exchange, regime, vix_level, portfolio_weight)
                   VALUES (?,?,?,?,?, ?,?, ?,?,?,?, ?,?,?,?,?,?,?,?, ?,?, ?,?, ?,?,
                           ?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?,?)""",
                (now, ticker, r.get("name", ""), "portfolio", r.get("current_price", 0),
                 r.get("action"), r.get("aggregate_score"),
                 r.get("technical_score"), r.get("fundamental_score"),
                 r.get("sentiment_score"), r.get("forecast_score"),
                 r.get("rsi"), r.get("adx"), r.get("bb_pct"),
                 r.get("pe_ratio"), r.get("peg_ratio"),
                 r.get("revenue_growth"), r.get("roe"), r.get("short_pct"),
                 r.get("news_score"), None,
                 r.get("forecast_price"), r.get("forecast_price_long"),
                 r.get("stop_loss"), r.get("take_profit"),
                 # Point-in-time raw features
                 r.get("macd_signal"),
                 r.get("sma_50"),
                 r.get("sma_200"),
                 r.get("stoch_k"), r.get("williams_r"), r.get("obv_trend"), r.get("atr"),
                 r.get("profit_margin"), r.get("fcf_yield"), r.get("debt_to_equity"),
                 r.get("inst_ownership"), r.get("insider_net"),
                 r.get("analyst_upside"), r.get("analyst_rec"),
                 r.get("reddit_score"), r.get("fmp_news_score"),
                 r.get("earnings_proximity_days"),
                 ret_10d, ret_30d, ret_90d, vol_20d,
                 vix_pct, tnx_level, opt_map.get(ticker), weights_json,
                 # Context
                 None, None, regime_label, vix,
                 pw_map.get(ticker, 0)),
            )
            count += 1

    logger.info("Recorded %d portfolio signals for backtest (with feature store).", count)
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

            # Get signal price
            try:
                data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
                signal_price = float(data["Close"].iloc[-1]) if data is not None and not data.empty else 0
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

            conn.execute(
                """INSERT INTO signal_backtest
                   (run_date, ticker, name, source, signal_price,
                    action, aggregate_score,
                    technical_score, fundamental_score, sentiment_score, forecast_score,
                    momentum_score, sector, exchange)
                   VALUES (?,?,?,?,?, ?,?, ?,?,?,?, ?,?,?)""",
                (now, ticker, name, "discovery", signal_price,
                 action, agg, tech, fund, sent, fcast, mom, sector, exchange),
            )
            count += 1

    logger.info("Recorded %d discovery picks for backtest.", count)
    return count


# ---------------------------------------------------------------------------
# Evaluate matured signals — multi-horizon
# ---------------------------------------------------------------------------

def _fetch_price_at_offset(ticker: str, run_date: str, offset_days: int) -> float | None:
    """Fetch the closing price approximately offset_days after run_date."""
    try:
        base = datetime.strptime(run_date[:10], "%Y-%m-%d")
        start = (base + timedelta(days=offset_days - 3)).strftime("%Y-%m-%d")
        end = (base + timedelta(days=offset_days + 5)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            return float(data["Close"].iloc[0])
    except Exception:
        pass
    return None


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


def evaluate_matured_signals(horizon_days: int = 90) -> int:
    """Evaluate signals that have matured at the given horizon.

    Runs multi-horizon evaluation: 30d, 60d, and 90d checks.
    """
    init_backtest_db()
    evaluated_total = 0

    for horizon, col_price, col_return, col_flag in [
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

            # On 90d evaluation, also do full analysis
            if horizon == 90:
                # SPY benchmark
                rd = run_date[:10]
                if rd not in spy_cache:
                    spy_price_start = _fetch_price_at_offset("SPY", run_date, 0)
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

                # Forecast accuracy (5d and 63d)
                if sig["forecast_price_5d"]:
                    actual_5d = _fetch_price_at_offset(ticker, run_date, 5)
                    if actual_5d:
                        err_5d = abs(actual_5d - sig["forecast_price_5d"]) / signal_price * 100
                        updates["actual_price_5d"] = actual_5d
                        updates["forecast_error_5d"] = round(err_5d, 2)

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
    with _connect() as conn:
        all_signals = conn.execute(
            "SELECT * FROM signal_backtest WHERE evaluated_90d = 1"
        ).fetchall()

    if len(all_signals) < 5:
        return

    now = datetime.now().isoformat(timespec="seconds")

    # --- Pillar effectiveness per source and horizon ---
    pillars = ["technical_score", "fundamental_score", "sentiment_score", "forecast_score"]

    with _connect() as conn:
        conn.execute("DELETE FROM pillar_effectiveness")

        for source in ["portfolio", "discovery", "all"]:
            if source == "all":
                signals = all_signals
            else:
                signals = [s for s in all_signals if s["source"] == source]

            if len(signals) < 5:
                continue

            for horizon_col, horizon_label in [("return_30d", "30d"), ("return_60d", "60d"), ("return_90d", "90d")]:
                for pillar in pillars:
                    scores = np.array([s[pillar] or 0 for s in signals])
                    returns = np.array([s[horizon_col] or 0 for s in signals])

                    valid = ~(np.isnan(scores) | np.isnan(returns))
                    scores, returns = scores[valid], returns[valid]

                    if len(scores) < 5 or np.std(scores) == 0:
                        continue

                    median = np.median(scores)
                    high = scores >= median
                    low = scores < median

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
                           (updated_at, source, pillar, horizon, information_coefficient,
                            hit_rate, avg_return_high, avg_return_low, sample_size)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (now, source, pillar.replace("_score", ""), horizon_label,
                         round(ic, 4), round(hit_rate, 3),
                         round(avg_high, 2), round(avg_low, 2), len(signals)),
                    )

    # --- Action calibration ---
    with _connect() as conn:
        conn.execute("DELETE FROM action_calibration")

        for source in ["portfolio", "discovery", "all"]:
            signals = all_signals if source == "all" else [s for s in all_signals if s["source"] == source]
            if not signals:
                continue

            actions = set(s["action"] for s in signals if s["action"])
            for action in actions:
                action_signals = [s for s in signals if s["action"] == action]
                if not action_signals:
                    continue

                returns = [s["return_90d"] or 0 for s in action_signals]
                avg_ret = sum(returns) / len(returns)

                if action in ("STRONG BUY", "BUY"):
                    hits = sum(1 for r in returns if r > 0)
                elif action in ("SELL", "STRONG SELL"):
                    hits = sum(1 for r in returns if r < 0)
                else:
                    hits = sum(1 for r in returns if abs(r) < 15)

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

        regimes = set(s["regime"] for s in all_signals if s["regime"])
        for regime in regimes:
            reg_signals = [s for s in all_signals if s["regime"] == regime]
            if len(reg_signals) < 3:
                continue

            returns = [s["return_90d"] or 0 for s in reg_signals]
            avg_ret = sum(returns) / len(returns)

            # Find best pillar for this regime
            best_pillar, best_ic = None, -1
            for pillar in pillars:
                scores = np.array([s[pillar] or 0 for s in reg_signals])
                rets = np.array([s["return_90d"] or 0 for s in reg_signals])
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


# ---------------------------------------------------------------------------
# Adaptive weights — feeds back into main scoring and discovery
# ---------------------------------------------------------------------------

def get_adaptive_weights(source: str = "all", horizon: str = "90d") -> dict[str, float] | None:
    """Return IC-adjusted weights for scoring pillars.

    Used by both main scoring engine and discovery to adapt weights
    based on what actually predicts returns.

    Args:
        source: 'portfolio', 'discovery', or 'all'
        horizon: '30d', '60d', or '90d'

    Returns dict like {"technical": 0.35, "fundamental": 0.20, ...} or None.
    """
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT pillar, information_coefficient, sample_size
               FROM pillar_effectiveness
               WHERE source=? AND horizon=?""",
            (source, horizon),
        ).fetchall()

    if not rows:
        return None

    # Need minimum sample size to trust the data
    min_samples = rows[0]["sample_size"] if rows else 0
    if min_samples < 20:
        return None

    # IC-proportional weighting with floor
    raw = {}
    for r in rows:
        # Floor at 0.05 to never fully zero out a pillar
        ic = max(r["information_coefficient"], 0.05)
        raw[r["pillar"]] = ic

    total = sum(raw.values())
    if total <= 0:
        return None

    weights = {k: round(v / total, 4) for k, v in raw.items()}

    # Apply shrinkage toward equal weights (prevents overfitting)
    shrinkage = getattr(config, "WEIGHT_SHRINKAGE", 0.40)
    n = len(weights)
    equal = 1.0 / n
    shrunk = {k: round(shrinkage * equal + (1 - shrinkage) * v, 4) for k, v in weights.items()}

    # Normalize
    total_s = sum(shrunk.values())
    shrunk = {k: round(v / total_s, 4) for k, v in shrunk.items()}

    logger.info("Adaptive weights (%s/%s, n=%d): %s", source, horizon, min_samples, shrunk)
    return shrunk


# Backward compatible alias
def get_adaptive_discovery_weights() -> dict[str, float] | None:
    """Return adaptive weights specifically for discovery ranking."""
    return get_adaptive_weights(source="discovery", horizon="90d")


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
