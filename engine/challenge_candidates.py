"""Curated challenge-candidate generator for discovery.

The challenge list is an exploration sleeve, not a recommendation list.  It
promotes plausible candidates into deeper scoring when they are easy to miss
through source gaps, temporary momentum crowding, or cold-start data sparsity.
Every candidate still has to pass the normal deep scoring and action gates.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import config
from utils.atomic_io import atomic_write_json
from utils.global_universe import get_dynamic_entries, get_full_universe, is_excluded_ticker

logger = logging.getLogger(__name__)


def _safe_float(value, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        f = float(value)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clip(value: float | None, lo: float = 0.0, hi: float = 1.0) -> float:
    value = _safe_float(value, 0.0) or 0.0
    return max(lo, min(hi, value))


def _norm_symbol(value) -> str:
    return str(value or "").upper().strip()


def _as_date(value=None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return _as_date(value)
    except (TypeError, ValueError):
        return None


def _feature_cache_dir(path: str | Path | None = None) -> Path:
    return Path(path or "feature_cache")


def _latest_feature_file(feature_cache_dir: str | Path | None = None) -> Path | None:
    base = _feature_cache_dir(feature_cache_dir)
    files = sorted(base.glob("features_*.json"), reverse=True)
    return files[0] if files else None


def _load_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _load_latest_features(feature_cache_dir: str | Path | None = None) -> tuple[dict[str, dict], str | None]:
    path = _latest_feature_file(feature_cache_dir)
    if path is None:
        return {}, None
    raw = _load_json(path)
    features = raw.get("features", {})
    return (features if isinstance(features, dict) else {}, raw.get("date") or path.stem.replace("features_", ""))


def _load_latest_pit_snapshots(pit_path: str | Path | None = None) -> dict[str, dict]:
    raw = _load_json(pit_path or "feature_cache/pit_fundamentals.json")
    tickers = raw.get("tickers", {})
    out: dict[str, dict] = {}
    if not isinstance(tickers, dict):
        return out
    for symbol, history in tickers.items():
        if not isinstance(history, dict) or not history:
            continue
        try:
            latest_key = sorted(history)[-1]
            latest = dict(history.get(latest_key) or {})
            latest["_snapshot_date"] = latest_key
            out[_norm_symbol(symbol)] = latest
        except Exception:
            continue
    return out


def _universe_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    try:
        for entry in get_full_universe():
            meta[_norm_symbol(entry.ticker)] = {
                "country": entry.country,
                "sector": entry.sector,
                "exchange": entry.exchange,
                "index_source": entry.index_source,
            }
    except Exception:
        pass
    try:
        for entry in get_dynamic_entries():
            symbol = _norm_symbol(entry.get("ticker"))
            if not symbol:
                continue
            meta.setdefault(symbol, {}).update({
                "country": entry.get("country", ""),
                "sector": entry.get("sector", ""),
                "exchange": entry.get("exchange", ""),
                "index_source": "DYNAMIC",
            })
    except Exception:
        pass
    return meta


def _infer_country(symbol: str, metadata: dict | None = None) -> str:
    if metadata and metadata.get("country"):
        return str(metadata["country"])
    if symbol.endswith(".L"):
        return "GB"
    if symbol.endswith(".T"):
        return "JP"
    if symbol.endswith(".TO"):
        return "CA"
    if symbol.endswith(".HK"):
        return "HK"
    if symbol.endswith(".SI"):
        return "SG"
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        return "KR"
    return "US" if "." not in symbol else ""


def _free_cashflow(snapshot: dict) -> float | None:
    fcf = _safe_float(snapshot.get("free_cashflow"), None)
    if fcf is not None:
        return fcf
    cfo = _safe_float(snapshot.get("operating_cashflow"), None)
    capex = _safe_float(snapshot.get("capital_expenditure"), None)
    if cfo is None or capex is None:
        return None
    return cfo + capex if capex < 0 else cfo - capex


def _score_factor_candidate(symbol: str, feature: dict, pit: dict, metadata: dict) -> dict | None:
    country = _infer_country(symbol, metadata)
    sector = str(metadata.get("sector") or feature.get("sector") or "").strip()
    if is_excluded_ticker(symbol, country=country, region=country):
        return None

    price = _safe_float(feature.get("last_price"), 0.0) or 0.0
    dollar_volume = _safe_float(feature.get("avg_dollar_volume"), 0.0) or 0.0
    min_dv = float(getattr(config, "DISCOVERY_CHALLENGE_MIN_DOLLAR_VOLUME", 5_000_000))
    if price <= 1.0 or dollar_volume < min_dv:
        return None

    ret_90d = _safe_float(feature.get("ret_90d"), 0.0) or 0.0
    ret_30d = _safe_float(feature.get("ret_30d"), 0.0) or 0.0
    ret_10d = _safe_float(feature.get("ret_10d"), 0.0) or 0.0
    rel = _safe_float(feature.get("relative_strength"), 0.0) or 0.0
    vol_20d = _safe_float(feature.get("vol_20d"), 0.35) or 0.35
    pct_high = _safe_float(feature.get("pct_from_high_252d"), feature.get("pct_from_high"),) or 0.0
    above_50 = bool(feature.get("above_sma50"))
    above_200 = bool(feature.get("above_sma200"))

    momentum_score = (
        0.35 * _clip((ret_90d + 0.05) / 0.30)
        + 0.20 * _clip((ret_30d + 0.03) / 0.15)
        + 0.10 * _clip((ret_10d + 0.02) / 0.08)
        + 0.15 * _clip((rel + 0.05) / 0.20)
        + 0.10 * (1.0 if above_200 else 0.0)
        + 0.10 * _clip((pct_high - 0.65) / 0.35)
    )
    readiness_score = (
        0.35 * (1.0 if above_50 else 0.0)
        + 0.35 * (1.0 if above_200 else 0.0)
        + 0.15 * _clip((pct_high - 0.75) / 0.20)
        + 0.15 * _clip((0.55 - vol_20d) / 0.45)
    )

    assets = _safe_float(pit.get("total_assets"), None)
    gross_profit = _safe_float(pit.get("gross_profit"), None)
    gpa = gross_profit / assets if gross_profit is not None and assets and assets > 0 else None
    fcf = _free_cashflow(pit)
    fcf_assets = fcf / assets if fcf is not None and assets and assets > 0 else None
    ni = _safe_float(pit.get("net_income"), None)
    cfo = _safe_float(pit.get("operating_cashflow"), None)
    debt = _safe_float(pit.get("total_debt"), None)
    if debt is None:
        debt = (_safe_float(pit.get("long_term_debt"), 0.0) or 0.0) + (_safe_float(pit.get("short_term_debt"), 0.0) or 0.0)
    debt_assets = debt / assets if debt is not None and assets and assets > 0 else None

    quality_bits = []
    if gpa is not None:
        quality_bits.append(_clip((gpa - 0.10) / 0.25))
    if fcf_assets is not None:
        quality_bits.append(_clip((fcf_assets + 0.01) / 0.08))
    if ni is not None and cfo is not None:
        quality_bits.append(1.0 if cfo > ni else 0.25)
    if debt_assets is not None:
        quality_bits.append(_clip((0.55 - debt_assets) / 0.55))
    quality_score = sum(quality_bits) / len(quality_bits) if quality_bits else 0.0

    mcap = _safe_float(pit.get("market_cap"), None)
    pe = _safe_float(pit.get("trailing_pe"), None)
    value_bits = []
    if pe is not None and pe > 0:
        value_bits.append(_clip((30.0 - pe) / 25.0))
    if mcap and mcap > 0 and ni is not None and ni > 0:
        value_bits.append(_clip((ni / mcap - 0.01) / 0.08))
    if mcap and mcap > 0 and fcf is not None and fcf > 0:
        value_bits.append(_clip((fcf / mcap - 0.01) / 0.08))
    ebit = _safe_float(pit.get("ebit"), None)
    cash = _safe_float(pit.get("cash"), 0.0) or 0.0
    if mcap and mcap > 0 and ebit is not None and ebit > 0:
        ev = mcap + max(debt or 0.0, 0.0) - max(cash, 0.0)
        if ev > 0:
            value_bits.append(_clip((ebit / ev - 0.03) / 0.10))
    value_score = sum(value_bits) / len(value_bits) if value_bits else 0.0

    rev_growth = _safe_float(pit.get("revenue_growth"), None)
    earn_growth = _safe_float(pit.get("earnings_growth"), None)
    revision_score = 0.0
    revision_parts = []
    if rev_growth is not None:
        revision_parts.append(_clip((rev_growth + 0.02) / 0.25))
    if earn_growth is not None:
        revision_parts.append(_clip((earn_growth + 0.05) / 0.40))
    if revision_parts:
        revision_score = sum(revision_parts) / len(revision_parts)

    liquidity_score = _clip((dollar_volume - min_dv) / max(min_dv * 20.0, 1.0))
    signal_groups = {
        "value": value_score >= 0.45,
        "quality": quality_score >= 0.45,
        "momentum": momentum_score >= 0.55,
        "revision": revision_score >= 0.45,
        "ready": readiness_score >= 0.55,
    }
    min_groups = int(getattr(config, "DISCOVERY_CHALLENGE_MIN_FACTOR_GROUPS", 2))
    if sum(1 for ok in signal_groups.values() if ok) < min_groups:
        return None

    score = (
        0.25 * value_score
        + 0.25 * quality_score
        + 0.20 * momentum_score
        + 0.15 * revision_score
        + 0.10 * readiness_score
        + 0.05 * liquidity_score
    )
    reasons = [name for name, ok in signal_groups.items() if ok]
    return {
        "symbol": symbol,
        "companyName": symbol,
        "country": country,
        "sector": sector,
        "exchange": metadata.get("exchange", ""),
        "source": "factor_validated",
        "challenge_source": "factor_validated",
        "challenge_score": round(float(score), 4),
        "challenge_reason": "Factor validated: " + ", ".join(reasons),
        "required_checks": ["liquidity", "health", "ready_entry"],
        "factor_groups": reasons,
        "expires_after_days": int(getattr(config, "DISCOVERY_CHALLENGE_EXPIRY_DAYS", 30)),
    }


def _manual_candidates(raw_entries: Iterable, *, as_of: date) -> list[dict]:
    out: list[dict] = []
    default_expiry = int(getattr(config, "DISCOVERY_CHALLENGE_EXPIRY_DAYS", 30))
    for raw in raw_entries or []:
        if isinstance(raw, dict):
            symbol = _norm_symbol(raw.get("symbol") or raw.get("ticker"))
            if not symbol:
                continue
            expires_at = _parse_date(raw.get("expires_at"))
            added_at = _parse_date(raw.get("added_at")) or as_of
            if expires_at is None:
                days = int(raw.get("expires_after_days", default_expiry) or default_expiry)
                expires_at = added_at + timedelta(days=days)
            if expires_at < as_of:
                continue
            country = str(raw.get("country") or _infer_country(symbol, None))
            if is_excluded_ticker(symbol, country=country, region=country):
                continue
            out.append({
                "symbol": symbol,
                "companyName": raw.get("companyName") or raw.get("name") or symbol,
                "country": country,
                "sector": raw.get("sector", ""),
                "exchange": raw.get("exchange", ""),
                "source": "manual_override",
                "challenge_source": "manual_override",
                "challenge_score": float(raw.get("score", 1.0) or 1.0),
                "challenge_reason": raw.get("reason", "Manual research override"),
                "required_checks": raw.get("required_checks", ["liquidity", "health", "ready_entry"]),
                "expires_at": expires_at.isoformat(),
            })
        else:
            symbol = _norm_symbol(raw)
            if not symbol or is_excluded_ticker(symbol):
                continue
            out.append({
                "symbol": symbol,
                "companyName": symbol,
                "country": _infer_country(symbol, None),
                "sector": "",
                "exchange": "",
                "source": "manual_override",
                "challenge_source": "manual_override",
                "challenge_score": 1.0,
                "challenge_reason": "Manual research override",
                "required_checks": ["liquidity", "health", "ready_entry"],
                "expires_at": (as_of + timedelta(days=default_expiry)).isoformat(),
            })
    return out


def _near_miss_candidates(*, as_of: date, metadata: dict[str, dict]) -> list[dict]:
    lookback = int(getattr(config, "DISCOVERY_CHALLENGE_LOOKBACK_DAYS", 21))
    cutoff = (as_of - timedelta(days=lookback)).isoformat()
    rows: list[dict] = []
    try:
        from engine.paper_trading import _connect
        with _connect() as conn:
            query = """
                SELECT p.*
                  FROM signal_backtest p
                 WHERE p.source = 'discovery_panel'
                   AND substr(p.run_date, 1, 10) >= ?
                   AND NOT EXISTS (
                       SELECT 1
                         FROM signal_backtest d
                        WHERE d.source = 'discovery'
                          AND d.ticker = p.ticker
                          AND substr(d.run_date, 1, 10) = substr(p.run_date, 1, 10)
                   )
                 ORDER BY p.run_date DESC
                 LIMIT 1000
            """
            rows = [dict(row) for row in conn.execute(query, (cutoff,)).fetchall()]
    except Exception as exc:
        logger.debug("Challenge near-miss query skipped: %s", exc)
        return []

    best: dict[str, dict] = {}
    for row in rows:
        symbol = _norm_symbol(row.get("ticker"))
        if not symbol:
            continue
        meta = metadata.get(symbol, {})
        country = _infer_country(symbol, meta)
        if is_excluded_ticker(symbol, country=country, region=country):
            continue
        sleeve_scores = [
            _safe_float(row.get("sleeve_momentum"), None),
            _safe_float(row.get("sleeve_quality"), None),
            _safe_float(row.get("sleeve_value"), None),
            _safe_float(row.get("sleeve_low_risk"), None),
            _safe_float(row.get("sleeve_pead"), None),
            _safe_float(row.get("sleeve_ready"), None),
        ]
        sleeve_scores = [s for s in sleeve_scores if s is not None]
        sleeve_avg = sum(sleeve_scores) / len(sleeve_scores) if sleeve_scores else 0.0
        aggregate = _safe_float(row.get("aggregate_score"), 0.0) or 0.0
        momentum = _safe_float(row.get("momentum_score"), 0.0) or 0.0
        value = _safe_float(row.get("value_factor_score"), 0.0) or 0.0
        quality = _safe_float(row.get("quality_factor_score"), 0.0) or 0.0
        ready = _safe_float(row.get("sleeve_ready"), 0.0) or 0.0
        score = 0.45 * aggregate + 0.20 * sleeve_avg + 0.15 * momentum + 0.10 * quality + 0.05 * value + 0.05 * ready
        candidate = {
            "symbol": symbol,
            "companyName": row.get("name") or symbol,
            "country": country,
            "sector": row.get("sector") or meta.get("sector", ""),
            "exchange": row.get("exchange") or meta.get("exchange", ""),
            "source": "near_miss",
            "challenge_source": "near_miss",
            "challenge_score": round(float(score), 4),
            "challenge_reason": f"Near miss: Stage 5b score {aggregate:.2f}, lens {row.get('entry_lens') or 'unknown'}",
            "required_checks": ["liquidity", "health", "ready_entry"],
            "last_seen": str(row.get("run_date") or "")[:10],
            "expires_after_days": int(getattr(config, "DISCOVERY_CHALLENGE_EXPIRY_DAYS", 30)),
        }
        if symbol not in best or candidate["challenge_score"] > best[symbol]["challenge_score"]:
            best[symbol] = candidate
    return sorted(best.values(), key=lambda c: c.get("challenge_score", 0.0), reverse=True)


def _factor_candidates(
    *,
    feature_cache_dir: str | Path | None = None,
    pit_path: str | Path | None = None,
) -> list[dict]:
    features, _feature_date = _load_latest_features(feature_cache_dir)
    pit_latest = _load_latest_pit_snapshots(pit_path)
    metadata = _universe_metadata()
    candidates: list[dict] = []
    for symbol, feature in features.items():
        symbol = _norm_symbol(symbol)
        if not symbol or not isinstance(feature, dict):
            continue
        candidate = _score_factor_candidate(
            symbol,
            feature,
            pit_latest.get(symbol, {}),
            metadata.get(symbol, {}),
        )
        if candidate:
            candidates.append(candidate)
    return sorted(candidates, key=lambda c: c.get("challenge_score", 0.0), reverse=True)


def _with_expiry(candidate: dict, as_of: date) -> dict:
    out = dict(candidate)
    if not out.get("expires_at"):
        days = int(out.pop("expires_after_days", getattr(config, "DISCOVERY_CHALLENGE_EXPIRY_DAYS", 30)) or 30)
        out["expires_at"] = (as_of + timedelta(days=days)).isoformat()
    return out


def _apply_caps(
    candidates: list[dict],
    *,
    limit: int,
    seen: set[str] | None = None,
    sector_counts: dict[str, int] | None = None,
    country_counts: dict[str, int] | None = None,
) -> list[dict]:
    max_sector = int(getattr(config, "DISCOVERY_CHALLENGE_MAX_PER_SECTOR", 8))
    max_country = int(getattr(config, "DISCOVERY_CHALLENGE_MAX_PER_COUNTRY", 10))
    seen = set(seen or set())
    sector_counts = sector_counts if sector_counts is not None else {}
    country_counts = country_counts if country_counts is not None else {}
    selected: list[dict] = []
    for candidate in candidates:
        symbol = _norm_symbol(candidate.get("symbol"))
        if not symbol or symbol in seen:
            continue
        country = str(candidate.get("country") or _infer_country(symbol, None))
        sector = str(candidate.get("sector") or "Unknown")
        if is_excluded_ticker(symbol, country=country, region=country):
            continue
        if country_counts.get(country, 0) >= max_country:
            continue
        if sector_counts.get(sector, 0) >= max_sector:
            continue
        candidate["symbol"] = symbol
        candidate["country"] = country
        selected.append(candidate)
        seen.add(symbol)
        country_counts[country] = country_counts.get(country, 0) + 1
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def build_challenge_candidates(
    *,
    target_n: int | None = None,
    as_of=None,
    feature_cache_dir: str | Path | None = None,
    pit_path: str | Path | None = None,
    manual_overrides: Iterable | None = None,
    include_near_miss: bool = True,
    write_cache: bool = True,
) -> list[dict]:
    """Build the rotating curated challenge list."""
    as_of_date = _as_date(as_of)
    target = int(target_n or getattr(config, "DISCOVERY_CHALLENGE_TARGET_N", 60))
    target = max(0, target)
    if target <= 0:
        return []

    near_quota = int(round(target * float(getattr(config, "DISCOVERY_CHALLENGE_NEAR_MISS_PCT", 0.50))))
    factor_quota = int(round(target * float(getattr(config, "DISCOVERY_CHALLENGE_FACTOR_PCT", 0.35))))
    manual_quota = max(0, target - near_quota - factor_quota)
    configured_manual_pct = float(getattr(config, "DISCOVERY_CHALLENGE_MANUAL_PCT", 0.15))
    manual_quota = min(manual_quota, int(round(target * configured_manual_pct)))

    metadata = _universe_metadata()
    legacy_manual = list(getattr(config, "DISCOVERY_CHALLENGE_TICKERS", []) or [])
    configured_manual = list(getattr(config, "DISCOVERY_CHALLENGE_MANUAL_OVERRIDES", []) or [])
    if manual_overrides is not None:
        configured_manual.extend(list(manual_overrides))
    configured_manual.extend(legacy_manual)

    near = _near_miss_candidates(as_of=as_of_date, metadata=metadata) if include_near_miss else []
    factors = _factor_candidates(feature_cache_dir=feature_cache_dir, pit_path=pit_path)
    manual = _manual_candidates(configured_manual, as_of=as_of_date)

    selected: list[dict] = []
    seen: set[str] = set()
    sector_counts: dict[str, int] = {}
    country_counts: dict[str, int] = {}
    for sleeve, quota in ((near, near_quota), (factors, factor_quota), (manual, manual_quota)):
        picked = _apply_caps(
            sleeve,
            limit=max(0, quota),
            seen=seen,
            sector_counts=sector_counts,
            country_counts=country_counts,
        )
        selected.extend(picked)
        seen.update(_norm_symbol(c.get("symbol")) for c in picked)

    if len(selected) < target:
        remainder = sorted(
            [
                c for c in [*near, *factors]
                if _norm_symbol(c.get("symbol")) not in seen
            ],
            key=lambda c: c.get("challenge_score", 0.0),
            reverse=True,
        )
        picked = _apply_caps(
            remainder,
            limit=target - len(selected),
            seen=seen,
            sector_counts=sector_counts,
            country_counts=country_counts,
        )
        selected.extend(picked)

    selected = [_with_expiry(c, as_of_date) for c in selected[:target]]
    selected.sort(key=lambda c: c.get("challenge_score", 0.0), reverse=True)

    if write_cache:
        payload = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "as_of": as_of_date.isoformat(),
            "target_n": target,
            "counts": {
                "near_miss_pool": len(near),
                "factor_pool": len(factors),
                "manual_pool": len(manual),
                "selected": len(selected),
            },
            "candidates": selected,
        }
        try:
            cache_path = Path(getattr(config, "DISCOVERY_CHALLENGE_CACHE_PATH", "feature_cache/challenge_candidates.json"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cache_path, payload, indent=2)
        except Exception as exc:
            logger.debug("Challenge candidate cache write skipped: %s", exc)

    return selected


def get_challenge_candidates(*, force_refresh: bool = False) -> list[dict]:
    """Return cached/generated challenge candidates for discovery."""
    if not bool(getattr(config, "DISCOVERY_CHALLENGE_AUTO_ENABLED", True)):
        return _manual_candidates(
            list(getattr(config, "DISCOVERY_CHALLENGE_MANUAL_OVERRIDES", []) or [])
            + list(getattr(config, "DISCOVERY_CHALLENGE_TICKERS", []) or []),
            as_of=_as_date(),
        )

    cache_path = Path(getattr(config, "DISCOVERY_CHALLENGE_CACHE_PATH", "feature_cache/challenge_candidates.json"))
    max_age_hours = float(getattr(config, "DISCOVERY_CHALLENGE_CACHE_MAX_AGE_HOURS", 12))
    if not force_refresh and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            generated_at = datetime.fromisoformat(str(payload.get("generated_at")))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
            candidates = payload.get("candidates", [])
            if age_hours <= max_age_hours and isinstance(candidates, list):
                return [
                    c for c in candidates
                    if not is_excluded_ticker(c.get("symbol", ""), country=c.get("country"), region=c.get("country"))
                    and (_parse_date(c.get("expires_at")) or _as_date()) >= _as_date()
                ]
        except Exception:
            pass
    return build_challenge_candidates(write_cache=True)
