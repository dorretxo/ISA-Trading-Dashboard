"""Cache-only sleeve composites for discovery cheap ranking.

The discovery funnel already computes or caches the academic factors; this
module is only the assembly layer.  It avoids provider calls and normalises
each sleeve cross-sectionally, with a sector-neutral blend so quality/value
names are compared against their real peer set before Stage 6 spends time on
them.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

import config
from engine.factors import cross_sectional_zscore

SLEEVE_NAMES = ("quality", "momentum", "value", "low_risk", "pead", "ready")


def _float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float | None, lo: float = -1.0, hi: float = 1.0) -> float | None:
    if value is None:
        return None
    return float(min(hi, max(lo, value)))


def _pm1(value) -> float | None:
    """Convert common [0, 1] scores to [-1, 1]; preserve signed scores."""
    v = _float(value)
    if v is None:
        return None
    if 0.0 <= v <= 1.0:
        return float(2.0 * v - 1.0)
    return _clip(v)


def _norm_return(value, scale: float) -> float | None:
    v = _float(value)
    if v is None:
        return None
    if abs(v) > 2.0:
        v /= 100.0
    return _clip(v / scale)


def _bounded_mean(values: list[float | None]) -> tuple[float | None, float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None, 0.0
    if not valid:
        return None, 0.0
    return float(np.mean(np.clip(valid, -1.0, 1.0))), len(valid) / len(values)


def _feature(candidate: Mapping, feature_cache: Mapping[str, Mapping], key: str):
    ticker = str(candidate.get("symbol") or "").upper()
    cached = feature_cache.get(ticker) or {}
    return candidate.get(key, cached.get(key))


def _sector_key(candidate: Mapping) -> str:
    return str(candidate.get("sector") or candidate.get("_sector") or "Unknown")


def _low_better(value, good: float, bad: float) -> float | None:
    v = _float(value)
    if v is None:
        return None
    if abs(v) > 2.0 and bad <= 2.0:
        v /= 100.0
    if good == bad:
        return None
    return _clip(1.0 - 2.0 * ((v - good) / (bad - good)))


def _raw_sleeves(candidate: Mapping, feature_cache: Mapping[str, Mapping]) -> tuple[dict[str, float | None], dict[str, float]]:
    beta = _feature(candidate, feature_cache, "beta_90d")
    if beta is None:
        beta = candidate.get("_beta", candidate.get("beta"))
    vol_20d = _feature(candidate, feature_cache, "vol_20d")
    if vol_20d is None:
        vol_20d = candidate.get("_vol_20d")

    momentum, momentum_cov = _bounded_mean([
        _pm1(candidate.get("_momentum_score")),
        _norm_return(candidate.get("_ret_90d", _feature(candidate, feature_cache, "ret_90d")), 0.25),
        _norm_return(candidate.get("_ret_30d", _feature(candidate, feature_cache, "ret_30d")), 0.15),
        _norm_return(candidate.get("_ret_10d", _feature(candidate, feature_cache, "ret_10d")), 0.08),
        _norm_return(candidate.get("_relative_strength", _feature(candidate, feature_cache, "relative_strength")), 0.12),
        _pm1(candidate.get("residual_momentum_score", candidate.get("_residual_momentum_score"))),
    ])

    f_score = _float(candidate.get("_f_score", candidate.get("f_score")))
    f_cov = _float(candidate.get("_f_score_coverage", candidate.get("f_score_coverage")), 0.0) or 0.0
    quality, quality_cov = _bounded_mean([
        _pm1(candidate.get("_quality_score")),
        _pm1(candidate.get("quality_factor_score")),
        _pm1(candidate.get("qmj_factor_score")),
        _clip(candidate.get("_pit_quality_score")),
        _pm1(candidate.get("gpa_score")),
        _clip((f_score - 4.5) / 4.5) if f_score is not None and f_cov >= 0.45 else None,
        _low_better(candidate.get("debt_to_assets", candidate.get("_debt_to_assets")), 0.15, 0.70),
    ])

    pe = _float(candidate.get("_pe_ratio", candidate.get("pe_ratio")))
    fcf_yield = _float(candidate.get("_fcf_yield", candidate.get("fcf_yield")))
    value, value_cov = _bounded_mean([
        _pm1(candidate.get("_value_score")),
        _pm1(candidate.get("value_factor_score")),
        _clip(candidate.get("_pit_value_score")),
        _pm1(candidate.get("ev_ebit_score")),
        _clip((fcf_yield - 0.03) / 0.08) if fcf_yield is not None else None,
        _clip((18.0 - pe) / 18.0) if pe is not None and pe > 0 else None,
    ])

    beta_score = None
    beta_f = _float(beta)
    if beta_f is not None:
        beta_score = _clip((1.0 - min(2.5, max(-0.5, beta_f))) / 0.8)
    low_risk, low_risk_cov = _bounded_mean([
        _pm1(candidate.get("_pit_low_risk_score")),
        _pm1(candidate.get("bab_factor_score")),
        _pm1(candidate.get("volatility_factor_score")),
        beta_score,
        _low_better(vol_20d, 0.12, 0.45),
        _low_better(candidate.get("downside_vol_60d"), 0.08, 0.35),
        _low_better(abs(_float(candidate.get("max_dd_252d"), 0.0) or 0.0), 0.08, 0.45),
    ])

    pead, pead_cov = _bounded_mean([
        _pm1(candidate.get("_pit_pead_score")),
        _pm1(candidate.get("pead_factor_score")),
        _pm1(candidate.get("sue_score")),
        _pm1(candidate.get("revision_momentum_3m")),
    ])

    stretch = _float(candidate.get("price_vs_sma200_stretch"))
    if stretch is None:
        last_price = _float(candidate.get("_last_price", candidate.get("price")))
        sma200 = _float(candidate.get("sma_200", candidate.get("_sma_200")))
        if last_price and sma200 and sma200 > 0:
            stretch = last_price / sma200 - 1.0
    rr = _float(candidate.get("r_r_ratio"))
    fill = _float(candidate.get("fill_probability"))
    ready, ready_cov = _bounded_mean([
        _pm1(candidate.get("_cheap_ready_score")),
        _clip((rr - 1.5) / 1.5) if rr is not None else None,
        _pm1(fill),
        0.45 if candidate.get("_above_sma50") else -0.20,
        0.35 if candidate.get("_above_sma200") else -0.25,
        _clip((0.50 - abs(stretch)) / 0.50) if stretch is not None else None,
    ])

    raw = {
        "momentum": momentum,
        "quality": quality,
        "value": value,
        "low_risk": low_risk,
        "pead": pead,
        "ready": ready,
    }
    coverage = {
        "momentum": momentum_cov,
        "quality": quality_cov,
        "value": value_cov,
        "low_risk": low_risk_cov,
        "pead": pead_cov,
        "ready": ready_cov,
    }
    return raw, coverage


def _normalise_by_sleeve(
    candidates: list[Mapping],
    raw_by_symbol: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    blend = float(getattr(config, "DISCOVERY_SLEEVE_SECTOR_NEUTRAL_BLEND", 0.35))
    blend = max(0.0, min(1.0, blend))
    out: dict[str, dict[str, float]] = {
        str(c.get("symbol") or "").upper(): {} for c in candidates
    }
    symbols = [str(c.get("symbol") or "").upper() for c in candidates]
    sectors = [_sector_key(c) for c in candidates]

    for sleeve in SLEEVE_NAMES:
        raw = np.array([
            np.nan if raw_by_symbol.get(sym, {}).get(sleeve) is None else raw_by_symbol[sym][sleeve]
            for sym in symbols
        ], dtype=np.float64)
        valid = np.isfinite(raw)
        global_scores = np.zeros(len(symbols), dtype=np.float64)
        if valid.any():
            z = cross_sectional_zscore(raw[valid])
            global_scores[valid] = np.clip(z / 2.5, -1.0, 1.0)

        sector_scores = np.zeros(len(symbols), dtype=np.float64)
        for sector in sorted(set(sectors)):
            idx = np.array([i for i, s in enumerate(sectors) if s == sector and valid[i]], dtype=int)
            if len(idx) < 4:
                sector_scores[idx] = global_scores[idx]
                continue
            z = cross_sectional_zscore(raw[idx])
            sector_scores[idx] = np.clip(z / 2.5, -1.0, 1.0)

        blended = (1.0 - blend) * global_scores + blend * sector_scores
        for i, sym in enumerate(symbols):
            out.setdefault(sym, {})[sleeve] = float(np.clip(blended[i], -1.0, 1.0))
    return out


def _normalised_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    configured = weights or getattr(config, "DISCOVERY_SLEEVE_WEIGHTS", {}) or {}
    out = {name: max(0.0, float(configured.get(name, 0.0))) for name in SLEEVE_NAMES}
    if sum(out.values()) <= 0:
        out = {"quality": 0.24, "momentum": 0.20, "value": 0.18, "low_risk": 0.15, "ready": 0.15, "pead": 0.08}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def compute_sleeve_scores(
    candidates: list[dict],
    *,
    sleeve_weights: Mapping[str, float] | None = None,
    feature_cache: Mapping[str, Mapping] | None = None,
) -> dict[str, dict]:
    """Return sleeve composites keyed by ticker symbol.

    Output per ticker:
    ``{"composite": float, "per_sleeve": dict, "coverage": dict,
    "low_coverage_sleeves": list[str]}``.
    """
    if feature_cache is None:
        try:
            from utils.feature_store import FeatureStore
            store = FeatureStore()
            store.load()
            feature_cache = store._data
        except Exception:
            feature_cache = {}

    raw_by_symbol: dict[str, dict[str, float | None]] = {}
    coverage_by_symbol: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        sym = str(candidate.get("symbol") or "").upper()
        raw, cov = _raw_sleeves(candidate, feature_cache or {})
        raw_by_symbol[sym] = raw
        coverage_by_symbol[sym] = cov

    norm = _normalise_by_sleeve(candidates, raw_by_symbol)
    weights = _normalised_weights(sleeve_weights)
    min_cov = float(getattr(config, "DISCOVERY_SLEEVE_MIN_COVERAGE", 0.50))
    results: dict[str, dict] = {}
    for candidate in candidates:
        sym = str(candidate.get("symbol") or "").upper()
        sleeves = norm.get(sym, {})
        coverage = coverage_by_symbol.get(sym, {})
        low_cov = [name for name in SLEEVE_NAMES if coverage.get(name, 0.0) < min_cov]

        composite = 0.0
        weight_total = 0.0
        for name, weight in weights.items():
            value = sleeves.get(name, 0.0)
            cov = coverage.get(name, 0.0)
            confidence = 0.50 + 0.50 * max(0.0, min(1.0, cov))
            composite += weight * confidence * value
            weight_total += weight * confidence
        composite = composite / weight_total if weight_total > 0 else 0.0
        # Stage 5a expects higher-is-better roughly [0, 1].
        results[sym] = {
            "composite": float(np.clip(0.5 + 0.5 * composite, 0.0, 1.0)),
            "per_sleeve": {k: float(v) for k, v in sleeves.items()},
            "coverage": {k: float(v) for k, v in coverage.items()},
            "low_coverage_sleeves": low_cov,
        }
    return results
