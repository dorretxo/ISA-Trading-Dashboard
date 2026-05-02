"""Institutional cold-start prior for discovery ranking.

The model deliberately uses proven, commercially implemented equity factors
instead of waiting for this app's local paper-trading labels to mature.  Local
learning can adjust these priors later, but this layer is the immediate
"quality strong buy" guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping

import numpy as np

import config


@dataclass(frozen=True)
class InstitutionalPrior:
    score: float
    percentile: float
    confidence: float
    coverage: float
    components: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def passes_strong_buy_bar(self) -> bool:
        return (
            self.percentile >= float(getattr(config, "INSTITUTIONAL_PRIOR_STRONG_BUY_PERCENTILE", 0.90))
            and self.confidence >= float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_CONFIDENCE", 0.55))
            and self.coverage >= float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45))
        )


_COMPONENT_WEIGHTS = {
    # Quality Minus Junk: profitability, growth/stability, safety, payout proxy.
    "qmj": 0.20,
    "quality": 0.10,
    "value": 0.16,
    "momentum": 0.18,
    # Betting-Against-Beta style defensive risk control.
    "bab": 0.10,
    "low_risk": 0.06,
    "health": 0.12,
    # Post-earnings drift / revisions should help timing without dominating.
    "pead": 0.06,
    "earnings": 0.03,
    "liquidity": 0.05,
    # Trading-cost / churn proxy.  Kept small because it is a gate, not alpha.
    "turnover": 0.04,
}


def _float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float | None, lo: float = -1.0, hi: float = 1.0) -> float | None:
    if value is None:
        return None
    return float(min(hi, max(lo, value)))


def _bounded_mean(values: Iterable[float | None]) -> tuple[float | None, float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return None, 0.0
    return float(np.mean(np.clip(valid, -1.0, 1.0))), min(1.0, len(valid) / max(1, len(list(values)) or 1))


def _norm_ratio(value, scale: float, center: float = 0.0) -> float | None:
    v = _float(value)
    if v is None:
        return None
    return _clip((v - center) / scale)


def _inverse_positive(value, good: float, bad: float) -> float | None:
    v = _float(value)
    if v is None or v <= 0:
        return None
    if good == bad:
        return None
    return _clip(1.0 - 2.0 * ((v - good) / (bad - good)))


def _beta_score(value) -> float | None:
    beta = _float(value)
    if beta is None:
        return None
    # Low beta is good, but do not over-reward negative/hedge-like beta.
    beta = min(2.5, max(-0.5, beta))
    return _clip((1.0 - beta) / 0.8)


def _turnover_capacity_score(row: Mapping) -> float | None:
    if not getattr(config, "INSTITUTIONAL_PRIOR_TURNOVER_COST_ENABLED", True):
        return None
    avg_dollar_volume = _float(row.get("avg_dollar_volume", row.get("_avg_dollar_volume")))
    market_cap = _float(row.get("market_cap", row.get("_market_cap")))
    vol_20d = _float(row.get("vol_20d", row.get("_vol_20d")))
    ret_10d = _float(row.get("return_10d_prior", row.get("_ret_10d", row.get("ret_10d"))))
    ret_90d = _float(row.get("return_90d_prior", row.get("_ret_90d", row.get("ret_90d"))))

    components: list[float] = []
    if avg_dollar_volume is not None and avg_dollar_volume > 0:
        components.append(_clip((math.log10(max(avg_dollar_volume, 1.0)) - 7.0) / 2.0) or 0.0)
    if market_cap is not None and market_cap > 0:
        components.append(_clip((math.log10(max(market_cap, 1.0)) - 9.0) / 2.5) or 0.0)
    if vol_20d is not None:
        vol = vol_20d / 100.0 if abs(vol_20d) > 2.0 else vol_20d
        components.append(_clip((0.35 - vol) / 0.35) or 0.0)
    if ret_10d is not None:
        r10 = ret_10d / 100.0 if abs(ret_10d) > 2.0 else ret_10d
        r90 = 0.0
        if ret_90d is not None:
            r90 = ret_90d / 100.0 if abs(ret_90d) > 2.0 else ret_90d
        short_chase = max(0.0, abs(r10) - abs(r90) / 3.0)
        components.append(_clip(-short_chase / 0.12) or 0.0)

    if not components:
        return None
    return float(np.clip(np.mean(components), -1.0, 1.0))


def _component_scores(row: Mapping) -> tuple[dict[str, float | None], float]:
    """Return component scores in [-1, +1] plus raw feature coverage."""
    f_score = _float(row.get("f_score"))
    f_cov = _float(row.get("f_score_coverage"), 0.0) or 0.0
    if f_score is not None and f_cov >= 6.0 / 9.0:
        f_score_norm = _clip((f_score - 4.5) / 4.5)
    else:
        f_score_norm = None

    quality_vals = [
        _float(row.get("quality_factor_score")),
        _float(row.get("quality_score_fundamental")),
        _float(row.get("gpa_score")),
        _norm_ratio(row.get("gross_profitability"), 0.35),
        _norm_ratio(row.get("fcf_to_assets"), 0.12),
        _norm_ratio(row.get("roe"), 0.25),
    ]
    qmj_vals = [
        _float(row.get("qmj_factor_score")),
        _float(row.get("quality_factor_score")),
        _float(row.get("quality_score_fundamental")),
        _float(row.get("gpa_score")),
        _norm_ratio(row.get("gross_profitability"), 0.35),
        _norm_ratio(row.get("fcf_to_assets"), 0.12),
        _norm_ratio(row.get("roe"), 0.25),
        _float(row.get("earnings_stability")),
        f_score_norm,
        _clip(-((_float(row.get("debt_to_equity", row.get("debt_equity"))) or 0.0) / 2.0)),
    ]
    value_vals = [
        _float(row.get("value_factor_score")),
        _float(row.get("ev_ebit_score")),
        _norm_ratio(row.get("fcf_yield"), 0.10),
        _inverse_positive(row.get("pe_ratio"), good=8.0, bad=35.0),
        _inverse_positive(row.get("peg_ratio"), good=0.8, bad=3.0),
    ]
    stretch = _float(row.get("price_vs_sma200_stretch"))
    stretch_penalty = _clip(-stretch / 0.70) if stretch is not None and stretch > 0.25 else 0.0
    momentum_vals = [
        _float(row.get("momentum_factor_score")),
        _norm_ratio(row.get("relative_strength"), 0.25),
        _norm_ratio(row.get("return_90d_prior", row.get("_ret_90d", row.get("ret_90d"))), 0.35),
        _norm_ratio(row.get("return_30d_prior", row.get("_ret_30d", row.get("ret_30d"))), 0.20),
        0.4 if row.get("above_sma200") else (-0.2 if row.get("above_sma200") is False else None),
        _norm_ratio(row.get("sma50_slope"), 0.04),
        stretch_penalty,
    ]
    low_risk_vals = [
        _float(row.get("volatility_factor_score")),
        _clip(-(max((_float(row.get("beta_90d", row.get("_beta"))) or 1.0) - 1.0, 0.0) / 1.0)),
        _clip(-((_float(row.get("vol_20d", row.get("_vol_20d"))) or 0.25) - 0.25) / 0.40),
        _clip(-((_float(row.get("debt_to_equity", row.get("debt_equity"))) or 0.0) / 2.0)),
        _clip(-((_float(row.get("short_pct")) or 0.0) / 0.25)),
    ]
    bab_vals = [
        _float(row.get("bab_factor_score")),
        _float(row.get("volatility_factor_score")),
        _beta_score(row.get("beta_90d", row.get("_beta", row.get("beta")))),
        _clip(-((_float(row.get("vol_20d", row.get("_vol_20d"))) or 0.25) - 0.25) / 0.40),
        _float(row.get("idiosyncratic_vol_score")),
    ]
    health_vals = [
        f_score_norm,
        _float(row.get("gpa_score")),
        _clip(((_float(row.get("current_ratio")) or 1.0) - 1.0) / 1.5),
        _clip(((_float(row.get("cash_to_debt")) or 0.0) - 0.5) / 1.5),
        _clip(-((_float(row.get("net_debt_ebitda")) or 0.0) / 4.0)),
    ]
    earnings_vals = [
        _clip(-((_float(row.get("eps_growth_variance_5y")) or 0.0) / 0.30)),
        _float(row.get("earnings_stability")),
    ]
    pead_vals = [
        _float(row.get("pead_factor_score")),
        _float(row.get("sue_score")),
        _norm_ratio(row.get("revision_momentum_3m"), 0.15),
    ]
    liquidity_vals = [
        _clip((math.log10(max(_float(row.get("avg_dollar_volume", row.get("_avg_dollar_volume"))) or 1.0, 1.0)) - 6.0) / 2.0),
        _clip((math.log10(max(_float(row.get("market_cap", row.get("_market_cap"))) or 1.0, 1.0)) - 8.0) / 3.0),
    ]
    turnover_vals = [_float(row.get("turnover_cost_score")), _turnover_capacity_score(row)]

    raw_groups = {
        "qmj": qmj_vals,
        "quality": quality_vals,
        "value": value_vals,
        "momentum": momentum_vals,
        "bab": bab_vals,
        "low_risk": low_risk_vals,
        "health": health_vals,
        "pead": pead_vals,
        "earnings": earnings_vals,
        "liquidity": liquidity_vals,
        "turnover": turnover_vals,
    }
    components: dict[str, float | None] = {}
    coverage_parts = []
    for name, vals in raw_groups.items():
        valid_count = sum(1 for v in vals if v is not None)
        coverage_parts.append(valid_count / len(vals))
        components[name] = float(np.mean([v for v in vals if v is not None])) if valid_count else None
    return components, float(np.mean(coverage_parts)) if coverage_parts else 0.0


def _weighted_score(components: Mapping[str, float | None]) -> float:
    total = 0.0
    weight = 0.0
    for name, w in _COMPONENT_WEIGHTS.items():
        val = components.get(name)
        if val is None:
            continue
        total += float(val) * w
        weight += w
    if weight <= 0:
        return 0.0
    return float(np.clip(total / weight, -1.0, 1.0))


def _percentiles(rows: list[Mapping], scores: list[float]) -> list[float]:
    if not scores:
        return []
    global_order = np.argsort(np.argsort(scores))
    global_pct = (global_order + 1) / max(1, len(scores))
    out = list(float(v) for v in global_pct)

    sectors: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        sector = str(row.get("sector") or row.get("_sector") or "Unknown")
        sectors.setdefault(sector, []).append(idx)
    for idxs in sectors.values():
        if len(idxs) < 8:
            continue
        vals = [scores[i] for i in idxs]
        order = np.argsort(np.argsort(vals))
        for pos, row_idx in enumerate(idxs):
            out[row_idx] = float((order[pos] + 1) / len(idxs))
    return out


def _component_percentile_scores(
    rows: list[Mapping],
    component_rows: list[dict[str, float | None]],
) -> list[dict[str, float | None]]:
    """Return component ranks mapped to [-1, +1], sector-relative when stable."""
    if not component_rows:
        return []

    min_sector_n = int(getattr(config, "INSTITUTIONAL_PRIOR_SECTOR_COMPONENT_MIN_N", 8))
    out = [{k: None for k in _COMPONENT_WEIGHTS} for _ in component_rows]

    sectors: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        sector = str(row.get("sector") or row.get("_sector") or "Unknown")
        sectors.setdefault(sector, []).append(idx)

    for comp in _COMPONENT_WEIGHTS:
        valid_global = [
            (idx, comps.get(comp))
            for idx, comps in enumerate(component_rows)
            if comps.get(comp) is not None and math.isfinite(float(comps.get(comp)))
        ]
        if len(valid_global) < 3:
            continue

        def assign(idxs: list[int]) -> None:
            vals = [(idx, component_rows[idx].get(comp)) for idx in idxs if component_rows[idx].get(comp) is not None]
            vals = [(idx, float(v)) for idx, v in vals if math.isfinite(float(v))]
            if len(vals) < 3:
                return
            order = np.argsort(np.argsort([v for _, v in vals]))
            denom = max(1, len(vals) - 1)
            for rank_pos, (idx, _v) in zip(order, vals):
                pct = rank_pos / denom
                out[idx][comp] = float(np.clip(2.0 * pct - 1.0, -1.0, 1.0))

        assign([idx for idx, _ in valid_global])
        for idxs in sectors.values():
            if len(idxs) >= min_sector_n:
                assign(idxs)

    return out


def _blend_dynamic_components(
    static_components: list[dict[str, float | None]],
    dynamic_components: list[dict[str, float | None]],
) -> list[dict[str, float | None]]:
    if not dynamic_components or not getattr(config, "INSTITUTIONAL_PRIOR_DYNAMIC_COMPONENTS", True):
        return static_components
    blend = float(np.clip(getattr(config, "INSTITUTIONAL_PRIOR_DYNAMIC_BLEND", 0.50), 0.0, 1.0))
    blended: list[dict[str, float | None]] = []
    for static, dynamic in zip(static_components, dynamic_components):
        row: dict[str, float | None] = {}
        for comp in _COMPONENT_WEIGHTS:
            s = static.get(comp)
            d = dynamic.get(comp)
            if s is None:
                row[comp] = d
            elif d is None:
                row[comp] = s
            else:
                row[comp] = float(np.clip((1.0 - blend) * float(s) + blend * float(d), -1.0, 1.0))
        blended.append(row)
    return blended


def score_universe(rows: list[Mapping]) -> dict[str, InstitutionalPrior]:
    """Score a discovery batch and return ticker -> institutional prior."""
    if not rows:
        return {}

    row_list = [r for r in rows if isinstance(r, Mapping)]
    component_rows: list[dict[str, float | None]] = []
    coverage = []
    raw_scores = []
    for row in row_list:
        comps, cov = _component_scores(row)
        component_rows.append(comps)
        coverage.append(cov)
        raw_scores.append(_weighted_score(comps))

    dynamic_components = _component_percentile_scores(row_list, component_rows)
    component_rows = _blend_dynamic_components(component_rows, dynamic_components)
    raw_scores = [_weighted_score(comps) for comps in component_rows]
    pcts = _percentiles(row_list, raw_scores)
    result: dict[str, InstitutionalPrior] = {}
    for row, comps, cov, score, pct in zip(row_list, component_rows, coverage, raw_scores, pcts):
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        available_components = sum(1 for v in comps.values() if v is not None)
        confidence = float(np.clip(0.45 + 0.45 * cov + 0.10 * min(1.0, available_components / 5), 0.0, 1.0))
        reasons = []
        if pct < float(getattr(config, "INSTITUTIONAL_PRIOR_STRONG_BUY_PERCENTILE", 0.90)):
            reasons.append(f"institutional prior below top-decile bar ({pct:.0%})")
        if confidence < float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_CONFIDENCE", 0.55)):
            reasons.append(f"prior confidence low ({confidence:.0%})")
        if cov < float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)):
            reasons.append(f"prior coverage thin ({cov:.0%})")
        result[ticker] = InstitutionalPrior(
            score=round(score, 4),
            percentile=round(float(pct), 4),
            confidence=round(confidence, 4),
            coverage=round(cov, 4),
            components={k: (round(v, 4) if v is not None else None) for k, v in comps.items()},
            reasons=reasons,
        )
    return result


def neutral_prior() -> InstitutionalPrior:
    return InstitutionalPrior(score=0.0, percentile=0.5, confidence=0.0, coverage=0.0)
