"""Conservative self-learning overlays for discovery.

This module uses realised signal IC as a Bayesian update around the app's
configured priors.  It is intentionally slow-moving: small local samples can
nudge weights, but they cannot dominate the academic/commercial prior.
"""

from __future__ import annotations

import logging
import math
from statistics import median

import numpy as np

import config
from engine.paper_trading import _connect
from engine.pillar_weighting import (
    apply_weight_guardrails,
    blend_with_default_prior,
    pillar_parity_gate,
    positive_ic_allocation,
)

logger = logging.getLogger(__name__)


_PILLARS = ("technical", "fundamental", "sentiment", "forecast")


def _rank_ic(scores: list[float], returns: list[float]) -> float | None:
    if len(scores) < 5:
        return None
    s = np.asarray(scores, dtype=float)
    r = np.asarray(returns, dtype=float)
    valid = np.isfinite(s) & np.isfinite(r)
    s = s[valid]
    r = r[valid]
    if len(s) < 5 or np.std(s) < 1e-9 or np.std(r) < 1e-9:
        return None
    sr = np.argsort(np.argsort(s))
    rr = np.argsort(np.argsort(r))
    ic = float(np.corrcoef(sr, rr)[0, 1])
    return ic if math.isfinite(ic) else None


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def _load_horizon_rows(source: str, horizon: str) -> list:
    flag = f"evaluated_{horizon}"
    ret = f"return_{horizon}"
    try:
        with _connect() as conn:
            cols = ", ".join(f"{p}_score" for p in _PILLARS)
            return conn.execute(
                f"""SELECT {cols}, {ret}
                    FROM signal_backtest
                    WHERE {flag} = 1
                      AND {ret} IS NOT NULL
                      AND (source = ? OR ? = 'all')""",
                (source, source),
            ).fetchall()
    except Exception as exc:
        logger.debug("Bayesian learning: no rows for %s/%s (%s)", source, horizon, exc)
        return []


def _load_pillar_ic(min_n: int = 200) -> dict[tuple[str, str, str, str | None], tuple[float, int]]:
    """Load measured pillar IC from the pre-aggregated effectiveness table.

    Keys are ``(source, pillar, horizon, regime)`` and values are ``(ic, sample_size)``.
    The table is populated by the backtest/evaluation layer and is preferable
    to row-level recomputation when older labelled rows are missing pillar
    columns.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT source, pillar, horizon, regime, information_coefficient, sample_size
                   FROM pillar_effectiveness
                   WHERE sample_size >= ?
                     AND information_coefficient IS NOT NULL""",
                (int(min_n),),
            ).fetchall()
    except Exception as exc:
        logger.debug("Bayesian learning: pillar_effectiveness unavailable (%s)", exc)
        return {}

    out: dict[tuple[str, str, str], tuple[float, int]] = {}
    for row in rows:
        source = str(row["source"] or "").strip()
        pillar = str(row["pillar"] or "").strip()
        horizon = str(row["horizon"] or "").strip()
        regime = str(row["regime"] or "").strip().upper() or None
        if pillar not in _PILLARS or not source or not horizon:
            continue
        try:
            ic = float(row["information_coefficient"])
            n = int(row["sample_size"] or 0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic) or n < min_n:
            continue
        out[(source, pillar, horizon, regime)] = (ic, n)
    return out


def _cap_relative_delta(
    target: dict[str, float],
    base: dict[str, float],
    *,
    max_relative: float = 0.20,
) -> dict[str, float]:
    capped: dict[str, float] = {}
    for pillar in _PILLARS:
        prior = float(base.get(pillar, 0.0))
        value = float(target.get(pillar, prior))
        if prior > 0:
            lower = prior * (1.0 - max_relative)
            upper = prior * (1.0 + max_relative)
            value = float(np.clip(value, lower, upper))
        capped[pillar] = value
    return _normalise(capped)


def _persist_weight_deltas(
    *,
    source: str,
    horizon: str,
    regime: str | None,
    n: int,
    prior: dict[str, float],
    empirical: dict[str, float],
    posterior: dict[str, float],
    ic_by_pillar: dict[str, float],
) -> None:
    try:
        from datetime import datetime
        from utils.state_manager import load_state, save_state

        state = load_state()
        state["bayesian_weight_deltas"] = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "horizon": horizon,
            "regime": regime,
            "sample_size": n,
            "prior": {p: round(float(prior.get(p, 0.0)), 4) for p in _PILLARS},
            "empirical": {p: round(float(empirical.get(p, 0.0)), 4) for p in _PILLARS},
            "posterior": {p: round(float(posterior.get(p, 0.0)), 4) for p in _PILLARS},
            "delta": {
                p: round(float(posterior.get(p, 0.0) - prior.get(p, 0.0)), 4)
                for p in _PILLARS
            },
            "ic": {p: round(float(ic_by_pillar.get(p, 0.0)), 4) for p in _PILLARS},
        }
        save_state(state)
    except Exception as exc:
        logger.debug("Bayesian learning: could not persist weight deltas (%s)", exc)


def _weights_from_pillar_effectiveness(
    base_weights: dict[str, float],
    *,
    source: str,
    regime: str | None = None,
) -> dict[str, float] | None:
    min_samples = int(getattr(config, "BAYESIAN_MIN_SAMPLES", 30))
    table_min = int(getattr(config, "BAYESIAN_EFFECTIVENESS_MIN_SAMPLES", 200))
    horizon_chain = list(getattr(config, "BAYESIAN_HORIZON_CHAIN", ["30d", "10d", "5d"]))
    min_by_horizon = getattr(config, "BAYESIAN_MIN_SAMPLES_BY_HORIZON", {})
    source_chain = [source] if source == "all" else [source, "all"]
    regime_norm = str(regime or "").strip().upper() or None
    regime_chain: list[str | None] = (
        [regime_norm, None] if regime_norm and getattr(config, "BAYESIAN_REGIME_CONDITIONAL", True)
        else [None]
    )
    rows = _load_pillar_ic(min_n=min(table_min, min_samples))
    if not rows:
        return None

    chosen: dict[str, tuple[float, int]] = {}
    chosen_horizon = None
    chosen_source = None
    chosen_regime = None
    for horizon in horizon_chain:
        try:
            horizon_min_samples = max(table_min, int(min_by_horizon.get(horizon, min_samples)))
        except AttributeError:
            horizon_min_samples = max(table_min, min_samples)
        for src in source_chain:
            for reg in regime_chain:
                per_pillar = {
                    p: rows[(src, p, horizon, reg)]
                    for p in _PILLARS
                    if (src, p, horizon, reg) in rows and rows[(src, p, horizon, reg)][1] >= horizon_min_samples
                }
                if len(per_pillar) >= 2:
                    chosen = per_pillar
                    chosen_horizon = horizon
                    chosen_source = src
                    chosen_regime = reg
                    break
            if chosen:
                break
        if chosen:
            break
    if not chosen or chosen_horizon is None or chosen_source is None:
        return None

    gate_ok, gate_reasons, _gate_summary = pillar_parity_gate()
    if not gate_ok:
        logger.info("Bayesian discovery weights blocked by parity gate: %s", "; ".join(gate_reasons))
        return None

    prior = _normalise({p: float(base_weights.get(p, 0.0)) for p in _PILLARS})
    ic_by_pillar = {p: chosen.get(p, (0.0, 0))[0] for p in _PILLARS}
    n_eff = int(median([v[1] for v in chosen.values()]))

    min_positive = int(getattr(config, "ADAPTIVE_WEIGHTS_GATE_MIN_POSITIVE_PILLARS", 1))
    empirical = positive_ic_allocation(ic_by_pillar, min_positive_pillars=min_positive)
    if empirical is None:
        logger.info(
            "Bayesian discovery weights blocked (%s/%s): no positive signed IC evidence (%s)",
            chosen_source,
            chosen_horizon,
            {k: round(float(v), 4) for k, v in ic_by_pillar.items()},
        )
        return dict(prior)

    k0 = float(getattr(config, "BAYESIAN_EFFECTIVENESS_PRIOR_STRENGTH", 200))
    max_blend = float(getattr(config, "BAYESIAN_MAX_LIVE_BLEND", 0.25))
    horizon_blend_mult = getattr(config, "BAYESIAN_HORIZON_BLEND_MULT", {"30d": 1.0, "10d": 0.60, "5d": 0.35})
    try:
        horizon_mult = float(horizon_blend_mult.get(chosen_horizon, 1.0))
    except AttributeError:
        horizon_mult = 1.0
    blend = min(max_blend * float(np.clip(horizon_mult, 0.0, 1.0)), n_eff / max(n_eff + k0, 1.0))
    posterior = {
        p: (1.0 - blend) * prior.get(p, 0.0) + blend * empirical.get(p, 0.0)
        for p in _PILLARS
    }
    max_rel = float(getattr(config, "BAYESIAN_DAILY_MAX_REL_DELTA", 0.20))
    posterior = _cap_relative_delta(posterior, prior, max_relative=max_rel)
    posterior = blend_with_default_prior(posterior)
    posterior = apply_weight_guardrails(posterior, horizon=chosen_horizon, ic_by_pillar=ic_by_pillar)
    _persist_weight_deltas(
        source=chosen_source,
        horizon=chosen_horizon,
        regime=chosen_regime,
        n=n_eff,
        prior=prior,
        empirical=empirical,
        posterior=posterior,
        ic_by_pillar=ic_by_pillar,
    )
    logger.info(
        "Bayesian discovery weights from pillar_effectiveness: source=%s horizon=%s regime=%s n=%d blend=%.2f weights=%s ic=%s",
        chosen_source,
        chosen_horizon,
        chosen_regime or "pooled",
        n_eff,
        blend,
        {k: round(v, 4) for k, v in posterior.items()},
        {k: round(v, 4) for k, v in ic_by_pillar.items()},
    )
    return {k: round(v, 4) for k, v in posterior.items()}


def get_bayesian_pillar_weights(
    base_weights: dict[str, float],
    *,
    source: str = "discovery",
    regime: str | None = None,
) -> dict[str, float]:
    """Return a shrinkage-weighted blend of configured priors and local IC."""
    if not getattr(config, "BAYESIAN_SELF_LEARNING_ENABLED", True):
        return dict(base_weights)

    if regime is None and getattr(config, "BAYESIAN_REGIME_CONDITIONAL", True):
        try:
            from engine.regime import get_vix_regime
            regime = (get_vix_regime() or {}).get("regime_label")
        except Exception:
            regime = None

    table_weights = _weights_from_pillar_effectiveness(base_weights, source=source, regime=regime)
    if table_weights:
        return table_weights

    min_samples = int(getattr(config, "BAYESIAN_MIN_SAMPLES", 30))
    horizon_chain = list(getattr(config, "BAYESIAN_HORIZON_CHAIN", ["30d", "10d", "5d"]))
    min_by_horizon = getattr(config, "BAYESIAN_MIN_SAMPLES_BY_HORIZON", {})
    source_chain = [source] if source == "all" else [source, "all"]

    chosen_rows = []
    chosen_horizon = None
    chosen_source = None
    for horizon in horizon_chain:
        try:
            horizon_min_samples = int(min_by_horizon.get(horizon, min_samples))
        except AttributeError:
            horizon_min_samples = min_samples
        for src in source_chain:
            rows = _load_horizon_rows(src, horizon)
            if len(rows) >= horizon_min_samples:
                chosen_rows = rows
                chosen_horizon = horizon
                chosen_source = src
                break
        if chosen_rows:
            break

    if not chosen_rows:
        return dict(base_weights)

    gate_ok, gate_reasons, _gate_summary = pillar_parity_gate()
    if not gate_ok:
        logger.info("Bayesian discovery weights blocked by parity gate: %s", "; ".join(gate_reasons))
        return dict(base_weights)

    returns = [float(r[f"return_{chosen_horizon}"] or 0.0) for r in chosen_rows]
    ic_by_pillar: dict[str, float] = {}
    for pillar in _PILLARS:
        scores = [float(r[f"{pillar}_score"] or 0.0) for r in chosen_rows]
        ic = _rank_ic(scores, returns)
        ic_by_pillar[pillar] = float(ic or 0.0)

    min_positive = int(getattr(config, "ADAPTIVE_WEIGHTS_GATE_MIN_POSITIVE_PILLARS", 1))
    empirical = positive_ic_allocation(ic_by_pillar, min_positive_pillars=min_positive)
    if empirical is None:
        logger.info(
            "Bayesian discovery weights blocked (%s/%s): no positive signed IC evidence (%s)",
            chosen_source,
            chosen_horizon,
            {k: round(float(v), 4) for k, v in ic_by_pillar.items()},
        )
        return dict(base_weights)
    prior = _normalise({p: float(base_weights.get(p, 0.0)) for p in _PILLARS})

    prior_strength = float(getattr(config, "BAYESIAN_PRIOR_STRENGTH", 125))
    max_blend = float(getattr(config, "BAYESIAN_MAX_LIVE_BLEND", 0.25))
    horizon_blend_mult = getattr(config, "BAYESIAN_HORIZON_BLEND_MULT", {"30d": 1.0, "10d": 0.60, "5d": 0.35})
    try:
        horizon_mult = float(horizon_blend_mult.get(chosen_horizon, 1.0))
    except AttributeError:
        horizon_mult = 1.0
    horizon_mult = float(np.clip(horizon_mult, 0.0, 1.0))
    n = len(chosen_rows)
    blend = min(max_blend * horizon_mult, n / max(n + prior_strength, 1.0))
    posterior = {
        p: (1.0 - blend) * prior.get(p, 0.0) + blend * empirical.get(p, 0.0)
        for p in _PILLARS
    }
    posterior = blend_with_default_prior(_normalise(posterior))
    posterior = apply_weight_guardrails(posterior, horizon=chosen_horizon, ic_by_pillar=ic_by_pillar)
    logger.info(
        "Bayesian discovery weights: source=%s horizon=%s n=%d blend=%.2f weights=%s",
        chosen_source,
        chosen_horizon,
        n,
        blend,
        {k: round(v, 4) for k, v in posterior.items()},
    )
    return {k: round(v, 4) for k, v in posterior.items()}
