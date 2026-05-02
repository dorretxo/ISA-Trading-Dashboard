"""Tier 4 — momentum-sanity gates and limit-price advisor.

References:

- Wilder (1978) — Relative Strength Index; classical overbought thresholds.
- Vanstone & Hahn (2017, *J. Empirical Finance*) — RSI overbought reversal.
- Faber (2007, *J. Wealth Mgmt*) — 200-DMA-relative trend filters.
- George & Hwang (2004, *JFE*) — 52-week-high momentum subsumes 12-month.
- Daniel & Moskowitz (2016, *JFE*) — momentum crashes; volatility scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MomentumGateResult:
    action_ceiling: str = "STRONG BUY"
    reasons: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    needs_pullback: bool = False    # When True, UI should render a limit-price


_RANK = {"STRONG BUY": 4, "BUY": 3, "NEUTRAL": 2, "AVOID": 1, "MANUAL REVIEW": 0}


def _cap(current: str, candidate: str) -> str:
    return candidate if _RANK.get(candidate, 4) < _RANK.get(current, 4) else current


def _f(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def evaluate_momentum_gates(
    *,
    rsi: float | None,
    stretch_200dma: float | None,    # E.g. +0.35 for "35% above 200-DMA"
    entry_stance: str | None = None,
    realized_vol_pctile: float | None = None,
    config_module=None,
) -> MomentumGateResult:
    """Apply RSI / stretch / entry-stance / vol gates.

    ``entry_stance == "Pullback Preferred"`` is treated as a hard block on
    STRONG BUY (downgrades to BUY-with-limit) when
    :data:`config.ENTRY_STANCE_BLOCKS_STRONG_BUY` is True.  This is the
    single change that would have caught Trustpilot at 251p.
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    result = MomentumGateResult()
    if not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return result

    # RSI cap
    rsi_v = _f(rsi)
    if rsi_v is not None:
        rsi_strong = float(getattr(cfg, "RSI_STRONG_BUY_MAX", 75))
        rsi_neutral = float(getattr(cfg, "RSI_NEUTRAL_CAP", 80))
        if rsi_v > rsi_neutral:
            result.action_ceiling = _cap(result.action_ceiling, "NEUTRAL")
            result.reasons.append(f"RSI {rsi_v:.0f} above NEUTRAL cap ({rsi_neutral:.0f})")
            result.flags["rsi"] = "fail"
            result.needs_pullback = True
        elif rsi_v > rsi_strong:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"RSI {rsi_v:.0f} above STRONG-BUY cap ({rsi_strong:.0f})")
            result.flags["rsi"] = "borderline"
            result.needs_pullback = True
        else:
            result.flags["rsi"] = "pass"
    else:
        result.flags["rsi"] = "skip"

    # Stretch above 200-DMA
    s = _f(stretch_200dma)
    if s is not None:
        cap = float(getattr(cfg, "STRETCH_200DMA_STRONG_BUY_MAX", 0.35))
        if s > cap:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Price {s:+.0%} above 200-DMA exceeds cap ({cap:+.0%})")
            result.flags["stretch_200dma"] = "fail"
            result.needs_pullback = True
        else:
            result.flags["stretch_200dma"] = "pass"
    else:
        result.flags["stretch_200dma"] = "skip"

    # Entry stance — promoted from UI hint to gate
    if getattr(cfg, "ENTRY_STANCE_BLOCKS_STRONG_BUY", True):
        stance = (entry_stance or "").strip().lower()
        if stance and stance != "ready":
            # "pullback preferred", "wait", etc. all block
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Entry stance '{entry_stance}' blocks STRONG BUY")
            result.flags["entry_stance"] = "fail"
            result.needs_pullback = True
        else:
            result.flags["entry_stance"] = "pass"

    # Volatility-scaled momentum (Daniel-Moskowitz)
    if getattr(cfg, "MOMENTUM_VOL_SCALING_ENABLED", True):
        vp = _f(realized_vol_pctile)
        if vp is not None and vp >= 0.90:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Realised vol pctile {vp:.0%} — momentum-crash risk")
            result.flags["vol_pctile"] = "fail"
        else:
            result.flags["vol_pctile"] = "pass" if vp is not None else "skip"

    return result


# ---------------------------------------------------------------------------
# Limit-price advisor — what price to wait for
# ---------------------------------------------------------------------------

@dataclass
class LimitPriceSuggestion:
    limit_price: float | None
    method: str
    rationale: str


def suggest_limit_price(
    *,
    current_price: float | None,
    sma_200: float | None,
    support_level: float | None = None,
    atr: float | None = None,
    placement_price: float | None = None,
    config_module=None,
) -> LimitPriceSuggestion | None:
    """Suggest a pullback-wait price.

    Strategy: use the *highest* of available reference points, so we never
    propose a limit higher than today's price (which would be a market
    order, not a wait):

      - 200-DMA × (1 + buffer)
      - support_level × 1.02
      - placement_price × 1.0
      - current_price - 1.5 × ATR

    The *highest* of these that is also strictly below ``current_price`` is
    returned.  If nothing is below current price, returns None (no wait
    possible — the candidate must be capped at NEUTRAL).
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    cp = _f(current_price)
    if cp is None or cp <= 0:
        return None

    buffer = float(getattr(cfg, "DISCOVERY_LIMIT_PRICE_BUFFER", 0.05))

    candidates: list[tuple[float, str, str]] = []
    sma = _f(sma_200)
    if sma is not None and sma > 0:
        candidates.append((sma * (1 + buffer), "200-DMA + buffer", f"200-DMA {sma:.2f} × (1+{buffer:.0%})"))
    sup = _f(support_level)
    if sup is not None and sup > 0:
        candidates.append((sup * 1.02, "support + 2%", f"Support {sup:.2f} × 1.02"))
    plc = _f(placement_price)
    if plc is not None and plc > 0:
        candidates.append((plc, "recent placement", f"Recent placement {plc:.2f}"))
    a = _f(atr)
    if a is not None and a > 0:
        candidates.append((cp - 1.5 * a, "current - 1.5×ATR", f"Current {cp:.2f} − 1.5 × ATR {a:.2f}"))

    # Keep only candidates strictly below current price
    below = [c for c in candidates if c[0] < cp]
    if not below:
        return None
    # Take the highest one — the *closest* pullback target
    below.sort(key=lambda c: c[0], reverse=True)
    price, method, rationale = below[0]
    return LimitPriceSuggestion(limit_price=round(price, 4), method=method, rationale=rationale)
