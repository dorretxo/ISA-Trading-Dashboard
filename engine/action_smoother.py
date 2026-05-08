"""Exit-action smoother — hysteresis bands + persistence gate (EXIT side only).

Pure logic.  No I/O.  The caller assembles inputs (from
``engine.score_history`` + ``engine.stops._get_vix_percentile``) and passes them
to :func:`smooth_exit_action`.

Replaces the scalar threshold cross at ``engine/scoring.py`` with a
two-threshold no-trade region (Constantinides 1986; Davis-Norman 1990) gated by
a fixed-window persistence rule (Wald 1947 SPRT, simplified count form).  Band
width scales with realised σ_score and VIX percentile (Kaminski-Lo 2014).
CUSUM-confirmed urgent signals from ``exit_engine`` override the band (Page
1954, Lorden 1971-optimal).

The smoother only adjusts ``KEEP / SELL / STRONG SELL`` — BUY / STRONG BUY pass
through untouched per design (the user's complaint is exit churn).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

EXIT_ACTIONS: frozenset[str] = frozenset({"KEEP", "SELL", "STRONG SELL"})
DOWNSIDE_ACTIONS: frozenset[str] = frozenset({"SELL", "STRONG SELL"})


# Reason codes — also exposed in the UI chip.
REASON_DISABLED = "smoother_disabled"
REASON_PASSTHROUGH_BUY = "passthrough_buy_side"
REASON_COLD_START = "cold_start_legacy"
REASON_BAND_HELD_KEEP = "band_held_keep"
REASON_BAND_HELD_SELL = "band_held_sell"
REASON_PERSISTENCE_SHORT = "persistence_short"
REASON_PERSISTENCE_SATISFIED = "persistence_satisfied"
REASON_RECOVERED = "recovered_to_keep"
REASON_CUSUM_OVERRIDE = "cusum_override"


@dataclass
class SmoothedAction:
    """Output of the smoother."""
    action: str
    reason: str
    since_date: str | None
    band_low: float | None
    band_high: float | None
    persistence_m: int
    persistence_n: int


def _band_width(score_vol: float | None, vix_pct: float | None) -> float:
    """Half-width of the dead-zone in score units.

    band = max(BAND_MIN, K_VOL · σ_score) · (1 + VIX_SCALE · vix_pct/100)
    """
    sigma = (
        score_vol
        if score_vol is not None and score_vol >= 0
        else config.EXIT_SMOOTHER_VOL_FALLBACK
    )
    base = max(config.EXIT_SMOOTHER_BAND_MIN, config.EXIT_SMOOTHER_BAND_K_VOL * sigma)
    vp = vix_pct if vix_pct is not None else 50.0
    vp = max(0.0, min(100.0, float(vp)))
    return float(base * (1.0 + config.EXIT_SMOOTHER_VIX_SCALE * vp / 100.0))


def _legacy_action_from_score(raw_score: float) -> str:
    """Mirror of the legacy threshold map at engine/scoring.py:340-349.

    Used for cold-start (no history) and as a sanity check against base_action.
    """
    if raw_score >= config.SCORE_STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    if raw_score >= config.SCORE_BUY_THRESHOLD:
        return "BUY"
    if raw_score >= config.SCORE_KEEP_THRESHOLD:
        return "KEEP"
    if raw_score >= config.SCORE_SELL_THRESHOLD:
        return "SELL"
    return "STRONG SELL"


def smooth_exit_action(
    *,
    raw_score: float,
    base_action: str,
    prev_smoothed_action: str | None,
    score_vol: float | None,
    vix_pct: float | None,
    persistence_m_sell: int = 0,
    persistence_m_strong: int = 0,
    persistence_n: int = 0,
    persistence_since: str | None = None,
    cusum_urgent: bool = False,
    cusum_score: float = 0.0,
) -> SmoothedAction:
    """Apply hysteresis + persistence to a raw exit action.

    Parameters
    ----------
    raw_score:
        Today's ``aggregate_score`` (post risk-overlay, pre-action mapping).
    base_action:
        The legacy action label that ``analyse_holding`` would emit today.
    prev_smoothed_action:
        Yesterday's smoothed action for this ticker (or fallback to yesterday's
        base action if cold-start during deployment).  ``None`` = true
        cold-start: no history at all.
    score_vol:
        Stdev of ``aggregate_score`` over the lookback window.  ``None`` →
        fallback constant from config.
    vix_pct:
        VIX percentile 0..100.  ``None`` → assume neutral 50.
    persistence_m_sell, persistence_m_strong:
        How many of the last N days the base action was SELL/STRONG SELL or
        STRONG SELL respectively (computed by caller via
        :func:`engine.score_history.count_recent_action`).
    persistence_n:
        Window size used for the M-of-N counts (informational only).
    persistence_since:
        ``run_date`` of the first qualifying row in the persistence window —
        powers the UI "SELL since YYYY-MM-DD" chip.
    cusum_urgent:
        ``True`` when ``exit_engine`` emitted an *urgent* exit signal whose
        CUSUM detector alarmed in the negative direction.
    cusum_score:
        Composite ``exit_score`` from that signal (0..1).
    """
    # ── Master toggle ────────────────────────────────────────────────────
    if not getattr(config, "EXIT_SMOOTHER_ENABLED", True):
        return SmoothedAction(
            action=base_action,
            reason=REASON_DISABLED,
            since_date=None,
            band_low=None,
            band_high=None,
            persistence_m=0,
            persistence_n=0,
        )

    # ── EXIT-side gate: BUY / STRONG BUY pass straight through ──────────
    if base_action not in EXIT_ACTIONS:
        return SmoothedAction(
            action=base_action,
            reason=REASON_PASSTHROUGH_BUY,
            since_date=None,
            band_low=None,
            band_high=None,
            persistence_m=0,
            persistence_n=persistence_n,
        )

    # ── CUSUM structural-break override (beats band + persistence) ──────
    if (
        cusum_urgent
        and cusum_score >= getattr(config, "EXIT_SMOOTHER_CUSUM_OVERRIDE_MIN", 0.6)
    ):
        return SmoothedAction(
            action="STRONG SELL",
            reason=REASON_CUSUM_OVERRIDE,
            since_date=persistence_since,
            band_low=None,
            band_high=None,
            persistence_m=persistence_m_strong,
            persistence_n=persistence_n,
        )

    # ── True cold-start: no prior smoothed action — use legacy mapping ──
    if not prev_smoothed_action or prev_smoothed_action not in EXIT_ACTIONS:
        legacy = base_action  # caller already applied legacy thresholds
        return SmoothedAction(
            action=legacy,
            reason=REASON_COLD_START,
            since_date=None,
            band_low=None,
            band_high=None,
            persistence_m=0,
            persistence_n=persistence_n,
        )

    # ── Compute the regime-scaled band ───────────────────────────────────
    band = _band_width(score_vol, vix_pct)
    delta_down = band
    delta_up = band * float(getattr(config, "EXIT_SMOOTHER_RECOVERY_RATIO", 0.5))

    # Effective thresholds — Davis-Norman asymmetry: harder to enter SELL,
    # tighter to recover (the "no-trade region" inside which prior holds).
    sell_entry = config.SCORE_KEEP_THRESHOLD - delta_down            # cross to enter SELL
    strong_entry = config.SCORE_SELL_THRESHOLD - delta_down          # cross to enter STRONG SELL
    keep_recovery = config.SCORE_KEEP_THRESHOLD + delta_up           # cross up to recover to KEEP
    sell_recovery = config.SCORE_SELL_THRESHOLD + delta_up           # cross up to recover to SELL from STRONG SELL

    band_low = round(sell_entry, 4)
    band_high = round(keep_recovery, 4)

    persistence_m_required_sell = int(
        getattr(config, "EXIT_SMOOTHER_PERSISTENCE_M", 3)
    )
    persistence_m_required_strong = int(
        getattr(config, "EXIT_SMOOTHER_STRONG_SELL_M", 4)
    )

    # ── Apply hysteresis based on prior smoothed action ──────────────────
    if prev_smoothed_action == "KEEP":
        # Need a *clear* downward break + persistence to flip to SELL.
        if raw_score < strong_entry:
            # Even STRONG SELL territory still requires persistence at the
            # stronger threshold to escalate from KEEP directly.
            if persistence_m_strong >= persistence_m_required_strong:
                return SmoothedAction(
                    action="STRONG SELL",
                    reason=REASON_PERSISTENCE_SATISFIED,
                    since_date=persistence_since,
                    band_low=band_low,
                    band_high=band_high,
                    persistence_m=persistence_m_strong,
                    persistence_n=persistence_n,
                )
            if persistence_m_sell >= persistence_m_required_sell:
                return SmoothedAction(
                    action="SELL",
                    reason=REASON_PERSISTENCE_SATISFIED,
                    since_date=persistence_since,
                    band_low=band_low,
                    band_high=band_high,
                    persistence_m=persistence_m_sell,
                    persistence_n=persistence_n,
                )
            return SmoothedAction(
                action="KEEP",
                reason=REASON_PERSISTENCE_SHORT,
                since_date=None,
                band_low=band_low,
                band_high=band_high,
                persistence_m=persistence_m_sell,
                persistence_n=persistence_n,
            )
        if raw_score < sell_entry:
            # Inside SELL band — needs persistence; otherwise hold KEEP.
            if persistence_m_sell >= persistence_m_required_sell:
                return SmoothedAction(
                    action="SELL",
                    reason=REASON_PERSISTENCE_SATISFIED,
                    since_date=persistence_since,
                    band_low=band_low,
                    band_high=band_high,
                    persistence_m=persistence_m_sell,
                    persistence_n=persistence_n,
                )
            return SmoothedAction(
                action="KEEP",
                reason=REASON_PERSISTENCE_SHORT,
                since_date=None,
                band_low=band_low,
                band_high=band_high,
                persistence_m=persistence_m_sell,
                persistence_n=persistence_n,
            )
        # Score still in KEEP territory after band — stay put.
        return SmoothedAction(
            action="KEEP",
            reason=REASON_BAND_HELD_KEEP,
            since_date=None,
            band_low=band_low,
            band_high=band_high,
            persistence_m=persistence_m_sell,
            persistence_n=persistence_n,
        )

    if prev_smoothed_action == "SELL":
        # Recovery up to KEEP — easier (tighter band, no persistence).
        if raw_score >= keep_recovery:
            return SmoothedAction(
                action="KEEP",
                reason=REASON_RECOVERED,
                since_date=None,
                band_low=band_low,
                band_high=band_high,
                persistence_m=persistence_m_sell,
                persistence_n=persistence_n,
            )
        # Escalation to STRONG SELL — needs the stronger persistence count.
        if raw_score < strong_entry:
            if persistence_m_strong >= persistence_m_required_strong:
                return SmoothedAction(
                    action="STRONG SELL",
                    reason=REASON_PERSISTENCE_SATISFIED,
                    since_date=persistence_since,
                    band_low=band_low,
                    band_high=band_high,
                    persistence_m=persistence_m_strong,
                    persistence_n=persistence_n,
                )
        # Anything else — hold SELL.
        return SmoothedAction(
            action="SELL",
            reason=REASON_BAND_HELD_SELL,
            since_date=None,
            band_low=band_low,
            band_high=band_high,
            persistence_m=persistence_m_sell,
            persistence_n=persistence_n,
        )

    # prev_smoothed_action == "STRONG SELL"
    # Recovery to SELL — needs to clear sell_recovery band.
    if raw_score >= sell_recovery:
        # Further recovery to KEEP — only when comfortably above keep_recovery.
        if raw_score >= keep_recovery:
            return SmoothedAction(
                action="KEEP",
                reason=REASON_RECOVERED,
                since_date=None,
                band_low=band_low,
                band_high=band_high,
                persistence_m=persistence_m_sell,
                persistence_n=persistence_n,
            )
        return SmoothedAction(
            action="SELL",
            reason=REASON_RECOVERED,
            since_date=None,
            band_low=band_low,
            band_high=band_high,
            persistence_m=persistence_m_sell,
            persistence_n=persistence_n,
        )
    # Score still below SELL recovery — hold STRONG SELL.
    return SmoothedAction(
        action="STRONG SELL",
        reason=REASON_BAND_HELD_SELL,
        since_date=None,
        band_low=band_low,
        band_high=band_high,
        persistence_m=persistence_m_strong,
        persistence_n=persistence_n,
    )


__all__ = [
    "SmoothedAction",
    "smooth_exit_action",
    "EXIT_ACTIONS",
    "DOWNSIDE_ACTIONS",
    "REASON_DISABLED",
    "REASON_PASSTHROUGH_BUY",
    "REASON_COLD_START",
    "REASON_BAND_HELD_KEEP",
    "REASON_BAND_HELD_SELL",
    "REASON_PERSISTENCE_SHORT",
    "REASON_PERSISTENCE_SATISFIED",
    "REASON_RECOVERED",
    "REASON_CUSUM_OVERRIDE",
]
