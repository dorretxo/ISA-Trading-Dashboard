"""Shared discovery decision-packet helpers for UI and email surfaces."""

from __future__ import annotations

from utils.safe_numeric import safe_float


_VALID_ENTRY_STANCES = {"Ready", "Pullback Preferred", "Watch Only"}


def candidate_value(candidate, key: str, default=None):
    """Read a field from either a dict payload or a scored-candidate object."""
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def discovery_confidence(candidate) -> tuple[str, str, float]:
    """Return (label, tone, score) for discovery confidence."""
    sent_conf = safe_float(candidate_value(candidate, "sentiment_score", 0))
    data_discount = safe_float(candidate_value(candidate, "confidence_discount", 1.0), default=1.0)
    has_data = 0.0 if candidate_value(candidate, "action", "") == "INSUFFICIENT DATA" else 1.0
    score = max(
        0.0,
        min(
            1.0,
            0.35 * has_data
            + 0.30 * data_discount
            + 0.35 * (0.5 + 0.5 * max(min(sent_conf, 1.0), -1.0)),
        ),
    )

    if candidate_value(candidate, "action", "") == "INSUFFICIENT DATA":
        return "Data Gap", "data", score
    if score >= 0.75:
        return "High Confidence", "high", score
    if score >= 0.55:
        return "Medium Confidence", "medium", score
    return "Watch Carefully", "low", score


def candidate_entry_stance(candidate) -> str:
    """Infer the preferred entry stance for a discovery candidate."""
    stance = str(candidate_value(candidate, "entry_stance", "") or "").strip()
    if stance in _VALID_ENTRY_STANCES:
        return stance

    analyst_upside_raw = candidate_value(candidate, "analyst_upside", None)
    analyst_upside = (
        safe_float(analyst_upside_raw)
        if analyst_upside_raw is not None
        else None
    )
    insider_buys = int(safe_float(candidate_value(candidate, "insider_buys", 0), default=0))
    insider_sells = int(safe_float(candidate_value(candidate, "insider_sells", 0), default=0))
    return_30d = safe_float(candidate_value(candidate, "return_30d", 0))

    if (
        candidate_value(candidate, "governance_flag", False)
        or candidate_value(candidate, "asymmetric_risk_flag", False)
        or candidate_value(candidate, "earnings_imminent", False)
        or (
            candidate_value(candidate, "is_parabolic", False)
            and analyst_upside is not None
            and analyst_upside < 0
        )
        or (
            candidate_value(candidate, "near_52w_high", False)
            and return_30d >= 0.25
        )
        or (
            insider_sells > insider_buys
            and analyst_upside is not None
            and analyst_upside < 0
        )
    ):
        return "Watch Only"

    if (
        candidate_value(candidate, "is_parabolic", False)
        or candidate_value(candidate, "near_52w_high", False)
        or (analyst_upside is not None and analyst_upside < 5)
        or insider_sells > insider_buys
        or candidate_value(candidate, "earnings_near", False)
    ):
        return "Pullback Preferred"

    return "Ready"


def candidate_key_risk(candidate) -> str:
    """Return the highest-priority gating risk in one short phrase."""
    if candidate_value(candidate, "ticker_identity_warning", None):
        return "Ticker identity"
    ready_status = str(candidate_value(candidate, "ready_contract_status", "") or "")
    ready_reasons = candidate_value(candidate, "ready_contract_reasons", None)
    if ready_status == "FAIL" and ready_reasons:
        if isinstance(ready_reasons, (list, tuple)) and ready_reasons:
            return str(ready_reasons[0])[:48]
        return str(ready_reasons)[:48]
    if candidate_value(candidate, "gate_v2_status", "") == "REJECT":
        return "Discovery gate rejected"
    if candidate_value(candidate, "trap_safeguard_triggered", False):
        return "Commodity cycle review"
    if candidate_value(candidate, "governance_flag", False) and candidate_value(candidate, "asymmetric_risk_flag", False):
        return "Governance / asymmetric risk"
    if candidate_value(candidate, "governance_flag", False):
        return "Governance risk"
    if candidate_value(candidate, "asymmetric_risk_flag", False):
        return "Asymmetric risk"
    if candidate_value(candidate, "earnings_imminent", False):
        return f"Earnings in {candidate_value(candidate, 'earnings_days', '?')}d"
    if candidate_value(candidate, "earnings_near", False):
        return f"Earnings soon ({candidate_value(candidate, 'earnings_days', '?')}d)"
    if candidate_value(candidate, "earnings_miss", False):
        return "Recent earnings miss"
    if candidate_value(candidate, "is_parabolic", False):
        return "Parabolic move"
    if candidate_value(candidate, "near_52w_high", False):
        return "Near 52-week high"
    if safe_float(candidate_value(candidate, "max_correlation", 0)) >= 0.70:
        return f"High corr {safe_float(candidate_value(candidate, 'max_correlation', 0)):.2f}"
    if candidate_value(candidate, "action", "") == "INSUFFICIENT DATA":
        return "Incomplete signal"
    return ""


def candidate_ready_reasons(candidate, limit: int = 3) -> list[str]:
    """Return short, plain-English reasons a candidate is not ready now."""
    raw = candidate_value(candidate, "ready_contract_reasons", None)
    if not raw:
        return []
    if isinstance(raw, str):
        reasons = [raw]
    else:
        try:
            reasons = [str(r) for r in raw if r]
        except TypeError:
            reasons = [str(raw)]
    cleaned = []
    seen = set()
    for reason in reasons:
        text = reason.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def candidate_entry_ready(candidate) -> bool:
    """True when a discovery BUY/STRONG BUY is executable now.

    This is deliberately separate from alpha direction.  A candidate may be a
    good watchlist BUY while still failing the entry contract because the
    pullback, reward/risk, confidence, or gate checks have not cleared.
    """
    action = str(candidate_value(candidate, "action", "") or "").upper()
    if action not in {"BUY", "STRONG BUY"}:
        return False
    if candidate_value(candidate, "ticker_identity_warning", None):
        return False
    if candidate_value(candidate, "trap_safeguard_triggered", False):
        return False
    if str(candidate_value(candidate, "gate_v2_status", "") or "").upper() == "REJECT":
        return False
    if str(candidate_value(candidate, "ready_contract_status", "") or "").upper() == "FAIL":
        return False
    if candidate_entry_stance(candidate) != "Ready":
        return False

    entry_price = safe_float(candidate_value(candidate, "entry_price", None), default=0.0)
    stop_loss = safe_float(candidate_value(candidate, "stop_loss", None), default=0.0)
    take_profit = safe_float(candidate_value(candidate, "take_profit", None), default=0.0)
    rr_ratio = safe_float(candidate_value(candidate, "r_r_ratio", None), default=0.0)
    position_weight = safe_float(candidate_value(candidate, "position_weight", None), default=0.0)
    if not (entry_price > 0 and stop_loss > 0 and take_profit > 0):
        return False
    if rr_ratio and rr_ratio < 1.5:
        return False
    if position_weight <= 0:
        return False
    return True


def candidate_entry_trigger(candidate) -> str:
    """Explain the practical trigger that would make a buy candidate actionable."""
    stance = candidate_entry_stance(candidate)
    currency = str(candidate_value(candidate, "currency", "USD") or "USD")
    entry = safe_float(candidate_value(candidate, "entry_price", None), default=0.0)
    rr = safe_float(candidate_value(candidate, "r_r_ratio", None), default=0.0)
    size = safe_float(candidate_value(candidate, "position_weight", None), default=0.0)
    entry_text = f"{currency} {entry:,.2f}" if entry > 0 else "the planned entry zone"

    reasons = " | ".join(candidate_ready_reasons(candidate, limit=4)).lower()
    if "r/r below" in reasons or "reward/risk" in reasons:
        return "Wait until reward/risk is at least 1.5x."
    if "position size unavailable" in reasons or size <= 0:
        return "Wait for a valid stop and position size."
    if stance == "Pullback Preferred":
        return f"Wait for a pullback toward {entry_text}."
    if stance == "Watch Only":
        return f"Wait. Re-check only if price reaches {entry_text} and the watch-only warning clears."
    if rr > 0 and rr < 1.5:
        return "Wait until the target/stop plan improves to at least 1.5x reward/risk."
    if entry > 0:
        return f"Entry trigger is around {entry_text}, with the ready checks cleared."
    return "Wait for the ready checks to clear."


def candidate_readiness_summary(candidate) -> str:
    """One-sentence status for BUY names that are not ready to buy yet."""
    action = str(candidate_value(candidate, "action", "") or "").upper()
    ready_status = str(candidate_value(candidate, "ready_contract_status", "") or "")
    strong_buy_eligible = candidate_value(candidate, "strong_buy_eligible", None)
    stance = candidate_entry_stance(candidate)
    reasons = candidate_ready_reasons(candidate, limit=2)

    if candidate_entry_ready(candidate):
        return "Ready to consider now if the suggested size and risk limit fit your plan."
    if action != "BUY":
        if reasons:
            return "Not ready: " + "; ".join(reasons) + "."
        return "Not a fresh buy setup right now."

    if ready_status == "FAIL" or strong_buy_eligible is False or stance != "Ready":
        reason_text = "; ".join(reasons) if reasons else f"entry view is {stance.lower()}"
        return f"Buy candidate, not ready: {reason_text}. Trigger: {candidate_entry_trigger(candidate)}"

    return "Buy candidate. Check the entry price, stop, and position size before acting."


def is_top_pick(candidate) -> bool:
    """Check if a candidate satisfies the strict Top Pick definition."""
    if candidate_value(candidate, "ticker_identity_warning", None):
        return False
    if candidate_value(candidate, "action") != "STRONG BUY":
        return False
    strong_buy_eligible = candidate_value(candidate, "strong_buy_eligible", True)
    if strong_buy_eligible is not None and not bool(strong_buy_eligible):
        return False
    if candidate_value(candidate, "ready_contract_status", "PASS") == "FAIL":
        return False
    if candidate_entry_stance(candidate) != "Ready":
        return False
    _, _, score = discovery_confidence(candidate)
    if score < 0.75:
        return False
    return True


def build_trade_packet(candidate) -> dict:
    """Build a comparable decision packet used across UI and email."""
    confidence_label, confidence_tone, confidence_score = discovery_confidence(candidate)
    stance = candidate_entry_stance(candidate)
    action = str(candidate_value(candidate, "action", "NEUTRAL") or "NEUTRAL")
    identity_warning = candidate_value(candidate, "ticker_identity_warning", None)
    final_rank = safe_float(
        candidate_value(candidate, "final_rank", candidate_value(candidate, "aggregate_score", 0))
    )
    aggregate_score = safe_float(candidate_value(candidate, "aggregate_score", 0))
    entry_price = safe_float(candidate_value(candidate, "entry_price", None), default=0.0)
    stop_loss = safe_float(candidate_value(candidate, "stop_loss", None), default=0.0)
    take_profit = safe_float(candidate_value(candidate, "take_profit", None), default=0.0)
    rr_ratio = safe_float(candidate_value(candidate, "r_r_ratio", None), default=0.0)
    fill_probability = safe_float(candidate_value(candidate, "fill_probability", None), default=0.0)
    position_weight = safe_float(candidate_value(candidate, "position_weight", None), default=0.0)
    prior_pct = safe_float(candidate_value(candidate, "institutional_prior_percentile", None), default=0.0)
    prior_conf = safe_float(candidate_value(candidate, "institutional_prior_confidence", None), default=0.0)
    ready_status = str(candidate_value(candidate, "ready_contract_status", "") or "")
    plan_complete = entry_price > 0 and stop_loss > 0 and take_profit > 0
    buy_eligible = action in {"BUY", "STRONG BUY"}
    entry_ready = candidate_entry_ready(candidate)
    clean_entry = (
        buy_eligible
        and not identity_warning
        and stance in {"Ready", "Pullback Preferred"}
        and confidence_label != "Data Gap"
        and ready_status != "FAIL"
    )
    trade_ready = entry_ready
    top_pick = is_top_pick(candidate) and plan_complete

    status_rank = 0
    if clean_entry:
        status_rank = 2
    if trade_ready:
        status_rank = 3
    if top_pick:
        status_rank = 4

    return {
        "ticker": str(candidate_value(candidate, "ticker", "") or ""),
        "name": str(candidate_value(candidate, "name", "") or ""),
        "sector": str(candidate_value(candidate, "sector", "") or ""),
        "exchange": str(candidate_value(candidate, "exchange", "") or ""),
        "currency": str(candidate_value(candidate, "currency", "USD") or "USD"),
        "action": action,
        "why": str(candidate_value(candidate, "why", "") or ""),
        "entry_stance": stance,
        "confidence_label": confidence_label,
        "confidence_tone": confidence_tone,
        "confidence_score": confidence_score,
        "final_rank": final_rank,
        "aggregate_score": aggregate_score,
        "expected_return_90d": safe_float(candidate_value(candidate, "expected_return_90d", 0)),
        "entry_price": entry_price if entry_price > 0 else None,
        "stop_loss": stop_loss if stop_loss > 0 else None,
        "take_profit": take_profit if take_profit > 0 else None,
        "r_r_ratio": rr_ratio if rr_ratio > 0 else None,
        "fill_probability": fill_probability if fill_probability > 0 else None,
        "position_weight": position_weight if position_weight > 0 else None,
        "portfolio_fit_score": safe_float(candidate_value(candidate, "portfolio_fit_score", 0)),
        "max_correlation": safe_float(candidate_value(candidate, "max_correlation", 0)),
        "institutional_prior_percentile": prior_pct if prior_pct > 0 else None,
        "institutional_prior_confidence": prior_conf if prior_conf > 0 else None,
        "ready_contract_status": ready_status or None,
        "ready_contract_reasons": candidate_ready_reasons(candidate, limit=4),
        "entry_trigger": candidate_entry_trigger(candidate),
        "readiness_summary": candidate_readiness_summary(candidate),
        "identity_warning": identity_warning,
        "top_pick": top_pick,
        "plan_complete": plan_complete,
        "buy_eligible": buy_eligible,
        "entry_ready": entry_ready,
        "clean_entry": clean_entry,
        "trade_ready": trade_ready,
        "status_rank": status_rank,
        "key_risk": candidate_key_risk(candidate),
        # Action-gate telemetry — surfaced in UI and email so the user can
        # see *why* a candidate isn't STRONG BUY at a glance.
        "action_gate_ceiling": str(candidate_value(candidate, "action_gate_ceiling", "STRONG BUY")),
        "action_gate_reasons": list(candidate_value(candidate, "action_gate_reasons", []) or [])[:4],
        "action_gate_flags": dict(candidate_value(candidate, "action_gate_flags", {}) or {}),
        "altman_z": safe_float(candidate_value(candidate, "altman_z", None), default=None),
        "altman_zone": str(candidate_value(candidate, "altman_zone", "unknown") or "unknown"),
        "beneish_m": safe_float(candidate_value(candidate, "beneish_m", None), default=None),
        "limit_price": safe_float(candidate_value(candidate, "limit_price", None), default=None) or None,
        "limit_price_method": candidate_value(candidate, "limit_price_method", None),
        "limit_price_rationale": candidate_value(candidate, "limit_price_rationale", None),
        "threshold_profile": str(candidate_value(candidate, "threshold_profile", "moderate") or "moderate"),
    }


def trade_packet_sort_key(packet: dict) -> tuple:
    """Return a stable sort key for actionable discovery packets."""
    return (
        safe_float(packet.get("status_rank", 0)),
        1 if packet.get("top_pick") else 0,
        safe_float(packet.get("confidence_score", 0)),
        safe_float(packet.get("final_rank", 0)),
        safe_float(packet.get("aggregate_score", 0)),
    )


def sort_trade_packets(candidates: list) -> list[dict]:
    """Convert candidates into packets ordered by actionability then quality."""
    packets = [build_trade_packet(candidate) for candidate in candidates]
    return sorted(packets, key=trade_packet_sort_key, reverse=True)


_TIER1_GATE_KEYS = ("altman_z", "beneish_m", "f_score")


def packet_has_tier1_failure(packet: dict) -> bool:
    """True if any Tier-1 distress / quality-of-earnings flag is "fail"."""
    flags = packet.get("action_gate_flags") or {}
    return any(flags.get(k) == "fail" for k in _TIER1_GATE_KEYS)


def split_packets_by_tier1_health(packets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition packets into (clean, near_miss).

    A "near miss" is any packet that the percentile cut would label
    BUY/STRONG BUY but a Tier-1 gate (Altman Z, Beneish M, F-Score)
    confirmed as a fail.  When ``DISCOVERY_DIGEST_HIDE_GATE_FAILURES``
    is True (default), the email/UI digest hides the near-miss list to
    avoid noise on the decision surface — they remain available in the
    secondary diagnostic section.
    """
    clean: list[dict] = []
    near_miss: list[dict] = []
    for p in packets:
        if (p.get("action") in ("BUY", "STRONG BUY")) and packet_has_tier1_failure(p):
            near_miss.append(p)
        else:
            clean.append(p)
    return clean, near_miss
