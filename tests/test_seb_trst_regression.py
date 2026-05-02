"""End-to-end regression test fixed to the SEB SA / Trustpilot snapshots.

These two stocks topped the screener as STRONG BUY in the April 2026
human review and were both flagged as wrong by an external four-factor
framework analysis.  This test pins the new Tier 1-4 action gates to the
exact factor values from that review so any future regression that lets
either stock through is caught immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.action_gates import apply_action_gates


@dataclass
class _Cand:
    """Minimal stand-in for ScoredCandidate — only the fields the gates read."""
    ticker: str
    sector: str
    action: str = "STRONG BUY"
    aggregate_score: float = 0.45
    altman_z: float | None = None
    altman_zone: str = "unknown"
    beneish_m: float | None = None
    f_score: int | None = None
    f_score_coverage: float = 1.0
    accruals_factor_score: float | None = None
    investment_factor_score: float | None = None
    net_debt_ebitda: float | None = None
    ev_ebit: float | None = None
    ev_sales: float | None = None
    pe_ratio: float | None = None
    pe_forward: float | None = None
    eps_growth_3y_cagr: float | None = None
    qmj_factor_score: float | None = None
    gpa: float | None = None
    gpa_score: float | None = None
    roic: float | None = None
    wacc: float | None = None
    op_margin_yoy_delta: float | None = None
    rsi: float | None = None
    price_vs_sma200_stretch: float | None = None
    entry_stance: str = "Ready"
    realized_vol_pctile: float | None = None
    sma_200: float | None = None
    current_price: float | None = None
    atr: float | None = None
    support_levels: dict = field(default_factory=dict)
    action_gate_ceiling: str = "STRONG BUY"
    action_gate_reasons: list = field(default_factory=list)
    action_gate_flags: dict = field(default_factory=dict)
    limit_price: float | None = None
    limit_price_method: str | None = None
    limit_price_rationale: str | None = None
    strong_buy_eligible: bool = True
    ready_contract_reasons: list = field(default_factory=list)
    ready_contract_status: str = "PASS"


def test_seb_sa_april_2026_snapshot_blocks_strong_buy():
    """SEB SA (SK.PA) at €53.65 on 24-Apr-2026.

    Framework verdict: 1.5/4 — Altman Z=1.72 (distress), F=5/9, ROIC-WACC
    only 120 bps, op-margin -230 bps, net debt/EBITDA 2.7x.  All Tier-1
    distress flags should fire; Tier-3 ROIC-WACC and op-margin should
    further confirm the downgrade.
    """
    seb = _Cand(
        ticker="SK.PA", sector="Consumer Cyclical",
        altman_z=1.72, f_score=5, f_score_coverage=1.0,
        accruals_factor_score=-0.20, investment_factor_score=0.10,
        net_debt_ebitda=2.7,
        ev_ebit=10.2, pe_ratio=9.1, pe_forward=9.1,
        qmj_factor_score=0.10, gpa=0.42,
        roic=0.071, wacc=0.059,
        op_margin_yoy_delta=-0.023,
        rsi=69, price_vs_sma200_stretch=0.05, entry_stance="Ready",
    )
    apply_action_gates([seb])
    # Altman Z=1.72 sits in the distress zone (< 1.81) → NEUTRAL ceiling
    assert seb.action_gate_ceiling == "NEUTRAL", (
        f"Expected NEUTRAL ceiling for SEB SA (Z=1.72), got {seb.action_gate_ceiling}"
    )
    # All three primary failure modes must surface in the reasons
    reasons_blob = " | ".join(seb.action_gate_reasons)
    assert "Altman" in reasons_blob, "Altman Z gate must fire on SEB"
    assert "Piotroski" in reasons_blob, "F-Score floor must fire on SEB"
    assert "ROIC" in reasons_blob, "ROIC-WACC spread must fire on SEB"


def test_trustpilot_april_2026_snapshot_blocks_strong_buy_and_offers_pullback():
    """Trustpilot (TRST.L) at 251p on 24-Apr-2026.

    Framework verdict: 3/4 — quality + momentum strong, but EV/EBIT 78x,
    EV/Sales 4.8x, fwd P/E 49x, RSI 70, +35% above 200-DMA, and
    "Pullback Preferred" entry stance.  Should be downgraded and
    accompanied by a limit-price recommendation.
    """
    trst = _Cand(
        ticker="TRST.L", sector="Communication Services",
        altman_z=4.5, f_score=8, f_score_coverage=1.0,
        accruals_factor_score=0.30, investment_factor_score=0.15,
        net_debt_ebitda=-1.0,
        ev_ebit=78.0, ev_sales=4.8, pe_ratio=49.0, pe_forward=49.0,
        eps_growth_3y_cagr=0.20,
        qmj_factor_score=0.85, gpa=1.69, roic=0.20, wacc=0.10,
        op_margin_yoy_delta=0.05,
        rsi=70, price_vs_sma200_stretch=0.35, entry_stance="Pullback Preferred",
        sma_200=185.0, current_price=251.0, atr=8.0,
        support_levels={"placement": 214.0},
    )
    apply_action_gates([trst])
    # EV/EBIT 78x exceeds the 50x absolute ceiling → NEUTRAL
    assert trst.action_gate_ceiling == "NEUTRAL", (
        f"Expected NEUTRAL ceiling for Trustpilot (EV/EBIT 78x), got {trst.action_gate_ceiling}"
    )
    reasons_blob = " | ".join(trst.action_gate_reasons)
    assert "EV/EBIT" in reasons_blob
    assert "Entry stance" in reasons_blob, "Pullback Preferred must block STRONG BUY"
    # Limit-price suggestion should be produced and below current price
    assert trst.limit_price is not None
    assert trst.limit_price < trst.current_price


def test_clean_compounder_passes_strong_buy_after_gates():
    """A genuinely-clean compounder should keep its STRONG BUY label."""
    clean = _Cand(
        ticker="ABCD", sector="Technology",
        altman_z=5.0, f_score=8, f_score_coverage=1.0,
        accruals_factor_score=0.40, investment_factor_score=0.20,
        net_debt_ebitda=0.3,
        ev_ebit=18.0, ev_sales=4.0, pe_ratio=22.0, pe_forward=22.0,
        eps_growth_3y_cagr=0.18,
        qmj_factor_score=0.85, gpa=0.70, roic=0.25, wacc=0.10,
        op_margin_yoy_delta=0.03,
        rsi=58, price_vs_sma200_stretch=0.15, entry_stance="Ready",
    )
    apply_action_gates([clean])
    assert clean.action_gate_ceiling == "STRONG BUY"
    assert clean.action_gate_reasons == []
    assert clean.limit_price is None


def test_digest_split_hides_tier1_failures():
    """The digest helper separates clean STRONG BUYs from Tier-1 near-misses."""
    from utils.discovery_digest import (
        packet_has_tier1_failure,
        split_packets_by_tier1_health,
    )
    clean_packet = {
        "ticker": "GOOD", "action": "STRONG BUY",
        "action_gate_flags": {"altman_z": "pass", "f_score": "pass"},
    }
    bad_packet = {
        "ticker": "SK.PA", "action": "STRONG BUY",
        "action_gate_flags": {"altman_z": "fail", "f_score": "fail"},
    }
    borderline_packet = {
        "ticker": "MID", "action": "BUY",
        "action_gate_flags": {"altman_z": "borderline"},
    }
    assert not packet_has_tier1_failure(clean_packet)
    assert packet_has_tier1_failure(bad_packet)
    assert not packet_has_tier1_failure(borderline_packet)

    clean, near_miss = split_packets_by_tier1_health(
        [clean_packet, bad_packet, borderline_packet]
    )
    assert clean_packet in clean
    assert borderline_packet in clean
    assert bad_packet in near_miss
    assert len(clean) == 2
    assert len(near_miss) == 1
