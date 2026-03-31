"""Reusable HTML helpers for the Streamlit dashboard."""

from __future__ import annotations

import html as _html

from utils.safe_numeric import safe_float, is_valid_number, format_currency, format_pct, format_score


def format_price(val, currency: str) -> str:
    return format_currency(val, currency, decimals=2)


def format_change(val) -> str:
    return format_pct(val, decimals=2)


def render_action_pill(action: str) -> str:
    cls = action.lower().replace(" ", "-")
    return f'<span class="action-pill action-pill-{cls}">{action}</span>'


def render_score_bar(score: float) -> str:
    pct = ((score + 1) / 2) * 100
    pct = max(0, min(100, pct))
    return f"""
    <div style="margin:4px 0;">
        <div class="score-bar-outer">
            <div class="score-bar-marker" style="left:{pct}%"></div>
        </div>
        <div class="score-bar-labels"><span>-1</span><span>0</span><span>+1</span></div>
    </div>"""


def render_pillar_bars(tech: float, fund: float, sent: float, fcast: float) -> str:
    rows = ""
    for label, val in [("Tech", tech), ("Fund", fund), ("Sent", sent), ("Fcast", fcast)]:
        pct = ((val + 1) / 2) * 100
        pct = max(0, min(100, pct))
        if val > 0.1:
            color = "#10b981"
        elif val < -0.1:
            color = "#ef4444"
        else:
            color = "#6b7280"
        if pct >= 50:
            left = 50
            width = pct - 50
        else:
            left = pct
            width = 50 - pct
        rows += f"""
        <div class="pillar-row">
            <span class="pillar-label">{label}</span>
            <div class="pillar-bar-bg">
                <div class="pillar-bar-fill" style="left:{left}%;width:{width}%;background:{color};"></div>
            </div>
            <span class="pillar-val" style="color:{color}">{val:+.2f}</span>
        </div>"""
    return f'<div style="margin:4px 0">{rows}</div>'


def render_rsi_gauge(rsi_val: float) -> str:
    if rsi_val is None:
        return ""
    pct = max(0, min(100, rsi_val))
    return f"""
    <div style="margin:6px 0;">
        <div class="rsi-gauge-outer">
            <div class="rsi-gauge-marker" style="left:{pct}%"></div>
        </div>
        <div class="rsi-gauge-labels">
            <span>Oversold</span><span>RSI {rsi_val:.0f}</span><span>Overbought</span>
        </div>
    </div>"""


def render_news_card(title: str, sentiment: float, extra: str = "") -> str:
    if sentiment > 0.1:
        cls = "news-positive"
    elif sentiment < -0.1:
        cls = "news-negative"
    else:
        cls = "news-neutral"
    safe_title = _html.escape(title)
    return f"""<div class="news-card {cls}">
        <span class="news-score">{sentiment:+.2f}</span>
        {safe_title} {extra}
    </div>"""


def render_metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-card-label">{label}</div>'
        f'<div class="metric-card-value">{value}</div>'
        f"{sub_html}</div>"
    )


def render_weight_bar(label: str, weight: float) -> str:
    pct = weight * 100
    return f"""<div class="weight-bar-row">
        <span class="weight-bar-label">{label}</span>
        <div class="weight-bar-bg"><div class="weight-bar-fill" style="width:{pct}%"></div></div>
        <span class="weight-bar-pct">{pct:.1f}%</span>
    </div>"""


def discovery_confidence(cand) -> tuple[str, str, float]:
    sent_conf = safe_float(getattr(cand, "sentiment_score", 0))
    data_discount = safe_float(getattr(cand, "confidence_discount", 1.0), default=1.0)
    has_data = 0.0 if getattr(cand, "action", "") == "INSUFFICIENT DATA" else 1.0
    score = max(
        0.0,
        min(
            1.0,
            0.35 * has_data + 0.30 * data_discount + 0.35 * (0.5 + 0.5 * max(min(sent_conf, 1.0), -1.0)),
        ),
    )
    if getattr(cand, "action", "") == "INSUFFICIENT DATA":
        return "Data Gap", "data", score
    if score >= 0.75:
        return "High Confidence", "high", score
    if score >= 0.55:
        return "Medium Confidence", "medium", score
    return "Watch Carefully", "low", score


def candidate_risk_tags(cand) -> list[str]:
    tags = []
    if getattr(cand, "is_parabolic", False):
        tags.append("Parabolic move")
    if getattr(cand, "earnings_imminent", False):
        tags.append(f"Earnings in {getattr(cand, 'earnings_days', '?')}d")
    elif getattr(cand, "earnings_near", False):
        tags.append(f"Earnings soon ({getattr(cand, 'earnings_days', '?')}d)")
    if getattr(cand, "earnings_miss", False):
        miss_pct = getattr(cand, "earnings_miss_pct", None)
        tags.append(f"Recent earnings miss{f' {miss_pct:+.0f}%' if miss_pct is not None else ''}")
    if getattr(cand, "near_52w_high", False):
        tags.append("Near 52-week high")
    if getattr(cand, "fx_penalty_applied", False):
        tags.append(f"FX drag {safe_float(getattr(cand, 'fx_penalty_pct', 0)):.1f}%")
    if safe_float(getattr(cand, "max_correlation", 0)) >= 0.70:
        tags.append(f"High correlation {safe_float(getattr(cand, 'max_correlation', 0)):.2f}")
    if getattr(cand, "action", "") == "INSUFFICIENT DATA":
        tags.append("Incomplete signal")
    return tags[:4]


def candidate_evidence_tags(cand) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    ret_90 = safe_float(getattr(cand, "return_90d", 0)) * 100
    ret_30 = safe_float(getattr(cand, "return_30d", 0)) * 100
    exp_90 = safe_float(getattr(cand, "expected_return_90d", 0)) * 100
    fit = safe_float(getattr(cand, "portfolio_fit_score", 0))
    corr = safe_float(getattr(cand, "max_correlation", 0))
    fund = safe_float(getattr(cand, "fundamental_score", 0))
    tech = safe_float(getattr(cand, "technical_score", 0))

    if fit >= 0.80:
        tags.append((f"Fit {fit:.2f}", "info"))
    if corr and corr < 0.40:
        tags.append((f"Low corr {corr:.2f}", "info"))
    if ret_90 > 20:
        tags.append((f"90d {ret_90:+.1f}%", "good"))
    if ret_30 > 10:
        tags.append((f"30d {ret_30:+.1f}%", "good"))
    if fund > 0.15:
        tags.append((f"Fundamentals {fund:+.2f}", "good"))
    if tech > 0.15:
        tags.append((f"Technicals {tech:+.2f}", "good"))
    if exp_90 > 5:
        tags.append((f"90d model {exp_90:+.1f}%", "good"))
    if safe_float(getattr(cand, "volume_ratio", 1.0)) > 1.5:
        tags.append((f"Volume {safe_float(getattr(cand, 'volume_ratio', 1.0)):.1f}x", "info"))
    return tags[:5]


def candidate_thesis(cand) -> str:
    positives = []
    if safe_float(getattr(cand, "portfolio_fit_score", 0)) >= 0.80:
        positives.append("improves diversification")
    if safe_float(getattr(cand, "momentum_score", 0)) >= 0.80 or safe_float(getattr(cand, "return_90d", 0)) > 0.25:
        positives.append("trend strength is still intact")
    if safe_float(getattr(cand, "fundamental_score", 0)) > 0.15:
        positives.append("fundamentals support the move")
    if safe_float(getattr(cand, "forecast_score", 0)) > 0.15 or safe_float(getattr(cand, "expected_return_90d", 0)) > 0.06:
        positives.append("the forward model still sees upside")

    risks = candidate_risk_tags(cand)
    if getattr(cand, "action", "") == "INSUFFICIENT DATA":
        return "Signal quality is incomplete, so this name should stay in watch mode until fresh analysis fills the missing pillars."
    if not positives:
        positives.append("it remains one of the cleaner ideas in the current universe")
    lead = ", ".join(positives[:2])
    if risks:
        return f"This idea stands out because it {lead}, but {risks[0].lower()} needs monitoring."
    return f"This idea stands out because it {lead}, with no immediate red-flag overlays in the current pass."


def render_html_chips(chips: list[tuple[str, str]], class_name: str = "signal-chip") -> str:
    if not chips:
        return ""
    rendered = "".join(
        f'<span class="{class_name} {tone}">{_html.escape(text)}</span>'
        for text, tone in chips
    )
    return f'<div class="chip-row">{rendered}</div>'


def lens_sorted_candidates(candidates: list, lens: str) -> list:
    if lens == "Best Diversifiers":
        return sorted(
            candidates,
            key=lambda c: (
                safe_float(getattr(c, "portfolio_fit_score", 0)) * 0.60
                + safe_float(getattr(c, "final_rank", 0)) * 0.25
                + safe_float(getattr(c, "momentum_score", 0)) * 0.15
            ),
            reverse=True,
        )
    if lens == "Momentum Leaders":
        return sorted(
            candidates,
            key=lambda c: (
                safe_float(getattr(c, "momentum_score", 0)),
                safe_float(getattr(c, "return_90d", 0)),
                safe_float(getattr(c, "return_30d", 0)),
            ),
            reverse=True,
        )
    if lens == "Value / Quality":
        return sorted(
            candidates,
            key=lambda c: (
                safe_float(getattr(c, "fundamental_score", 0)) * 0.60
                + safe_float(getattr(c, "aggregate_score", 0)) * 0.25
                + safe_float(getattr(c, "portfolio_fit_score", 0)) * 0.15
            ),
            reverse=True,
        )
    return sorted(candidates, key=lambda c: safe_float(getattr(c, "final_rank", 0)), reverse=True)


def exit_card_tags(exit_signal: dict) -> list[tuple[str, str]]:
    chips: list[tuple[str, str]] = []
    score = exit_signal.get("current_score")
    price = exit_signal.get("current_price")
    stop_loss = exit_signal.get("stop_loss")
    take_profit = exit_signal.get("take_profit")
    signal_type = str(exit_signal.get("signal_type", "")).lower()

    if is_valid_number(score):
        chips.append((f"Score {format_score(score)}", "info"))
    if is_valid_number(price):
        chips.append((f"Price {format_currency(price, 'GBP')}", "info"))
    if is_valid_number(stop_loss):
        chips.append((f"Stop {format_currency(stop_loss, 'GBP')}", "warn"))
    if is_valid_number(take_profit):
        chips.append((f"Target {format_currency(take_profit, 'GBP')}", "good"))
    if "stop" in signal_type:
        chips.append(("Risk control", "risk"))
    elif "target" in signal_type or "profit" in signal_type:
        chips.append(("Lock gains", "good"))
    elif "decay" in signal_type:
        chips.append(("Signal weakening", "warn"))
    return chips[:5]
