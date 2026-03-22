"""ISA Portfolio Dashboard — Premium Streamlit Frontend."""

import html as _html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from engine.scoring import analyse_portfolio
from engine.backtest import optimize_weights, _score_to_action
from utils.data_fetch import clear_cache, load_portfolio
from utils.cache_loader import load_dashboard_data, format_freshness

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ISA Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system — comprehensive CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ── Base overrides ─────────────────────────────────────────────── */
div[data-testid="stMetricValue"] { font-size: 1.15rem; }
.stDataFrame { font-size: 13px; }

/* ── Hero section ───────────────────────────────────────────────── */
.hero-value {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 2px;
}
.hero-pl {
    font-size: 1.3rem;
    font-weight: 600;
}
.hero-pl-positive { color: #10b981; }
.hero-pl-negative { color: #ef4444; }

/* ── Action pill badges ─────────────────────────────────────────── */
.action-pill {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    color: white;
    text-align: center;
}
.action-pill-strong-buy { background: linear-gradient(135deg, #059669, #10b981); }
.action-pill-buy { background: linear-gradient(135deg, #10b981, #34d399); }
.action-pill-keep { background: linear-gradient(135deg, #3b82f6, #60a5fa); }
.action-pill-sell { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #1a1a1a; }
.action-pill-strong-sell { background: linear-gradient(135deg, #dc2626, #ef4444); }

/* ── Score bar (horizontal gauge -1 to +1) ──────────────────────── */
.score-bar-outer {
    position: relative;
    width: 100%;
    height: 10px;
    background: linear-gradient(to right, #ef4444 0%, #fbbf24 40%, #6b7280 50%, #fbbf24 60%, #10b981 100%);
    border-radius: 5px;
    margin: 6px 0 2px 0;
    opacity: 0.35;
}
.score-bar-marker {
    position: absolute;
    top: -3px;
    width: 16px;
    height: 16px;
    background: white;
    border: 3px solid #1a1a1a;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.score-bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    opacity: 0.5;
    margin-top: 1px;
}

/* ── Pillar mini bars ───────────────────────────────────────────── */
.pillar-row {
    display: flex;
    align-items: center;
    margin: 2px 0;
    font-size: 0.78rem;
}
.pillar-label {
    width: 42px;
    font-weight: 600;
    opacity: 0.7;
    flex-shrink: 0;
}
.pillar-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(128,128,128,0.2);
    border-radius: 3px;
    position: relative;
    margin: 0 6px;
}
.pillar-bar-fill {
    position: absolute;
    top: 0;
    height: 100%;
    border-radius: 3px;
}
.pillar-val {
    width: 38px;
    text-align: right;
    font-weight: 600;
    font-size: 0.75rem;
}

/* ── RSI gauge ──────────────────────────────────────────────────── */
.rsi-gauge-outer {
    position: relative;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #10b981 0%, #10b981 30%, #6b7280 30%, #6b7280 70%, #ef4444 70%, #ef4444 100%);
    opacity: 0.5;
    margin: 4px 0;
}
.rsi-gauge-marker {
    position: absolute;
    top: -4px;
    width: 14px;
    height: 14px;
    background: white;
    border: 2px solid #333;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.rsi-gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    opacity: 0.45;
}

/* ── News / headline cards ──────────────────────────────────────── */
.news-card {
    border-left: 4px solid;
    padding: 6px 10px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    background: rgba(128,128,128,0.05);
}
.news-positive { border-color: #10b981; }
.news-negative { border-color: #ef4444; }
.news-neutral  { border-color: #6b7280; }
.news-score {
    float: right;
    font-weight: 600;
    font-size: 0.8rem;
    opacity: 0.7;
}

/* ── Metric card (for details grid) ─────────────────────────────── */
.metric-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
    margin: 3px 0;
}
.metric-card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.6;
    margin-bottom: 2px;
}
.metric-card-value {
    font-size: 1.1rem;
    font-weight: 700;
}
.metric-card-sub {
    font-size: 0.7rem;
    opacity: 0.5;
    margin-top: 1px;
}

/* ── Holding card action tint ───────────────────────────────────── */
.holding-header {
    padding: 2px 0;
}

/* ── Sidebar weight bars ────────────────────────────────────────── */
.weight-bar-row {
    display: flex;
    align-items: center;
    margin: 5px 0;
    font-size: 0.82rem;
}
.weight-bar-label {
    width: 90px;
    font-weight: 600;
}
.weight-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(128,128,128,0.2);
    border-radius: 4px;
    overflow: hidden;
    margin: 0 8px;
}
.weight-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
}
.weight-bar-pct {
    width: 42px;
    text-align: right;
    font-weight: 600;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly shared theme
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

_ACTION_COLORS = {
    "STRONG BUY": "#059669",
    "BUY": "#10b981",
    "KEEP": "#3b82f6",
    "SELL": "#f59e0b",
    "STRONG SELL": "#ef4444",
}

# ---------------------------------------------------------------------------
# Reusable HTML component functions
# ---------------------------------------------------------------------------

def _format_price(val, currency: str) -> str:
    if val is None:
        return "N/A"
    symbol = "£" if currency in ("GBP", "GBX") else "$" if currency == "USD" else "€"
    if currency == "GBX":
        return f"{val:,.0f}p"
    return f"{symbol}{val:,.2f}"


def _format_change(val) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _render_action_pill(action: str) -> str:
    cls = action.lower().replace(" ", "-")
    return f'<span class="action-pill action-pill-{cls}">{action}</span>'


def _render_score_bar(score: float) -> str:
    """Horizontal bar from -1 to +1 with a marker at score position."""
    pct = ((score + 1) / 2) * 100  # Map -1..+1 to 0..100%
    pct = max(0, min(100, pct))
    return f"""
    <div style="margin:4px 0;">
        <div class="score-bar-outer">
            <div class="score-bar-marker" style="left:{pct}%"></div>
        </div>
        <div class="score-bar-labels"><span>-1</span><span>0</span><span>+1</span></div>
    </div>"""


def _render_pillar_bars(tech: float, fund: float, sent: float, fcast: float) -> str:
    """4 mini horizontal score bars for the 4 pillars."""
    rows = ""
    for label, val in [("Tech", tech), ("Fund", fund), ("Sent", sent), ("Fcast", fcast)]:
        pct = ((val + 1) / 2) * 100
        pct = max(0, min(100, pct))
        # Color: green for positive, red for negative
        if val > 0.1:
            color = "#10b981"
        elif val < -0.1:
            color = "#ef4444"
        else:
            color = "#6b7280"
        # Bar fills from center (50%) to the value position
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


def _render_rsi_gauge(rsi_val: float) -> str:
    """RSI gauge with oversold/neutral/overbought zones."""
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


def _render_news_card(title: str, sentiment: float, extra: str = "") -> str:
    """Styled headline card with colored left border."""
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


def _render_metric_card(label: str, value: str, sub: str = "") -> str:
    """Mini metric card for detail grids."""
    sub_html = f'<div class="metric-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-card-label">{label}</div>'
        f'<div class="metric-card-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _render_weight_bar(label: str, weight: float) -> str:
    """Sidebar weight visualization bar."""
    pct = weight * 100
    return f"""<div class="weight-bar-row">
        <span class="weight-bar-label">{label}</span>
        <div class="weight-bar-bg"><div class="weight-bar-fill" style="width:{pct}%"></div></div>
        <span class="weight-bar-pct">{pct:.1f}%</span>
    </div>"""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 📊 ISA Portfolio Dashboard")
st.caption("Multi-factor analysis engine — Technical · Fundamental · Sentiment · MoE Forecast")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    # Refresh triggers live recomputation; normal load uses cache
    _force_refresh = st.button("🔄 Refresh Analysis", use_container_width=True, type="primary")
    if _force_refresh:
        clear_cache()

    st.divider()

    # Scoring weights — visual bars
    st.markdown("**Scoring Weights**")
    weight_html = ""
    for pillar, w in config.WEIGHTS.items():
        weight_html += _render_weight_bar(pillar.title(), w)
    st.markdown(weight_html, unsafe_allow_html=True)

    st.divider()

    # Sort & Filter
    sort_option = st.selectbox(
        "Sort by",
        ["Score ↓", "Score ↑", "P&L % ↓", "P&L % ↑", "Ticker A-Z"],
        index=0,
    )
    action_filter = st.multiselect(
        "Filter by Action",
        ["STRONG BUY", "BUY", "KEEP", "SELL", "STRONG SELL"],
        default=[],
    )

    st.divider()

    # FMP status with colored dot
    try:
        from utils.fmp_client import (
            is_available as fmp_is_available,
            get_remaining_budget, get_calls_today,
        )
        if fmp_is_available():
            remaining = get_remaining_budget()
            today_calls = get_calls_today()
            plan = getattr(config, "FMP_PLAN", "free").title()
            st.markdown(f":green_circle: **FMP {plan}** — {remaining}/min avail · {today_calls} today")
        elif config.FMP_API_KEY:
            st.markdown(":orange_circle: **FMP** — Rate limit reached")
        else:
            st.markdown(":red_circle: **FMP** — Not configured")
    except ImportError:
        st.markdown(":red_circle: **FMP** — Not available")

    st.divider()
    st.caption("Data: yfinance + FMP · News: Google RSS + Reddit + FMP · Sentiment: FinBERT")
    st.caption("Record sales below in the Trade History section.")

# ---------------------------------------------------------------------------
# Load data: cache-first, live on Refresh
# ---------------------------------------------------------------------------
if _force_refresh:
    with st.spinner("Running live analysis... (this may take a few minutes)"):
        _dash = load_dashboard_data(force_refresh=True)
else:
    _dash = load_dashboard_data(force_refresh=False)

holdings = _dash.holdings
results = _dash.results
risk_data = _dash.risk_data
position_weights = _dash.position_weights

# Freshness status bar
_freshness_parts = []
if _dash.from_cache:
    _freshness_parts.append(f"Portfolio: **{format_freshness(_dash.portfolio_timestamp)}**")
else:
    _freshness_parts.append("Portfolio: **live**")
if _dash.optimizer_timestamp:
    _freshness_parts.append(f"Optimizer: **{format_freshness(_dash.optimizer_timestamp)}**")
if _dash.discovery_timestamp:
    _freshness_parts.append(f"Screener: **{format_freshness(_dash.discovery_timestamp)}**")

# VIX regime from cached data
_regime = _dash.vix_regime
if _regime:
    _regime_colors_map = {"BULL": "🟢", "NEUTRAL": "⚪", "BEAR": "🔴"}
    _regime_icon = _regime_colors_map.get(_regime.get("regime_label", ""), "⚪")
    _freshness_parts.append(f"Regime: {_regime_icon} {_regime.get('regime_label', 'N/A')}")

st.caption(" · ".join(_freshness_parts))
if _dash.from_cache and not results:
    st.warning("No cached data available. Click **Refresh Analysis** to run the first analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute portfolio-level P&L
# ---------------------------------------------------------------------------
total_cost = 0.0
total_value = 0.0
per_holding_pl = []
for r in results:
    _cp = r.get("current_price")
    _ap = r.get("avg_buy_price")
    _qty = r.get("quantity", 0)
    _cur = r.get("currency", "GBP")
    if _cp is not None and _ap is not None and _ap > 0 and _qty > 0:
        factor = 0.01 if _cur == "GBX" else 1.0
        cost = _ap * _qty * factor
        value = _cp * _qty * factor
        total_cost += cost
        total_value += value
        pl = value - cost
        pl_pct = ((value - cost) / cost) * 100 if cost > 0 else 0
        per_holding_pl.append((r["ticker"], pl, pl_pct))

total_pl = total_value - total_cost
total_pl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
avg_score = np.mean([r["aggregate_score"] for r in results]) if results else 0

# Action counts
action_counts = {}
for a in ["STRONG BUY", "BUY", "KEEP", "SELL", "STRONG SELL"]:
    action_counts[a] = sum(1 for r in results if r["action"] == a)

# ---------------------------------------------------------------------------
# Hero section — portfolio summary
# ---------------------------------------------------------------------------
hero_left, hero_right = st.columns([2, 1])

with hero_left:
    pl_color_cls = "hero-pl-positive" if total_pl >= 0 else "hero-pl-negative"
    pl_sign = "+" if total_pl >= 0 else ""
    st.markdown(
        f'<div class="hero-value">£{total_value:,.0f}</div>'
        f'<div class="hero-pl {pl_color_cls}">'
        f'{pl_sign}£{total_pl:,.0f} ({total_pl_pct:+.1f}%)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Total portfolio value · unrealised P&L")

    # Best / worst performer
    if per_holding_pl:
        best = max(per_holding_pl, key=lambda x: x[2])
        worst = min(per_holding_pl, key=lambda x: x[2])
        st.markdown(
            f"**Best:** {best[0]} ({best[2]:+.1f}%) · "
            f"**Worst:** {worst[0]} ({worst[2]:+.1f}%)"
        )

with hero_right:
    # Donut chart — allocation by action type
    donut_labels = []
    donut_values = []
    donut_colors = []
    for action_name in ["STRONG BUY", "BUY", "KEEP", "SELL", "STRONG SELL"]:
        cnt = action_counts[action_name]
        if cnt > 0:
            donut_labels.append(action_name)
            donut_values.append(cnt)
            donut_colors.append(_ACTION_COLORS[action_name])

    if donut_values:
        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.65,
            marker=dict(colors=donut_colors),
            textinfo="value",
            textfont=dict(size=14, color="white"),
            hovertemplate="%{label}: %{value} holdings<extra></extra>",
        ))
        fig_donut.update_layout(
            **_PLOTLY_LAYOUT,
            height=200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=10)),
            annotations=[dict(
                text=f"<b>{len(results)}</b><br>holdings",
                x=0.5, y=0.5, font_size=16, showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

# Action count cards
act_cols = st.columns(5)
for i, (action_name, color) in enumerate([
    ("STRONG BUY", "#059669"), ("BUY", "#10b981"), ("KEEP", "#3b82f6"),
    ("SELL", "#f59e0b"), ("STRONG SELL", "#ef4444"),
]):
    cnt = action_counts[action_name]
    act_cols[i].metric(action_name, cnt)

# Portfolio health gauge
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(avg_score, 3),
    number=dict(font=dict(size=28), valueformat="+.3f"),
    gauge=dict(
        axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        bar=dict(color="#3b82f6", thickness=0.6),
        bgcolor="rgba(128,128,128,0.1)",
        steps=[
            dict(range=[-1, -0.3], color="rgba(239,68,68,0.15)"),
            dict(range=[-0.3, 0.25], color="rgba(59,130,246,0.1)"),
            dict(range=[0.25, 1], color="rgba(16,185,129,0.15)"),
        ],
        threshold=dict(line=dict(color="white", width=2), thickness=0.8, value=avg_score),
    ),
    title=dict(text="Portfolio Health", font=dict(size=14)),
))
gauge_fig.update_layout(**{**_PLOTLY_LAYOUT, "margin": dict(l=30, r=30, t=50, b=10)}, height=160)
st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Portfolio Risk Analysis
# ---------------------------------------------------------------------------
if risk_data and risk_data.get("sector_weights"):
    with st.expander("🛡️ **Portfolio Risk Analysis**", expanded=bool(risk_data.get("concentration_warnings"))):
        risk_col1, risk_col2 = st.columns(2)

        # Sector allocation pie chart
        with risk_col1:
            sector_w = risk_data["sector_weights"]
            if sector_w:
                sector_fig = go.Figure(go.Pie(
                    labels=list(sector_w.keys()),
                    values=[round(v * 100, 1) for v in sector_w.values()],
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set2),
                    textinfo="label+percent",
                    textposition="outside",
                ))
                sector_fig.update_layout(
                    **{**_PLOTLY_LAYOUT, "margin": dict(l=10, r=10, t=35, b=10)},
                    height=300,
                    title=dict(text="Sector Allocation", font=dict(size=14)),
                    showlegend=False,
                )
                st.plotly_chart(sector_fig, use_container_width=True, config={"displayModeBar": False})

        # Correlation heatmap
        with risk_col2:
            corr_matrix = risk_data.get("correlation_matrix")
            if corr_matrix is not None and (not hasattr(corr_matrix, "empty") or not corr_matrix.empty) and getattr(corr_matrix, "size", 0) > 0:
                corr_fig = px.imshow(
                    corr_matrix.round(2),
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto",
                )
                corr_fig.update_layout(
                    **{**_PLOTLY_LAYOUT, "margin": dict(l=10, r=10, t=35, b=10)},
                    height=300,
                    title=dict(text="Return Correlations (90d)", font=dict(size=14)),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False})

        # Risk warnings
        warnings = risk_data.get("concentration_warnings", [])
        high_corrs = risk_data.get("high_correlations", [])

        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")

        if high_corrs:
            corr_strs = [f"{t1}↔{t2} ({c:+.2f})" for t1, t2, c in high_corrs[:5]]
            st.info(f"📊 Highly correlated pairs: {', '.join(corr_strs)}")

        # Risk score
        risk_score = risk_data.get("risk_score", 0)
        risk_label = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.6 else "High"
        risk_color = "#10b981" if risk_score < 0.3 else "#f59e0b" if risk_score < 0.6 else "#ef4444"
        st.markdown(
            f'<div style="text-align:center; padding:8px;">'
            f'<span style="font-size:1.1rem; font-weight:600;">Portfolio Risk Score: </span>'
            f'<span style="font-size:1.3rem; font-weight:700; color:{risk_color};">'
            f'{risk_score:.0%} ({risk_label})</span></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Exit Intelligence (from cache or live)
# ---------------------------------------------------------------------------
_exit_list = _dash.cached_exit_signals
if _exit_list is None and not _dash.from_cache:
    # Live mode but no exit signals computed yet — run now
    try:
        from engine.exit_engine import assess_exits as _assess_exits
        _exit_objs = _assess_exits(results, holdings)
        _exit_list = [
            {"ticker": e.ticker, "name": e.name, "signal_type": e.signal_type,
             "severity": e.severity, "message": e.message,
             "current_score": e.current_score, "current_price": e.current_price}
            for e in _exit_objs
        ]
    except Exception:
        _exit_list = []

if _exit_list:
    _severity_colors = {"urgent": "#ef4444", "action_needed": "#f59e0b", "warning": "#6b7280"}
    _severity_icons = {"urgent": "🔴", "action_needed": "🟡", "warning": "⚪"}

    with st.expander(
        f"🚪 **Exit Intelligence** ({len(_exit_list)} signals)"
        + (f" · {format_freshness(_dash.exit_signals_timestamp)}" if _dash.exit_signals_timestamp else ""),
        expanded=any(e.get("severity") == "urgent" for e in _exit_list),
    ):
        for _es in _exit_list:
            _scolor = _severity_colors.get(_es.get("severity", ""), "#6b7280")
            _sicon = _severity_icons.get(_es.get("severity", ""), "⚪")
            st.markdown(
                f'{_sicon} **{_es.get("ticker", "")}** ({_es.get("name", "")}) — '
                f'<span style="color:{_scolor};font-weight:600">{_es.get("signal_type", "").replace("_", " ").title()}</span>: '
                f'{_es.get("message", "")}',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Suggested Allocation (Inverse-Volatility)
# ---------------------------------------------------------------------------
if position_weights:
    with st.expander("⚖️ **Suggested Allocation (Inverse-Volatility)**"):
        alloc_tickers = [pw["ticker"] for pw in position_weights]
        alloc_current = [pw["current_weight"] * 100 for pw in position_weights]
        alloc_suggested = [pw["suggested_weight"] * 100 for pw in position_weights]

        alloc_fig = go.Figure()
        alloc_fig.add_trace(go.Bar(
            name="Current",
            x=alloc_tickers, y=alloc_current,
            marker_color="#6b7280",
            text=[f"{v:.1f}%" for v in alloc_current],
            textposition="auto",
        ))
        alloc_fig.add_trace(go.Bar(
            name="Suggested",
            x=alloc_tickers, y=alloc_suggested,
            marker_color="#3b82f6",
            text=[f"{v:.1f}%" for v in alloc_suggested],
            textposition="auto",
        ))
        alloc_fig.update_layout(
            **{**_PLOTLY_LAYOUT, "margin": dict(l=40, r=10, t=35, b=10)},
            barmode="group",
            yaxis_title="Weight %",
            height=300,
            title=dict(text="Current vs Suggested Allocation", font=dict(size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(alloc_fig, use_container_width=True, config={"displayModeBar": False})

        # Rebalance table
        alloc_df = pd.DataFrame([
            {
                "Ticker": pw["ticker"],
                "Current %": f"{pw['current_weight'] * 100:.1f}%",
                "Suggested %": f"{pw['suggested_weight'] * 100:.1f}%",
                "Delta": f"{pw['rebalance_delta'] * 100:+.1f}%",
                "Ann. Vol": f"{pw['volatility'] * 100:.1f}%",
            }
            for pw in position_weights
        ])
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Portfolio Optimizer (Mean-Variance) — from cache or live
# ---------------------------------------------------------------------------
_opt_data = _dash.cached_optimizer
if _opt_data and _opt_data.get("holdings"):
    _opt_title = "🎯 **Portfolio Optimizer (Mean-Variance)**"
    if _dash.optimizer_timestamp:
        _opt_title += f" · {format_freshness(_dash.optimizer_timestamp)}"
    with st.expander(_opt_title, expanded=True):
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Expected Return", f"{_opt_data['portfolio_expected_return'] * 100:+.1f}%")
        oc2.metric("Portfolio Vol", f"{_opt_data['portfolio_volatility'] * 100:.1f}%")
        oc3.metric("Sharpe Ratio", f"{_opt_data['portfolio_sharpe']:.2f}")
        oc4.metric("Turnover", f"{_opt_data['turnover'] * 100:.1f}%")

        _regime_label = (_regime or {}).get("regime_label", "NEUTRAL") if _regime else "N/A"
        st.caption(f"Risk-free rate: {_opt_data.get('risk_free_rate', 0)*100:.1f}% | "
                   f"Regime: {_regime_label} | Method: {_opt_data.get('method', 'N/A')}")

        for w in _opt_data.get("warnings", []):
            st.info(w)

        # Current vs Optimal chart
        _opt_h = _opt_data["holdings"]
        _opt_tickers = [h["ticker"] for h in _opt_h]
        _opt_current = [h["current_weight"] * 100 for h in _opt_h]
        _opt_optimal = [h["optimal_weight"] * 100 for h in _opt_h]

        _opt_fig = go.Figure()
        _opt_fig.add_trace(go.Bar(
            name="Current", x=_opt_tickers, y=_opt_current,
            marker_color="#6b7280",
            text=[f"{v:.1f}%" for v in _opt_current], textposition="auto",
        ))
        _opt_fig.add_trace(go.Bar(
            name="Optimal", x=_opt_tickers, y=_opt_optimal,
            marker_color="#10b981",
            text=[f"{v:.1f}%" for v in _opt_optimal], textposition="auto",
        ))
        _opt_fig.update_layout(
            **{**_PLOTLY_LAYOUT, "margin": dict(l=40, r=10, t=35, b=10)},
            barmode="group", yaxis_title="Weight %", height=300,
            title=dict(text="Current vs Mean-Variance Optimal Allocation", font=dict(size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(_opt_fig, use_container_width=True, config={"displayModeBar": False})

        # Per-holding table
        _opt_rows = pd.DataFrame([{
            "Ticker": h["ticker"], "Action": h.get("action", ""),
            "Score": f"{h.get('aggregate_score', 0):.3f}",
            "Current %": f"{h['current_weight'] * 100:.1f}%",
            "Optimal %": f"{h['optimal_weight'] * 100:.1f}%",
            "Delta": f"{h['rebalance_delta'] * 100:+.1f}%",
            "E[Return]": f"{h.get('expected_return', 0) * 100:+.1f}%",
            "Vol": f"{h.get('volatility', 0) * 100:.1f}%",
            "Sector": h.get("sector", ""),
            "FX Cost": f"{h['fx_cost_if_rebalanced'] * 100:.2f}%" if h.get("fx_cost_if_rebalanced", 0) > 0 else "—",
        } for h in _opt_h])
        st.dataframe(_opt_rows, hide_index=True, use_container_width=True)

        # Rebalance trades
        _trades = _opt_data.get("rebalance_trades", [])
        if _trades:
            st.markdown("**Suggested Rebalance Trades** (> 2% delta)")
            for t in _trades:
                _dir_color = "#10b981" if t["direction"] == "BUY" else "#ef4444"
                st.markdown(
                    f'<span style="color:{_dir_color};font-weight:600">{t["direction"]}</span> '
                    f'**{t["ticker"]}** ({t["name"]}) — '
                    f'{t["current_weight"]:.1f}% → {t["optimal_weight"]:.1f}% '
                    f'({t["delta_pct"]:+.1f}%, ~£{t["trade_value"]:,.0f})',
                    unsafe_allow_html=True,
                )
        else:
            st.success("Portfolio is near-optimal — no significant rebalancing needed.")

        # Sector & FX exposure
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Optimal Sector Weights**")
            for sector, weight in sorted(_opt_data.get("sector_weights", {}).items(), key=lambda x: -x[1]):
                bar_w = int(weight * 200)
                st.markdown(f"`{sector:20s}` {'█' * max(bar_w // 5, 1)} {weight*100:.1f}%")
        with sc2:
            st.markdown("**FX Exposure**")
            for ccy, weight in sorted(_opt_data.get("fx_exposure", {}).items(), key=lambda x: -x[1]):
                st.markdown(f"`{ccy:5s}` {weight*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Sort & filter holdings
# ---------------------------------------------------------------------------
filtered_results = list(results)
if action_filter:
    filtered_results = [r for r in filtered_results if r["action"] in action_filter]

# Compute P&L% for sorting
for r in filtered_results:
    _cp = r.get("current_price")
    _ap = r.get("avg_buy_price")
    r["_pl_pct"] = ((_cp - _ap) / _ap * 100) if (_cp and _ap and _ap > 0) else 0

sort_map = {
    "Score ↓": (lambda r: r["aggregate_score"], True),
    "Score ↑": (lambda r: r["aggregate_score"], False),
    "P&L % ↓": (lambda r: r["_pl_pct"], True),
    "P&L % ↑": (lambda r: r["_pl_pct"], False),
    "Ticker A-Z": (lambda r: r["ticker"], False),
}
sort_key, sort_reverse = sort_map.get(sort_option, (lambda r: r["aggregate_score"], True))
filtered_results.sort(key=sort_key, reverse=sort_reverse)

# ---------------------------------------------------------------------------
# Holding cards
# ---------------------------------------------------------------------------
st.markdown(f"### Portfolio Analysis ({len(filtered_results)} holdings)")

for r in filtered_results:
    action = r["action"]
    currency = r.get("currency", "GBP")
    _cp = r.get("current_price")
    _ap = r.get("avg_buy_price")
    _qty = r.get("quantity", 0)

    with st.container(border=True):
        # ── Header row: Ticker | Score bar | Action pill ──
        hdr1, hdr2, hdr3 = st.columns([3, 2.5, 1.2])

        with hdr1:
            # Ticker + name + daily change
            change_val = r.get("daily_change_pct")
            change_color = "#10b981" if change_val and change_val >= 0 else "#ef4444"
            change_txt = f"{change_val:+.2f}%" if change_val is not None else ""
            st.markdown(
                f"**{r['ticker']}** &nbsp;·&nbsp; {r['name']} &nbsp;"
                f'<span style="color:{change_color};font-weight:600;font-size:0.85rem">{change_txt}</span>',
                unsafe_allow_html=True,
            )

        with hdr2:
            _er90 = r.get("expected_return_90d")
            _er90_txt = f" · 90d: {_er90 * 100:+.1f}%" if _er90 is not None else ""
            st.markdown(f"**Score: {r['aggregate_score']:+.3f}**{_er90_txt}")
            st.markdown(_render_score_bar(r["aggregate_score"]), unsafe_allow_html=True)

        with hdr3:
            st.markdown(
                f'<div style="text-align:center;padding-top:4px">{_render_action_pill(action)}</div>',
                unsafe_allow_html=True,
            )

        # ── Body row: Price | P&L | Targets | Pillars ──
        b1, b2, b3, b4 = st.columns([1.2, 1.5, 1.3, 1.5])

        with b1:
            price_str = _format_price(_cp, currency)
            st.metric("Price", price_str)
            if _ap:
                st.caption(f"Avg buy: {_format_price(_ap, currency)}")

        with b2:
            if _cp is not None and _ap is not None and _ap > 0:
                _pl = _cp - _ap
                _pl_pct = (_pl / _ap) * 100
                _pl_str = _format_price(abs(_pl), currency)
                if _pl < 0:
                    _pl_str = f"-{_pl_str}"
                factor = 0.01 if currency == "GBX" else 1.0
                _total_pl = _pl * _qty * factor
                _sym = "£" if currency in ("GBP", "GBX") else "$" if currency == "USD" else "€"
                _total_str = f"{_sym}{abs(_total_pl):,.0f}"
                if _total_pl < 0:
                    _total_str = f"-{_total_str}"
                st.metric("P&L / Share", _pl_str, f"{_pl_pct:+.1f}%")
                st.metric("Total P&L", _total_str, f"{_pl_pct:+.1f}%",
                    help=f"{_qty} shares × {_format_price(abs(_pl), currency)} per share")
            else:
                st.metric("P&L / Share", "N/A")

        with b3:
            tp_str = _format_price(r["take_profit"], currency)
            sl_str = _format_price(r["stop_loss"], currency)
            st.metric("Target ↑", tp_str, help=f"Method: {r['target_method']}")
            st.metric("Stop ↓", sl_str, help=f"Method: {r['stop_method']}")

        with b4:
            st.markdown("**Pillar Scores**")
            st.markdown(
                _render_pillar_bars(
                    r["technical_score"],
                    r["fundamental_score"],
                    r["sentiment_score"],
                    r.get("forecast_score", 0),
                ),
                unsafe_allow_html=True,
            )

        # ── Risk overlay flags ──
        _risk_flags = []
        if r.get("is_parabolic"):
            _risk_flags.append(
                f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:600">PARABOLIC (penalty {r.get("parabolic_penalty", 0):.2f})</span>'
            )
        if r.get("earnings_imminent"):
            _risk_flags.append(
                f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:600">EARNINGS IN {r.get("earnings_proximity_days", "?")}d</span>'
            )
        elif r.get("earnings_near"):
            _risk_flags.append(
                f'<span style="background:#f0f9ff;color:#1e40af;padding:2px 8px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:600">EARNINGS IN {r.get("earnings_proximity_days", "?")}d</span>'
            )
        if r.get("cap_tier") in ("small", "micro"):
            _tier_label = r["cap_tier"].upper()
            _risk_flags.append(
                f'<span style="background:#f5f3ff;color:#6d28d9;padding:2px 8px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:600">{_tier_label} CAP (max wt {r.get("max_weight_scale", 1.0):.0%})</span>'
            )
        if _risk_flags:
            st.markdown(" ".join(_risk_flags), unsafe_allow_html=True)

        # ── Why row ──
        st.caption(f"**Why:** {r['why']}")

        # ── Tabbed details ──
        with st.expander("Details"):
            tab_scores, tab_fund, tab_sent, tab_fcast = st.tabs([
                "📊 Scores", "📈 Fundamentals", "📰 Sentiment", "🔮 Forecast"
            ])

            # ─── Tab 1: Scores ───
            with tab_scores:
                sc1, sc2 = st.columns([1, 1])

                with sc1:
                    # 4 pillar metrics
                    st.metric("Technical", f"{r['technical_score']:+.2f}",
                        help="SMA, RSI, MACD, Bollinger Bands, Stochastic RSI, OBV, ADX, Williams %R")
                    st.metric("Fundamental", f"{r['fundamental_score']:+.2f}",
                        help="P/E, EPS growth, D/E, margins, ROE, FCF, analyst target, short interest, insider activity")
                    st.metric("Sentiment", f"{r['sentiment_score']:+.2f}",
                        help="VADER NLP on Google News + Reddit + FMP News")
                    st.metric("Forecast", f"{r.get('forecast_score', 0):+.2f}",
                        help="MoE 5-day price forecast (7 experts)")

                with sc2:
                    # Radar chart for 4 pillars
                    pillars = ["Technical", "Fundamental", "Sentiment", "Forecast"]
                    raw_vals = [r["technical_score"], r["fundamental_score"],
                                r["sentiment_score"], r.get("forecast_score", 0)]
                    # Map -1..+1 to 0..1 for radar
                    radar_vals = [(v + 1) / 2 for v in raw_vals]
                    radar_vals.append(radar_vals[0])  # Close the polygon
                    pillars_closed = pillars + [pillars[0]]

                    fill_color = _ACTION_COLORS.get(action, "#3b82f6")
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_vals,
                        theta=pillars_closed,
                        fill="toself",
                        fillcolor=f"rgba{tuple(list(int(fill_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.2])}",
                        line=dict(color=fill_color, width=2),
                        name=r["ticker"],
                    ))
                    fig_radar.update_layout(
                        **_PLOTLY_LAYOUT,
                        height=250,
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(128,128,128,0.2)"),
                            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

                # RSI gauge
                if r.get("rsi") is not None:
                    st.markdown(f"**RSI** — {r['rsi']:.1f}")
                    st.markdown(_render_rsi_gauge(r["rsi"]), unsafe_allow_html=True)

                # Technical indicators grid
                tech_metrics = []
                if r.get("bb_pct") is not None:
                    tech_metrics.append(("BB%", f"{r['bb_pct']:.0%}"))
                if r.get("stoch_k") is not None:
                    tech_metrics.append(("StochRSI", f"{r['stoch_k']:.0%}"))
                if r.get("obv_divergence"):
                    tech_metrics.append(("OBV", f"{r['obv_divergence']} div"))
                elif r.get("obv_trend"):
                    tech_metrics.append(("OBV", r["obv_trend"]))
                if r.get("adx") is not None:
                    tech_metrics.append(("ADX", f"{r['adx']:.0f}"))
                if r.get("williams_r") is not None:
                    tech_metrics.append(("W%R", f"{r['williams_r']:.0f}"))

                if tech_metrics:
                    cols_per_row = 3
                    for i in range(0, len(tech_metrics), cols_per_row):
                        chunk = tech_metrics[i:i + cols_per_row]
                        tcols = st.columns(cols_per_row)
                        for j, (lbl, val) in enumerate(chunk):
                            tcols[j].markdown(
                                _render_metric_card(lbl, val),
                                unsafe_allow_html=True,
                            )

            # ─── Tab 2: Fundamentals ───
            with tab_fund:
                # Quality metrics grid
                fund_metrics = []
                if r.get("pe_ratio"):
                    fund_metrics.append(("P/E", f"{r['pe_ratio']:.1f}"))
                if r.get("revenue_growth") is not None:
                    fund_metrics.append(("Rev Growth", f"{r['revenue_growth']:.0%}"))
                if r.get("profit_margin") is not None:
                    fund_metrics.append(("Margin", f"{r['profit_margin']:.0%}"))
                if r.get("roe") is not None:
                    fund_metrics.append(("ROE", f"{r['roe']:.0%}"))
                if r.get("fcf_yield") is not None:
                    fund_metrics.append(("FCF Yield", f"{r['fcf_yield']:.1%}"))
                if r.get("short_pct") is not None:
                    fund_metrics.append(("Short Interest", f"{r['short_pct']:.1%}"))
                if r.get("inst_ownership") is not None:
                    fund_metrics.append(("Inst. Ownership", f"{r['inst_ownership']:.0%}"))

                if fund_metrics:
                    cols_per_row = 4
                    for i in range(0, len(fund_metrics), cols_per_row):
                        chunk = fund_metrics[i:i + cols_per_row]
                        fcols = st.columns(cols_per_row)
                        for j, (lbl, val) in enumerate(chunk):
                            fcols[j].markdown(
                                _render_metric_card(lbl, val),
                                unsafe_allow_html=True,
                            )

                # Analyst consensus
                if r.get("analyst_target"):
                    rec_str = r.get("analyst_rec", "").replace("_", " ") if r.get("analyst_rec") else ""
                    upside = r.get("analyst_upside")
                    upside_str = f"({upside:+.0f}%)" if upside is not None else ""
                    analysts = f" · {r['num_analysts']} analysts" if r.get("num_analysts") else ""
                    st.markdown(
                        f"**Analyst Target:** {_format_price(r['analyst_target'], currency)} "
                        f"{upside_str} · {rec_str}{analysts}"
                    )

                # Insider activity
                if r.get("insider_net") and r["insider_net"] != "N/A":
                    st.markdown(
                        f"**Insider Activity:** {r['insider_net']} "
                        f"({r.get('insider_buys', 0)} buys / {r.get('insider_sells', 0)} sells)"
                    )

                # Insider transactions
                if r.get("insider_transactions"):
                    st.markdown("**Recent Insider Transactions**")
                    for txn in r["insider_transactions"][:5]:
                        emoji = "🟢" if txn["type"] == "Buy" else "🔴" if txn["type"] == "Sell" else "⚪"
                        shares_str = f"{txn['shares']:,}" if txn["shares"] else "?"
                        st.markdown(f"- {emoji} {txn['insider']}: {txn['type']} {shares_str} shares ({txn['date']})")

                # FMP Insights
                if r.get("fmp_available"):
                    st.markdown("---")
                    st.markdown("**FMP Insights**")
                    fmp_metrics = []
                    if r.get("peg_ratio") is not None:
                        fmp_metrics.append(("PEG", f"{r['peg_ratio']:.1f}"))
                    if r.get("earnings_beat_rate"):
                        fmp_metrics.append(("Beat Rate", r["earnings_beat_rate"]))
                    if r.get("quarterly_trend"):
                        fmp_metrics.append(("Trend", r["quarterly_trend"]))
                    if r.get("estimate_revision"):
                        fmp_metrics.append(("Revisions", r["estimate_revision"]))
                    if r.get("pe_vs_sector"):
                        fmp_metrics.append(("P/E vs Sector", r["pe_vs_sector"]))
                    if r.get("next_earnings_date"):
                        fmp_metrics.append(("Next Earnings", r["next_earnings_date"]))

                    if fmp_metrics:
                        cols_per_row = 3
                        for i in range(0, len(fmp_metrics), cols_per_row):
                            chunk = fmp_metrics[i:i + cols_per_row]
                            fmp_cols = st.columns(cols_per_row)
                            for j, (lbl, val) in enumerate(chunk):
                                fmp_cols[j].markdown(
                                    _render_metric_card(lbl, val),
                                    unsafe_allow_html=True,
                                )

                    if r.get("recent_upgrades", 0) > 0 or r.get("recent_downgrades", 0) > 0:
                        st.markdown(
                            f"Upgrades: **{r['recent_upgrades']}** · "
                            f"Downgrades: **{r['recent_downgrades']}** (90 days)"
                        )

                # Position info
                st.markdown("---")
                st.caption(
                    f"Avg Buy: {_format_price(r['avg_buy_price'], currency)} · "
                    f"Qty: {r['quantity']} · "
                    f"Stop: {r['stop_method']} · Target: {r['target_method']}"
                )

            # ─── Tab 3: Sentiment ───
            with tab_sent:
                # Sentiment scores overview
                sent_cols = st.columns(3)
                if r.get("news_score") is not None:
                    sent_cols[0].metric("News", f"{r['news_score']:+.2f}")
                if r.get("reddit_score") is not None:
                    sent_cols[1].metric("Reddit", f"{r['reddit_score']:+.2f}")
                if r.get("fmp_news_score") is not None:
                    sent_cols[2].metric("FMP News", f"{r['fmp_news_score']:+.2f}")

                # News headlines
                if r.get("news_headlines"):
                    st.markdown("**News Headlines**")
                    news_html = ""
                    for h in r["news_headlines"]:
                        news_html += _render_news_card(h["title"], h["sentiment"])
                    st.markdown(news_html, unsafe_allow_html=True)

                # Reddit
                if r.get("reddit_headlines"):
                    st.markdown("**Reddit**")
                    reddit_html = ""
                    for h in r["reddit_headlines"]:
                        sub = f"r/{_html.escape(str(h.get('subreddit', '?')))}" if h.get("subreddit") else ""
                        ups = f" [{int(h.get('upvotes', 0))}pts]" if h.get("upvotes") else ""
                        reddit_html += _render_news_card(h["title"], h["sentiment"], f"<small>{sub}{ups}</small>")
                    st.markdown(reddit_html, unsafe_allow_html=True)

                # FMP News
                if r.get("fmp_headlines"):
                    st.markdown("**FMP Stock News**")
                    fmp_html = ""
                    for h in r["fmp_headlines"]:
                        fmp_html += _render_news_card(h["title"], h["sentiment"])
                    st.markdown(fmp_html, unsafe_allow_html=True)

            # ─── Tab 4: Forecast ───
            with tab_fcast:
                if r.get("forecast_price") is not None:
                    horizon = r.get("forecast_horizon", 5)
                    st.markdown(f"**MoE Price Forecast ({horizon}-day)**")

                    fc_cols = st.columns(4)
                    fc_cols[0].metric("Predicted", _format_price(r["forecast_price"], currency),
                                     f"{r.get('forecast_pct_change', 0):+.1f}%")
                    fc_cols[1].metric("Low (80%)", _format_price(r["forecast_low"], currency))
                    fc_cols[2].metric("High (80%)", _format_price(r["forecast_high"], currency))
                    if r.get("forecast_ensemble_mae") is not None:
                        fc_cols[3].metric("Ensemble MAE", f"{r['forecast_ensemble_mae']:.2f}")
                    else:
                        fc_cols[3].metric("Ensemble MAE", "Building...")

                    # Expert weights as horizontal bar chart
                    if r.get("forecast_experts") and r.get("forecast_expert_weights"):
                        expert_names = []
                        expert_weights = []
                        expert_preds = []
                        for e in r["forecast_experts"]:
                            name = e["name"].replace("_", " ").title()
                            w = r["forecast_expert_weights"].get(e["name"], 0)
                            expert_names.append(name)
                            expert_weights.append(w * 100)
                            expert_preds.append(e["price"])

                        fig_expert = go.Figure(go.Bar(
                            x=expert_weights,
                            y=expert_names,
                            orientation="h",
                            marker=dict(
                                color=expert_weights,
                                colorscale=[[0, "#6b7280"], [1, "#3b82f6"]],
                            ),
                            text=[f"{w:.1f}%" for w in expert_weights],
                            textposition="auto",
                            hovertemplate="%{y}: %{x:.1f}% weight<extra></extra>",
                        ))
                        fig_expert.update_layout(
                            **_PLOTLY_LAYOUT,
                            height=220,
                            xaxis=dict(title="Weight %", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
                            yaxis=dict(autorange="reversed"),
                            title=dict(text="Expert Weight Allocation", font=dict(size=13)),
                        )
                        st.plotly_chart(fig_expert, use_container_width=True, config={"displayModeBar": False})

                    # Expert table
                    if r.get("forecast_experts"):
                        expert_rows = []
                        for e in r["forecast_experts"]:
                            weight = r.get("forecast_expert_weights", {}).get(e["name"], 0)
                            mae_val = r.get("forecast_expert_maes", {}).get(e["name"])
                            expert_rows.append({
                                "Expert": e["name"].replace("_", " ").title(),
                                "Prediction": round(e["price"], 2),
                                "Low": round(e["low"], 2),
                                "High": round(e["high"], 2),
                                "Weight": f"{weight:.1%}",
                                "MAE": f"{mae_val:.2f}" if mae_val is not None else "—",
                            })
                        st.dataframe(pd.DataFrame(expert_rows), hide_index=True, use_container_width=True)

                    # Long-horizon forecast (if available)
                    if r.get("forecast_price_long") is not None:
                        st.divider()
                        horizon_long = r.get("forecast_horizon_long", 63)
                        st.markdown(f"**Long-Term Forecast ({horizon_long}-day)**")
                        lc = st.columns(4)
                        lc[0].metric("Predicted", _format_price(r["forecast_price_long"], currency),
                                     f"{r.get('forecast_pct_change_long', 0):+.1f}%")
                        lc[1].metric("Low (80%)", _format_price(r.get("forecast_low_long", 0), currency))
                        lc[2].metric("High (80%)", _format_price(r.get("forecast_high_long", 0), currency))
                        if r.get("forecast_ensemble_mae_long") is not None:
                            lc[3].metric("Ensemble MAE", f"{r['forecast_ensemble_mae_long']:.2f}")
                        else:
                            lc[3].metric("Ensemble MAE", "Building...")
                else:
                    st.info("Forecast data not available for this holding.")


# ---------------------------------------------------------------------------
# 90-Day Portfolio Return Projection
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 90-Day Return Projection")
st.caption(
    "Monte Carlo simulation (5,000 paths) combining MoE directional forecasts with "
    "historical volatility and cross-asset correlations. Shows the distribution of "
    "portfolio returns over the next ~90 calendar days (63 trading days)."
)

_run_projection = st.button("Run 90-Day Projection (Monte Carlo)", type="secondary")
if _run_projection:
    from engine.portfolio_projection import project_portfolio_return, project_swap_impact

    with st.spinner("Running Monte Carlo simulation (5,000 paths × 63 days)..."):
        _proj = project_portfolio_return(results, holdings, position_weights)

    # Portfolio-level summary
    st.markdown("#### Portfolio Return Distribution")
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Expected Return", f"{_proj.expected_return_pct:+.1f}%")
    pc2.metric("P(Positive)", f"{_proj.prob_positive:.0%}")
    pc3.metric("Current Value", f"£{_proj.current_value:,.0f}")
    pc4.metric("Expected Value", f"£{_proj.expected_value:,.0f}")
    pc5.metric("Expected Gain", f"£{_proj.expected_value - _proj.current_value:+,.0f}")

    # Confidence interval table
    st.markdown("#### Confidence Intervals")
    ci_data = []
    for pctile, label in [(0.10, "Bear (10th)"), (0.25, "Cautious (25th)"),
                           (0.50, "Median (50th)"), (0.75, "Optimistic (75th)"),
                           (0.90, "Bull (90th)")]:
        ci_data.append({
            "Scenario": label,
            "Portfolio Return": f"{_proj.projected_returns[pctile]:+.1f}%",
            "Portfolio Value": f"£{_proj.projected_values[pctile]:,.0f}",
            "Gain / Loss": f"£{_proj.projected_values[pctile] - _proj.current_value:+,.0f}",
        })
    st.dataframe(pd.DataFrame(ci_data), hide_index=True, use_container_width=True)

    # Per-ticker breakdown
    st.markdown("#### Per-Ticker Projections")
    ticker_rows = []
    for tp in sorted(_proj.ticker_projections, key=lambda x: x.expected_return_pct, reverse=True):
        ticker_rows.append({
            "Ticker": tp.ticker,
            "Current": f"{tp.current_price:.2f}",
            "MoE Forecast": f"{tp.moe_predicted_price:.2f}",
            "MoE Return": f"{tp.moe_pct_change:+.1f}%",
            "MC Expected": f"{tp.expected_return_pct:+.1f}%",
            "10th pct": f"{tp.projected_returns[0.10]:+.1f}%",
            "50th pct": f"{tp.projected_returns[0.50]:+.1f}%",
            "90th pct": f"{tp.projected_returns[0.90]:+.1f}%",
            "P(>0)": f"{tp.prob_positive:.0%}",
            "Annual Vol": f"{tp.annual_volatility:.0%}",
        })
    st.dataframe(pd.DataFrame(ticker_rows), hide_index=True, use_container_width=True)

    # Swap impact analysis — uses cached discovery candidates from orchestrator state
    st.markdown("#### Swap Impact Analysis")
    st.caption("Compare projected returns before and after a proposed swap.")

    _state_path = Path(__file__).parent / config.ORCHESTRATOR_STATE_FILE
    _cached_candidates = []
    if _state_path.exists():
        try:
            with open(_state_path, "r") as f:
                _orch_state = json.load(f)
            _cached_candidates = _orch_state.get("cached_discovery", [])
        except Exception:
            pass

    if _cached_candidates:
        # Find weakest holdings as swap-out candidates
        _sorted_results = sorted(results, key=lambda r: r.get("aggregate_score", 0))
        _swap_out_options = [f"{r['ticker']} (score: {r.get('aggregate_score', 0):.3f})" for r in _sorted_results]
        _swap_in_options = [f"{c['ticker']} (score: {c.get('aggregate_score', 0):.3f})" for c in _cached_candidates[:10]]

        sc1, sc2 = st.columns(2)
        with sc1:
            _swap_out_sel = st.selectbox("Sell (swap out)", _swap_out_options, index=0)
        with sc2:
            _swap_in_sel = st.selectbox("Buy (swap in)", _swap_in_options, index=0)

        if st.button("Compare Swap Impact"):
            _swap_out_ticker = _swap_out_sel.split(" (")[0]
            _swap_in_ticker = _swap_in_sel.split(" (")[0]

            with st.spinner(f"Simulating swap: {_swap_out_ticker} → {_swap_in_ticker}..."):
                _proj_before, _proj_after = project_swap_impact(
                    results, holdings, _swap_out_ticker, _swap_in_ticker, position_weights,
                )

            # Side-by-side comparison
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**Current Portfolio**")
                st.metric("Expected Return", f"{_proj_before.expected_return_pct:+.1f}%")
                st.metric("P(Positive)", f"{_proj_before.prob_positive:.0%}")
                st.metric("Expected Value", f"£{_proj_before.expected_value:,.0f}")
            with bc2:
                st.markdown(f"**After Swap ({_swap_out_ticker} → {_swap_in_ticker})**")
                delta_ret = _proj_after.expected_return_pct - _proj_before.expected_return_pct
                st.metric("Expected Return", f"{_proj_after.expected_return_pct:+.1f}%",
                          delta=f"{delta_ret:+.1f}%")
                delta_prob = _proj_after.prob_positive - _proj_before.prob_positive
                st.metric("P(Positive)", f"{_proj_after.prob_positive:.0%}",
                          delta=f"{delta_prob:+.0%}")
                delta_val = _proj_after.expected_value - _proj_before.expected_value
                st.metric("Expected Value", f"£{_proj_after.expected_value:,.0f}",
                          delta=f"£{delta_val:+,.0f}")
    else:
        st.info("No discovery candidates cached. Run the Global Discovery Engine first to enable swap impact analysis.")


# ---------------------------------------------------------------------------
# Signal Analytics
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Signal Analytics")

_store_path = Path(__file__).parent / "forecast_store.json"
_store = {}
if _store_path.exists():
    try:
        with open(_store_path, "r") as f:
            _store = json.load(f)
    except json.JSONDecodeError:
        st.warning("forecast_store.json is corrupted — run `python fix_forecast_store.py` to repair it.")

_rolling_maes = _store.get("rolling_maes", {})

if _rolling_maes:
    tab_accuracy, tab_weights, tab_impact, tab_backtest, tab_performance = st.tabs([
        "📉 Expert Accuracy", "⚖️ Weight Distribution", "🎯 Signal Impact",
        "🔧 Weight Optimization", "📊 Forecast Performance",
    ])

    _all_experts = set()
    for ticker_data in _rolling_maes.values():
        _all_experts.update(k for k in ticker_data.keys() if k != "ensemble")
    _all_experts = sorted(_all_experts)

    # ── Tab 1: Expert Accuracy (MAE) ──
    with tab_accuracy:
        st.caption("Lower MAE = more accurate predictions. Experts with lower MAE get higher weight.")

        mae_data = {}
        for ticker, experts in _rolling_maes.items():
            ticker_short = ticker.split(".")[0]
            col_data = {}
            for expert_name in _all_experts:
                mae_list = experts.get(expert_name, [])
                if mae_list:
                    col_data[expert_name.replace("_", " ").title()] = round(sum(mae_list) / len(mae_list), 2)
            if col_data:
                mae_data[ticker_short] = col_data

        if mae_data:
            mae_df = pd.DataFrame(mae_data)
            mae_df.index.name = "Expert"

            ticker_select = st.selectbox("Select holding:", list(mae_data.keys()), key="mae_ticker")

            if ticker_select:
                ticker_mae = mae_df[ticker_select].dropna().sort_values()

                fig_mae = px.bar(
                    x=ticker_mae.values, y=ticker_mae.index,
                    orientation="h",
                    color=ticker_mae.values,
                    color_continuous_scale=[[0, "#10b981"], [0.5, "#fbbf24"], [1, "#ef4444"]],
                )
                fig_mae.update_layout(
                    **_PLOTLY_LAYOUT,
                    height=300,
                    xaxis_title="MAE",
                    yaxis_title="",
                    coloraxis_showscale=False,
                    title=dict(text=f"Expert MAE — {ticker_select}", font=dict(size=14)),
                )
                fig_mae.update_traces(hovertemplate="%{y}: MAE %{x:.2f}<extra></extra>")
                st.plotly_chart(fig_mae, use_container_width=True, config={"displayModeBar": False})

                # Ensemble MAE
                for full_ticker, experts in _rolling_maes.items():
                    if full_ticker.split(".")[0] == ticker_select:
                        ens_list = experts.get("ensemble", [])
                        if ens_list:
                            st.metric(f"Ensemble MAE ({ticker_select})", f"{sum(ens_list)/len(ens_list):.2f}")
                        break

            with st.expander("Full MAE comparison table"):
                st.dataframe(mae_df, use_container_width=True)

    # ── Tab 2: Weight Distribution ──
    with tab_weights:
        st.caption("How the gating network distributes weight across experts per holding.")

        weight_data = {}
        for r in results:
            weights = r.get("forecast_expert_weights", {})
            if weights:
                ticker_short = r["ticker"].split(".")[0]
                weight_data[ticker_short] = {
                    k.replace("_", " ").title(): round(v * 100, 1)
                    for k, v in weights.items()
                }

        if weight_data:
            weight_df = pd.DataFrame(weight_data)
            weight_df.index.name = "Expert"

            ticker_select_w = st.selectbox("Select holding:", list(weight_data.keys()), key="weight_ticker")

            if ticker_select_w:
                w_series = weight_df[ticker_select_w].dropna().sort_values(ascending=False)

                # Bar chart
                fig_w = px.bar(
                    x=w_series.index, y=w_series.values,
                    color=w_series.values,
                    color_continuous_scale=[[0, "#6b7280"], [1, "#3b82f6"]],
                )
                fig_w.update_layout(
                    **_PLOTLY_LAYOUT, height=300,
                    xaxis_title="", yaxis_title="Weight %",
                    coloraxis_showscale=False,
                    title=dict(text=f"Expert Weights — {ticker_select_w}", font=dict(size=14)),
                )
                st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})

                # Radar chart
                radar_experts = list(w_series.index)
                radar_weights = list(w_series.values)
                radar_weights.append(radar_weights[0])
                radar_experts_closed = radar_experts + [radar_experts[0]]

                fig_radar_w = go.Figure(go.Scatterpolar(
                    r=radar_weights,
                    theta=radar_experts_closed,
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.15)",
                    line=dict(color="#3b82f6", width=2),
                ))
                fig_radar_w.update_layout(
                    **_PLOTLY_LAYOUT, height=300,
                    polar=dict(
                        radialaxis=dict(visible=True, gridcolor="rgba(128,128,128,0.2)"),
                        angularaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False,
                    title=dict(text=f"Weight Profile — {ticker_select_w}", font=dict(size=14)),
                )
                st.plotly_chart(fig_radar_w, use_container_width=True, config={"displayModeBar": False})

            with st.expander("Full weight distribution table (%)"):
                st.dataframe(weight_df, use_container_width=True)

    # ── Tab 3: Signal Impact Analysis ──
    with tab_impact:
        st.caption(
            "Compares each expert's MAE against the ensemble. "
            "**Negative = expert improves accuracy** · **Positive = expert hurts accuracy**"
        )

        impact_data = {}
        for ticker, experts in _rolling_maes.items():
            ticker_short = ticker.split(".")[0]
            ens_list = experts.get("ensemble", [])
            if not ens_list:
                continue
            ens_mae = sum(ens_list) / len(ens_list)
            col = {}
            for expert_name in _all_experts:
                mae_list = experts.get(expert_name, [])
                if mae_list:
                    expert_mae = sum(mae_list) / len(mae_list)
                    col[expert_name.replace("_", " ").title()] = round(expert_mae - ens_mae, 2)
            if col:
                impact_data[ticker_short] = col

        if impact_data:
            impact_df = pd.DataFrame(impact_data)

            ticker_select_i = st.selectbox("Select holding:", list(impact_data.keys()), key="impact_ticker")

            if ticker_select_i:
                impact_series = impact_df[ticker_select_i].dropna().sort_values()

                # Diverging bar chart
                colors = ["#10b981" if v < 0 else "#ef4444" for v in impact_series.values]
                fig_impact = go.Figure(go.Bar(
                    x=impact_series.values,
                    y=impact_series.index,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertemplate="%{y}: %{x:+.2f} vs ensemble<extra></extra>",
                ))
                fig_impact.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,0.5)")
                fig_impact.update_layout(
                    **_PLOTLY_LAYOUT, height=300,
                    xaxis_title="MAE vs Ensemble",
                    title=dict(text=f"Signal Impact — {ticker_select_i}", font=dict(size=14)),
                )
                st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})

                best_expert = impact_series.idxmin()
                worst_expert = impact_series.idxmax()
                bcol, wcol = st.columns(2)
                bcol.metric("Most Accurate", best_expert, f"{impact_series[best_expert]:+.2f} vs ensemble")
                wcol.metric("Least Accurate", worst_expert, f"{impact_series[worst_expert]:+.2f} vs ensemble", delta_color="inverse")

            with st.expander("Full signal impact table"):
                st.dataframe(impact_df, use_container_width=True)

            st.markdown("---")
            st.markdown("**Cross-Portfolio Expert Ranking**")
            avg_impact = impact_df.mean(axis=1).sort_values()
            summary_df = pd.DataFrame({
                "Expert": avg_impact.index,
                "Avg MAE Delta": [f"{v:+.2f}" for v in avg_impact.values],
                "Verdict": [
                    "✅ Improves" if v < -0.5
                    else "⚠️ Neutral" if abs(v) <= 0.5
                    else "❌ Hurts"
                    for v in avg_impact.values
                ],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ── Tab 4: Weight Optimization ──
    with tab_backtest:
        st.markdown(
            "**Multi-method weight optimization** — IC-based weighting with shrinkage + "
            "constrained grid search across ~40 stocks."
        )
        st.caption(
            f"Shrinkage: {config.WEIGHT_SHRINKAGE:.0%} toward equal · "
            f"Min floor: {config.WEIGHT_MIN_FLOOR:.0%} per pillar"
        )

        if st.button("🚀 Run Optimization", key="run_backtest", type="secondary"):
            progress_bar = st.progress(0, text="Starting...")

            def _progress(current, total, ticker):
                pct = min(current / max(total, 1), 1.0)
                progress_bar.progress(pct, text=f"{ticker} ({current + 1}/{total})")

            opt_result = optimize_weights(progress_callback=_progress)
            progress_bar.progress(1.0, text="Done!")
            st.session_state["opt_result"] = opt_result

        if "opt_result" in st.session_state:
            opt = st.session_state["opt_result"]

            if opt.universe_size >= 10:
                # Grouped bar chart — weight comparison
                pillar_names = ["Technical", "Fundamental", "Sentiment", "Forecast"]
                pillar_keys = ["technical", "fundamental", "sentiment", "forecast"]

                fig_wcomp = go.Figure()
                for label, weights_dict, color in [
                    ("Current", dict(config.WEIGHTS), "#6b7280"),
                    ("IC-Based", opt.ic_based_weights, "#f59e0b"),
                    ("Grid Search", opt.grid_search_weights, "#8b5cf6"),
                    ("Recommended", opt.recommended_weights, "#3b82f6"),
                ]:
                    fig_wcomp.add_trace(go.Bar(
                        name=label,
                        x=pillar_names,
                        y=[weights_dict[k] * 100 for k in pillar_keys],
                        marker_color=color,
                        text=[f"{weights_dict[k]*100:.1f}%" for k in pillar_keys],
                        textposition="auto",
                    ))
                fig_wcomp.update_layout(
                    **_PLOTLY_LAYOUT,
                    height=320,
                    barmode="group",
                    yaxis_title="Weight %",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    title=dict(text="Weight Comparison", font=dict(size=14)),
                )
                st.plotly_chart(fig_wcomp, use_container_width=True, config={"displayModeBar": False})

                # Weight table
                weight_comparison = pd.DataFrame({
                    "Pillar": pillar_names,
                    "Current": [config.WEIGHTS[k] for k in pillar_keys],
                    "IC-Based": [opt.ic_based_weights[k] for k in pillar_keys],
                    "Grid Search": [opt.grid_search_weights[k] for k in pillar_keys],
                    "Recommended": [opt.recommended_weights[k] for k in pillar_keys],
                })
                weight_comparison["Change"] = weight_comparison["Recommended"] - weight_comparison["Current"]
                st.dataframe(
                    weight_comparison.style.format({
                        "Current": "{:.1%}", "IC-Based": "{:.1%}",
                        "Grid Search": "{:.1%}", "Recommended": "{:.1%}",
                        "Change": "{:+.1%}",
                    }),
                    hide_index=True, use_container_width=True,
                )

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Universe", f"{opt.universe_size} stocks")
                m2.metric("Shrinkage", f"{opt.shrinkage_factor:.0%}")
                m3.metric("Current Fitness", f"{opt.fitness_current:.3f}")
                m4.metric("Recommended Fitness", f"{opt.fitness_recommended:.3f}",
                          delta=f"{opt.fitness_recommended - opt.fitness_current:+.3f}")

                # Pillar ICs
                st.markdown("#### Pillar Information Coefficients (IC)")
                if opt.pillar_ics:
                    ic_rows = []
                    for pillar in pillar_keys:
                        pic = opt.pillar_ics.get(pillar)
                        if pic:
                            ic_rows.append({
                                "Pillar": pillar.title(),
                                "Avg IC": f"{pic.avg_ic:+.3f}",
                                "IC Std": f"{pic.ic_std:.3f}" if pic.ic_std > 0 else "—",
                                "IC IR": f"{pic.ic_ir:+.2f}" if pic.ic_ir != 0 else "—",
                                "Snapshots": pic.num_snapshots,
                                "Method": pic.method,
                                "Signal": (
                                    "🟢 Strong" if abs(pic.avg_ic) > 0.2
                                    else "🟡 Moderate" if abs(pic.avg_ic) > 0.1
                                    else "🔴 Weak"
                                ),
                            })
                    st.dataframe(pd.DataFrame(ic_rows), hide_index=True, use_container_width=True)

                # Expandable sections
                with st.expander("Per-Stock Score Breakdown"):
                    stock_rows = []
                    for s in sorted(opt.current_snapshot_scores, key=lambda x: x.forward_return_pct, reverse=True):
                        agg = (
                            s.technical_score * opt.recommended_weights["technical"]
                            + s.fundamental_score * opt.recommended_weights["fundamental"]
                            + s.sentiment_score * opt.recommended_weights["sentiment"]
                            + s.forecast_score * opt.recommended_weights["forecast"]
                        )
                        stock_rows.append({
                            "Ticker": s.ticker,
                            "Tech": f"{s.technical_score:+.2f}",
                            "Fund": f"{s.fundamental_score:+.2f}",
                            "Sent": f"{s.sentiment_score:+.2f}",
                            "Fcast": f"{s.forecast_score:+.2f}",
                            "Aggregate": f"{agg:+.3f}",
                            "Action": _score_to_action(agg),
                            f"Actual {config.FORECAST_HORIZON_DAYS}d Return": f"{s.forward_return_pct:+.2f}%",
                        })
                    st.dataframe(pd.DataFrame(stock_rows), hide_index=True, use_container_width=True)

                with st.expander("Top 10 Grid Search Combinations"):
                    top_rows = []
                    for i, entry in enumerate(opt.weight_grid_top_n, 1):
                        w = entry["weights"]
                        top_rows.append({
                            "#": i,
                            "Technical": f"{w['technical']:.0%}",
                            "Fundamental": f"{w['fundamental']:.0%}",
                            "Sentiment": f"{w['sentiment']:.0%}",
                            "Forecast": f"{w['forecast']:.0%}",
                            "Fitness": f"{entry['fitness']:.4f}",
                        })
                    st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

                if opt.skipped_tickers:
                    with st.expander(f"Skipped tickers ({len(opt.skipped_tickers)})"):
                        st.write(", ".join(opt.skipped_tickers))

                st.info(
                    "**How it works:** IC-based weights measure each pillar's predictive power, "
                    "shrunk toward equal weights to prevent overfitting, with a minimum floor "
                    "to preserve signal diversity. Grid search cross-checks by finding the best "
                    "combo. Recommended = average of both methods. Re-run monthly."
                )
            else:
                st.warning(f"Only {opt.universe_size} stocks scored (need 10+). Check connectivity.")

    # ── Tab 5: Forecast Performance (OOS) ──
    with tab_performance:
        try:
            from engine.performance import get_forecast_performance
            perf = get_forecast_performance()

            if perf.get("sufficient_data"):
                st.caption(f"Based on {perf['total_predictions']} evaluated predictions.")

                # Key metrics row
                pm1, pm2, pm3, pm4 = st.columns(4)
                hit_pct = perf["hit_rate"] * 100
                hit_color = "normal" if hit_pct >= 50 else "inverse"
                pm1.metric("Hit Rate", f"{hit_pct:.1f}%",
                           delta=f"{'above' if hit_pct >= 50 else 'below'} 50%",
                           delta_color=hit_color)
                pm2.metric("Avg Error", f"{perf['avg_error_pct']:.1f}%")
                pm3.metric("RMSE", f"{perf['rmse']:.2f}")
                pm4.metric("Predictions", perf["total_predictions"])

                # Rolling accuracy chart
                if perf.get("rolling_accuracy") and len(perf["rolling_accuracy"]) >= 2:
                    roll_df = pd.DataFrame(perf["rolling_accuracy"])
                    fig_roll = go.Figure(go.Scatter(
                        x=roll_df["date"], y=[r * 100 for r in roll_df["hit_rate"]],
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=4),
                        hovertemplate="Date: %{x}<br>Hit Rate: %{y:.1f}%<extra></extra>",
                    ))
                    fig_roll.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                                       annotation_text="50% baseline")
                    fig_roll.update_layout(
                        **_PLOTLY_LAYOUT,
                        height=280,
                        yaxis=dict(title="Hit Rate %", range=[0, 100]),
                        xaxis=dict(title="Date"),
                        title=dict(text="Rolling Directional Accuracy (30-prediction window)",
                                   font=dict(size=13)),
                    )
                    st.plotly_chart(fig_roll, use_container_width=True, config={"displayModeBar": False})

                # Expert comparison table
                if perf.get("expert_comparison"):
                    st.markdown("**Expert Performance Comparison**")
                    expert_perf_rows = []
                    for name, stats in sorted(
                        perf["expert_comparison"].items(),
                        key=lambda x: x[1]["hit_rate"],
                        reverse=True,
                    ):
                        expert_perf_rows.append({
                            "Expert": name.replace("_", " ").title(),
                            "Hit Rate": f"{stats['hit_rate']:.1%}",
                            "Avg Error %": f"{stats['avg_error_pct']:.1f}%",
                            "RMSE": f"{stats['rmse']:.2f}",
                            "Predictions": stats["count"],
                        })
                    st.dataframe(pd.DataFrame(expert_perf_rows), hide_index=True,
                                 use_container_width=True)

                # Per-ticker breakdown
                if perf.get("per_ticker"):
                    st.markdown("**Per-Ticker Accuracy**")
                    ticker_perf_rows = []
                    for ticker, stats in sorted(
                        perf["per_ticker"].items(),
                        key=lambda x: x[1]["hit_rate"],
                        reverse=True,
                    ):
                        ticker_perf_rows.append({
                            "Ticker": ticker,
                            "Hit Rate": f"{stats['hit_rate']:.1%}",
                            "Avg Error %": f"{stats['avg_error_pct']:.1f}%",
                            "Predictions": stats["count"],
                        })
                    st.dataframe(pd.DataFrame(ticker_perf_rows), hide_index=True,
                                 use_container_width=True)
            else:
                st.info(
                    f"📊 Need at least 5 evaluated predictions for performance metrics. "
                    f"Currently: {perf['total_predictions']} predictions tracked."
                )
        except Exception as e:
            st.warning(f"Could not load forecast performance: {e}")

else:
    st.info("Signal analytics will appear after the first forecast run builds MAE history.")

# ---------------------------------------------------------------------------
# Global Discovery Engine
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 🔍 Global Discovery Engine")
st.caption(
    "Screens small/mid-cap stocks across LSE, XETRA, Euronext, TSX, NYSE & NASDAQ. "
    "Applies a multi-stage funnel: FMP screening → sector/beta filter → correlation filter "
    "→ fundamental quick-rank → full 4-pillar analysis → FX penalty + portfolio fit scoring."
)

# Auto-load cached discovery results from dashboard data on first visit
if "discovery_results" not in st.session_state:
    if True:
        try:
            _cached_disc = _dash.cached_discovery
            _last_disc_run = _dash.discovery_timestamp
            if _cached_disc:
                from engine.discovery import ScoredCandidate, DiscoveryResult
                # Reconstruct ScoredCandidate objects from cached dicts
                _restored = []
                for c in _cached_disc:
                    _restored.append(ScoredCandidate(
                        ticker=c.get("ticker", ""),
                        name=c.get("name", ""),
                        exchange=c.get("exchange", ""),
                        country=c.get("country", ""),
                        sector=c.get("sector", ""),
                        industry=c.get("industry", ""),
                        market_cap=c.get("market_cap", 0),
                        currency=c.get("currency", "USD"),
                        aggregate_score=c.get("aggregate_score", 0),
                        technical_score=c.get("technical_score", 0),
                        fundamental_score=c.get("fundamental_score", 0),
                        sentiment_score=c.get("sentiment_score", 0),
                        forecast_score=c.get("forecast_score", 0),
                        action=c.get("action", ""),
                        why=c.get("why", ""),
                        fx_penalty_applied=c.get("fx_penalty_applied", False),
                        fx_penalty_pct=c.get("fx_penalty_pct", 0),
                        max_correlation=c.get("max_correlation", 0),
                        correlated_with=c.get("correlated_with", ""),
                        sector_weight_if_added=c.get("sector_weight_if_added", 0),
                        portfolio_fit_score=c.get("portfolio_fit_score", 0),
                        momentum_score=c.get("momentum_score", 0),
                        return_90d=c.get("return_90d", 0),
                        return_30d=c.get("return_30d", 0),
                        volume_ratio=c.get("volume_ratio", 1.0),
                        expected_return_90d=c.get("expected_return_90d", 0),
                        parabolic_penalty=c.get("parabolic_penalty", 0),
                        is_parabolic=c.get("is_parabolic", False),
                        earnings_near=c.get("earnings_near", False),
                        earnings_imminent=c.get("earnings_imminent", False),
                        earnings_days=c.get("earnings_days"),
                        cap_tier=c.get("cap_tier", "unknown"),
                        confidence_discount=c.get("confidence_discount", 1.0),
                        max_weight_scale=c.get("max_weight_scale", 1.0),
                        final_rank=c.get("final_rank", c.get("aggregate_score", 0)),
                    ))
                st.session_state["discovery_results"] = DiscoveryResult(
                    candidates=_restored,
                    screened_count=0,
                    after_momentum_screen=0,
                    after_quick_filter=0,
                    after_corr_filter=0,
                    after_quick_rank=0,
                    fully_scored=len(_restored),
                )
                st.session_state["discovery_cached_from"] = _last_disc_run
        except Exception:
            pass

_disc_col1, _disc_col2 = st.columns([1, 3])
with _disc_col1:
    _run_discovery = st.button("🔎 Re-run Screener", type="primary", use_container_width=True)
with _disc_col2:
    _cached_ts = st.session_state.get("discovery_cached_from")
    if _cached_ts and "discovery_results" in st.session_state:
        st.info(f"Showing cached results · {format_freshness(_cached_ts)} · Click Re-run to refresh")
    else:
        st.info(
            f"Screens {len(config.DISCOVERY_EXCHANGES)} US exchanges + global universe · "
            f"Market cap ≥ £{config.DISCOVERY_MIN_MCAP / 1e6:.0f}M (no upper cap) · "
            f"Top {config.DISCOVERY_TOP_N_FULL_SCORE} fully scored · "
            f"~60-90 min runtime"
        )

if _run_discovery:
    from engine.discovery import run_discovery, DiscoveryResult

    _disc_progress = st.progress(0, text="Starting discovery...")
    _disc_status = st.empty()

    def _disc_progress_cb(message, current, total):
        pct = min(current / max(total, 1), 1.0) if total > 0 else 0
        _disc_progress.progress(pct, text=message)

    disc_result = run_discovery(
        holdings=holdings,
        risk_data=risk_data if "risk_data" in dir() else None,
        progress_callback=_disc_progress_cb,
    )
    _disc_progress.progress(1.0, text="Discovery complete!")
    st.session_state["discovery_results"] = disc_result
    st.session_state.pop("discovery_cached_from", None)

# Show results (cached from orchestrator or fresh from manual run)
if "discovery_results" in st.session_state:
    disc: "DiscoveryResult" = st.session_state["discovery_results"]

    if disc.error:
        st.warning(f"Discovery issue: {disc.error}")

    if disc.candidates:
        # Funnel summary
        momentum_count = getattr(disc, "after_momentum_screen", "?")
        st.markdown(
            f"**Funnel:** {disc.screened_count} screened → "
            f"{momentum_count} momentum → "
            f"{disc.after_quick_filter} filtered → "
            f"{disc.after_corr_filter} uncorrelated → "
            f"{disc.after_quick_rank} ranked → "
            f"{disc.fully_scored} scored · "
            f"⏱️ {disc.run_time_seconds:.0f}s"
        )

        # Top 3 recommendations
        top_3 = disc.candidates[:3]
        if top_3:
            st.markdown("#### Top Recommendations")
            rec_cols = st.columns(len(top_3))
            for idx, (col, cand) in enumerate(zip(rec_cols, top_3)):
                with col:
                    # Country flag approximation
                    country_flags = {
                        "US": "🇺🇸", "UK": "🇬🇧", "GB": "🇬🇧", "CA": "🇨🇦",
                        "DE": "🇩🇪", "FR": "🇫🇷", "IT": "🇮🇹", "ES": "🇪🇸",
                        "NL": "🇳🇱", "JP": "🇯🇵",
                    }
                    flag = country_flags.get(cand.country, "🌍")

                    # Action color
                    action_colors = {
                        "STRONG BUY": "🟢", "BUY": "🟡",
                        "NEUTRAL": "⚪", "AVOID": "🔴",
                    }
                    dot = action_colors.get(cand.action, "⚪")

                    st.markdown(f"**#{idx + 1} {flag} {cand.ticker}**")
                    st.markdown(f"*{cand.name}*")
                    st.metric("Final Rank", f"{cand.final_rank:.3f}", delta=f"{cand.action}")
                    st.caption(f"📍 {cand.exchange} · {cand.sector}")
                    st.caption(f"Market Cap: £{cand.market_cap / 1e9:.1f}B")

                    # Sub-scores
                    st.markdown(
                        f"Tech: `{cand.technical_score:.2f}` · "
                        f"Fund: `{cand.fundamental_score:.2f}` · "
                        f"Sent: `{cand.sentiment_score:.2f}` · "
                        f"Fcast: `{cand.forecast_score:.2f}`"
                    )

                    # Momentum metrics
                    if hasattr(cand, "momentum_score"):
                        ret_90 = getattr(cand, "return_90d", 0) * 100
                        ret_30 = getattr(cand, "return_30d", 0) * 100
                        vol_r = getattr(cand, "volume_ratio", 1.0)
                        st.caption(
                            f"📈 Momentum: {cand.momentum_score:.2f} · "
                            f"90d: {ret_90:+.1f}% · 30d: {ret_30:+.1f}% · "
                            f"Vol: {vol_r:.1f}x"
                        )

                    # Risk overlay flags
                    _cand_risk = []
                    if getattr(cand, "is_parabolic", False):
                        _cand_risk.append(f"⚠️ PARABOLIC (-{cand.parabolic_penalty:.2f})")
                    if getattr(cand, "earnings_imminent", False):
                        _cand_risk.append(f"⚠️ Earnings {cand.earnings_days}d")
                    elif getattr(cand, "earnings_near", False):
                        _cand_risk.append(f"📅 Earnings {cand.earnings_days}d")
                    _tier = getattr(cand, "cap_tier", "unknown")
                    if _tier in ("small", "micro"):
                        _cand_risk.append(f"🏷️ {_tier.upper()} cap")
                    if _cand_risk:
                        st.caption(" · ".join(_cand_risk))

                    # FX note
                    if cand.fx_penalty_applied:
                        st.caption(f"⚠️ FX penalty: -{cand.fx_penalty_pct:.1f}% ({cand.currency})")
                    else:
                        st.caption(f"✅ No FX fee (GBP-denominated)")

                    # Portfolio fit
                    fit_desc = []
                    if cand.max_correlation < 0.40:
                        fit_desc.append("low correlation to portfolio")
                    elif cand.max_correlation < 0.60:
                        fit_desc.append(f"moderate correlation ({cand.max_correlation:.2f})")
                    else:
                        fit_desc.append(f"correlation {cand.max_correlation:.2f} with {cand.correlated_with}")

                    if cand.sector_weight_if_added < 0.25:
                        fit_desc.append(f"adds {cand.sector} diversification")
                    st.caption(f"🎯 Fit ({cand.portfolio_fit_score:.2f}): {'; '.join(fit_desc)}")

                    # Why
                    st.caption(f"**Why:** {cand.why}")

        # Full results table
        with st.expander("📋 All Scored Candidates"):
            disc_rows = []
            for c in disc.candidates:
                disc_rows.append({
                    "Rank": c.final_rank,
                    "Ticker": c.ticker,
                    "Name": c.name,
                    "Exchange": c.exchange,
                    "Sector": c.sector,
                    "Score": c.aggregate_score,
                    "FX Penalty": f"-{c.fx_penalty_pct:.1f}%" if c.fx_penalty_applied else "—",
                    "Fit": c.portfolio_fit_score,
                    "Max Corr": c.max_correlation,
                    "Action": c.action,
                    "Currency": c.currency,
                    "Cap": getattr(c, "cap_tier", "—"),
                    "Parabolic": f"-{c.parabolic_penalty:.2f}" if getattr(c, "is_parabolic", False) else "—",
                    "Earnings": f"{c.earnings_days}d" if getattr(c, "earnings_days", None) else "—",
                })
            if disc_rows:
                st.dataframe(pd.DataFrame(disc_rows), hide_index=True, use_container_width=True)

        # Rejection reasons
        if disc.rejections:
            with st.expander(f"🚫 Rejected Candidates ({len(disc.rejections)})"):
                rej_rows = []
                for r in disc.rejections[:100]:  # Cap display
                    rej_rows.append({
                        "Ticker": r.ticker,
                        "Name": r.name,
                        "Exchange": r.exchange,
                        "Stage": r.stage,
                        "Reason": r.reason,
                    })
                st.dataframe(pd.DataFrame(rej_rows), hide_index=True, use_container_width=True)

    elif not disc.error:
        st.info("No candidates found meeting the criteria. Try adjusting discovery parameters in config.py.")

    # Comprehensive Signal Backtest Performance
    try:
        from engine.discovery_backtest import (
            get_pick_performance, get_pillar_stats, get_pending_picks_count,
            get_action_calibration, get_regime_stats, get_stop_target_stats,
            get_forecast_accuracy,
        )

        _perf = get_pick_performance(limit=50)
        _pstats = get_pillar_stats()
        _pending = get_pending_picks_count()
        _acal = get_action_calibration()
        _rstats = get_regime_stats()
        _st_stats = get_stop_target_stats()
        _fcast_acc = get_forecast_accuracy()

        if _perf or _pending:
            with st.expander(f"📊 Signal Track Record ({len(_perf)} evaluated, {_pending} pending)"):
                # --- Pillar Effectiveness ---
                if _pstats:
                    st.markdown("**Pillar Effectiveness (which signals predict 90-day returns)**")
                    ps_rows = [{
                        "Pillar": s["pillar"].title(),
                        "IC": f"{s['information_coefficient']:+.3f}",
                        "Hit Rate": f"{s['hit_rate']:.0%}",
                        "Avg Return (High)": f"{s['avg_return_high']:+.1f}%",
                        "Avg Return (Low)": f"{s['avg_return_low']:+.1f}%",
                        "Samples": s["sample_size"],
                    } for s in _pstats]
                    st.dataframe(pd.DataFrame(ps_rows), hide_index=True, use_container_width=True)

                # --- Action Calibration ---
                if _acal:
                    st.markdown("**Action Calibration (are action labels accurate?)**")
                    ac_rows = [{
                        "Action": a["action"],
                        "Avg 90d Return": f"{a['avg_return_90d']:+.1f}%",
                        "Accuracy": f"{a['hit_rate']:.0%}",
                        "Samples": a["sample_size"],
                    } for a in _acal]
                    st.dataframe(pd.DataFrame(ac_rows), hide_index=True, use_container_width=True)

                # --- Regime Effectiveness ---
                if _rstats:
                    st.markdown("**Regime Effectiveness (which regime works best?)**")
                    re_rows = [{
                        "Regime": r["regime"],
                        "Avg 90d Return": f"{r['avg_return_90d']:+.1f}%",
                        "Best Pillar": (r["best_pillar"] or "—").title(),
                        "Samples": r["sample_size"],
                    } for r in _rstats]
                    st.dataframe(pd.DataFrame(re_rows), hide_index=True, use_container_width=True)

                # --- Stop/Target + Forecast Stats ---
                st_col, fc_col = st.columns(2)
                with st_col:
                    if _st_stats and _st_stats.get("with_stops"):
                        st.markdown("**Stop-Loss / Take-Profit Hits**")
                        total_w = _st_stats["with_stops"]
                        s_hit = _st_stats.get("stops_hit") or 0
                        t_hit = _st_stats.get("targets_hit") or 0
                        st.caption(
                            f"Stops hit: {s_hit}/{total_w} ({s_hit/total_w:.0%})"
                            + (f" — avg day {_st_stats['avg_stop_day']:.0f}" if _st_stats.get("avg_stop_day") else "")
                        )
                        st.caption(
                            f"Targets hit: {t_hit}/{total_w} ({t_hit/total_w:.0%})"
                            + (f" — avg day {_st_stats['avg_target_day']:.0f}" if _st_stats.get("avg_target_day") else "")
                        )
                with fc_col:
                    if _st_stats and _st_stats.get("avg_forecast_err_5d") is not None:
                        st.markdown("**Forecast Accuracy**")
                        st.caption(f"5-day avg error: {_st_stats['avg_forecast_err_5d']:.1f}%")
                        if _st_stats.get("avg_forecast_err_63d") is not None:
                            st.caption(f"63-day avg error: {_st_stats['avg_forecast_err_63d']:.1f}%")

                # --- Multi-horizon signal performance ---
                if _perf:
                    st.markdown("**Signal Performance — Multi-Horizon Returns**")
                    perf_rows = [{
                        "Date": p["run_date"][:10],
                        "Ticker": p["ticker"],
                        "Source": p.get("source", "—"),
                        "Action": p.get("action", "—"),
                        "Score": f"{p['aggregate_score']:.3f}",
                        "30d": f"{p['return_30d']:+.1f}%" if p.get("return_30d") is not None else "—",
                        "60d": f"{p['return_60d']:+.1f}%" if p.get("return_60d") is not None else "—",
                        "90d": f"{p['return_90d']:+.1f}%" if p.get("return_90d") is not None else "—",
                        "Beat SPY": "Yes" if p.get("beat_market") else "No",
                        "Action OK": "Yes" if p.get("action_correct") else "No",
                    } for p in _perf]
                    st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)

                    # Summary metrics
                    returns = [p["return_90d"] for p in _perf if p.get("return_90d") is not None]
                    if returns:
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        bc1.metric("Avg Return", f"{sum(returns)/len(returns):+.1f}%")
                        bc2.metric("Win Rate", f"{sum(1 for r in returns if r > 0)/len(returns):.0%}")
                        bc3.metric("Best Pick", f"{max(returns):+.1f}%")
                        bc4.metric("Worst Pick", f"{min(returns):+.1f}%")

                    # Beat market rate
                    beat = [p for p in _perf if p.get("beat_market") is not None]
                    if beat:
                        beat_rate = sum(1 for p in beat if p["beat_market"]) / len(beat)
                        st.caption(f"Beat SPY: {beat_rate:.0%} of signals | "
                                   f"Action accuracy: {sum(1 for p in _perf if p.get('action_correct'))/len(_perf):.0%}")
    except Exception:
        pass

    # Evaluation Harness — Scorecard
    try:
        from engine.evaluation_harness import compute_scorecard

        _sc = compute_scorecard(source="all", min_signals=5)
        if _sc and _sc.evaluated_signals >= 5 and _sc.sharpe_ratio is not None:
            with st.expander(f"📈 **Performance Scorecard** ({_sc.evaluated_signals} signals evaluated)"):
                # Top-level risk/return metrics
                ev1, ev2, ev3, ev4 = st.columns(4)
                ev1.metric("Sharpe Ratio", f"{_sc.sharpe_ratio:.2f}")
                ev2.metric("Sortino Ratio", f"{_sc.sortino_ratio:.2f}" if _sc.sortino_ratio else "—")
                ev3.metric("Max Drawdown", f"{_sc.max_drawdown:+.1f}%" if _sc.max_drawdown else "—")
                ev4.metric("Calmar Ratio", f"{_sc.calmar_ratio:.2f}" if _sc.calmar_ratio else "—")

                ev5, ev6, ev7, ev8 = st.columns(4)
                ev5.metric("Hit Rate (90d)", f"{_sc.overall_hit_rate:.0%}" if _sc.overall_hit_rate else "—")
                ev6.metric("Action Accuracy", f"{_sc.action_accuracy:.0%}" if _sc.action_accuracy else "—")
                ev7.metric("Beat SPY Rate", f"{_sc.beat_benchmark_rate:.0%}" if _sc.beat_benchmark_rate else "—")
                ev8.metric("IC Stability", f"{_sc.ic_stability:.4f}" if _sc.ic_stability else "—")

                # Per-horizon table
                if _sc.horizons:
                    st.markdown("**Returns by Horizon**")
                    _h_rows = pd.DataFrame([{
                        "Horizon": h.horizon,
                        "Avg Return": f"{h.avg_return:+.1f}%",
                        "Median": f"{h.median_return:+.1f}%",
                        "Std Dev": f"{h.std_return:.1f}%",
                        "Hit Rate": f"{h.hit_rate:.0%}",
                        "Alpha vs SPY": f"{h.alpha:+.1f}%" if h.horizon == "90d" else "—",
                        "Best": f"{h.best:+.1f}%",
                        "Worst": f"{h.worst:+.1f}%",
                        "N": h.sample_size,
                    } for h in _sc.horizons])
                    st.dataframe(_h_rows, hide_index=True, use_container_width=True)

                # Per-regime table
                if _sc.regimes:
                    st.markdown("**Performance by Market Regime**")
                    _r_rows = pd.DataFrame([{
                        "Regime": r.regime,
                        "Avg 90d Return": f"{r.avg_return_90d:+.1f}%",
                        "Hit Rate": f"{r.hit_rate:.0%}",
                        "Best Pillar": (r.best_pillar or "—").title(),
                        "N": r.sample_size,
                    } for r in _sc.regimes])
                    st.dataframe(_r_rows, hide_index=True, use_container_width=True)

                # Stop/target + forecast
                st_c, fc_c = st.columns(2)
                with st_c:
                    if _sc.stop_hit_rate is not None:
                        st.markdown("**Stop/Target Effectiveness**")
                        st.caption(f"Stop-loss hit rate: {_sc.stop_hit_rate:.0%}"
                                   + (f" (avg day {_sc.avg_stop_day:.0f})" if _sc.avg_stop_day else ""))
                        st.caption(f"Take-profit hit rate: {_sc.target_hit_rate:.0%}"
                                   + (f" (avg day {_sc.avg_target_day:.0f})" if _sc.avg_target_day else ""))
                with fc_c:
                    if _sc.avg_forecast_error_5d is not None:
                        st.markdown("**Forecast Accuracy**")
                        st.caption(f"5-day avg error: {_sc.avg_forecast_error_5d:.1f}%")
                        if _sc.avg_forecast_error_63d is not None:
                            st.caption(f"63-day avg error: {_sc.avg_forecast_error_63d:.1f}%")

    except Exception as _sc_err:
        import logging as _logging
        _logging.getLogger(__name__).warning("Performance scorecard failed: %s", _sc_err)


# ---------------------------------------------------------------------------
# Trade History — Record Sales
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Trade History")

from utils.data_fetch import load_portfolio_full, record_sale

_portfolio_full = load_portfolio_full()
_trade_history = _portfolio_full.get("trade_history", [])

tab_record, tab_history = st.tabs(["Record a Sale", "Past Trades"])

with tab_record:
    st.caption("When you sell a stock on Interactive Investor, record it here to track P&L and update your portfolio.")
    _holding_options = [f"{r['ticker']} — {r.get('name', r['ticker'])}" for r in results]

    if _holding_options:
        rc1, rc2 = st.columns(2)
        with rc1:
            _sell_ticker_sel = st.selectbox("Stock to sell", _holding_options)
            _sell_ticker = _sell_ticker_sel.split(" — ")[0]

            # Show current info for the selected holding
            _sel_result = next((r for r in results if r["ticker"] == _sell_ticker), None)
            _sel_holding = next((h for h in holdings if h["ticker"] == _sell_ticker), None)
            if _sel_result and _sel_holding:
                st.caption(
                    f"Current price: {_sel_result.get('current_price', 0):.2f} · "
                    f"Avg buy: {_sel_holding['avg_buy_price']:.2f} · "
                    f"Qty held: {_sel_holding['quantity']}"
                )

        with rc2:
            _sell_price = st.number_input(
                "Sell price", min_value=0.001, value=float(_sel_result.get("current_price", 0)) if _sel_result else 0.0,
                format="%.4f",
            )
            _sell_qty = st.number_input(
                "Quantity sold", min_value=1,
                value=int(_sel_holding["quantity"]) if _sel_holding else 1,
            )

        rc3, rc4 = st.columns(2)
        with rc3:
            _sell_date = st.date_input("Sale date")
        with rc4:
            _sell_notes = st.text_input("Notes (optional)", placeholder="e.g. stop-loss triggered")

        # Preview P&L
        if _sel_holding:
            _preview_pnl = (_sell_price - _sel_holding["avg_buy_price"]) * _sell_qty
            _preview_pct = (_sell_price - _sel_holding["avg_buy_price"]) / _sel_holding["avg_buy_price"] * 100
            _pnl_color = "green" if _preview_pnl >= 0 else "red"
            st.markdown(
                f"**Estimated P&L:** :{_pnl_color}[{_preview_pnl:+,.2f} ({_preview_pct:+.1f}%)]"
            )

        if st.button("Confirm Sale", type="primary"):
            trade = record_sale(
                ticker=_sell_ticker,
                sell_price=_sell_price,
                quantity=_sell_qty,
                sell_date=str(_sell_date),
                notes=_sell_notes,
            )
            if trade:
                st.success(
                    f"Recorded: Sold {trade['quantity']} × {trade['ticker']} @ {trade['sell_price']:.4f} "
                    f"— P&L: {trade['pnl']:+,.2f} ({trade['pnl_pct']:+.1f}%)"
                )
                st.rerun()
            else:
                st.error(f"Ticker {_sell_ticker} not found in portfolio.")
    else:
        st.info("No holdings in portfolio.")

with tab_history:
    if _trade_history:
        _th_rows = []
        _total_realized = 0.0
        for t in reversed(_trade_history):  # Most recent first
            _th_rows.append({
                "Date": t.get("sell_date", "—"),
                "Ticker": t["ticker"],
                "Name": t.get("name", ""),
                "Qty": t["quantity"],
                "Buy Price": f"{t['buy_price']:.4f}",
                "Sell Price": f"{t['sell_price']:.4f}",
                "P&L": f"{t['pnl']:+,.2f}",
                "Return": f"{t['pnl_pct']:+.1f}%",
                "Notes": t.get("notes", ""),
            })
            _total_realized += t.get("pnl", 0)

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Total Trades", len(_trade_history))
        tc2.metric("Realized P&L", f"{_total_realized:+,.2f}")
        _winners = sum(1 for t in _trade_history if t.get("pnl", 0) > 0)
        tc3.metric("Win Rate", f"{_winners / len(_trade_history) * 100:.0f}%" if _trade_history else "—")

        st.dataframe(pd.DataFrame(_th_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No trades recorded yet. Use the 'Record a Sale' tab when you sell a stock on Interactive Investor.")


# ---------------------------------------------------------------------------
# Paper Trading Ledger
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Paper Trading Ledger")

if getattr(config, "PAPER_TRADING_ENABLED", False):
    from engine.paper_trading import (
        get_all_signals, get_slippage_stats, get_pnl_summary,
        get_slippage_by_ticker, get_open_positions, get_realized_pnl,
        get_unrealized_pnl, init_db as _pt_init,
    )
    _pt_init()

    tab_overview, tab_signals, tab_positions, tab_slippage = st.tabs([
        "Overview", "Signal Log", "Paper Positions", "Slippage Analysis",
    ])

    # --- Overview tab ---
    with tab_overview:
        pnl_sum = get_pnl_summary()
        slip_stats = get_slippage_stats()

        col1, col2, col3, col4 = st.columns(4)
        total_trades = pnl_sum.get("total_trades") or 0
        with col1:
            st.metric("Total Closed Trades", total_trades)
        with col2:
            total_pnl = pnl_sum.get("total_pnl") or 0
            st.metric("Total P&L", f"{'+'if total_pnl>=0 else ''}{total_pnl:,.2f}")
        with col3:
            win_rate = (
                (pnl_sum["winners"] / total_trades * 100)
                if total_trades > 0 and pnl_sum.get("winners") is not None else 0
            )
            st.metric("Win Rate", f"{win_rate:.0f}%")
        with col4:
            avg_slip = slip_stats.get("avg_slippage_bps") or 0
            st.metric("Avg Slippage", f"{avg_slip:+.1f} bps")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Avg Return", f"{pnl_sum.get('avg_return_pct') or 0:+.1f}%")
        with col6:
            st.metric("Best Trade", f"{pnl_sum.get('best_trade_pct') or 0:+.1f}%")
        with col7:
            st.metric("Worst Trade", f"{pnl_sum.get('worst_trade_pct') or 0:+.1f}%")
        with col8:
            st.metric("Avg Hold", f"{pnl_sum.get('avg_hold_days') or 0:.0f} days")

        # Unrealized P&L for open positions
        unrealized = get_unrealized_pnl()
        if unrealized:
            st.markdown("#### Open Paper Positions — Unrealized P&L")
            ur_rows = []
            for u in unrealized:
                ur_rows.append({
                    "Ticker": u["ticker"],
                    "Qty": u["quantity"],
                    "Entry": f"{u['avg_entry_price']:.4f}",
                    "Current": f"{u['current_price']:.4f}",
                    "P&L": f"{u['unrealized_pnl']:+,.2f}",
                    "Return": f"{u['unrealized_pnl_pct']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(ur_rows), hide_index=True, use_container_width=True)

    # --- Signal Log tab ---
    with tab_signals:
        signals = get_all_signals(limit=200)
        if signals:
            sig_rows = []
            for s in signals:
                sig_rows.append({
                    "Time": s["timestamp"],
                    "Ticker": s["ticker"],
                    "Side": s["side"],
                    "Source": s["source"],
                    "Signal Price": f"{s['signal_price']:.4f}" if s["signal_price"] else "—",
                    "Fill Price": f"{s['fill_price']:.4f}" if s.get("fill_price") else "Pending",
                    "Slippage (bps)": f"{s['slippage_bps']:+.1f}" if s.get("slippage_bps") is not None else "—",
                    "Score": f"{s['score']:.3f}" if s.get("score") is not None else "—",
                    "Action": s.get("action") or "—",
                    "Swap From": s.get("swap_from") or "—",
                })
            st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No paper trade signals recorded yet. Signals are logged automatically by the daily orchestrator.")

    # --- Paper Positions tab ---
    with tab_positions:
        positions = get_open_positions()
        realized = get_realized_pnl()

        if positions:
            st.markdown("#### Open Positions")
            pos_rows = [{
                "Ticker": p["ticker"],
                "Qty": p["quantity"],
                "Avg Entry": f"{p['avg_entry_price']:.4f}",
                "Opened": p["opened_at"],
            } for p in positions]
            st.dataframe(pd.DataFrame(pos_rows), hide_index=True, use_container_width=True)

        if realized:
            st.markdown("#### Closed Trades")
            rl_rows = [{
                "Ticker": r["ticker"],
                "Entry": f"{r['entry_price']:.4f}",
                "Exit": f"{r['exit_price']:.4f}",
                "Qty": r["quantity"],
                "P&L": f"{r['pnl']:+,.2f}",
                "Return": f"{r['pnl_pct']:+.1f}%",
                "Hold": f"{r['hold_days']}d" if r.get("hold_days") else "—",
                "Closed": r["closed_at"],
            } for r in realized]
            st.dataframe(pd.DataFrame(rl_rows), hide_index=True, use_container_width=True)

        if not positions and not realized:
            st.info("No paper positions yet. Positions are created when pending signals are filled at next-session open.")

    # --- Slippage Analysis tab ---
    with tab_slippage:
        slip_by_ticker = get_slippage_by_ticker()
        if slip_by_ticker:
            st.markdown("#### Slippage by Ticker")
            sl_rows = [{
                "Ticker": t["ticker"],
                "Fills": t["fills"],
                "Avg Slippage (bps)": f"{t['avg_slippage_bps']:+.1f}",
                "Avg |Slippage| (bps)": f"{t['avg_abs_slippage_bps']:.1f}",
                "Min (bps)": f"{t['min_slippage_bps']:+.1f}",
                "Max (bps)": f"{t['max_slippage_bps']:+.1f}",
            } for t in slip_by_ticker]
            st.dataframe(pd.DataFrame(sl_rows), hide_index=True, use_container_width=True)

            # Overall stats
            stats = get_slippage_stats()
            if stats.get("total_fills"):
                st.markdown("#### Overall Slippage Distribution")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Total Fills", stats["total_fills"])
                    st.metric("Buy Fills", stats.get("buy_fills") or 0)
                with mc2:
                    st.metric("Avg Slippage", f"{stats['avg_slippage_bps']:+.1f} bps")
                    st.metric("Avg Buy Slippage", f"{stats.get('avg_buy_slippage') or 0:+.1f} bps")
                with mc3:
                    st.metric("Avg |Slippage|", f"{stats['avg_abs_slippage_bps']:.1f} bps")
                    st.metric("Avg Sell Slippage", f"{stats.get('avg_sell_slippage') or 0:+.1f} bps")
        else:
            st.info("No fill data yet. Slippage is calculated when pending signals are resolved at next-session open.")
else:
    st.info("Paper trading is disabled. Set `PAPER_TRADING_ENABLED = True` in config.py to enable.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "This dashboard is for informational purposes only and does not constitute financial advice. "
    "Always do your own research. Trades must be executed manually on Interactive Investor."
)
