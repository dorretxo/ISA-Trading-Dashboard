"""ISA Portfolio Dashboard — Premium Streamlit Frontend."""

import html as _html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from engine.scoring import analyse_portfolio
from engine.backtest import optimize_weights, _score_to_action
from utils.data_fetch import clear_cache, load_portfolio, get_ticker_info, get_price_history
from utils.cache_loader import load_dashboard_data, format_freshness
from utils.safe_numeric import safe_float, is_valid_number, format_currency, format_pct, format_score

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

/* ── Top-level tab navigation ──────────────────────────────────── */
div[data-testid="stTabs"] > div[role="tablist"] {
    background: linear-gradient(145deg, rgba(15,23,42,0.85), rgba(30,41,59,0.75));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 14px;
    padding: 5px 6px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18), 0 2px 6px rgba(0, 0, 0, 0.12);
    gap: 4px;
    margin-bottom: 14px;
}
div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"] {
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em;
    border: none !important;
    color: rgba(203, 213, 225, 0.7) !important;
    background: transparent !important;
    transition: all 0.2s ease;
}
div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    color: rgba(255, 255, 255, 0.9) !important;
}
div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(99,102,241,0.18)) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255,255,255,0.08);
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}
/* Remove default Streamlit tab underline */
div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]::after,
div[data-testid="stTabs"] > div[role="tablist"] > div[data-testid="stTabsGap"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
    height: 0 !important;
}
/* Tab content area: subtle card-like container */
div[data-testid="stTabs"] > div[data-testid="stTabContent"] {
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 12px;
    padding: 16px 8px 8px 8px;
    background: rgba(15, 23, 42, 0.25);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* ── Nested section tabs (inside holding cards, analytics, etc.) ── */
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] {
    background: linear-gradient(145deg, rgba(30,41,59,0.70), rgba(51,65,85,0.55));
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 4px 5px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.14), 0 1px 4px rgba(0, 0, 0, 0.10);
    gap: 3px;
    margin-bottom: 10px;
}
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"] {
    border-radius: 9px !important;
    padding: 9px 18px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.01em;
    border: none !important;
    color: rgba(203, 213, 225, 0.65) !important;
    background: transparent !important;
    transition: all 0.2s ease;
}
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: rgba(255, 255, 255, 0.85) !important;
}
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.20), rgba(139,92,246,0.15)) !important;
    color: #e2e8f0 !important;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.20), inset 0 1px 0 rgba(255,255,255,0.06);
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
}
/* Remove underline on nested tabs too */
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]::after,
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[role="tablist"] > div[data-testid="stTabsGap"],
div[data-testid="stTabContent"] div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
    height: 0 !important;
}
/* Nested tab content: lighter card */
div[data-testid="stTabContent"] div[data-testid="stTabs"] > div[data-testid="stTabContent"] {
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 10px;
    padding: 12px 6px 6px 6px;
    background: rgba(30, 41, 59, 0.18);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

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

/* Discovery command center + recommendation cards */
.section-kicker {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.35rem;
    font-weight: 700;
}
.insight-card {
    border-radius: 18px;
    padding: 16px 18px;
    min-height: 128px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background:
        radial-gradient(circle at top right, rgba(255,255,255,0.10), transparent 38%),
        linear-gradient(145deg, rgba(15,23,42,0.92), rgba(30,41,59,0.88));
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
}
.insight-card.best { background:
        radial-gradient(circle at top right, rgba(16,185,129,0.22), transparent 38%),
        linear-gradient(145deg, rgba(6,78,59,0.96), rgba(15,118,110,0.90)); }
.insight-card.fit { background:
        radial-gradient(circle at top right, rgba(59,130,246,0.20), transparent 38%),
        linear-gradient(145deg, rgba(30,64,175,0.96), rgba(30,41,59,0.90)); }
.insight-card.momentum { background:
        radial-gradient(circle at top right, rgba(245,158,11,0.22), transparent 38%),
        linear-gradient(145deg, rgba(120,53,15,0.96), rgba(146,64,14,0.90)); }
.insight-card.risk { background:
        radial-gradient(circle at top right, rgba(239,68,68,0.20), transparent 38%),
        linear-gradient(145deg, rgba(127,29,29,0.96), rgba(68,64,60,0.92)); }
.insight-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    opacity: 0.82;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.insight-title {
    font-size: 1.25rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.insight-sub {
    font-size: 0.88rem;
    line-height: 1.45;
    opacity: 0.92;
}
.insight-meta {
    margin-top: 0.75rem;
    font-size: 0.78rem;
    opacity: 0.78;
}
.lens-note {
    font-size: 0.84rem;
    color: #64748b;
    margin-top: -0.25rem;
    margin-bottom: 0.5rem;
}
.rec-hero {
    border-radius: 20px;
    padding: 18px 18px 14px 18px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background:
        radial-gradient(circle at top right, rgba(255,255,255,0.10), transparent 34%),
        linear-gradient(150deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.14);
    margin-bottom: 0.75rem;
}
.rec-rankline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.rec-rank {
    font-size: 0.77rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 700;
}
.rec-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 0.2rem;
}
.rec-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.35;
    margin-bottom: 0.75rem;
}
.rec-thesis {
    font-size: 0.92rem;
    line-height: 1.5;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
}
.badge-row, .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.4rem 0 0 0;
}
.signal-badge, .signal-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.26rem 0.58rem;
    border-radius: 999px;
    font-size: 0.74rem;
    line-height: 1;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
}
.signal-badge.buy { background: rgba(16,185,129,0.16); color: #86efac; border-color: rgba(16,185,129,0.28); }
.signal-badge.neutral { background: rgba(148,163,184,0.16); color: #cbd5e1; border-color: rgba(148,163,184,0.24); }
.signal-badge.avoid { background: rgba(239,68,68,0.16); color: #fca5a5; border-color: rgba(239,68,68,0.24); }
.signal-badge.data { background: rgba(107,114,128,0.22); color: #e5e7eb; border-color: rgba(148,163,184,0.20); }
.signal-badge.ready { background: rgba(16,185,129,0.16); color: #86efac; border-color: rgba(16,185,129,0.28); }
.signal-badge.pullback { background: rgba(245,158,11,0.16); color: #fde68a; border-color: rgba(245,158,11,0.26); }
.signal-badge.watch { background: rgba(239,68,68,0.16); color: #fca5a5; border-color: rgba(239,68,68,0.24); }
.signal-chip.good { background: rgba(16,185,129,0.12); color: #10b981; border-color: rgba(16,185,129,0.18); }
.signal-chip.info { background: rgba(59,130,246,0.12); color: #60a5fa; border-color: rgba(59,130,246,0.18); }
.signal-chip.warn { background: rgba(245,158,11,0.14); color: #fbbf24; border-color: rgba(245,158,11,0.22); }
.signal-chip.risk { background: rgba(239,68,68,0.14); color: #fca5a5; border-color: rgba(239,68,68,0.22); }
.confidence-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.26rem 0.58rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.confidence-chip.high { background: rgba(16,185,129,0.12); color: #86efac; border-color: rgba(16,185,129,0.24); }
.confidence-chip.medium { background: rgba(59,130,246,0.12); color: #93c5fd; border-color: rgba(59,130,246,0.20); }
.confidence-chip.low { background: rgba(245,158,11,0.14); color: #fde68a; border-color: rgba(245,158,11,0.22); }
.confidence-chip.data { background: rgba(107,114,128,0.20); color: #e5e7eb; border-color: rgba(148,163,184,0.18); }
.exit-card {
    border-radius: 18px;
    padding: 16px 18px;
    border: 1px solid rgba(148,163,184,0.16);
    background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
    box-shadow: 0 14px 26px rgba(15,23,42,0.10);
    margin-bottom: 0.75rem;
}
.exit-card.urgent {
    background:
        radial-gradient(circle at top right, rgba(239,68,68,0.18), transparent 38%),
        linear-gradient(145deg, rgba(69,10,10,0.98), rgba(30,41,59,0.92));
}
.exit-card.action {
    background:
        radial-gradient(circle at top right, rgba(245,158,11,0.18), transparent 38%),
        linear-gradient(145deg, rgba(120,53,15,0.98), rgba(30,41,59,0.92));
}
.exit-card.warning {
    background:
        radial-gradient(circle at top right, rgba(148,163,184,0.18), transparent 38%),
        linear-gradient(145deg, rgba(31,41,55,0.98), rgba(51,65,85,0.92));
}
.exit-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}
.exit-title {
    font-size: 1.15rem;
    font-weight: 800;
    line-height: 1.1;
}
.exit-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}
.severity-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.62rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.severity-pill.urgent { background: rgba(239,68,68,0.16); color: #fecaca; border: 1px solid rgba(239,68,68,0.28); }
.severity-pill.action { background: rgba(245,158,11,0.16); color: #fde68a; border: 1px solid rgba(245,158,11,0.24); }
.severity-pill.warning { background: rgba(148,163,184,0.16); color: #e2e8f0; border: 1px solid rgba(148,163,184,0.20); }
.exit-message {
    font-size: 0.9rem;
    line-height: 1.45;
    color: #e2e8f0;
    margin: 0.55rem 0 0.65rem 0;
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
    "INSUFFICIENT DATA": "#6b7280",
    "SELL": "#f59e0b",
    "STRONG SELL": "#ef4444",
}

# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _get_earnings_dates(ticker: str):
    """Fetch earnings dates for a ticker, cached for 1 hour."""
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            return ed
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Reusable HTML component functions
# ---------------------------------------------------------------------------

def _format_price(val, currency: str) -> str:
    return format_currency(val, currency, decimals=2)


def _format_change(val) -> str:
    return format_pct(val, decimals=2)


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


def _discovery_confidence(cand) -> tuple[str, str, float]:
    sent_conf = safe_float(getattr(cand, "sentiment_score", 0))
    data_discount = safe_float(getattr(cand, "confidence_discount", 1.0), default=1.0)
    has_data = 0.0 if getattr(cand, "action", "") == "INSUFFICIENT DATA" else 1.0
    score = max(0.0, min(1.0, 0.35 * has_data + 0.30 * data_discount + 0.35 * (0.5 + 0.5 * max(min(sent_conf, 1.0), -1.0))))

    if getattr(cand, "action", "") == "INSUFFICIENT DATA":
        return "Data Gap", "data", score
    if score >= 0.75:
        return "High Confidence", "high", score
    if score >= 0.55:
        return "Medium Confidence", "medium", score
    return "Watch Carefully", "low", score


def _candidate_risk_tags(cand) -> list[str]:
    analyst_upside_raw = getattr(cand, "analyst_upside", None)
    analyst_upside = (
        safe_float(analyst_upside_raw)
        if analyst_upside_raw is not None
        else None
    )
    insider_buys = int(safe_float(getattr(cand, "insider_buys", 0), default=0))
    insider_sells = int(safe_float(getattr(cand, "insider_sells", 0), default=0))
    tags = []
    if getattr(cand, "ticker_identity_warning", None):
        tags.append("Ticker identity check")
    if getattr(cand, "governance_flag", False) and getattr(cand, "asymmetric_risk_flag", False):
        tags.append("Governance / asymmetric risk")
    elif getattr(cand, "governance_flag", False):
        tags.append("Governance risk")
    elif getattr(cand, "asymmetric_risk_flag", False):
        tags.append("Asymmetric risk")
    if analyst_upside is not None and analyst_upside < 0:
        tags.append("Above analyst target")
    if insider_sells > insider_buys:
        tags.append("Net insider selling")
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
    return tags[:5]


def _candidate_evidence_tags(cand) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    ret_90 = safe_float(getattr(cand, "return_90d", 0)) * 100
    ret_30 = safe_float(getattr(cand, "return_30d", 0)) * 100
    exp_90 = safe_float(getattr(cand, "expected_return_90d", 0)) * 100
    fit = safe_float(getattr(cand, "portfolio_fit_score", 0))
    corr = safe_float(getattr(cand, "max_correlation", 0))
    fund = safe_float(getattr(cand, "fundamental_score", 0))
    tech = safe_float(getattr(cand, "technical_score", 0))
    beta = getattr(cand, "beta_90d", None)
    dividend_yield = getattr(cand, "dividend_yield", None)
    balance_grade = str(getattr(cand, "balance_sheet_grade", "") or "").upper()

    if fit >= 0.80:
        tags.append((f"Fit {fit:.2f}", "info"))
    if corr and corr < 0.40:
        tags.append((f"Low corr {corr:.2f}", "info"))
    if beta is not None and safe_float(beta, default=9.9) <= 1.10:
        tags.append((f"Beta {safe_float(beta):.2f}", "info"))
    if dividend_yield is not None and safe_float(dividend_yield) >= 0.02:
        tags.append((f"Yield {safe_float(dividend_yield) * 100:.1f}%", "info"))
    if balance_grade in {"A", "B"}:
        tags.append((f"Balance {balance_grade}", "info"))
    ex_div_days = getattr(cand, "ex_dividend_days", None)
    if ex_div_days is not None and 0 <= ex_div_days <= 14:
        tags.append((f"Ex-div {ex_div_days}d", "info"))
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


def _candidate_entry_stance(cand) -> str:
    stance = str(getattr(cand, "entry_stance", "") or "").strip()
    if stance in {"Ready", "Pullback Preferred", "Watch Only"}:
        return stance

    analyst_upside_raw = getattr(cand, "analyst_upside", None)
    analyst_upside = (
        safe_float(analyst_upside_raw)
        if analyst_upside_raw is not None
        else None
    )
    insider_buys = int(safe_float(getattr(cand, "insider_buys", 0), default=0))
    insider_sells = int(safe_float(getattr(cand, "insider_sells", 0), default=0))
    return_30d = safe_float(getattr(cand, "return_30d", 0))

    if (
        getattr(cand, "governance_flag", False)
        or getattr(cand, "asymmetric_risk_flag", False)
        or getattr(cand, "earnings_imminent", False)
        or (
            getattr(cand, "is_parabolic", False)
            and analyst_upside is not None
            and analyst_upside < 0
        )
        or (
            getattr(cand, "near_52w_high", False)
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
        getattr(cand, "is_parabolic", False)
        or getattr(cand, "near_52w_high", False)
        or (analyst_upside is not None and analyst_upside < 5)
        or insider_sells > insider_buys
        or getattr(cand, "earnings_near", False)
    ):
        return "Pullback Preferred"

    return "Ready"


def _entry_stance_tone(stance: str) -> str:
    return {
        "Ready": "ready",
        "Pullback Preferred": "pullback",
        "Watch Only": "watch",
    }.get(stance, "neutral")


def _candidate_is_gated(cand) -> bool:
    return bool(getattr(cand, "ticker_identity_warning", None)) or _candidate_entry_stance(cand) == "Watch Only"


def _pick_best_new_opportunity(candidates: list):
    """Return (candidate_or_none, meta, subtitle) for the command-center opportunity card."""
    verified = [c for c in candidates if not getattr(c, "ticker_identity_warning", None)]
    ready = [c for c in verified if _candidate_entry_stance(c) == "Ready"]
    if ready:
        cand = max(ready, key=lambda c: safe_float(getattr(c, "final_rank", 0)))
        meta = (
            f"Ready entry · Final rank {safe_float(getattr(cand, 'final_rank', 0)):.3f} "
            f"· {getattr(cand, 'action', 'NEUTRAL')}"
        )
        return cand, meta, _candidate_thesis(cand)

    pullback = [c for c in verified if _candidate_entry_stance(c) == "Pullback Preferred"]
    if pullback:
        cand = max(pullback, key=lambda c: safe_float(getattr(c, "final_rank", 0)))
        meta = (
            f"Pullback preferred · Final rank {safe_float(getattr(cand, 'final_rank', 0)):.3f} "
            f"· {getattr(cand, 'action', 'NEUTRAL')}"
        )
        return cand, meta, _candidate_thesis(cand)

    return (
        None,
        "No clean entry right now",
        "The current top set is gated by timing or ticker-identity risk, so the dashboard is withholding a fresh entry candidate.",
    )


def _candidate_thesis(cand) -> str:
    positives = []
    if safe_float(getattr(cand, "portfolio_fit_score", 0)) >= 0.80:
        positives.append("improves diversification")
    if safe_float(getattr(cand, "momentum_score", 0)) >= 0.80 or safe_float(getattr(cand, "return_90d", 0)) > 0.25:
        positives.append("trend strength is still intact")
    if safe_float(getattr(cand, "fundamental_score", 0)) > 0.15:
        positives.append("fundamentals support the move")
    if safe_float(getattr(cand, "forecast_score", 0)) > 0.15 or safe_float(getattr(cand, "expected_return_90d", 0)) > 0.06:
        positives.append("the forward model still sees upside")

    stance = _candidate_entry_stance(cand)
    risks = _candidate_risk_tags(cand)
    if getattr(cand, "ticker_identity_warning", None):
        return "Ticker identity needs verification before treating this as a live idea."
    if getattr(cand, "action", "") == "INSUFFICIENT DATA":
        return "Signal quality is incomplete, so this name should stay in watch mode until fresh analysis fills the missing pillars."
    if not positives:
        positives.append("it remains one of the cleaner ideas in the current universe")
    lead = ", ".join(positives[:2])
    if stance == "Watch Only":
        return f"The business case may be interesting because it {lead}, but the current setup looks too crowded or binary for a fresh 3-6 month entry."
    if stance == "Pullback Preferred":
        return f"The core thesis still works because it {lead}, but the current setup looks better on a pullback than at today's price."
    if risks:
        return f"This idea stands out because it {lead}, but {risks[0].lower()} needs monitoring."
    return f"This idea stands out because it {lead}, with no immediate red-flag overlays in the current pass."


def _render_html_chips(chips: list[tuple[str, str]], class_name: str = "signal-chip") -> str:
    if not chips:
        return ""
    rendered = "".join(
        f'<span class="{class_name} {tone}">{_html.escape(text)}</span>'
        for text, tone in chips
    )
    return f'<div class="chip-row">{rendered}</div>'


def _lens_sorted_candidates(candidates: list, lens: str) -> list:
    if lens == "Balanced Growth / Downside Protection":
        stance_rank = {
            "Ready": 2,
            "Pullback Preferred": 1,
            "Watch Only": 0,
        }
        return sorted(
            candidates,
            key=lambda c: (
                stance_rank.get(_candidate_entry_stance(c), 0),
                -safe_float(getattr(c, "beta_90d", None), default=9.9),
                safe_float(getattr(c, "portfolio_fit_score", 0)),
                safe_float(getattr(c, "final_rank", 0)),
                safe_float(getattr(c, "dividend_yield", 0)),
            ),
            reverse=True,
        )
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


def _render_candidate_detail_card(cand, label: str = "Selected") -> None:
    """Render a full discovery candidate card with metrics and pillar bars.

    Reused by Featured Recommendations and the All Scored detail view.
    """
    _country_flags = {
        "US": "US", "UK": "UK", "GB": "UK", "CA": "CA",
        "DE": "DE", "FR": "FR", "IT": "IT", "ES": "ES",
        "NL": "NL", "JP": "JP",
    }
    _conf_label, _conf_tone, _ = _discovery_confidence(cand)
    _action = getattr(cand, "action", "NEUTRAL")
    if _action in ("STRONG BUY", "BUY"):
        _action_tone = "buy"
    elif _action == "INSUFFICIENT DATA":
        _action_tone = "data"
    elif _action in ("AVOID", "SELL", "STRONG SELL"):
        _action_tone = "avoid"
    else:
        _action_tone = "neutral"
    _mcap = safe_float(cand.market_cap)
    _geo = _country_flags.get(cand.country, "Global")
    _entry_stance = _candidate_entry_stance(cand)
    _entry_tone = _entry_stance_tone(_entry_stance)
    _subtitle = (
        f"{_geo} · {cand.exchange} · {cand.sector}"
        + (f" · {format_currency(_mcap / 1e9, 'GBP', decimals=1)}B mcap" if _mcap > 0 else "")
    )
    _badge_html = (
        f'<div class="badge-row">'
        f'<span class="signal-badge {_action_tone}">{_html.escape(_action)}</span>'
        f'<span class="signal-badge {_entry_tone}">{_html.escape(_entry_stance)}</span>'
        f'<span class="confidence-chip {_conf_tone}">{_html.escape(_conf_label)}</span>'
        f'</div>'
    )
    st.markdown(
        f"""
        <div class="rec-hero">
            <div class="rec-rankline">
                <div class="rec-rank">{_html.escape(label)}</div>
                <div class="rec-rank">Final Rank {safe_float(cand.final_rank):.3f}</div>
            </div>
            <div class="rec-title">{_html.escape(cand.ticker)}</div>
            <div class="rec-subtitle">{_html.escape(cand.name)}<br>{_html.escape(_subtitle)}</div>
            {_badge_html}
            <div class="rec-thesis">{_html.escape(_candidate_thesis(cand))}</div>
            {_render_html_chips(_candidate_evidence_tags(cand))}
            {_render_html_chips([(tag, "risk") for tag in _candidate_risk_tags(cand)])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Momentum", f"{safe_float(cand.momentum_score):.2f}")
    mc2.metric("Fit", f"{safe_float(cand.portfolio_fit_score):.2f}")
    mc3.metric("90d Model", format_pct(safe_float(getattr(cand, "expected_return_90d", 0)) * 100))
    st.markdown(
        _render_pillar_bars(
            safe_float(cand.technical_score),
            safe_float(cand.fundamental_score),
            safe_float(cand.sentiment_score),
            safe_float(cand.forecast_score),
        ),
        unsafe_allow_html=True,
    )
    # --- Trading Plan section ---
    _entry_p = safe_float(getattr(cand, "entry_price", None))
    _stop_p = safe_float(getattr(cand, "stop_loss", None))
    _tp_p = safe_float(getattr(cand, "take_profit", None))
    _has_plan = _entry_p and _entry_p > 0
    if _has_plan:
        st.markdown("**Trading Plan**")
        _cur = getattr(cand, "currency", "USD")
        _plan_cols = st.columns(4)
        with _plan_cols[0]:
            st.metric("Entry Price",
                      _format_price(_entry_p, _cur),
                      help=f"Method: {getattr(cand, 'entry_method', 'N/A')}")
        with _plan_cols[1]:
            st.metric("Stop Loss",
                      _format_price(_stop_p, _cur) if _stop_p else "N/A",
                      help=f"Method: {getattr(cand, 'stop_method', 'N/A')}")
        with _plan_cols[2]:
            st.metric("Take Profit",
                      _format_price(_tp_p, _cur) if _tp_p else "N/A",
                      help=f"Method: {getattr(cand, 'target_method', 'N/A')}")
        with _plan_cols[3]:
            _rr = safe_float(getattr(cand, "r_r_ratio", None))
            st.metric("R/R Ratio",
                      f"{_rr:.1f}x" if _rr and _rr > 0 else "N/A")
        with st.expander("Entry Details", expanded=False):
            _cur = getattr(cand, "currency", "USD")
            _tp_cols = st.columns(4)
            with _tp_cols[0]:
                st.metric("Entry Price",
                          _format_price(_entry_p, _cur),
                          help=f"Method: {getattr(cand, 'entry_method', 'N/A')}")
            with _tp_cols[1]:
                st.metric("Stop Loss",
                          _format_price(_stop_p, _cur) if _stop_p else "N/A",
                          help=f"Method: {getattr(cand, 'stop_method', 'N/A')}")
            with _tp_cols[2]:
                st.metric("Take Profit",
                          _format_price(_tp_p, _cur) if _tp_p else "N/A",
                          help=f"Method: {getattr(cand, 'target_method', 'N/A')}")
            with _tp_cols[3]:
                _rr = safe_float(getattr(cand, "r_r_ratio", None))
                st.metric("R/R Ratio",
                          f"{_rr:.1f}x" if _rr and _rr > 0 else "N/A")

            _tp_cols2 = st.columns(4)
            with _tp_cols2[0]:
                _fill = safe_float(getattr(cand, "fill_probability", None))
                st.metric("Fill Prob.",
                          format_pct(_fill * 100) if _fill else "N/A")
            with _tp_cols2[1]:
                _sdp = safe_float(getattr(cand, "stop_distance_pct", None))
                st.metric("Stop Distance",
                          format_pct(_sdp) if _sdp else "N/A")
            with _tp_cols2[2]:
                _shares = getattr(cand, "position_size_shares", 0) or 0
                _size_method = getattr(cand, "sizing_method", "") or "Stop-budget sizing"
                st.metric("Shares", f"{_shares:,}" if _shares > 0 else "N/A",
                          help=_size_method.replace("_", " "))
            with _tp_cols2[3]:
                _pw = safe_float(getattr(cand, "position_weight", 0))
                st.metric("Position Weight",
                          format_pct(_pw * 100) if _pw > 0 else "N/A")

            # Entry zone + support levels
            _support = getattr(cand, "support_levels", {}) or {}
            _regime = getattr(cand, "regime_info", {}) or {}
            if _support or _regime:
                _info_parts = []
                if _regime.get("vix_percentile"):
                    _vp = _regime["vix_percentile"]
                    _label = "calm" if _vp < 30 else ("elevated" if _vp < 70 else "stressed")
                    _info_parts.append(f"VIX regime: {_label} ({_vp:.0f}th pctl)")
                if _regime.get("atr_multiplier"):
                    _info_parts.append(f"ATR mult: {_regime['atr_multiplier']:.1f}x")
                _entry_lens = getattr(cand, "entry_lens", "") or ""
                if _entry_lens:
                    _info_parts.append(f"Lens: {_entry_lens}")
                _kelly_cap = safe_float(getattr(cand, "kelly_cap_fraction", None))
                if _kelly_cap > 0:
                    _info_parts.append(f"Kelly cap: {_kelly_cap * 100:.1f}%")
                for _sk, _sv in _support.items():
                    _info_parts.append(
                        f"{_sk}: {_format_price(_sv.get('price', 0), _cur)} "
                        f"({_sv.get('distance_pct', 0):.1f}% below)")
                st.caption(" · ".join(_info_parts))

    with st.expander("Why this ranked here"):
        st.markdown(f"**Thesis:** {_candidate_thesis(cand)}")
        st.caption(f"**System rationale:** {cand.why}")
        st.caption(
            f"Momentum: {format_pct(safe_float(getattr(cand, 'return_90d', 0)) * 100)} over 90d · "
            f"{format_pct(safe_float(getattr(cand, 'return_30d', 0)) * 100)} over 30d · "
            f"volume {safe_float(getattr(cand, 'volume_ratio', 1.0)):.1f}x"
        )
        if cand.fx_penalty_applied:
            st.caption(f"FX drag applied: -{safe_float(cand.fx_penalty_pct):.1f}% ({cand.currency})")
        else:
            st.caption("No FX drag for GBP-denominated exposure.")
        if safe_float(cand.max_correlation) < 0.40:
            st.caption("Portfolio fit: low correlation to the current book.")
        else:
            st.caption(
                f"Portfolio fit: correlation {safe_float(cand.max_correlation):.2f}"
                + (f" with {cand.correlated_with}" if cand.correlated_with else "")
            )


def _exit_card_tags(exit_signal: dict) -> list[tuple[str, str]]:
    chips: list[tuple[str, str]] = []
    score = exit_signal.get("current_score")
    price = exit_signal.get("current_price")
    currency = exit_signal.get("currency", "GBP")
    structural_stop = exit_signal.get("structural_stop_loss", exit_signal.get("stop_loss"))
    trailing_stop = exit_signal.get("trailing_exit_stop")
    take_profit = exit_signal.get("take_profit")
    signal_type = str(exit_signal.get("signal_type", "")).lower()
    base_action = exit_signal.get("base_action")
    final_action = exit_signal.get("final_action")

    if is_valid_number(score):
        chips.append((f"Score {format_score(score)}", "info"))
    if is_valid_number(price):
        chips.append((f"Price {format_currency(price, currency)}", "info"))
    if base_action:
        chips.append((f"Alpha {base_action}", "info"))
    if final_action and final_action != base_action:
        chips.append((f"Final {final_action}", "risk"))
    if is_valid_number(structural_stop):
        chips.append((f"Structural {format_currency(structural_stop, currency)}", "warn"))
    if is_valid_number(trailing_stop):
        chips.append((f"Trailing {format_currency(trailing_stop, currency)}", "risk"))
    if is_valid_number(take_profit):
        chips.append((f"Target {format_currency(take_profit, currency)}", "good"))
    if "score_sell" in signal_type:
        action = exit_signal.get("detail", {}).get("action", "SELL")
        chips.append(("Exit recommended" if action == "STRONG SELL" else "Reduce position", "risk"))
    elif "stop" in signal_type:
        chips.append(("Risk control", "risk"))
    elif "target" in signal_type or "profit" in signal_type:
        chips.append(("Lock gains", "good"))
    elif "decay" in signal_type:
        chips.append(("Signal weakening", "warn"))
    elif "momentum" in signal_type:
        chips.append(("Trend break", "warn"))
    elif "holding" in signal_type:
        chips.append(("Stale position", "warn"))
    return chips[:6]


def _exit_override_html(exit_signal: dict) -> str:
    prior = exit_signal.get("prior_score", exit_signal.get("aggregate_score"))
    exit_score = exit_signal.get("exit_score")
    penalty = exit_signal.get("exit_penalty", exit_signal.get("_exit_penalty"))
    posterior = exit_signal.get("posterior_score", exit_signal.get("_exit_posterior"))
    base_action = exit_signal.get("base_action")
    final_action = exit_signal.get("final_action")
    if not all(is_valid_number(v) for v in (prior, exit_score, penalty, posterior)):
        return ""
    action_txt = ""
    if base_action and final_action:
        action_txt = f"{_html.escape(str(base_action))} &rarr; {_html.escape(str(final_action))} &bull; "
    return (
        '<div style="font-size:12px;color:#94a3b8;margin-top:6px;">'
        f'{action_txt}Prior {safe_float(prior):+.3f} &bull; '
        f'Exit {safe_float(exit_score):.3f} &bull; '
        f'Penalty {safe_float(penalty):+.3f} &bull; '
        f'Posterior {safe_float(posterior):+.3f}'
        '</div>'
    )


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
        "Filter by Final Action",
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

# FX conversion: all values to GBP for accurate portfolio total
try:
    from engine.portfolio_optimizer import _get_fx_rate
except ImportError:
    def _get_fx_rate(currency: str) -> float:
        return {"USD": 0.79, "EUR": 0.86}.get(currency, 1.0) if currency not in ("GBP", "GBX") else 1.0

for r in results:
    _cp = r.get("current_price")
    _ap = r.get("avg_buy_price")
    _qty = r.get("quantity", 0)
    _cur = r.get("currency", "GBP")
    if _cp is not None and _ap is not None and _ap > 0 and _qty > 0:
        # GBX: divide by 100; foreign currencies: convert via FX rate
        factor = 0.01 if _cur == "GBX" else 1.0
        fx_rate = _get_fx_rate(_cur)  # 1.0 for GBP/GBX, ~0.79 for USD, etc.
        cost = _ap * _qty * factor * fx_rate
        value = _cp * _qty * factor * fx_rate
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
# TOP-LEVEL TAB NAVIGATION
# ---------------------------------------------------------------------------
tab_dashboard, tab_holdings, tab_discovery, tab_analytics = st.tabs([
    "📊 Dashboard",
    f"💼 Holdings ({len(results)})",
    "🔍 Discovery",
    "📈 Analytics",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD — Hero + Risk + Exit + Optimizer summary
# ═══════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    # ---------------------------------------------------------------------------
    # Hero section — portfolio summary
    # ---------------------------------------------------------------------------
    hero_left, hero_right = st.columns([2, 1])

    with hero_left:
        total_value = safe_float(total_value)
        total_pl = safe_float(total_pl)
        total_pl_pct = safe_float(total_pl_pct)
        pl_color_cls = "hero-pl-positive" if total_pl >= 0 else "hero-pl-negative"
        pl_sign = "+" if total_pl >= 0 else "-"
        st.markdown(
            f'<div class="hero-value">{format_currency(total_value, "GBP", decimals=0)}</div>'
            f'<div class="hero-pl {pl_color_cls}">'
            f'{pl_sign}{format_currency(abs(total_pl), "GBP", decimals=0)} ({format_pct(total_pl_pct)})</div>',
            unsafe_allow_html=True,
        )
        st.caption("Total portfolio value · unrealised P&L")

        # Best / worst performer
        if per_holding_pl:
            best = max(per_holding_pl, key=lambda x: x[2])
            worst = min(per_holding_pl, key=lambda x: x[2])
            st.markdown(
                f"**Best:** {best[0]} ({format_pct(safe_float(best[2]))}) · "
                f"**Worst:** {worst[0]} ({format_pct(safe_float(worst[2]))})"
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
            from engine.exit_engine import assess_exits as _assess_exits, reconcile_actions_with_exits as _reconcile_exits, exit_signal_to_dict as _exit_to_dict
            _exit_objs = _assess_exits(results, holdings)
            _reconcile_exits(results, _exit_objs)
            _result_map = {r["ticker"]: r for r in results}
            _exit_list = [_exit_to_dict(e, _result_map.get(e.ticker)) for e in _exit_objs]
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
            st.markdown("#### Exit Command Center")
            _urgent_count = sum(1 for e in _exit_list if e.get("severity") == "urgent")
            _action_count = sum(1 for e in _exit_list if e.get("severity") == "action_needed")
            _warning_count = sum(1 for e in _exit_list if e.get("severity") == "warning")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Urgent", _urgent_count)
            ec2.metric("Action Needed", _action_count)
            ec3.metric("Watchlist", _warning_count)

            _sorted_exit = sorted(
                _exit_list,
                key=lambda e: (
                    0 if e.get("severity") == "urgent" else 1 if e.get("severity") == "action_needed" else 2,
                    e.get("ticker", ""),
                ),
            )
            for _card_exit in _sorted_exit:
                _sev = _card_exit.get("severity", "warning")
                _card_cls = "urgent" if _sev == "urgent" else "action" if _sev == "action_needed" else "warning"
                _sev_label = "Urgent" if _sev == "urgent" else "Action Needed" if _sev == "action_needed" else "Watch"
                _title = _html.escape(_card_exit.get("ticker", ""))
                _subtitle = _html.escape(_card_exit.get("name", "")) + " · " + _html.escape(
                    _card_exit.get("signal_type", "").replace("_", " ").title()
                )
                _message = _html.escape(_card_exit.get("message", ""))
                _override = _exit_override_html(_card_exit)
                _chips = _render_html_chips(_exit_card_tags(_card_exit))
                _card_html = (
                    f'<div class="exit-card {_card_cls}">'
                    '<div class="exit-topline">'
                    '<div>'
                    f'<div class="exit-title">{_title}</div>'
                    f'<div class="exit-subtitle">{_subtitle}</div>'
                    '</div>'
                    f'<span class="severity-pill {_card_cls}">{_html.escape(_sev_label)}</span>'
                    '</div>'
                    f'<div class="exit-message">{_message}</div>'
                    f'{_override}'
                    f'{_chips}'
                    '</div>'
                )
                st.markdown(
                    _card_html,
                    unsafe_allow_html=True,
                )

            for _es in []:
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
                text=[f"{safe_float(v):.1f}%" for v in alloc_current],
                textposition="auto",
            ))
            alloc_fig.add_trace(go.Bar(
                name="Suggested",
                x=alloc_tickers, y=[safe_float(v) for v in alloc_suggested],
                marker_color="#3b82f6",
                text=[f"{safe_float(v):.1f}%" for v in alloc_suggested],
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
                    "Current %": format_pct(safe_float(pw['current_weight']) * 100, plus_sign=False),
                    "Suggested %": format_pct(safe_float(pw['suggested_weight']) * 100, plus_sign=False),
                    "Delta": format_pct(safe_float(pw['rebalance_delta']) * 100),
                    "Ann. Vol": format_pct(safe_float(pw['volatility']) * 100, plus_sign=False),
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
        with st.expander(_opt_title, expanded=False):
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Expected Return", format_pct(safe_float(_opt_data['portfolio_expected_return']) * 100))
            oc2.metric("Portfolio Vol", format_pct(safe_float(_opt_data['portfolio_volatility']) * 100, plus_sign=False))
            oc3.metric("Sharpe Ratio", f"{safe_float(_opt_data['portfolio_sharpe']):.2f}")
            oc4.metric("Turnover", format_pct(safe_float(_opt_data['turnover']) * 100, plus_sign=False))

            _regime_label = (_regime or {}).get("regime_label", "NEUTRAL") if _regime else "N/A"
            st.caption(f"Risk-free rate: {safe_float(_opt_data.get('risk_free_rate', 0))*100:.1f}% | "
                       f"Regime: {_regime_label} | Method: {_opt_data.get('method', 'N/A')}")

            for w in _opt_data.get("warnings", []):
                st.info(w)

            # Current vs Optimal chart
            _opt_h = _opt_data["holdings"]
            _opt_tickers = [h["ticker"] for h in _opt_h]
            _opt_current = [safe_float(h["current_weight"]) * 100 for h in _opt_h]
            _opt_optimal = [safe_float(h["optimal_weight"]) * 100 for h in _opt_h]

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
                "Score": format_score(h.get('aggregate_score', 0)),
                "Current %": format_pct(safe_float(h['current_weight']) * 100, plus_sign=False),
                "Optimal %": format_pct(safe_float(h['optimal_weight']) * 100, plus_sign=False),
                "Delta": format_pct(safe_float(h['rebalance_delta']) * 100),
                "E[Return]": format_pct(safe_float(h.get('expected_return', 0)) * 100),
                "Vol": format_pct(safe_float(h.get('volatility', 0)) * 100, plus_sign=False),
                "Sector": h.get("sector", ""),
                "FX Cost": format_pct(safe_float(h['fx_cost_if_rebalanced']) * 100, decimals=2, plus_sign=False) if safe_float(h.get("fx_cost_if_rebalanced", 0)) > 0 else "—",
            } for h in _opt_h])
            st.dataframe(_opt_rows, hide_index=True, use_container_width=True)

            # Rebalance trades
            _trades = _opt_data.get("rebalance_trades", [])
            if _trades:
                st.markdown("**Suggested Rebalance Trades** (> 2% delta)")
                for t in _trades:
                    _dir_color = "#10b981" if t["direction"] == "BUY" else "#ef4444"
                    _cw = safe_float(t["current_weight"])
                    _ow = safe_float(t["optimal_weight"])
                    _dp = safe_float(t["delta_pct"])
                    _tv = safe_float(t["trade_value"])
                    st.markdown(
                        f'<span style="color:{_dir_color};font-weight:600">{t["direction"]}</span> '
                        f'**{t["ticker"]}** ({t["name"]}) — '
                        f'{_cw:.1f}% → {_ow:.1f}% '
                        f'({_dp:+.1f}%, ~{format_currency(_tv, "GBP", decimals=0)})',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("Portfolio is near-optimal — no significant rebalancing needed.")

            # Sector & FX exposure
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Optimal Sector Weights**")
                for sector, weight in sorted(_opt_data.get("sector_weights", {}).items(), key=lambda x: -safe_float(x[1])):
                    weight = safe_float(weight)
                    bar_w = int(weight * 200)
                    st.markdown(f"`{sector:20s}` {'█' * max(bar_w // 5, 1)} {weight*100:.1f}%")
            with sc2:
                st.markdown("**FX Exposure**")
                for ccy, weight in sorted(_opt_data.get("fx_exposure", {}).items(), key=lambda x: -safe_float(x[1])):
                    weight = safe_float(weight)
                    st.markdown(f"`{ccy:5s}` {weight*100:.1f}%")


with tab_holdings:
    # ---------------------------------------------------------------------------
    # Sort & filter holdings
    # ---------------------------------------------------------------------------
    filtered_results = list(results)
    if action_filter:
        filtered_results = [
            r for r in filtered_results
            if r.get("final_action", r.get("action")) in action_filter
        ]

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
        final_action = r.get("final_action", r["action"])
        base_action = r.get("base_action", final_action)
        action = final_action
        currency = r.get("currency", "GBP")
        _cp = r.get("current_price")
        _ap = r.get("avg_buy_price")
        _qty = r.get("quantity", 0)
        _structural_stop = r.get("structural_stop_loss", r.get("stop_loss"))
        _trailing_stop = r.get("trailing_exit_stop")
        _structural_method = r.get("structural_stop_method", r.get("stop_method", "N/A"))
        _trailing_method = r.get("trailing_exit_method", "Not triggered")

        with st.container(border=True):
            # ── Header row: Ticker | Score bar | Action pill ──
            hdr1, hdr2, hdr3 = st.columns([3, 2.5, 1.2])

            with hdr1:
                # Ticker + name + daily change
                change_val = r.get("daily_change_pct")
                change_color = "#10b981" if is_valid_number(change_val) and change_val >= 0 else "#ef4444"
                change_txt = format_pct(change_val, decimals=2) if is_valid_number(change_val) else ""
                st.markdown(
                    f"**{r['ticker']}** &nbsp;·&nbsp; {r['name']} &nbsp;"
                    f'<span style="color:{change_color};font-weight:600;font-size:0.85rem">{change_txt}</span>',
                    unsafe_allow_html=True,
                )

            with hdr2:
                _er90 = r.get("expected_return_90d")
                _er90_txt = f" · 90d: {format_pct(safe_float(_er90) * 100)}" if is_valid_number(_er90) else ""
                st.markdown(f"**Score: {format_score(r['aggregate_score'])}**{_er90_txt}")
                st.markdown(_render_score_bar(r["aggregate_score"]), unsafe_allow_html=True)
                _action_caption = (
                    f"Alpha: {base_action} → Final: {final_action}"
                    if base_action != final_action
                    else f"Alpha = Final: {final_action}"
                )
                st.caption(_action_caption)

            with hdr3:
                st.markdown(
                    f'<div style="text-align:center;padding-top:4px">{_render_action_pill(final_action)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="text-align:center;font-size:12px;color:#94a3b8;margin-top:6px;">'
                    f'Base Alpha: {_html.escape(str(base_action))}'
                    f'</div>',
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
                _cp_safe = safe_float(_cp, default=0)
                _ap_safe = safe_float(_ap, default=0)
                if _cp_safe > 0 and _ap_safe > 0:
                    _pl = _cp_safe - _ap_safe
                    _pl_pct = (_pl / _ap_safe) * 100
                    _pl_str = _format_price(abs(_pl), currency)
                    if _pl < 0:
                        _pl_str = f"-{_pl_str}"
                    factor = 0.01 if currency == "GBX" else 1.0
                    _total_pl = _pl * safe_float(_qty) * factor
                    _total_str = format_currency(abs(_total_pl), currency if currency != "GBX" else "GBP", decimals=0)
                    if _total_pl < 0:
                        _total_str = f"-{_total_str}"
                    st.metric("P&L / Share", _pl_str, format_pct(_pl_pct))
                    st.metric("Total P&L", _total_str, format_pct(_pl_pct),
                        help=f"{safe_float(_qty):.0f} shares × {_format_price(abs(_pl), currency)} per share")
                else:
                    st.metric("P&L / Share", "N/A")

            with b3:
                tp_str = _format_price(r.get("take_profit"), currency)
                sl_str = _format_price(_structural_stop, currency) if is_valid_number(_structural_stop) else "N/A"
                trailing_str = _format_price(_trailing_stop, currency) if is_valid_number(_trailing_stop) else "N/A"
                _sdp = r.get("stop_distance_pct")
                _sdp_str = f" ({_sdp:.1f}%)" if _sdp else ""
                _regime = r.get("regime_info", {}) or {}
                _regime_hint = ""
                if _regime.get("vix_percentile"):
                    _vp = _regime["vix_percentile"]
                    _rl = "calm" if _vp < 30 else ("elevated" if _vp < 70 else "stressed")
                    _regime_hint = f" · VIX: {_rl}"
                st.metric("Target ↑", tp_str, help=f"Method: {r.get('target_method', 'N/A')}")
                st.metric("Structural Stop ↓", sl_str,
                          help=f"Method: {_structural_method}{_sdp_str}{_regime_hint}")
                st.metric("Trailing Exit ↓", trailing_str,
                          help=f"Method: {_trailing_method}")

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

            _override_html = _exit_override_html(r)
            if _override_html:
                st.markdown(_override_html, unsafe_allow_html=True)

            # ── Risk overlay flags ──
            _risk_flags = []
            if r.get("is_parabolic"):
                _risk_flags.append(
                    f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.75rem;font-weight:600">PARABOLIC (penalty {safe_float(r.get("parabolic_penalty", 0)):.2f})</span>'
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
            if r.get("governance_flag"):
                _risk_flags.append(
                    '<span style="background:#fef2f2;color:#991b1b;padding:2px 8px;border-radius:4px;'
                    'font-size:0.75rem;font-weight:600">GOVERNANCE CONCERN</span>'
                )
            if r.get("asymmetric_risk_flag"):
                _risk_flags.append(
                    '<span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:4px;'
                    'font-size:0.75rem;font-weight:600">ASYMMETRIC RISK</span>'
                )
            _ex_div_days = r.get("ex_dividend_days")
            if _ex_div_days is not None and 0 <= _ex_div_days <= 14:
                _risk_flags.append(
                    f'<span style="background:#ecfdf5;color:#065f46;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.75rem;font-weight:600">EX-DIV IN {_ex_div_days}d</span>'
                )
            if _risk_flags:
                st.markdown(" ".join(_risk_flags), unsafe_allow_html=True)

            # ── Why row ──
            st.caption(f"**Why:** {r['why']}")

            # ── Tabbed details ──
            with st.expander("Details"):
                tab_overview, tab_scores, tab_fund, tab_sent, tab_fcast = st.tabs([
                    "🏠 Overview", "📊 Scores", "📈 Fundamentals", "📰 Sentiment", "🔮 Forecast"
                ])

                # ─── Tab 0: Overview (Yahoo Finance-style) ───
                with tab_overview:
                    _ov_ticker = r["ticker"]
                    _ov_currency = r.get("currency", "GBP")
                    _ov_info = get_ticker_info(_ov_ticker) if get_ticker_info else {}

                    # --- 6-month price chart ---
                    _ov_df = get_price_history(_ov_ticker)
                    if _ov_df is not None and not _ov_df.empty:
                        _ov_closes = _ov_df["Close"]
                        if isinstance(_ov_closes, pd.DataFrame):
                            _ov_closes = _ov_closes.iloc[:, 0]
                        # Last 126 trading days ≈ 6 months
                        _ov_chart_data = _ov_closes.tail(126)

                        _ov_start = float(_ov_chart_data.iloc[0])
                        _ov_end = float(_ov_chart_data.iloc[-1])
                        _ov_pct = ((_ov_end - _ov_start) / _ov_start * 100) if _ov_start > 0 else 0
                        _ov_color = "#10b981" if _ov_pct >= 0 else "#ef4444"

                        fig_ov = go.Figure()
                        fig_ov.add_trace(go.Scatter(
                            x=_ov_chart_data.index,
                            y=_ov_chart_data.values,
                            mode="lines",
                            line=dict(color=_ov_color, width=2),
                            fill="tozeroy",
                            fillcolor=f"rgba{tuple(list(int(_ov_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}",
                            showlegend=False,
                            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
                        ))
                        # Add avg buy price reference line
                        if _ap and safe_float(_ap) > 0:
                            fig_ov.add_hline(
                                y=safe_float(_ap), line_dash="dash",
                                line_color="#6b7280", opacity=0.6,
                                annotation_text=f"Avg buy: {_format_price(_ap, _ov_currency)}",
                                annotation_position="bottom right",
                                annotation_font_size=10,
                                annotation_font_color="#9ca3af",
                            )

                        # --- Key Events overlay (Yahoo Finance-style circles) ---
                        _ev_dates, _ev_prices, _ev_colors, _ev_texts, _ev_symbols = [], [], [], [], []
                        _chart_start = _ov_chart_data.index[0]

                        # Event type 1: Earnings dates (blue circles, cached)
                        try:
                            _earnings_dates = _get_earnings_dates(_ov_ticker)
                            if _earnings_dates is not None and not _earnings_dates.empty:
                                for ed in _earnings_dates.index:
                                    ed_ts = pd.Timestamp(ed).tz_localize(None) if ed.tzinfo else pd.Timestamp(ed)
                                    chart_start_ts = pd.Timestamp(_chart_start).tz_localize(None) if _chart_start.tzinfo else pd.Timestamp(_chart_start)
                                    if ed_ts >= chart_start_ts:
                                        # Find nearest trading day price
                                        _nearest = _ov_chart_data.index.get_indexer([ed_ts], method="nearest")
                                        if len(_nearest) > 0 and _nearest[0] >= 0:
                                            _idx = _nearest[0]
                                            _surprise = _earnings_dates.iloc[_earnings_dates.index.get_loc(ed)].get("Surprise(%)")
                                            _label = "Earnings"
                                            if _surprise is not None and not pd.isna(_surprise):
                                                _label = f"Earnings: {float(_surprise):+.1f}% surprise"
                                                _ev_colors.append("#10b981" if float(_surprise) >= 0 else "#ef4444")
                                            else:
                                                _ev_colors.append("#60a5fa")  # blue for upcoming/no data
                                            _ev_dates.append(_ov_chart_data.index[_idx])
                                            _ev_prices.append(float(_ov_chart_data.iloc[_idx]))
                                            _ev_texts.append(_label)
                                            _ev_symbols.append("circle")
                        except Exception:
                            pass

                        # Event type 2: News headlines with dates (orange/green circles)
                        _news = r.get("news_headlines", [])
                        for _nh in _news:
                            _nh_date = _nh.get("date") or _nh.get("published")
                            if not _nh_date:
                                continue
                            try:
                                _nd = pd.Timestamp(_nh_date)
                                if _nd.tzinfo:
                                    _nd = _nd.tz_localize(None)
                                chart_start_ts = pd.Timestamp(_chart_start).tz_localize(None) if _chart_start.tzinfo else pd.Timestamp(_chart_start)
                                if _nd >= chart_start_ts:
                                    _nearest = _ov_chart_data.index.get_indexer([_nd], method="nearest")
                                    if len(_nearest) > 0 and _nearest[0] >= 0:
                                        _idx = _nearest[0]
                                        _sent = _nh.get("sentiment", 0)
                                        _ev_dates.append(_ov_chart_data.index[_idx])
                                        _ev_prices.append(float(_ov_chart_data.iloc[_idx]))
                                        _ev_texts.append(_nh.get("title", "News")[:60])
                                        _ev_colors.append("#f59e0b" if _sent >= 0 else "#ef4444")
                                        _ev_symbols.append("diamond")
                            except Exception:
                                continue

                        # Event type 3: Analyst target (green triangle)
                        if r.get("analyst_target") and r.get("num_analysts"):
                            _rec = r.get("analyst_rec", "").replace("_", " ")
                            _ev_dates.append(_ov_chart_data.index[-1])
                            _ev_prices.append(float(_ov_chart_data.iloc[-1]) * 1.02)
                            _ev_texts.append(f"Analyst: {_rec} (target {_format_price(r['analyst_target'], _ov_currency)})")
                            _ev_colors.append("#a78bfa")  # purple
                            _ev_symbols.append("triangle-up")

                        if _ev_dates:
                            fig_ov.add_trace(go.Scatter(
                                x=_ev_dates,
                                y=_ev_prices,
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=_ev_colors,
                                    line=dict(width=1.5, color="rgba(0,0,0,0.3)"),
                                    symbol=_ev_symbols,
                                ),
                                text=_ev_texts,
                                hovertemplate="%{text}<br>%{x|%d %b %Y}<br>Price: %{y:.2f}<extra></extra>",
                                showlegend=False,
                            ))

                        _ov_layout = {**_PLOTLY_LAYOUT, "margin": dict(l=50, r=10, t=25, b=25)}
                        fig_ov.update_layout(
                            **_ov_layout,
                            height=250,
                            yaxis=dict(title=None, gridcolor="rgba(128,128,128,0.15)"),
                            xaxis=dict(title=None, gridcolor="rgba(128,128,128,0.10)"),
                        )
                        fig_ov.update_layout(
                            title=dict(
                                text=f"6M: {format_pct(_ov_pct)}",
                                font=dict(size=12, color=_ov_color),
                                x=0.01, y=0.98,
                            ),
                        )
                        st.plotly_chart(fig_ov, use_container_width=True, config={"displayModeBar": False})

                    # --- Key stats grid (Yahoo Finance style) ---
                    _ov_prev_close = _ov_info.get("previousClose") or _ov_info.get("regularMarketPreviousClose")
                    _ov_open = _ov_info.get("open") or _ov_info.get("regularMarketOpen")
                    _ov_day_low = _ov_info.get("dayLow") or _ov_info.get("regularMarketDayLow")
                    _ov_day_high = _ov_info.get("dayHigh") or _ov_info.get("regularMarketDayHigh")
                    _ov_52w_low = _ov_info.get("fiftyTwoWeekLow")
                    _ov_52w_high = _ov_info.get("fiftyTwoWeekHigh")
                    _ov_mcap = _ov_info.get("marketCap")
                    _ov_beta = _ov_info.get("beta") or _ov_info.get("beta3Year")
                    _ov_pe = _ov_info.get("trailingPE") or _ov_info.get("forwardPE")
                    _ov_eps = _ov_info.get("trailingEps")
                    _ov_volume = _ov_info.get("volume") or _ov_info.get("regularMarketVolume")
                    _ov_avg_vol = _ov_info.get("averageVolume")
                    _ov_div_yield = _ov_info.get("dividendYield")
                    _ov_target = _ov_info.get("targetMeanPrice")
                    _ov_sector = _ov_info.get("sector", "")
                    _ov_industry = _ov_info.get("industry", "")

                    def _ov_fmt(val, fmt=",.2f", suffix="", prefix=""):
                        if val is None:
                            return "—"
                        try:
                            return f"{prefix}{float(val):{fmt}}{suffix}"
                        except (TypeError, ValueError):
                            return "—"

                    def _ov_fmt_mcap(val):
                        if val is None:
                            return "—"
                        try:
                            v = float(val)
                        except (TypeError, ValueError):
                            return "—"
                        if v >= 1e12:
                            return f"{v/1e12:.2f}T"
                        if v >= 1e9:
                            return f"{v/1e9:.2f}B"
                        if v >= 1e6:
                            return f"{v/1e6:.0f}M"
                        return f"{v:,.0f}"

                    _ov_day_range = (
                        f"{_ov_fmt(_ov_day_low)} - {_ov_fmt(_ov_day_high)}"
                        if _ov_day_low and _ov_day_high else "—"
                    )
                    _ov_52w_range = (
                        f"{_ov_fmt(_ov_52w_low)} - {_ov_fmt(_ov_52w_high)}"
                        if _ov_52w_low and _ov_52w_high else "—"
                    )

                    # Two-column key stats layout
                    kc1, kc2 = st.columns(2)
                    with kc1:
                        st.markdown(
                            f"""<table style="width:100%;font-size:0.82rem;border-collapse:collapse;color:#e2e8f0">
                            <tr><td style="padding:3px 8px;color:#94a3b8">Previous Close</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_prev_close)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Open</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_open)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Day's Range</td><td style="padding:3px 8px;text-align:right">{_ov_day_range}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">52-Week Range</td><td style="padding:3px 8px;text-align:right">{_ov_52w_range}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Volume</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_volume, ',.0f')}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Avg. Volume</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_avg_vol, ',.0f')}</td></tr>
                            </table>""",
                            unsafe_allow_html=True,
                        )
                    with kc2:
                        st.markdown(
                            f"""<table style="width:100%;font-size:0.82rem;border-collapse:collapse;color:#e2e8f0">
                            <tr><td style="padding:3px 8px;color:#94a3b8">Market Cap</td><td style="padding:3px 8px;text-align:right">{_ov_fmt_mcap(_ov_mcap)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Beta (5Y)</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_beta)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">PE Ratio (TTM)</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_pe)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">EPS (TTM)</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_eps)}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">Dividend Yield</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_div_yield, '.2%') if _ov_div_yield else '—'}</td></tr>
                            <tr><td style="padding:3px 8px;color:#94a3b8">1Y Target Est.</td><td style="padding:3px 8px;text-align:right">{_ov_fmt(_ov_target)}</td></tr>
                            </table>""",
                            unsafe_allow_html=True,
                        )

                    if _ov_sector or _ov_industry:
                        st.caption(f"{_ov_sector}{' / ' + _ov_industry if _ov_industry else ''}")

                # ─── Tab 1: Scores ───
                with tab_scores:
                    sc1, sc2 = st.columns([1, 1])

                    with sc1:
                        # 4 pillar metrics
                        st.metric("Technical", format_score(r.get('technical_score', 0), decimals=2),
                            help="SMA, RSI, MACD, Bollinger Bands, Stochastic RSI, OBV, ADX, Williams %R")
                        st.metric("Fundamental", format_score(r.get('fundamental_score', 0), decimals=2),
                            help="P/E, EPS growth, D/E, margins, ROE, FCF, analyst target, short interest, insider activity")
                        st.metric("Sentiment", format_score(r.get('sentiment_score', 0), decimals=2),
                            help="VADER NLP on Google News + Reddit + FMP News")
                        st.metric("Forecast", format_score(r.get('forecast_score', 0), decimals=2),
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
                    if r.get("dividend_yield") is not None:
                        fund_metrics.append(("Div Yield", f"{r['dividend_yield']:.1%}"))
                    if r.get("payout_ratio") is not None:
                        fund_metrics.append(("Payout Ratio", f"{r['payout_ratio']:.0%}"))
                    if r.get("current_ratio") is not None:
                        fund_metrics.append(("Current Ratio", f"{r['current_ratio']:.1f}"))
                    if r.get("net_debt_ebitda") is not None:
                        fund_metrics.append(("ND/EBITDA", f"{r['net_debt_ebitda']:.1f}x"))
                    if r.get("balance_sheet_grade"):
                        fund_metrics.append(("Balance Sheet", r["balance_sheet_grade"]))

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
                        f"Avg Buy: {_format_price(r.get('avg_buy_price'), currency)} · "
                        f"Qty: {r.get('quantity', 0)} · "
                        f"Stop: {r.get('stop_method', 'N/A')} · Target: {r.get('target_method', 'N/A')}"
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
                        fc_cols[0].metric("Predicted", _format_price(r.get("forecast_price"), currency),
                                         f"{r.get('forecast_pct_change', 0):+.1f}%")
                        fc_cols[1].metric("Low (80%)", _format_price(r.get("forecast_low"), currency))
                        fc_cols[2].metric("High (80%)", _format_price(r.get("forecast_high"), currency))
                        if r.get("forecast_ensemble_mae") is not None:
                            fc_cols[3].metric("Ensemble MAE", f"{r.get('forecast_ensemble_mae', 0):.2f}")
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
                            lc[0].metric("Predicted", _format_price(r.get("forecast_price_long"), currency),
                                         f"{r.get('forecast_pct_change_long', 0):+.1f}%")
                            lc[1].metric("Low (80%)", _format_price(r.get("forecast_low_long", 0), currency))
                            lc[2].metric("High (80%)", _format_price(r.get("forecast_high_long", 0), currency))
                            if r.get("forecast_ensemble_mae_long") is not None:
                                lc[3].metric("Ensemble MAE", f"{r.get('forecast_ensemble_mae_long', 0):.2f}")
                            else:
                                lc[3].metric("Ensemble MAE", "Building...")
                    else:
                        st.info("Forecast data not available for this holding.")



with tab_analytics:
    # ---------------------------------------------------------------------------
    # 90-Day Portfolio Return Projection
    # ---------------------------------------------------------------------------
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
        pc1.metric("Expected Return", format_pct(safe_float(_proj.expected_return_pct)))
        pc2.metric("P(Positive)", f"{safe_float(_proj.prob_positive):.0%}")
        pc3.metric("Current Value", format_currency(_proj.current_value, "GBP", decimals=0))
        pc4.metric("Expected Value", format_currency(_proj.expected_value, "GBP", decimals=0))
        _gain = safe_float(_proj.expected_value) - safe_float(_proj.current_value)
        pc5.metric("Expected Gain", format_currency(_gain, "GBP", decimals=0))

        # Confidence interval table
        st.markdown("#### Confidence Intervals")
        ci_data = []
        for pctile, label in [(0.10, "Bear (10th)"), (0.25, "Cautious (25th)"),
                               (0.50, "Median (50th)"), (0.75, "Optimistic (75th)"),
                               (0.90, "Bull (90th)")]:
            _pv = safe_float(_proj.projected_values[pctile])
            ci_data.append({
                "Scenario": label,
                "Portfolio Return": format_pct(safe_float(_proj.projected_returns[pctile])),
                "Portfolio Value": format_currency(_pv, "GBP", decimals=0),
                "Gain / Loss": format_currency(_pv - safe_float(_proj.current_value), "GBP", decimals=0),
            })
        st.dataframe(pd.DataFrame(ci_data), hide_index=True, use_container_width=True)

        # Per-ticker breakdown
        st.markdown("#### Per-Ticker Projections")
        ticker_rows = []
        for tp in sorted(_proj.ticker_projections, key=lambda x: safe_float(x.expected_return_pct), reverse=True):
            ticker_rows.append({
                "Ticker": tp.ticker,
                "Current": format_currency(tp.current_price, "GBP"),
                "MoE Forecast": format_currency(tp.moe_predicted_price, "GBP"),
                "MoE Return": format_pct(safe_float(tp.moe_pct_change)),
                "MC Expected": format_pct(safe_float(tp.expected_return_pct)),
                "10th pct": format_pct(safe_float(tp.projected_returns.get(0.10))),
                "50th pct": format_pct(safe_float(tp.projected_returns.get(0.50))),
                "90th pct": format_pct(safe_float(tp.projected_returns.get(0.90))),
                "P(>0)": f"{safe_float(tp.prob_positive):.0%}",
                "Annual Vol": f"{safe_float(tp.annual_volatility):.0%}",
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
                    st.metric("Expected Return", format_pct(safe_float(_proj_before.expected_return_pct)))
                    st.metric("P(Positive)", f"{safe_float(_proj_before.prob_positive):.0%}")
                    st.metric("Expected Value", format_currency(_proj_before.expected_value, "GBP", decimals=0))
                with bc2:
                    st.markdown(f"**After Swap ({_swap_out_ticker} → {_swap_in_ticker})**")
                    delta_ret = safe_float(_proj_after.expected_return_pct) - safe_float(_proj_before.expected_return_pct)
                    st.metric("Expected Return", format_pct(safe_float(_proj_after.expected_return_pct)),
                              delta=format_pct(delta_ret))
                    delta_prob = safe_float(_proj_after.prob_positive) - safe_float(_proj_before.prob_positive)
                    st.metric("P(Positive)", f"{safe_float(_proj_after.prob_positive):.0%}",
                              delta=f"{delta_prob:+.0%}")
                    delta_val = safe_float(_proj_after.expected_value) - safe_float(_proj_before.expected_value)
                    st.metric("Expected Value", format_currency(_proj_after.expected_value, "GBP", decimals=0),
                              delta=format_currency(delta_val, "GBP", decimals=0))
        else:
            st.info("No discovery candidates cached. Run the Global Discovery Engine first to enable swap impact analysis.")


    # ---------------------------------------------------------------------------
    # Signal Analytics
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("### Signal Analytics")

    @st.cache_data(ttl=300)
    def _load_forecast_store():
        _path = Path(__file__).parent / "forecast_store.json"
        if _path.exists():
            try:
                with open(_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"_corrupt": True}
        return {}

    _store = _load_forecast_store()
    if _store.get("_corrupt"):
        st.warning("forecast_store.json is corrupted — run `python fix_forecast_store.py` to repair it.")
        _store = {}

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


with tab_discovery:
    # ---------------------------------------------------------------------------
    # Global Discovery Engine
    # ---------------------------------------------------------------------------
    st.markdown("### 🔍 Global Discovery Engine")
    st.caption(
        "A portfolio-aware idea engine for surfacing the best new opportunities, strongest diversifiers, "
        "and cleanest momentum setups from the daily discovery universe."
    )

    # Auto-load cached discovery results from dashboard data on first visit
    if "discovery_results" not in st.session_state:
        if True:
            try:
                _cached_disc = _dash.cached_discovery
                _cached_disc_meta = getattr(_dash, "cached_discovery_meta", {}) or {}
                _last_disc_run = _dash.discovery_timestamp
                if _cached_disc:
                    from engine.discovery import ScoredCandidate, DiscoveryResult
                    # Reconstruct ScoredCandidate objects from cached dicts
                    _restored = []
                    for c in _cached_disc:
                        _aggregate_score = c.get("aggregate_score") or 0
                        _technical_score = c.get("technical_score") or 0
                        _fundamental_score = c.get("fundamental_score") or 0
                        _sentiment_score = c.get("sentiment_score") or 0
                        _forecast_score = c.get("forecast_score") or 0
                        _portfolio_fit_score = c.get("portfolio_fit_score") or 0
                        _momentum_score = c.get("momentum_score") or 0
                        _final_rank = c.get("final_rank")
                        if _final_rank is None:
                            _final_rank = _aggregate_score
                        _action = c.get("action") or "INSUFFICIENT DATA"

                        # Older cached discovery payloads could preserve a stale
                        # action/final_rank while missing the pillar scores shown
                        # in the UI. Apply the same quality gate as live ranking.
                        _pillars_all_zero = (
                            abs(_technical_score)
                            + abs(_fundamental_score)
                            + abs(_sentiment_score)
                            + abs(_forecast_score)
                        ) < 0.001
                        if _pillars_all_zero:
                            _action = "INSUFFICIENT DATA"
                            _final_rank = (_final_rank or 0) * 0.30

                        # Use `or` fallback so None values (from older cache
                        # formats that stored the key with a None value) are
                        # replaced with the safe default.
                        _restored.append(ScoredCandidate(
                            ticker=c.get("ticker") or "",
                            name=c.get("name") or c.get("ticker", ""),
                            exchange=c.get("exchange") or "",
                            country=c.get("country") or "",
                            sector=c.get("sector") or "",
                            industry=c.get("industry") or "",
                            market_cap=c.get("market_cap") or 0,
                            currency=c.get("currency") or "USD",
                            aggregate_score=_aggregate_score,
                            technical_score=_technical_score,
                            fundamental_score=_fundamental_score,
                            sentiment_score=_sentiment_score,
                            forecast_score=_forecast_score,
                            action=_action,
                            why=c.get("why") or "",
                            fx_penalty_applied=c.get("fx_penalty_applied") or False,
                            fx_penalty_pct=c.get("fx_penalty_pct") or 0,
                            max_correlation=c.get("max_correlation") or 0,
                            correlated_with=c.get("correlated_with") or "",
                            sector_weight_if_added=c.get("sector_weight_if_added") or 0,
                            portfolio_fit_score=_portfolio_fit_score,
                            momentum_score=_momentum_score,
                            return_90d=c.get("return_90d") or 0,
                            return_30d=c.get("return_30d") or 0,
                            volume_ratio=c.get("volume_ratio") or 1.0,
                            expected_return_90d=c.get("expected_return_90d") or 0,
                            analyst_target=c.get("analyst_target"),
                            analyst_upside=c.get("analyst_upside"),
                            num_analysts=c.get("num_analysts"),
                            insider_buys=c.get("insider_buys") or 0,
                            insider_sells=c.get("insider_sells") or 0,
                            insider_net=c.get("insider_net") or "",
                            beta_90d=c.get("beta_90d"),
                            debt_to_equity=c.get("debt_to_equity"),
                            entry_stance=c.get("entry_stance") or "",
                            ticker_identity_warning=c.get("ticker_identity_warning"),
                            parabolic_penalty=c.get("parabolic_penalty") or 0,
                            is_parabolic=c.get("is_parabolic") or False,
                            earnings_near=c.get("earnings_near") or False,
                            earnings_imminent=c.get("earnings_imminent") or False,
                            earnings_days=c.get("earnings_days"),
                            cap_tier=c.get("cap_tier") or "unknown",
                            confidence_discount=c.get("confidence_discount") or 1.0,
                            max_weight_scale=c.get("max_weight_scale") or 1.0,
                            post_earnings_recent=c.get("post_earnings_recent") or False,
                            post_earnings_days=c.get("post_earnings_days"),
                            earnings_miss=c.get("earnings_miss") or False,
                            earnings_miss_pct=c.get("earnings_miss_pct"),
                            near_52w_high=c.get("near_52w_high") or False,
                            pct_from_52w_high=c.get("pct_from_52w_high"),
                            entry_lens=c.get("entry_lens") or "momentum",
                            entry_price=c.get("entry_price"),
                            entry_method=c.get("entry_method") or "",
                            entry_zone_low=c.get("entry_zone_low"),
                            entry_zone_high=c.get("entry_zone_high"),
                            fill_probability=c.get("fill_probability"),
                            stop_loss=c.get("stop_loss"),
                            stop_method=c.get("stop_method") or "",
                            stop_distance_pct=c.get("stop_distance_pct"),
                            take_profit=c.get("take_profit"),
                            target_method=c.get("target_method") or "",
                            position_size_shares=c.get("position_size_shares") or 0,
                            position_weight=c.get("position_weight") or 0,
                            risk_amount=c.get("risk_amount") or 0,
                            r_r_ratio=c.get("r_r_ratio"),
                            sizing_method=c.get("sizing_method") or "",
                            kelly_cap_fraction=c.get("kelly_cap_fraction"),
                            support_levels=c.get("support_levels") or {},
                            regime_info=c.get("regime_info") or {},
                            # Dividend safety
                            dividend_yield=c.get("dividend_yield"),
                            payout_ratio=c.get("payout_ratio"),
                            ex_dividend_date=c.get("ex_dividend_date"),
                            ex_dividend_days=c.get("ex_dividend_days"),
                            five_year_avg_yield=c.get("five_year_avg_yield"),
                            # Balance sheet strength
                            balance_sheet_grade=c.get("balance_sheet_grade"),
                            net_debt_ebitda=c.get("net_debt_ebitda"),
                            current_ratio=c.get("current_ratio"),
                            cash_to_debt=c.get("cash_to_debt"),
                            # Governance red flag
                            governance_flag=c.get("governance_flag", False),
                            governance_reasons=c.get("governance_reasons") or [],
                            # Asymmetric / binary outcome flag
                            asymmetric_risk_flag=c.get("asymmetric_risk_flag", False),
                            asymmetric_risk_reason=c.get("asymmetric_risk_reason"),
                            final_rank=_final_rank,
                        ))
                    st.session_state["discovery_results"] = DiscoveryResult(
                        candidates=_restored,
                        screened_count=_cached_disc_meta.get("screened_count", 0),
                        after_momentum_screen=_cached_disc_meta.get("after_momentum_screen", 0),
                        after_quick_filter=_cached_disc_meta.get("after_quick_filter", 0),
                        after_corr_filter=_cached_disc_meta.get("after_corr_filter", 0),
                        after_quick_rank=_cached_disc_meta.get("after_quick_rank", 0),
                        fully_scored=_cached_disc_meta.get("fully_scored", len(_restored)),
                        run_time_seconds=_cached_disc_meta.get("run_time_seconds", 0.0),
                        fx_penalties_applied=_cached_disc_meta.get("fx_penalties_applied", 0),
                    )
                    st.session_state["discovery_cached_from"] = _last_disc_run
            except Exception:
                pass

    from utils.orchestrator_status import get_orchestrator_status
    _orch_status = get_orchestrator_status()

    _disc_col1, _disc_col2 = st.columns([1, 3])
    with _disc_col1:
        _run_discovery = st.button(
            "🔎 Re-run Screener",
            type="primary",
            use_container_width=True,
            disabled=_orch_status["running"],
        )
    with _disc_col2:
        if _orch_status["running"]:
            _ckpt = _orch_status.get("checkpoint")
            if _ckpt and _ckpt["total"] > 0:
                _pct = _ckpt["scored_count"] / _ckpt["total"]
                _eta = _orch_status.get("eta_minutes")
                _eta_str = f" · ~{_eta:.0f} min remaining" if _eta else ""
                st.warning(
                    f"⏳ Batch orchestrator running (PID {_orch_status['pid']}) · "
                    f"Scoring {_ckpt['scored_count']}/{_ckpt['total']} "
                    f"({_pct:.0%}){_eta_str}"
                )
            else:
                st.warning(
                    f"⏳ Batch orchestrator running (PID {_orch_status['pid']}) · "
                    f"Screening phase (no scoring progress yet)"
                )
        else:
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
        from daily_orchestrator import save_discovery_results
        from utils.state_manager import load_state

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
        _disc_progress.progress(1.0, text="Discovery complete! Saving results...")

        # Persist results — same as batch orchestrator
        _disc_state = load_state()
        save_discovery_results(disc_result, _disc_state)

        st.session_state["discovery_results"] = disc_result
        st.session_state["discovery_cached_from"] = datetime.now().isoformat()

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

            _best_idea, _best_idea_meta, _best_idea_sub = _pick_best_new_opportunity(disc.candidates)
            _best_fit = max(
                disc.candidates,
                key=lambda c: (
                    safe_float(getattr(c, "portfolio_fit_score", 0)),
                    safe_float(getattr(c, "final_rank", 0)),
                ),
            )
            _best_momentum = max(
                disc.candidates,
                key=lambda c: (
                    safe_float(getattr(c, "momentum_score", 0)),
                    safe_float(getattr(c, "return_90d", 0)),
                ),
            )
            _watch_candidate = max(
                disc.candidates[: min(10, len(disc.candidates))],
                key=lambda c: (
                    len(_candidate_risk_tags(c)),
                    safe_float(getattr(c, "parabolic_penalty", 0)),
                    safe_float(getattr(c, "fx_penalty_pct", 0)),
                ),
            )

            st.markdown("#### Discovery Command Center")
            cc1, cc2, cc3, cc4 = st.columns(4)
            _command_cards = [
                (
                    cc1, "best", "Best New Opportunity", _best_idea,
                    f"Final rank {safe_float(getattr(_best_idea, 'final_rank', 0)):.3f} · {getattr(_best_idea, 'action', 'NEUTRAL')}",
                ),
                (
                    cc2, "fit", "Best Diversifier", _best_fit,
                    f"Portfolio fit {safe_float(getattr(_best_fit, 'portfolio_fit_score', 0)):.2f} · Corr {safe_float(getattr(_best_fit, 'max_correlation', 0)):.2f}",
                ),
                (
                    cc3, "momentum", "Momentum Leader", _best_momentum,
                    f"Momentum {safe_float(getattr(_best_momentum, 'momentum_score', 0)):.2f} · 90d {format_pct(safe_float(getattr(_best_momentum, 'return_90d', 0)) * 100)}",
                ),
                (
                    cc4, "risk", "Biggest Watchout", _watch_candidate,
                    _candidate_risk_tags(_watch_candidate)[0] if _candidate_risk_tags(_watch_candidate) else "No major red-flag overlays in the current top tier",
                ),
            ]
            _command_cards = [
                (cc1, "best", "Best New Opportunity", _best_idea, _best_idea_meta, _best_idea_sub),
                (
                    cc2,
                    "fit",
                    "Best Diversifier",
                    _best_fit,
                    f"Portfolio fit {safe_float(getattr(_best_fit, 'portfolio_fit_score', 0)):.2f} · Corr {safe_float(getattr(_best_fit, 'max_correlation', 0)):.2f}",
                    _candidate_thesis(_best_fit),
                ),
                (
                    cc3,
                    "momentum",
                    "Momentum Leader",
                    _best_momentum,
                    f"Momentum {safe_float(getattr(_best_momentum, 'momentum_score', 0)):.2f} · 90d {format_pct(safe_float(getattr(_best_momentum, 'return_90d', 0)) * 100)}",
                    _candidate_thesis(_best_momentum),
                ),
                (
                    cc4,
                    "risk",
                    "Biggest Watchout",
                    _watch_candidate,
                    _candidate_risk_tags(_watch_candidate)[0] if _candidate_risk_tags(_watch_candidate) else "No major red-flag overlays in the current top tier",
                    _candidate_thesis(_watch_candidate),
                ),
            ]
            for _col, _tone, _label, _cand, _meta, _sub in _command_cards:
                with _col:
                    _title = _html.escape(getattr(_cand, "ticker", "")) if _cand else "No clean entry"
                    st.markdown(
                        f"""
                        <div class="insight-card {_tone}">
                            <div class="insight-label">{_html.escape(_label)}</div>
                            <div class="insight-title">{_title}</div>
                            <div class="insight-sub">{_html.escape(_sub)}</div>
                            <div class="insight-meta">{_html.escape(_meta)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Recommendation Lens")
            _lens = st.radio(
                "Recommendation lens",
                ["Balanced Growth / Downside Protection", "Best Ideas", "Best Diversifiers", "Momentum Leaders", "Value / Quality"],
                horizontal=True,
                label_visibility="collapsed",
                key="discovery_lens",
            )
            _lens_notes = {
                "Balanced Growth / Downside Protection": "Prioritises ready entries, lower beta, portfolio fit, and income support over raw excitement.",
                "Best Ideas": "Balanced view of conviction, confidence, momentum, and portfolio fit.",
                "Best Diversifiers": "Highlights names that improve portfolio shape without giving up too much quality.",
                "Momentum Leaders": "Pulls the strongest trend-following setups to the front.",
                "Value / Quality": "Pushes fundamental strength and cleaner business quality higher in the stack.",
            }
            st.markdown(f'<div class="lens-note">{_html.escape(_lens_notes[_lens])}</div>', unsafe_allow_html=True)

            _featured = _lens_sorted_candidates(disc.candidates, _lens)[:3]
            if _featured:
                st.markdown("#### Featured Recommendations")
                _feature_cols = st.columns(len(_featured))
                _country_flags = {
                    "US": "US", "UK": "UK", "GB": "UK", "CA": "CA",
                    "DE": "DE", "FR": "FR", "IT": "IT", "ES": "ES",
                    "NL": "NL", "JP": "JP",
                }
                for idx, (col, cand) in enumerate(zip(_feature_cols, _featured), start=1):
                    with col:
                        _render_candidate_detail_card(cand, label=f"#{idx} in {_lens}")

            # Top 3 recommendations
            top_3 = disc.candidates[:3]
            if False and top_3:
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
                            "INSUFFICIENT DATA": "⚫",
                        }
                        dot = action_colors.get(cand.action, "⚪")

                        st.markdown(f"**#{idx + 1} {flag} {cand.ticker}**")
                        st.markdown(f"*{cand.name}*")
                        st.metric("Final Rank", f"{cand.final_rank:.3f}", delta=f"{cand.action}")
                        st.caption(f"📍 {cand.exchange} · {cand.sector}")
                        _mcap = safe_float(cand.market_cap)
                        st.caption(f"Market Cap: {format_currency(_mcap / 1e9, 'GBP', decimals=1)}B" if _mcap > 0 else "Market Cap: N/A")

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
                        if getattr(cand, "earnings_miss", False):
                            _miss_pct = getattr(cand, "earnings_miss_pct", None)
                            _miss_str = f" ({_miss_pct:+.0f}%)" if _miss_pct is not None else ""
                            _days_ago = getattr(cand, "post_earnings_days", None)
                            _days_str = f" {_days_ago}d ago" if _days_ago else ""
                            _cand_risk.append(f"❌ EARNINGS MISS{_miss_str}{_days_str}")
                        elif getattr(cand, "post_earnings_recent", False):
                            _days_ago = getattr(cand, "post_earnings_days", None)
                            _cand_risk.append(f"📊 Reported {_days_ago}d ago" if _days_ago else "📊 Recent earnings")
                        if getattr(cand, "earnings_imminent", False):
                            _cand_risk.append(f"⚠️ Earnings in {cand.earnings_days}d")
                        elif getattr(cand, "earnings_near", False):
                            _cand_risk.append(f"📅 Earnings in {cand.earnings_days}d")
                        if getattr(cand, "near_52w_high", False):
                            _pct = getattr(cand, "pct_from_52w_high", None)
                            _pct_str = f" ({_pct:.1%} below)" if _pct is not None else ""
                            _cand_risk.append(f"📈 NEAR 52W HIGH{_pct_str}")
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

            # Full results table — click a row to see the detail card
            with st.expander("📋 All Scored Candidates"):
                disc_rows = []
                for c in disc.candidates:
                    _c_entry = safe_float(getattr(c, "entry_price", None))
                    _c_stop = safe_float(getattr(c, "stop_loss", None))
                    _c_rr = safe_float(getattr(c, "r_r_ratio", None))
                    disc_rows.append({
                        "Rank": c.final_rank,
                        "Ticker": c.ticker,
                        "Name": c.name,
                        "Exchange": c.exchange,
                        "Sector": c.sector,
                        "Entry Stance": _candidate_entry_stance(c),
                        "↓ Score": c.aggregate_score,
                        "Confidence": _discovery_confidence(c)[0],
                        "Fit": c.portfolio_fit_score,
                        "Beta": round(safe_float(getattr(c, "beta_90d", None), default=0.0), 2) if getattr(c, "beta_90d", None) is not None else "—",
                        "Yield": f"{safe_float(getattr(c, 'dividend_yield', 0)) * 100:.1f}%" if getattr(c, "dividend_yield", None) is not None else "—",
                        "Identity": "Verify" if getattr(c, "ticker_identity_warning", None) else "OK",
                        "Action": c.action,
                        "Tech": round(safe_float(c.technical_score), 2),
                        "Fund": round(safe_float(c.fundamental_score), 2),
                        "Sent": round(safe_float(c.sentiment_score), 2),
                        "Fcast": round(safe_float(c.forecast_score), 2),
                        "Entry": f"{_c_entry:.2f}" if _c_entry else "—",
                        "Stop": f"{_c_stop:.2f}" if _c_stop else "—",
                        "R/R": f"{_c_rr:.1f}x" if _c_rr and _c_rr > 0 else "—",
                    })
                if disc_rows:
                    _disc_df = pd.DataFrame(disc_rows)

                    def _pillar_color(val):
                        """Cell colour matching pillar bar logic."""
                        try:
                            v = float(val)
                        except (ValueError, TypeError):
                            return ""
                        if v > 0.1:
                            return "color: #10b981"
                        elif v < -0.1:
                            return "color: #ef4444"
                        return "color: #6b7280"

                    _pillar_cols = ["Tech", "Fund", "Sent", "Fcast"]
                    _styled = _disc_df.style.map(
                        _pillar_color, subset=_pillar_cols,
                    ).format(
                        {col: "{:+.2f}" for col in _pillar_cols},
                    )
                    _selection = st.dataframe(
                        _styled,
                        hide_index=True,
                        use_container_width=True,
                        selection_mode="single-row",
                        on_select="rerun",
                        key="disc_all_scored_table",
                    )

                    # Render detail card for selected row
                    _sel_rows = _selection.get("selection", {}).get("rows", [])
                    if _sel_rows:
                        _sel_idx = _sel_rows[0]
                        if 0 <= _sel_idx < len(disc.candidates):
                            _sel_cand = disc.candidates[_sel_idx]
                            st.divider()
                            _render_candidate_detail_card(_sel_cand, label=f"{_sel_cand.ticker} Detail")

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

        # Discovery-specific evaluation — ranked by final_rank
        try:
            from engine.discovery_eval import get_discovery_scorecard

            _disc_sc = get_discovery_scorecard()
            if _disc_sc and _disc_sc.get("total_evaluated", 0) >= 5:
                with st.expander(
                    f"🎯 **Discovery Quality** ({_disc_sc['total_evaluated']} evaluated picks)"
                ):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("Top-10 Hit Rate", f"{safe_float(_disc_sc.get('top10_hit_rate_90d')):.0%}")
                    dc2.metric("Top-10 Avg Return", format_pct(_disc_sc.get("top10_avg_return_90d")))
                    dc3.metric("Excess vs SPY", format_pct(_disc_sc.get("excess_vs_spy_90d")))
                    dc4.metric("Ranking Stability", f"{safe_float(_disc_sc.get('ranking_stability')):.0%}")

                    dc5, dc6 = st.columns(2)
                    dc5.metric("Top-30 Hit Rate", f"{safe_float(_disc_sc.get('top30_hit_rate_90d')):.0%}")
                    dc6.metric("Swap Success Rate", f"{safe_float(_disc_sc.get('swap_success_rate')):.0%}")

                    if _disc_sc.get("summary"):
                        st.caption(_disc_sc["summary"])
        except Exception as _disc_sc_err:
            import logging as _logging
            _logging.getLogger(__name__).warning("Discovery scorecard failed: %s", _disc_sc_err)



with tab_analytics:
    # ---------------------------------------------------------------------------
    # Trade History — Record Sales
    # ---------------------------------------------------------------------------
    st.markdown("### Trade History")

    from utils.data_fetch import load_portfolio_full, record_sale

    _portfolio_full = load_portfolio_full()
    _trade_history = _portfolio_full.get("trade_history", [])

    tab_record, tab_history = st.tabs(["Record a Sale", "Past Trades"])

    with tab_record:
        st.caption("When you sell a stock on your broker, record it here to track P&L and update your portfolio.")
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
            st.info("No trades recorded yet. Use the 'Record a Sale' tab when you sell a stock on your broker.")


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
    "Always do your own research. Trades must be executed manually on your broker."
)
