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
from utils.discovery_digest import (
    build_trade_packet,
    candidate_entry_stance as shared_candidate_entry_stance,
    candidate_entry_trigger as shared_candidate_entry_trigger,
    candidate_readiness_summary as shared_candidate_readiness_summary,
    discovery_confidence as shared_discovery_confidence,
    is_top_pick as shared_is_top_pick,
    sort_trade_packets,
)
from utils.safe_numeric import safe_float, is_valid_number, format_currency, format_pct, format_score

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ISA Portfolio Dashboard",
    page_icon=":chart_with_upwards_trend:",
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
    width: 76px;
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
/* Top Pick highlight */
.rec-hero.top-pick {
    border: 2px solid rgba(234, 179, 8, 0.55);
    background:
        radial-gradient(circle at top right, rgba(234, 179, 8, 0.12), transparent 38%),
        linear-gradient(150deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.14), 0 0 0 1px rgba(234, 179, 8, 0.18);
}
.top-pick-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.26rem 0.62rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: rgba(234, 179, 8, 0.18);
    color: #fde68a;
    border: 1px solid rgba(234, 179, 8, 0.35);
}
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

_ACTION_LABELS = {
    "STRONG BUY": "Ready to buy",
    "BUY": "Buy candidate",
    "KEEP": "Hold",
    "NEUTRAL": "Watch",
    "MANUAL REVIEW": "Review first",
    "AVOID": "Avoid for now",
    "SELL": "Consider trimming",
    "STRONG SELL": "Exit candidate",
    "INSUFFICIENT DATA": "Not enough data",
}

_PILLAR_LABELS = {
    "technical": "Price trend",
    "fundamental": "Business quality",
    "sentiment": "News mood",
    "forecast": "Model forecast",
}

_HELP_TEXT = {
    "opportunity_score": (
        "A combined score for new ideas. It blends business quality, price trend, "
        "forecast, and fit with your current holdings. Higher is better, but it is not a guarantee."
    ),
    "holding_score": (
        "The app's current view of a holding after combining price trend, business quality, "
        "news mood, forecast, and risk checks."
    ),
    "price_trend": "Price and volume behavior. Positive means the chart is helping the idea.",
    "business_quality": "Value, profitability, balance sheet strength, cash flow, and analyst support.",
    "news_mood": "Recent news and social-market tone. Positive means the latest headlines are supportive.",
    "model_forecast": "The app's short-term price estimate. Treat it as one input, not a promise.",
    "confidence": "How complete and reliable the available data is. Low confidence means size down or wait.",
    "entry_price": "The price area where the app thinks the trade has a better risk/reward.",
    "stop_loss": "The risk limit. If price reaches this area, the trade idea is probably wrong.",
    "take_profit": "The first planned upside target.",
    "reward_risk": "Reward-to-risk. 2.0x means the upside target is twice the planned loss.",
    "fill_probability": "Estimated chance that the suggested entry price is reached soon.",
    "position_weight": "Suggested portfolio size for the idea after risk controls.",
    "portfolio_fit": "How well the idea diversifies your current holdings.",
    "correlation": "How similarly this stock has moved versus your current holdings.",
    "beta": "Market sensitivity. 1.0 moves roughly like the market; below 1 is calmer; above 1 is more volatile.",
    "rsi": "Momentum heat gauge. Below 30 can be washed out; above 70 can be stretched.",
    "mae": "Average forecast miss. Lower means the forecast has been closer to reality.",
    "ic": "Information coefficient. It checks whether higher-ranked stocks later did better. Higher is better.",
    "weight": "Influence in the final score. A higher weight means that input matters more today.",
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


def _plain_action(action: str) -> str:
    """Translate engine action codes into trader-facing language."""
    return _ACTION_LABELS.get(str(action or "").upper(), str(action or "Watch").replace("_", " ").title())


def _plain_pillar(pillar: str) -> str:
    return _PILLAR_LABELS.get(str(pillar or "").lower(), str(pillar or "").replace("_", " ").title())


def _help(key: str) -> str:
    return _HELP_TEXT.get(key, "")


def _action_next_step(action: str, stance: str | None = None) -> str:
    action_code = str(action or "").upper()
    stance_code = str(stance or "").lower()
    if stance_code == "watch only" or action_code in {"AVOID", "INSUFFICIENT DATA"}:
        return "Do not buy yet. Keep it on the watchlist until the warning clears."
    if stance_code == "pullback preferred":
        return "Do not chase. Wait for the planned entry price or a cleaner setup."
    if action_code == "STRONG BUY":
        return "Ready to consider now if the suggested size and risk limit are acceptable."
    if action_code == "BUY":
        return "Buy candidate, but check the entry price and risk limit before acting."
    if action_code == "KEEP":
        return "Hold. No fresh action is needed unless your portfolio sizing has changed."
    if action_code == "SELL":
        return "Consider trimming or tightening the risk limit."
    if action_code == "STRONG SELL":
        return "Exit candidate. Review the sale plan before adding more risk."
    return "Watch. The setup is not clean enough for a clear buy or sell call."


def _render_action_pill(action: str) -> str:
    cls = action.lower().replace(" ", "-")
    return f'<span class="action-pill action-pill-{cls}">{_html.escape(_plain_action(action))}</span>'


def _render_plain_help() -> None:
    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            """
            **Recommendation** is the plain-English action: ready to buy, hold, watch, trim, or exit.

            **Opportunity score** ranks new ideas after business quality, price trend, model forecast, and portfolio fit are combined.

            **Buy around** is the preferred entry area. **Risk limit** is where the idea is probably wrong. **First target** is the first planned upside level.

            **Reward/risk** compares target upside with planned downside. A 2.0x setup aims for twice as much upside as downside.

            **Confidence** tells you how complete the app's evidence is. Low confidence should mean smaller size, manual review, or no trade.
            """
        )



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
    for label, val in [("Trend", tech), ("Quality", fund), ("News", sent), ("Forecast", fcast)]:
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
    return shared_discovery_confidence(cand)


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
        tags.append((f"Business quality {fund:+.2f}", "good"))
    if tech > 0.15:
        tags.append((f"Price trend {tech:+.2f}", "good"))
    if exp_90 > 5:
        tags.append((f"90d upside {exp_90:+.1f}%", "good"))
    if safe_float(getattr(cand, "volume_ratio", 1.0)) > 1.5:
        tags.append((f"Volume {safe_float(getattr(cand, 'volume_ratio', 1.0)):.1f}x", "info"))
    return tags[:5]


def _candidate_entry_stance(cand) -> str:
    return shared_candidate_entry_stance(cand)


def _candidate_entry_trigger(cand) -> str:
    return shared_candidate_entry_trigger(cand)


def _candidate_readiness_summary(cand) -> str:
    return shared_candidate_readiness_summary(cand)


def _entry_stance_tone(stance: str) -> str:
    return {
        "Ready": "ready",
        "Pullback Preferred": "pullback",
        "Watch Only": "watch",
    }.get(stance, "neutral")


def _is_top_pick(cand) -> bool:
    """True when a candidate meets all four 'Top Pick' criteria simultaneously."""
    return shared_is_top_pick(cand)


def _candidate_is_gated(cand) -> bool:
    return bool(getattr(cand, "ticker_identity_warning", None)) or _candidate_entry_stance(cand) == "Watch Only"


def _pick_best_new_opportunity(candidates: list):
    """Return (candidate_or_none, meta, subtitle) for the command-center opportunity card."""
    packets = [(cand, build_trade_packet(cand)) for cand in candidates]
    ready = [item for item in packets if item[1]["trade_ready"]]
    if ready:
        cand, packet = max(
            ready,
            key=lambda item: (
                safe_float(item[1]["confidence_score"]),
                safe_float(item[1]["final_rank"]),
            ),
        )
        meta = (
            f"Ready to buy | Opportunity score {safe_float(packet['final_rank']):.3f} "
            f"| {_plain_action(packet['action'])} | {packet['confidence_label']}"
        )
        return cand, meta, _candidate_thesis(cand)

    clean = [item for item in packets if item[1]["clean_entry"]]
    if clean:
        cand, packet = max(
            clean,
            key=lambda item: (
                safe_float(item[1]["confidence_score"]),
                safe_float(item[1]["final_rank"]),
            ),
        )
        meta = (
            f"{packet['entry_stance']} | Opportunity score {safe_float(packet['final_rank']):.3f} "
            f"| {_plain_action(packet['action'])} | {packet['confidence_label']}"
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


def _build_discovery_setup_rows(candidates: list) -> list[dict]:
    """Compact scan surface for trade-ready discovery ideas."""
    rows = []
    for packet in sort_trade_packets(candidates)[:10]:
        rows.append({
            "Top idea": "Yes" if packet["top_pick"] else "",
            "Ticker": packet["ticker"],
            "Recommendation": _plain_action(packet["action"]),
            "Entry view": packet["entry_stance"],
            "Ready now": "Yes" if packet["trade_ready"] else "No",
            "Next trigger": packet["entry_trigger"],
            "Confidence": f"{packet['confidence_label']} ({packet['confidence_score']:.0%})",
            "Opportunity score": f"{safe_float(packet['final_rank']):.3f}",
            "90-day upside": format_pct(safe_float(packet["expected_return_90d"]) * 100),
            "Buy around": _format_price(packet["entry_price"], packet["currency"]),
            "Risk limit": _format_price(packet["stop_loss"], packet["currency"]),
            "First target": _format_price(packet["take_profit"], packet["currency"]),
            "Reward/risk": f"{safe_float(packet['r_r_ratio']):.1f}x" if packet["r_r_ratio"] else "-",
            "Entry chance": format_pct(safe_float(packet["fill_probability"]) * 100) if packet["fill_probability"] else "-",
            "Position": format_pct(safe_float(packet["position_weight"]) * 100, plus_sign=False) if packet["position_weight"] else "-",
            "Portfolio fit": f"{safe_float(packet['portfolio_fit_score']):.2f}",
            "Similar to holdings": f"{safe_float(packet['max_correlation']):.2f}",
            "Main caution": packet["key_risk"] or "-",
            "Ticker check": "Verify" if packet["identity_warning"] else "OK",
        })
    return rows


def _render_html_chips(chips: list[tuple[str, str]], class_name: str = "signal-chip") -> str:
    if not chips:
        return ""
    rendered = "".join(
        f'<span class="{class_name} {tone}">{_html.escape(text)}</span>'
        for text, tone in chips
    )
    return f'<div class="chip-row">{rendered}</div>'


def _lens_sorted_candidates(candidates: list, lens: str) -> list:
    if lens == "Balanced growth and downside protection":
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
    if lens == "Best diversifiers":
        return sorted(
            candidates,
            key=lambda c: (
                safe_float(getattr(c, "portfolio_fit_score", 0)) * 0.60
                + safe_float(getattr(c, "final_rank", 0)) * 0.25
                + safe_float(getattr(c, "momentum_score", 0)) * 0.15
            ),
            reverse=True,
        )
    if lens == "Trend leaders":
        return sorted(
            candidates,
            key=lambda c: (
                safe_float(getattr(c, "momentum_score", 0)),
                safe_float(getattr(c, "return_90d", 0)),
                safe_float(getattr(c, "return_30d", 0)),
            ),
            reverse=True,
        )
    if lens == "Value and quality":
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
    _top_pick = _is_top_pick(cand)
    _subtitle = (
        f"{_geo} | {cand.exchange} | {cand.sector}"
        + (f" | {format_currency(_mcap / 1e9, 'GBP', decimals=1)}B mcap" if _mcap > 0 else "")
    )
    _top_pick_chip = '<span class="top-pick-badge">&#9733; Top Pick</span>' if _top_pick else ''
    _badge_html = (
        f'<div class="badge-row">'
        f'{_top_pick_chip}'
        f'<span class="signal-badge {_action_tone}">{_html.escape(_plain_action(_action))}</span>'
        f'<span class="signal-badge {_entry_tone}">{_html.escape(_entry_stance)}</span>'
        f'<span class="confidence-chip {_conf_tone}">{_html.escape(_conf_label)}</span>'
        f'</div>'
    )
    _hero_class = "rec-hero top-pick" if _top_pick else "rec-hero"
    st.markdown(
        f"""
        <div class="{_hero_class}">
            <div class="rec-rankline">
                <div class="rec-rank">{_html.escape(label)}</div>
                <div class="rec-rank">Opportunity score {safe_float(cand.final_rank):.3f}</div>
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
    if _action in {"BUY", "STRONG BUY"}:
        _readiness = _candidate_readiness_summary(cand)
        if _action == "BUY" or getattr(cand, "ready_contract_status", None) == "FAIL":
            st.warning(_readiness)
        else:
            st.success(_readiness)

    # Action-gate breakdown (Tier 1-4 hard gates) — surface why the action
    # isn't STRONG BUY at a glance.  The data the user used to fetch from
    # an external review now lives next to the recommendation.
    _gate_ceiling = str(getattr(cand, "action_gate_ceiling", "STRONG BUY") or "STRONG BUY")
    _gate_reasons = list(getattr(cand, "action_gate_reasons", []) or [])
    _gate_flags = dict(getattr(cand, "action_gate_flags", {}) or {})
    _limit_price = safe_float(getattr(cand, "limit_price", None), default=None)
    _limit_method = getattr(cand, "limit_price_method", None)

    if _gate_ceiling != "STRONG BUY" or _gate_reasons:
        with st.expander("Why not STRONG BUY?", expanded=(_gate_ceiling != "STRONG BUY")):
            st.caption(f"Gate ceiling: **{_gate_ceiling}** — strictest of Tier 1-4 distress, value, quality, and momentum gates.")
            if _gate_reasons:
                for _reason in _gate_reasons[:6]:
                    st.markdown(f"- {_html.escape(str(_reason))}")
            if _gate_flags:
                _failed = [k for k, v in _gate_flags.items() if v == "fail"]
                _borderline = [k for k, v in _gate_flags.items() if v == "borderline"]
                if _failed:
                    st.caption(f"Failed: {', '.join(_failed)}")
                if _borderline:
                    st.caption(f"Borderline: {', '.join(_borderline)}")
            _altman = safe_float(getattr(cand, "altman_z", None), default=None)
            if _altman is not None:
                _zone = str(getattr(cand, "altman_zone", "unknown") or "unknown")
                st.caption(f"Altman Z = {_altman:.2f} ({_zone})")

    if _limit_price and _limit_price > 0:
        _cur_price = safe_float(getattr(cand, "entry_price", None) or getattr(cand, "current_price", None), default=0.0)
        _cur = getattr(cand, "currency", "USD")
        st.info(
            f"Wait for pullback to **{_format_price(_limit_price, _cur)}** "
            f"({_limit_method or 'limit'})"
            + (f" — currently {_format_price(_cur_price, _cur)}" if _cur_price > 0 else "")
        )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Trend strength", f"{safe_float(cand.momentum_score):.2f}", help=_help("price_trend"))
    mc2.metric("Portfolio fit", f"{safe_float(cand.portfolio_fit_score):.2f}", help=_help("portfolio_fit"))
    mc3.metric("90-day upside", format_pct(safe_float(getattr(cand, "expected_return_90d", 0)) * 100), help=_help("model_forecast"))
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
        st.markdown("**Suggested trade plan**")
        st.caption(_action_next_step(_action, _entry_stance))
        _cur = getattr(cand, "currency", "USD")
        _plan_cols = st.columns(4)
        with _plan_cols[0]:
            st.metric("Buy around",
                      _format_price(_entry_p, _cur),
                      help=f"{_help('entry_price')} Method: {getattr(cand, 'entry_method', 'N/A')}")
        with _plan_cols[1]:
            st.metric("Risk limit",
                      _format_price(_stop_p, _cur) if _stop_p else "N/A",
                      help=f"{_help('stop_loss')} Method: {getattr(cand, 'stop_method', 'N/A')}")
        with _plan_cols[2]:
            st.metric("First target",
                      _format_price(_tp_p, _cur) if _tp_p else "N/A",
                      help=f"{_help('take_profit')} Method: {getattr(cand, 'target_method', 'N/A')}")
        with _plan_cols[3]:
            _rr = safe_float(getattr(cand, "r_r_ratio", None))
            st.metric("Reward/risk",
                      f"{_rr:.1f}x" if _rr and _rr > 0 else "N/A",
                      help=_help("reward_risk"))
        with st.expander("Entry details", expanded=False):
            _cur = getattr(cand, "currency", "USD")
            _tp_cols = st.columns(4)
            with _tp_cols[0]:
                st.metric("Buy around",
                          _format_price(_entry_p, _cur),
                          help=f"{_help('entry_price')} Method: {getattr(cand, 'entry_method', 'N/A')}")
            with _tp_cols[1]:
                st.metric("Risk limit",
                          _format_price(_stop_p, _cur) if _stop_p else "N/A",
                          help=f"{_help('stop_loss')} Method: {getattr(cand, 'stop_method', 'N/A')}")
            with _tp_cols[2]:
                st.metric("First target",
                          _format_price(_tp_p, _cur) if _tp_p else "N/A",
                          help=f"{_help('take_profit')} Method: {getattr(cand, 'target_method', 'N/A')}")
            with _tp_cols[3]:
                _rr = safe_float(getattr(cand, "r_r_ratio", None))
                st.metric("Reward/risk",
                          f"{_rr:.1f}x" if _rr and _rr > 0 else "N/A",
                          help=_help("reward_risk"))

            _tp_cols2 = st.columns(4)
            with _tp_cols2[0]:
                _fill = safe_float(getattr(cand, "fill_probability", None))
                st.metric("Entry chance",
                          format_pct(_fill * 100) if _fill else "N/A",
                          help=_help("fill_probability"))
            with _tp_cols2[1]:
                _sdp = safe_float(getattr(cand, "stop_distance_pct", None))
                st.metric("Risk distance",
                          format_pct(_sdp) if _sdp else "N/A")
            with _tp_cols2[2]:
                _shares = getattr(cand, "position_size_shares", 0) or 0
                _size_method = getattr(cand, "sizing_method", "") or "Stop-budget sizing"
                st.metric("Shares", f"{_shares:,}" if _shares > 0 else "N/A",
                          help=_size_method.replace("_", " "))
            with _tp_cols2[3]:
                _pw = safe_float(getattr(cand, "position_weight", 0))
                st.metric("Suggested size",
                          format_pct(_pw * 100) if _pw > 0 else "N/A",
                          help=_help("position_weight"))

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
                st.caption(" | ".join(_info_parts))

    with st.expander("Why this is on the list"):
        st.markdown(f"**Thesis:** {_candidate_thesis(cand)}")
        st.caption(f"**Model notes:** {cand.why}")
        st.caption(
            f"Trend: {format_pct(safe_float(getattr(cand, 'return_90d', 0)) * 100)} over 90d | "
            f"{format_pct(safe_float(getattr(cand, 'return_30d', 0)) * 100)} over 30d | "
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
        chips.append((f"Holding score {format_score(score)}", "info"))
    if is_valid_number(price):
        chips.append((f"Price {format_currency(price, currency)}", "info"))
    if base_action:
        chips.append((f"Before risk checks {_plain_action(base_action)}", "info"))
    if final_action and final_action != base_action:
        chips.append((f"Final {_plain_action(final_action)}", "risk"))
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
        action_txt = f"{_html.escape(_plain_action(str(base_action)))} -> {_html.escape(_plain_action(str(final_action)))} | "
    return (
        '<div style="font-size:12px;color:#94a3b8;margin-top:6px;">'
        f'{action_txt}Before risk checks {safe_float(prior):+.3f} | '
        f'Exit warning {safe_float(exit_score):.3f} | '
        f'Risk penalty {safe_float(penalty):+.3f} | '
        f'After risk checks {safe_float(posterior):+.3f}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## Portfolio command centre")
st.caption("Plain-English buy, hold, trim, and watchlist recommendations for your ISA portfolio.")
_render_plain_help()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### What to do")

    # Refresh triggers live recomputation; normal load uses cache
    _force_refresh = st.button("Refresh recommendations", use_container_width=True, type="primary")
    if _force_refresh:
        clear_cache()

    st.divider()

    # Scoring weights: plain-English model mix bars.
    st.markdown("**Model mix**")
    weight_html = ""
    for pillar, w in config.WEIGHTS.items():
        weight_html += _render_weight_bar(_plain_pillar(pillar), w)
    st.markdown(weight_html, unsafe_allow_html=True)

    st.divider()

    # Sort & Filter
    sort_option = st.selectbox(
        "Sort by",
        ["Best score first", "Lowest score first", "Best gain first", "Worst gain first", "Ticker A-Z"],
        index=0,
    )
    action_filter = st.multiselect(
        "Show only these recommendations",
        ["STRONG BUY", "BUY", "KEEP", "SELL", "STRONG SELL"],
        default=[],
        format_func=_plain_action,
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
            st.markdown(f":green_circle: **FMP {plan}** - {remaining}/min available, {today_calls} today")
        elif config.FMP_API_KEY:
            st.markdown(":orange_circle: **FMP** - Rate limit reached")
        else:
            st.markdown(":red_circle: **FMP** - Not configured")
    except ImportError:
        st.markdown(":red_circle: **FMP** - Not available")

    st.divider()
    st.caption("Prices from Yahoo Finance. Fundamentals and news from FMP when available.")
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

st.caption(" | ".join(_freshness_parts))
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
    action_counts[a] = sum(1 for r in results if r.get("final_action", r.get("action")) == a)

# ---------------------------------------------------------------------------
# TOP-LEVEL TAB NAVIGATION
# ---------------------------------------------------------------------------
tab_dashboard, tab_holdings, tab_discovery, tab_analytics = st.tabs([
    "Dashboard",
    f"Holdings ({len(results)})",
    "New ideas",
    "Learning and backtest",
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
        st.caption("Total portfolio value | unrealised gain/loss")

        # Best / worst performer
        if per_holding_pl:
            best = max(per_holding_pl, key=lambda x: x[2])
            worst = min(per_holding_pl, key=lambda x: x[2])
            st.markdown(
                f"**Best:** {best[0]} ({format_pct(safe_float(best[2]))}) | "
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
                donut_labels.append(_plain_action(action_name))
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
            st.plotly_chart(
                fig_donut,
                use_container_width=True,
                config={"displayModeBar": False},
                key="portfolio_action_donut_chart",
            )

    # Action count cards
    act_cols = st.columns(5)
    for i, (action_name, color) in enumerate([
        ("STRONG BUY", "#059669"), ("BUY", "#10b981"), ("KEEP", "#3b82f6"),
        ("SELL", "#f59e0b"), ("STRONG SELL", "#ef4444"),
    ]):
        cnt = action_counts[action_name]
        act_cols[i].metric(_plain_action(action_name), cnt)

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
        title=dict(text="Portfolio recommendation score", font=dict(size=14)),
    ))
    gauge_fig.update_layout(**{**_PLOTLY_LAYOUT, "margin": dict(l=30, r=30, t=50, b=10)}, height=160)
    st.plotly_chart(
        gauge_fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="portfolio_health_gauge_chart",
    )

    # ---------------------------------------------------------------------------
    # Portfolio Risk Analysis
    # ---------------------------------------------------------------------------
    if risk_data and risk_data.get("sector_weights"):
        with st.expander("Portfolio risk check", expanded=bool(risk_data.get("concentration_warnings"))):
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
                        title=dict(text="Sector mix", font=dict(size=14)),
                        showlegend=False,
                    )
                    st.plotly_chart(
                        sector_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="portfolio_sector_allocation_chart",
                    )

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
                        title=dict(text="How similarly holdings moved (90d)", font=dict(size=14)),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(
                        corr_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="portfolio_correlation_heatmap",
                    )

            # Risk warnings
            warnings = risk_data.get("concentration_warnings", [])
            high_corrs = risk_data.get("high_correlations", [])

            if warnings:
                for w in warnings:
                    st.warning(w)

            if high_corrs:
                corr_strs = [f"{t1}<->{t2} ({c:+.2f})" for t1, t2, c in high_corrs[:5]]
                st.info(f"Holdings moving very similarly: {', '.join(corr_strs)}")

            # Risk score
            risk_score = risk_data.get("risk_score", 0)
            risk_label = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.6 else "High"
            risk_color = "#10b981" if risk_score < 0.3 else "#f59e0b" if risk_score < 0.6 else "#ef4444"
            st.markdown(
                f'<div style="text-align:center; padding:8px;">'
                f'<span style="font-size:1.1rem; font-weight:600;">Portfolio risk score: </span>'
                f'<span style="font-size:1.3rem; font-weight:700; color:{risk_color};">'
                f'{risk_score:.0%} ({risk_label})</span></div>',
                unsafe_allow_html=True,
            )

            # ── Institutional Risk Metrics (VaR/ES/β/stress) ────────────
            _var_es = risk_data.get("var_es") or {}
            _beta_info = risk_data.get("beta") or {}
            _stress = risk_data.get("stress_scenarios") or []
            _worst = risk_data.get("worst_stress")

            if _var_es or _beta_info or _stress:
                st.markdown("##### Downside risk numbers")
                _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                _v = _var_es.get("var_1d")
                _e = _var_es.get("es_1d")
                _va = _var_es.get("vol_annual")
                _b = _beta_info.get("beta")
                _rc1.metric(
                    "Bad-day loss",
                    f"{_v*100:+.2f}%" if _v is not None else "-",
                    help="A historical estimate of a poor one-day move. Roughly 1 day in 20 may be worse.",
                )
                _rc2.metric(
                    "Worst-day average",
                    f"{_e*100:+.2f}%" if _e is not None else "-",
                    help="Average loss across the worst historical days in the sample.",
                )
                _rc3.metric(
                    "Annual volatility",
                    f"{_va*100:.1f}%" if _va is not None else "-",
                    help="How much the portfolio has typically moved over a year, based on daily returns.",
                )
                _rc4.metric(
                    f"Market sensitivity vs {_beta_info.get('benchmark','SPY')}",
                    f"{_b:.2f}" if _b is not None else "-",
                    help=_help("beta"),
                )

                if _stress:
                    _sc_rows = [
                        {
                            "Scenario": s["name"],
                            "Window": s["window"],
                            "Portfolio": f"{s['portfolio_return']*100:+.1f}%",
                            "Worst Name": s.get("worst_name") or "-",
                            "Worst Return": (
                                f"{s['worst_return']*100:+.1f}%"
                                if s.get("worst_return") is not None else "-"
                            ),
                            "Coverage": f"{s['coverage']*100:.0f}%",
                        }
                        for s in _stress
                    ]
                    st.caption("**Historical stress replays** - how the current portfolio would have behaved in past selloffs.")
                    st.dataframe(_sc_rows, use_container_width=True, hide_index=True)
                    if _worst and _worst.get("portfolio_return", 0) < -0.15:
                        st.warning(
                            f"Worst historical replay '{_worst['name']}': "
                            f"{_worst['portfolio_return']*100:+.1f}% on current book."
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
            f"Exit review ({len(_exit_list)} warnings)"
            + (f" | {format_freshness(_dash.exit_signals_timestamp)}" if _dash.exit_signals_timestamp else ""),
            expanded=any(e.get("severity") == "urgent" for e in _exit_list),
        ):
            st.markdown("#### Exit review")
            _urgent_count = sum(1 for e in _exit_list if e.get("severity") == "urgent")
            _action_count = sum(1 for e in _exit_list if e.get("severity") == "action_needed")
            _warning_count = sum(1 for e in _exit_list if e.get("severity") == "warning")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Urgent", _urgent_count)
            ec2.metric("Action needed", _action_count)
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
                _sev_label = "Urgent" if _sev == "urgent" else "Action needed" if _sev == "action_needed" else "Watch"
                _title = _html.escape(_card_exit.get("ticker", ""))
                _subtitle = _html.escape(_card_exit.get("name", "")) + " | " + _html.escape(
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
        with st.expander("Suggested position sizes"):
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
                yaxis_title="Portfolio %",
                height=300,
                title=dict(text="Current vs suggested position size", font=dict(size=14)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(
                alloc_fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key="portfolio_allocation_comparison_chart",
            )

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
    # Portfolio Optimizer (Ensemble) — from cache or live
    # ---------------------------------------------------------------------------
    _opt_data = _dash.cached_optimizer
    if _opt_data and _opt_data.get("holdings"):
        _per_method_w = _opt_data.get("per_method_weights") or {}
        _method_w = _opt_data.get("method_weights") or {}
        _history_runs = int(_opt_data.get("history_run_count") or 0)
        _is_ensemble = bool(_method_w)
        _opt_title = "🎯 **Portfolio Optimizer (Ensemble)**" if _is_ensemble else "🎯 **Portfolio Optimizer (Mean-Variance)**"
        if _dash.optimizer_timestamp:
            _opt_title += f" · {format_freshness(_dash.optimizer_timestamp)}"
        with st.expander(_opt_title, expanded=False):
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Expected Return", format_pct(safe_float(_opt_data['portfolio_expected_return']) * 100))
            oc2.metric("Portfolio Vol", format_pct(safe_float(_opt_data['portfolio_volatility']) * 100, plus_sign=False))
            oc3.metric("Sharpe Ratio", f"{safe_float(_opt_data['portfolio_sharpe']):.2f}")
            oc4.metric("Turnover", format_pct(safe_float(_opt_data['turnover']) * 100, plus_sign=False))

            _regime_label = (_regime or {}).get("regime_label", "NEUTRAL") if _regime else "N/A"
            _mu_ver = _opt_data.get("mu_version", "legacy")
            _cov_m = _opt_data.get("cov_method", "lw_ewma")
            st.caption(
                f"Risk-free rate: {safe_float(_opt_data.get('risk_free_rate', 0))*100:.1f}% | "
                f"Regime: {_regime_label} | μ: {_mu_ver} | Σ: {_cov_m} | "
                f"Method: {_opt_data.get('method', 'N/A')}"
            )

            # Ensemble method weights (Sharpe-realised combiner)
            if _is_ensemble:
                _method_labels = {
                    "mean_variance": "Mean-Variance",
                    "min_variance": "Min Variance",
                    "risk_parity": "Risk Parity",
                    "black_litterman": "Black-Litterman",
                    "hrp": "HRP",
                }
                _mw_items = sorted(_method_w.items(), key=lambda kv: -safe_float(kv[1]))
                _mw_rows = [
                    {
                        "Method": _method_labels.get(m, m),
                        "Weight": format_pct(safe_float(w) * 100, plus_sign=False),
                    }
                    for m, w in _mw_items
                ]
                st.markdown("**Ensemble method weights**")
                st.caption(
                    f"Ensemble of {len(_mw_items)} methods - weights from realised Sharpe over {_history_runs} runs"
                )

                _mw_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]
                _mw_fig = go.Figure()
                for i, (m, w) in enumerate(_mw_items):
                    _mw_fig.add_trace(go.Bar(
                        name=_method_labels.get(m, m),
                        y=["Ensemble"],
                        x=[safe_float(w) * 100],
                        orientation="h",
                        marker_color=_mw_colors[i % len(_mw_colors)],
                        text=[format_pct(safe_float(w) * 100, plus_sign=False)],
                        textposition="inside",
                    ))
                _mw_fig.update_layout(
                    **{**_PLOTLY_LAYOUT, "margin": dict(l=20, r=10, t=20, b=10)},
                    barmode="stack",
                    height=180,
                    xaxis_title="Method weight %",
                    yaxis_title="",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )

                mwc1, mwc2 = st.columns([1.6, 1.0])
                with mwc1:
                    st.plotly_chart(
                        _mw_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="portfolio_optimizer_method_weights_chart",
                    )
                with mwc2:
                    st.dataframe(pd.DataFrame(_mw_rows), hide_index=True, use_container_width=True)

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
                title=dict(text="Current vs Ensemble Target Allocation", font=dict(size=14)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(
                _opt_fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key="portfolio_optimizer_allocation_chart",
            )

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

            # Per-method comparison (ensemble only)
            if _is_ensemble and _per_method_w:
                _method_col_labels = {
                    "mean_variance": "MV",
                    "min_variance": "MinVar",
                    "risk_parity": "RP",
                    "black_litterman": "BL",
                    "hrp": "HRP",
                }
                _per_method_rows = []
                for h in _opt_h:
                    _t = h["ticker"]
                    _row = {
                        "Ticker": _t,
                        "Current": format_pct(safe_float(h["current_weight"]) * 100, plus_sign=False),
                    }
                    _pm = _per_method_w.get(_t, {}) or {}
                    for m, lbl in _method_col_labels.items():
                        if m in _pm:
                            _row[lbl] = format_pct(safe_float(_pm[m]) * 100, plus_sign=False)
                    _row["Ensemble"] = format_pct(safe_float(h["optimal_weight"]) * 100, plus_sign=False)
                    _per_method_rows.append(_row)
                st.markdown("**Per-method weights** — side-by-side comparison")
                st.dataframe(pd.DataFrame(_per_method_rows), hide_index=True, use_container_width=True)

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
        "Best score first": (lambda r: r["aggregate_score"], True),
        "Lowest score first": (lambda r: r["aggregate_score"], False),
        "Best gain first": (lambda r: r["_pl_pct"], True),
        "Worst gain first": (lambda r: r["_pl_pct"], False),
        "Ticker A-Z": (lambda r: r["ticker"], False),
    }
    sort_key, sort_reverse = sort_map.get(sort_option, (lambda r: r["aggregate_score"], True))
    filtered_results.sort(key=sort_key, reverse=sort_reverse)

    # ---------------------------------------------------------------------------
    # Holdings Summary Table — dense scan surface before the cards
    # ---------------------------------------------------------------------------
    _exit_list_for_holdings = _dash.cached_exit_signals or []
    _exit_by_ticker = {e.get("ticker"): e for e in _exit_list_for_holdings}

    if filtered_results:
        def _conf_label(c):
            if c is None:
                return "-"
            if c >= 0.75:
                return f"High ({c:.0%})"
            if c >= 0.55:
                return f"Medium ({c:.0%})"
            return f"Low ({c:.0%})"

        _summary_rows = []
        for _r in filtered_results:
            _cp_ = _r.get("current_price")
            _ap_ = _r.get("avg_buy_price")
            _pl_pct_ = ((_cp_ - _ap_) / _ap_ * 100) if (_cp_ and _ap_ and _ap_ > 0) else None
            _fin_ = _r.get("final_action", _r.get("action", ""))
            _base_ = _r.get("base_action", _fin_)
            _conf_ = _r.get("effective_data_confidence", _r.get("data_confidence"))
            _exit_ = _exit_by_ticker.get(_r.get("ticker"))
            _exit_flag = ""
            if _exit_:
                _sev = _exit_.get("severity", "")
                _exit_flag = _sev.replace("_", " ").title() if _sev else ""
            _summary_rows.append({
                "Ticker": _r.get("ticker", ""),
                "Holding score": format_score(_r.get("aggregate_score", 0)),
                "Recommendation": _plain_action(_fin_),
                "Before risk checks": _plain_action(_base_) if _base_ != _fin_ else "-",
                "Confidence": _conf_label(safe_float(_conf_) if _conf_ is not None else None),
                "Gain/loss": format_pct(_pl_pct_) if _pl_pct_ is not None else "-",
                "Price": _format_price(_cp_, _r.get("currency", "GBP")),
                "Risk limit": _format_price(
                    _r.get("structural_stop_loss", _r.get("stop_loss")),
                    _r.get("currency", "GBP"),
                ),
                "First target": _format_price(_r.get("take_profit"), _r.get("currency", "GBP")),
                "Exit warning": _exit_flag,
            })
        with st.expander(f"Summary table ({len(_summary_rows)} holdings)", expanded=False):
            st.dataframe(_summary_rows, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------------
    # Holding cards
    # ---------------------------------------------------------------------------
    st.markdown(f"### Holding recommendations ({len(filtered_results)} holdings)")

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
                    f"**{r['ticker']}** &nbsp;|&nbsp; {r['name']} &nbsp;"
                    f'<span style="color:{change_color};font-weight:600;font-size:0.85rem">{change_txt}</span>',
                    unsafe_allow_html=True,
                )

                # Confidence badge (Phase 1.10) — flags recommendations
                # derived from sparse or degraded inputs.
                _conf = safe_float(
                    r.get("effective_data_confidence", r.get("data_confidence")),
                    default=None,
                ) if r.get("effective_data_confidence") is not None or r.get("data_confidence") is not None else None
                if _conf is not None:
                    if _conf >= 0.75:
                        _cbg, _cfg, _clbl = "#064e3b", "#6ee7b7", f"High confidence ({_conf:.0%})"
                    elif _conf >= 0.55:
                        _cbg, _cfg, _clbl = "#3b2f12", "#fcd34d", f"Medium confidence ({_conf:.0%})"
                    else:
                        _cbg, _cfg, _clbl = "#3b1220", "#fca5a5", f"Low confidence ({_conf:.0%})"
                    st.markdown(
                        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
                        f'background:{_cbg};color:{_cfg};font-size:11px;font-weight:600;'
                        f'margin-top:4px">{_clbl}</span>',
                        unsafe_allow_html=True,
                    )

                # Inline exit signal (Phase 1.11) — surfaces urgent/action_needed
                # alerts on the card itself so the trader sees them without
                # scrolling to the Exit Intelligence expander.
                _holding_exit = _exit_by_ticker.get(r.get("ticker"))
                if _holding_exit and _holding_exit.get("severity") in ("urgent", "action_needed"):
                    _sev = _holding_exit.get("severity")
                    _icon = "🔴" if _sev == "urgent" else "🟡"
                    _msg = _holding_exit.get("message", "") or _holding_exit.get("signal_type", "")
                    _exit_color = "#ef4444" if _sev == "urgent" else "#f59e0b"
                    st.markdown(
                        f'<div style="margin-top:6px;padding:6px 10px;border-left:3px solid {_exit_color};'
                        f'background:rgba(239,68,68,0.08);font-size:12px;color:#f3f4f6;">'
                        f'<b>{_sev.replace("_"," ").title()}</b> | {_html.escape(_msg[:140])}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with hdr2:
                _er90 = r.get("expected_return_90d")
                _er90_txt = f" | 90-day upside: {format_pct(safe_float(_er90) * 100)}" if is_valid_number(_er90) else ""
                st.markdown(f"**Holding score: {format_score(r['aggregate_score'])}**{_er90_txt}")
                st.markdown(_render_score_bar(r["aggregate_score"]), unsafe_allow_html=True)
                _action_caption = (
                    f"Model view before risk checks: {_plain_action(base_action)} -> final recommendation: {_plain_action(final_action)}"
                    if base_action != final_action
                    else f"Recommendation: {_plain_action(final_action)}"
                )
                st.caption(_action_caption)

            with hdr3:
                st.markdown(
                    f'<div style="text-align:center;padding-top:4px">{_render_action_pill(final_action)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="text-align:center;font-size:12px;color:#94a3b8;margin-top:6px;">'
                    f'Before risk checks: {_html.escape(_plain_action(base_action))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Body row: Price | P&L | Targets | Pillars ──
            b1, b2, b3, b4 = st.columns([1.2, 1.5, 1.3, 1.5])

            with b1:
                price_str = _format_price(_cp, currency)
                st.metric("Current price", price_str)
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
                    st.metric("Gain/loss per share", _pl_str, format_pct(_pl_pct))
                    st.metric("Total gain/loss", _total_str, format_pct(_pl_pct),
                        help=f"{safe_float(_qty):.0f} shares x {_format_price(abs(_pl), currency)} per share")
                else:
                    st.metric("Gain/loss per share", "N/A")

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
                    _regime_hint = f" | Market stress: {_rl}"
                st.metric("First target", tp_str, help=f"{_help('take_profit')} Method: {r.get('target_method', 'N/A')}")
                st.metric("Risk limit", sl_str,
                          help=f"{_help('stop_loss')} Method: {_structural_method}{_sdp_str}{_regime_hint}")
                st.metric("Trailing exit", trailing_str,
                          help=f"A moving exit level used to protect gains. Method: {_trailing_method}")

            with b4:
                st.markdown("**What drove the score**")
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
            st.caption(f"**Reason:** {r['why']}")

            # ── Tabbed details ──
            with st.expander("Details"):
                tab_overview, tab_scores, tab_fund, tab_sent, tab_fcast = st.tabs([
                    "Overview", "Score drivers", "Business quality", "News mood", "Forecast"
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
                        st.plotly_chart(
                            fig_ov,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"holding_overview_chart_{r['ticker'].replace('.', '_')}",
                        )

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
                        st.metric("Price trend", format_score(r.get('technical_score', 0), decimals=2),
                            help=_help("price_trend"))
                        st.metric("Business quality", format_score(r.get('fundamental_score', 0), decimals=2),
                            help=_help("business_quality"))
                        st.metric("News mood", format_score(r.get('sentiment_score', 0), decimals=2),
                            help=_help("news_mood"))
                        st.metric("Model forecast", format_score(r.get('forecast_score', 0), decimals=2),
                            help=_help("model_forecast"))

                    with sc2:
                        # Radar chart for 4 pillars
                        pillars = ["Price trend", "Business quality", "News mood", "Forecast"]
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
                        st.plotly_chart(
                            fig_radar,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"holding_radar_chart_{r['ticker'].replace('.', '_')}",
                        )

                    # RSI gauge
                    if r.get("rsi") is not None:
                        st.markdown(f"**Momentum heat gauge (RSI)** - {r['rsi']:.1f}")
                        st.caption(_help("rsi"))
                        st.markdown(_render_rsi_gauge(r["rsi"]), unsafe_allow_html=True)

                    # Technical indicators grid
                    tech_metrics = []
                    if r.get("bb_pct") is not None:
                        tech_metrics.append(("Price band position", f"{r['bb_pct']:.0%}"))
                    if r.get("stoch_k") is not None:
                        tech_metrics.append(("Short-term momentum", f"{r['stoch_k']:.0%}"))
                    if r.get("obv_divergence"):
                        tech_metrics.append(("Volume pressure", f"{r['obv_divergence']} div"))
                    elif r.get("obv_trend"):
                        tech_metrics.append(("Volume pressure", r["obv_trend"]))
                    if r.get("adx") is not None:
                        tech_metrics.append(("Trend strength", f"{r['adx']:.0f}"))
                    if r.get("williams_r") is not None:
                        tech_metrics.append(("Pullback gauge", f"{r['williams_r']:.0f}"))

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
                        fund_metrics.append(("Price / earnings", f"{r['pe_ratio']:.1f}"))
                    if r.get("revenue_growth") is not None:
                        fund_metrics.append(("Revenue growth", f"{r['revenue_growth']:.0%}"))
                    if r.get("profit_margin") is not None:
                        fund_metrics.append(("Profit margin", f"{r['profit_margin']:.0%}"))
                    if r.get("roe") is not None:
                        fund_metrics.append(("Return on equity", f"{r['roe']:.0%}"))
                    if r.get("fcf_yield") is not None:
                        fund_metrics.append(("Free cash flow yield", f"{r['fcf_yield']:.1%}"))
                    if r.get("short_pct") is not None:
                        fund_metrics.append(("Short interest", f"{r['short_pct']:.1%}"))
                    if r.get("inst_ownership") is not None:
                        fund_metrics.append(("Fund ownership", f"{r['inst_ownership']:.0%}"))
                    if r.get("dividend_yield") is not None:
                        fund_metrics.append(("Dividend yield", f"{r['dividend_yield']:.1%}"))
                    if r.get("payout_ratio") is not None:
                        fund_metrics.append(("Dividend payout", f"{r['payout_ratio']:.0%}"))
                    if r.get("current_ratio") is not None:
                        fund_metrics.append(("Short-term cover", f"{r['current_ratio']:.1f}"))
                    if r.get("net_debt_ebitda") is not None:
                        fund_metrics.append(("Debt / profit", f"{r['net_debt_ebitda']:.1f}x"))
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
                            f"**Analyst target:** {_format_price(r['analyst_target'], currency)} "
                            f"{upside_str} | {rec_str}{analysts}"
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
                        st.markdown("**Analyst and earnings checks**")
                        fmp_metrics = []
                        if r.get("peg_ratio") is not None:
                            fmp_metrics.append(("Growth-adjusted value", f"{r['peg_ratio']:.1f}"))
                        if r.get("earnings_beat_rate"):
                            fmp_metrics.append(("Beat Rate", r["earnings_beat_rate"]))
                        if r.get("quarterly_trend"):
                            fmp_metrics.append(("Trend", r["quarterly_trend"]))
                        if r.get("estimate_revision"):
                            fmp_metrics.append(("Revisions", r["estimate_revision"]))
                        if r.get("pe_vs_sector"):
                            fmp_metrics.append(("Value vs peers", r["pe_vs_sector"]))
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
                                f"Upgrades: **{r['recent_upgrades']}** | "
                                f"Downgrades: **{r['recent_downgrades']}** (90 days)"
                            )

                    # Position info
                    st.markdown("---")
                    st.caption(
                        f"Avg buy: {_format_price(r.get('avg_buy_price'), currency)} | "
                        f"Shares: {r.get('quantity', 0)} | "
                        f"Risk limit method: {r.get('stop_method', 'N/A')} | Target method: {r.get('target_method', 'N/A')}"
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
                        st.markdown(f"**Price forecast ({horizon}-day)**")
                        st.caption("The app blends several forecast models and gives more influence to models that have recently been more accurate.")

                        fc_cols = st.columns(4)
                        fc_cols[0].metric("Expected price", _format_price(r.get("forecast_price"), currency),
                                         f"{r.get('forecast_pct_change', 0):+.1f}%")
                        fc_cols[1].metric("Likely low", _format_price(r.get("forecast_low"), currency))
                        fc_cols[2].metric("Likely high", _format_price(r.get("forecast_high"), currency))
                        if r.get("forecast_ensemble_mae") is not None:
                            fc_cols[3].metric("Avg miss", f"{r.get('forecast_ensemble_mae', 0):.2f}", help=_help("mae"))
                        else:
                            fc_cols[3].metric("Avg miss", "Building...", help=_help("mae"))

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
                                xaxis=dict(title="Influence %", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
                                yaxis=dict(autorange="reversed"),
                                title=dict(text="Forecast model influence", font=dict(size=13)),
                            )
                            st.plotly_chart(
                                fig_expert,
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"forecast_expert_weights_{r['ticker'].replace('.', '_')}_{horizon}",
                            )

                        # Expert table
                        if r.get("forecast_experts"):
                            expert_rows = []
                            for e in r["forecast_experts"]:
                                weight = r.get("forecast_expert_weights", {}).get(e["name"], 0)
                                mae_val = r.get("forecast_expert_maes", {}).get(e["name"])
                                expert_rows.append({
                                    "Model": e["name"].replace("_", " ").title(),
                                    "Prediction": round(e["price"], 2),
                                    "Low": round(e["low"], 2),
                                    "High": round(e["high"], 2),
                                    "Influence": f"{weight:.1%}",
                                    "Avg miss": f"{mae_val:.2f}" if mae_val is not None else "-",
                                })
                            st.dataframe(pd.DataFrame(expert_rows), hide_index=True, use_container_width=True)

                        # Long-horizon forecast (if available)
                        if r.get("forecast_price_long") is not None:
                            st.divider()
                            horizon_long = r.get("forecast_horizon_long", 63)
                            st.markdown(f"**Longer-term forecast ({horizon_long}-day)**")
                            lc = st.columns(4)
                            lc[0].metric("Expected price", _format_price(r.get("forecast_price_long"), currency),
                                         f"{r.get('forecast_pct_change_long', 0):+.1f}%")
                            lc[1].metric("Likely low", _format_price(r.get("forecast_low_long", 0), currency))
                            lc[2].metric("Likely high", _format_price(r.get("forecast_high_long", 0), currency))
                            if r.get("forecast_ensemble_mae_long") is not None:
                                lc[3].metric("Avg miss", f"{r.get('forecast_ensemble_mae_long', 0):.2f}", help=_help("mae"))
                            else:
                                lc[3].metric("Avg miss", "Building...", help=_help("mae"))
                    else:
                        st.info("Forecast data not available for this holding.")



with tab_analytics:
    # ---------------------------------------------------------------------------
    # 90-Day Portfolio Return Projection
    # ---------------------------------------------------------------------------
    st.markdown("### 90-day portfolio outlook")
    st.caption(
        "A scenario engine blends the app's forecasts with recent volatility and how holdings move together. "
        "It shows a range of possible portfolio outcomes over about 90 calendar days."
    )

    _run_projection = st.button("Run 90-day outlook", type="secondary")
    if _run_projection:
        from engine.portfolio_projection import project_portfolio_return, project_swap_impact

        with st.spinner("Running 5,000 possible 90-day paths..."):
            _proj = project_portfolio_return(results, holdings, position_weights)

        # Portfolio-level summary
        st.markdown("#### Portfolio outcome range")
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Expected return", format_pct(safe_float(_proj.expected_return_pct)))
        pc2.metric("Chance of gain", f"{safe_float(_proj.prob_positive):.0%}")
        pc3.metric("Current value", format_currency(_proj.current_value, "GBP", decimals=0))
        pc4.metric("Expected value", format_currency(_proj.expected_value, "GBP", decimals=0))
        _gain = safe_float(_proj.expected_value) - safe_float(_proj.current_value)
        pc5.metric("Expected gain", format_currency(_gain, "GBP", decimals=0))

        # Confidence interval table
        st.markdown("#### Scenario bands")
        ci_data = []
        for pctile, label in [(0.10, "Bear (10th)"), (0.25, "Cautious (25th)"),
                               (0.50, "Median (50th)"), (0.75, "Optimistic (75th)"),
                               (0.90, "Bull (90th)")]:
            _pv = safe_float(_proj.projected_values[pctile])
            ci_data.append({
                "Scenario": label,
                "Portfolio return": format_pct(safe_float(_proj.projected_returns[pctile])),
                "Portfolio value": format_currency(_pv, "GBP", decimals=0),
                "Gain / loss": format_currency(_pv - safe_float(_proj.current_value), "GBP", decimals=0),
            })
        st.dataframe(pd.DataFrame(ci_data), hide_index=True, use_container_width=True)

        # Per-ticker breakdown
        st.markdown("#### Holding-level outlook")
        ticker_rows = []
        for tp in sorted(_proj.ticker_projections, key=lambda x: safe_float(x.expected_return_pct), reverse=True):
            ticker_rows.append({
                "Ticker": tp.ticker,
                "Current": format_currency(tp.current_price, "GBP"),
                "Model forecast": format_currency(tp.moe_predicted_price, "GBP"),
                "Model return": format_pct(safe_float(tp.moe_pct_change)),
                "Scenario expected": format_pct(safe_float(tp.expected_return_pct)),
                "Bad case": format_pct(safe_float(tp.projected_returns.get(0.10))),
                "Middle case": format_pct(safe_float(tp.projected_returns.get(0.50))),
                "Good case": format_pct(safe_float(tp.projected_returns.get(0.90))),
                "Chance of gain": f"{safe_float(tp.prob_positive):.0%}",
                "Annual volatility": f"{safe_float(tp.annual_volatility):.0%}",
            })
        st.dataframe(pd.DataFrame(ticker_rows), hide_index=True, use_container_width=True)

        # Swap impact analysis — uses cached discovery candidates from orchestrator state
        st.markdown("#### Swap impact")
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
            "Forecast accuracy", "Model mix", "What helps",
            "Weight tuning", "Forecast history",
        ])

        _all_experts = set()
        for ticker_data in _rolling_maes.values():
            _all_experts.update(k for k in ticker_data.keys() if k != "ensemble")
        _all_experts = sorted(_all_experts)

        # ── Tab 1: Expert Accuracy (MAE) ──
        with tab_accuracy:
            st.caption("Lower average miss means better forecasts. More accurate models get more influence.")

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
                mae_df.index.name = "Model"

                ticker_select = st.selectbox("Choose holding:", list(mae_data.keys()), key="mae_ticker")

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
                        xaxis_title="Average miss",
                        yaxis_title="",
                        coloraxis_showscale=False,
                        title=dict(text=f"Forecast miss - {ticker_select}", font=dict(size=14)),
                    )
                    fig_mae.update_traces(hovertemplate="%{y}: average miss %{x:.2f}<extra></extra>")
                    st.plotly_chart(
                        fig_mae,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="forecast_mae_chart",
                    )

                    # Ensemble MAE
                    for full_ticker, experts in _rolling_maes.items():
                        if full_ticker.split(".")[0] == ticker_select:
                            ens_list = experts.get("ensemble", [])
                            if ens_list:
                                st.metric(f"Blended avg miss ({ticker_select})", f"{sum(ens_list)/len(ens_list):.2f}", help=_help("mae"))
                            break

                with st.expander("Full forecast-miss table"):
                    st.dataframe(mae_df, use_container_width=True)

        # ── Tab 2: Weight Distribution ──
        with tab_weights:
            st.caption("How the app splits influence across forecast models for each holding.")

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
                weight_df.index.name = "Model"

                ticker_select_w = st.selectbox("Choose holding:", list(weight_data.keys()), key="weight_ticker")

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
                        xaxis_title="", yaxis_title="Influence %",
                        coloraxis_showscale=False,
                        title=dict(text=f"Model influence - {ticker_select_w}", font=dict(size=14)),
                    )
                    st.plotly_chart(
                        fig_w,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="forecast_weight_bar_chart",
                    )

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
                        title=dict(text=f"Influence profile - {ticker_select_w}", font=dict(size=14)),
                    )
                    st.plotly_chart(
                        fig_radar_w,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="forecast_weight_radar_chart",
                    )

                with st.expander("Full model mix table (%)"):
                    st.dataframe(weight_df, use_container_width=True)

        # ── Tab 3: Signal Impact Analysis ──
        with tab_impact:
            st.caption(
                "Compares each model against the blended forecast. "
                "**Negative means the model helped accuracy. Positive means it hurt accuracy.**"
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

                ticker_select_i = st.selectbox("Choose holding:", list(impact_data.keys()), key="impact_ticker")

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
                        xaxis_title="Average miss vs blended forecast",
                        title=dict(text=f"What helped - {ticker_select_i}", font=dict(size=14)),
                    )
                    st.plotly_chart(
                        fig_impact,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="forecast_signal_impact_chart",
                    )

                    best_expert = impact_series.idxmin()
                    worst_expert = impact_series.idxmax()
                    bcol, wcol = st.columns(2)
                    bcol.metric("Most helpful", best_expert, f"{impact_series[best_expert]:+.2f} vs blend")
                    wcol.metric("Least helpful", worst_expert, f"{impact_series[worst_expert]:+.2f} vs blend", delta_color="inverse")

                with st.expander("Full model-impact table"):
                    st.dataframe(impact_df, use_container_width=True)

                st.markdown("---")
                st.markdown("**Model ranking across the portfolio**")
                avg_impact = impact_df.mean(axis=1).sort_values()
                summary_df = pd.DataFrame({
                    "Model": avg_impact.index,
                    "Avg miss vs blend": [f"{v:+.2f}" for v in avg_impact.values],
                    "Verdict": [
                        "Helps" if v < -0.5
                        else "Neutral" if abs(v) <= 0.5
                        else "Hurts"
                        for v in avg_impact.values
                    ],
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # ── Tab 4: Weight Optimization ──
        with tab_backtest:
            st.markdown(
                "**Weight tuning** - the app checks which score drivers have recently worked "
                "and gently shifts influence toward them."
            )
            st.caption(
                f"Stability guard: {config.WEIGHT_SHRINKAGE:.0%} toward equal weights | "
                f"Minimum influence: {config.WEIGHT_MIN_FLOOR:.0%} per score driver"
            )

            if st.button("Run weight tuning", key="run_backtest", type="secondary"):
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
                    pillar_names = ["Price trend", "Business quality", "News mood", "Forecast"]
                    pillar_keys = ["technical", "fundamental", "sentiment", "forecast"]

                    fig_wcomp = go.Figure()
                    for label, weights_dict, color in [
                        ("Current", dict(config.WEIGHTS), "#6b7280"),
                        ("Recent accuracy", opt.ic_based_weights, "#f59e0b"),
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
                        yaxis_title="Influence %",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        title=dict(text="Model mix comparison", font=dict(size=14)),
                    )
                    st.plotly_chart(
                        fig_wcomp,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="weight_optimization_comparison_chart",
                    )

                    # Weight table
                    weight_comparison = pd.DataFrame({
                        "Score driver": pillar_names,
                        "Current": [config.WEIGHTS[k] for k in pillar_keys],
                        "Recent accuracy": [opt.ic_based_weights[k] for k in pillar_keys],
                        "Grid Search": [opt.grid_search_weights[k] for k in pillar_keys],
                        "Recommended": [opt.recommended_weights[k] for k in pillar_keys],
                    })
                    weight_comparison["Change"] = weight_comparison["Recommended"] - weight_comparison["Current"]
                    st.dataframe(
                        weight_comparison.style.format({
                            "Current": "{:.1%}",
                            "Recent accuracy": "{:.1%}",
                            "Grid Search": "{:.1%}", "Recommended": "{:.1%}",
                            "Change": "{:+.1%}",
                        }),
                        hide_index=True, use_container_width=True,
                    )

                    # Summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Stocks checked", f"{opt.universe_size} stocks")
                    m2.metric("Stability guard", f"{opt.shrinkage_factor:.0%}")
                    m3.metric("Current fit", f"{opt.fitness_current:.3f}")
                    m4.metric("Recommended fit", f"{opt.fitness_recommended:.3f}",
                              delta=f"{opt.fitness_recommended - opt.fitness_current:+.3f}")

                    # Pillar ICs
                    st.markdown("#### Which score drivers have worked recently")
                    st.caption(_help("ic"))
                    if opt.pillar_ics:
                        ic_rows = []
                        for pillar in pillar_keys:
                            pic = opt.pillar_ics.get(pillar)
                            if pic:
                                ic_rows.append({
                                    "Score driver": _plain_pillar(pillar),
                                    "Predictive link": f"{pic.avg_ic:+.3f}",
                                    "Stability": f"{pic.ic_std:.3f}" if pic.ic_std > 0 else "-",
                                    "Signal quality": f"{pic.ic_ir:+.2f}" if pic.ic_ir != 0 else "-",
                                    "Snapshots": pic.num_snapshots,
                                    "Method": pic.method,
                                    "Signal": (
                                        "Strong" if abs(pic.avg_ic) > 0.2
                                        else "Moderate" if abs(pic.avg_ic) > 0.1
                                        else "Weak"
                                    ),
                                })
                        st.dataframe(pd.DataFrame(ic_rows), hide_index=True, use_container_width=True)

                    # Expandable sections
                    with st.expander("Per-stock score breakdown"):
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
                                "Price trend": f"{s.technical_score:+.2f}",
                                "Business quality": f"{s.fundamental_score:+.2f}",
                                "News mood": f"{s.sentiment_score:+.2f}",
                                "Forecast": f"{s.forecast_score:+.2f}",
                                "Combined score": f"{agg:+.3f}",
                                "Recommendation": _plain_action(_score_to_action(agg)),
                                f"Actual {config.FORECAST_HORIZON_DAYS}d Return": f"{s.forward_return_pct:+.2f}%",
                            })
                        st.dataframe(pd.DataFrame(stock_rows), hide_index=True, use_container_width=True)

                    with st.expander("Top 10 model-mix combinations"):
                        top_rows = []
                        for i, entry in enumerate(opt.weight_grid_top_n, 1):
                            w = entry["weights"]
                            top_rows.append({
                                "#": i,
                                "Price trend": f"{w['technical']:.0%}",
                                "Business quality": f"{w['fundamental']:.0%}",
                                "News mood": f"{w['sentiment']:.0%}",
                                "Forecast": f"{w['forecast']:.0%}",
                                "Fitness": f"{entry['fitness']:.4f}",
                            })
                        st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

                    if opt.skipped_tickers:
                        with st.expander(f"Skipped tickers ({len(opt.skipped_tickers)})"):
                            st.write(", ".join(opt.skipped_tickers))

                    st.info(
                        "**How it works:** the app checks which score drivers have recently predicted returns, "
                        "keeps the shifts small to avoid overfitting, and preserves a minimum influence for each "
                        "driver so one noisy signal cannot take over. Re-run monthly."
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
                    pm1.metric("Direction hit rate", f"{hit_pct:.1f}%",
                               delta=f"{'above' if hit_pct >= 50 else 'below'} 50%",
                               delta_color=hit_color)
                    pm2.metric("Avg miss", f"{perf['avg_error_pct']:.1f}%", help=_help("mae"))
                    pm3.metric("Large-miss penalty", f"{perf['rmse']:.2f}")
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
                            yaxis=dict(title="Direction hit rate %", range=[0, 100]),
                            xaxis=dict(title="Date"),
                            title=dict(text="Rolling direction accuracy (30-prediction window)",
                                       font=dict(size=13)),
                        )
                        st.plotly_chart(
                            fig_roll,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="forecast_rolling_accuracy_chart",
                        )

                    # Expert comparison table
                    if perf.get("expert_comparison"):
                        st.markdown("**Forecast model comparison**")
                        expert_perf_rows = []
                        for name, stats in sorted(
                            perf["expert_comparison"].items(),
                            key=lambda x: x[1]["hit_rate"],
                            reverse=True,
                        ):
                            expert_perf_rows.append({
                                "Model": name.replace("_", " ").title(),
                                "Direction hit rate": f"{stats['hit_rate']:.1%}",
                                "Avg miss": f"{stats['avg_error_pct']:.1f}%",
                                "Large-miss penalty": f"{stats['rmse']:.2f}",
                                "Predictions": stats["count"],
                            })
                        st.dataframe(pd.DataFrame(expert_perf_rows), hide_index=True,
                                     use_container_width=True)

                    # Per-ticker breakdown
                    if perf.get("per_ticker"):
                        st.markdown("**Accuracy by holding**")
                        ticker_perf_rows = []
                        for ticker, stats in sorted(
                            perf["per_ticker"].items(),
                            key=lambda x: x[1]["hit_rate"],
                            reverse=True,
                        ):
                            ticker_perf_rows.append({
                                "Ticker": ticker,
                                "Direction hit rate": f"{stats['hit_rate']:.1%}",
                                "Avg miss": f"{stats['avg_error_pct']:.1f}%",
                                "Predictions": stats["count"],
                            })
                        st.dataframe(pd.DataFrame(ticker_perf_rows), hide_index=True,
                                     use_container_width=True)
                else:
                    st.info(
                        f"Need at least 5 evaluated predictions for performance metrics. "
                        f"Currently: {perf['total_predictions']} predictions tracked."
                    )
            except Exception as e:
                st.warning(f"Could not load forecast performance: {e}")

    else:
        st.info("Learning and backtest results will appear after the first forecast run builds accuracy history.")


with tab_discovery:
    # ---------------------------------------------------------------------------
    # Global Discovery Engine
    # ---------------------------------------------------------------------------
    st.markdown("### New stock ideas")
    st.caption(
        "Fresh buy candidates, wait-for-pullback ideas, and watchlist names ranked for your current portfolio."
    )

    # ── Regime banner (Phase 4.5) — surfaces which factors are being
    # up/down-weighted so traders can calibrate how aggressively to act
    # on discovery output in the current macro regime.
    try:
        from engine.regime import get_vix_regime as _get_vix_regime, get_multi_macro_regime as _get_multi_regime
        _vix_r = _get_vix_regime()
        _macro_r = _get_multi_regime()
        _regime_label = _vix_r.get("regime_label", "NEUTRAL")
        _vix_pctl = _vix_r.get("vix_percentile", 50.0)
        _macro_label = _macro_r.get("regime_label", "NEUTRAL")
        _tilts = _macro_r.get("factor_tilts", {}) or {}
        _regime_color = {
            "BULL": "#10b981", "NEUTRAL": "#3b82f6", "BEAR": "#ef4444",
            "RISK_ON": "#10b981", "RISK_OFF": "#ef4444",
            "TRANSITION_UP": "#84cc16", "TRANSITION_DOWN": "#f59e0b",
        }
        _vc = _regime_color.get(_regime_label, "#6b7280")
        _mc = _regime_color.get(_macro_label, "#6b7280")
        _tilt_strs = [
            f"{_plain_pillar(k.replace('_tilt',''))}: {v:+.0%}"
            for k, v in _tilts.items() if v
        ]
        _tilt_txt = " | ".join(_tilt_strs) if _tilt_strs else "no extra model tilt"
        st.markdown(
            f'<div style="margin:8px 0;padding:10px 14px;border-radius:8px;'
            f'background:rgba(59,130,246,0.08);border-left:3px solid {_mc};'
            f'font-size:13px;">'
            f'<b>Market backdrop:</b> '
            f'<span style="color:{_vc};font-weight:600">VIX {_regime_label}</span> '
            f'(stress percentile {_vix_pctl:.0f}) | '
            f'<span style="color:{_mc};font-weight:600">Macro {_macro_label}</span> '
            f'<span style="color:#94a3b8;margin-left:8px">Model emphasis: {_tilt_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass  # Regime data unavailable — banner is optional

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
                            return_10d=c.get("return_10d") or 0,
                            volume_ratio=c.get("volume_ratio") or 1.0,
                            vol_20d=c.get("vol_20d"),
                            pe_ratio=c.get("pe_ratio"),
                            peg_ratio=c.get("peg_ratio"),
                            revenue_growth=c.get("revenue_growth"),
                            roe=c.get("roe"),
                            short_pct=c.get("short_pct"),
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
                            quality_score_fundamental=c.get("quality_score_fundamental") or 0.0,
                            gross_profitability=c.get("gross_profitability"),
                            fcf_to_assets=c.get("fcf_to_assets"),
                            earnings_stability=c.get("earnings_stability"),
                            eps_growth_variance_5y=c.get("eps_growth_variance_5y"),
                            factor_momentum_tilt=c.get("factor_momentum_tilt") or {},
                            network_momentum=c.get("network_momentum"),
                            qa_sentiment_score=c.get("qa_sentiment_score"),
                            ml_alpha_raw=c.get("ml_alpha_raw"),
                            analysis_degraded=c.get("analysis_degraded", False),
                            analysis_degraded_reason=c.get("analysis_degraded_reason"),
                            alpha_rank=c.get("alpha_rank") or 0.0,
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
            except Exception as _cache_err:
                import logging as _logging
                _logging.getLogger("app").warning(
                    "Failed to restore cached discovery: %s: %s",
                    type(_cache_err).__name__, _cache_err,
                )

    from utils.orchestrator_status import get_orchestrator_status
    _orch_status = get_orchestrator_status()

    _disc_col1, _disc_col2 = st.columns([1, 3])
    with _disc_col1:
        _run_discovery = st.button(
            "Re-run screener",
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
                _eta_str = f" | about {_eta:.0f} min remaining" if _eta else ""
                st.warning(
                    f"Batch orchestrator running (PID {_orch_status['pid']}) | "
                    f"Scoring {_ckpt['scored_count']}/{_ckpt['total']} "
                    f"({_pct:.0%}){_eta_str}"
                )
            else:
                st.warning(
                    f"Batch orchestrator running (PID {_orch_status['pid']}) | "
                    f"Screening phase (no scoring progress yet)"
                )
        else:
            _cached_ts = st.session_state.get("discovery_cached_from")
            if _cached_ts and "discovery_results" in st.session_state:
                st.info(f"Showing cached results | {format_freshness(_cached_ts)} | Click Re-run to refresh")
            else:
                st.info(
                    f"Screens {len(config.DISCOVERY_EXCHANGES)} US exchanges plus global universe | "
                    f"Market cap at least GBP {config.DISCOVERY_MIN_MCAP / 1e6:.0f}M | "
                    f"Top {config.DISCOVERY_TOP_N_FULL_SCORE} fully scored | "
                    f"about 60-90 min runtime"
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
                f"**Screener path:** {disc.screened_count} checked -> "
                f"{momentum_count} with trend -> "
                f"{disc.after_quick_filter} passed quick checks -> "
                f"{disc.after_corr_filter} not too similar -> "
                f"{disc.after_quick_rank} ranked -> "
                f"{disc.fully_scored} fully scored | "
                f"{disc.run_time_seconds:.0f}s"
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

            st.markdown("#### Recommendation summary")
            cc1, cc2, cc3, cc4 = st.columns(4)
            _command_cards = [
                (
                    cc1, "best", "Best new opportunity", _best_idea,
                    f"Opportunity score {safe_float(getattr(_best_idea, 'final_rank', 0)):.3f} | {_plain_action(getattr(_best_idea, 'action', 'NEUTRAL'))}",
                ),
                (
                    cc2, "fit", "Best diversifier", _best_fit,
                    f"Portfolio fit {safe_float(getattr(_best_fit, 'portfolio_fit_score', 0)):.2f} | Similarity {safe_float(getattr(_best_fit, 'max_correlation', 0)):.2f}",
                ),
                (
                    cc3, "momentum", "Trend leader", _best_momentum,
                    f"Trend {safe_float(getattr(_best_momentum, 'momentum_score', 0)):.2f} | 90d {format_pct(safe_float(getattr(_best_momentum, 'return_90d', 0)) * 100)}",
                ),
                (
                    cc4, "risk", "Biggest caution", _watch_candidate,
                    _candidate_risk_tags(_watch_candidate)[0] if _candidate_risk_tags(_watch_candidate) else "No major red-flag overlays in the current top tier",
                ),
            ]
            _command_cards = [
                (cc1, "best", "Best new opportunity", _best_idea, _best_idea_meta, _best_idea_sub),
                (
                    cc2,
                    "fit",
                    "Best diversifier",
                    _best_fit,
                    f"Portfolio fit {safe_float(getattr(_best_fit, 'portfolio_fit_score', 0)):.2f} | Similarity {safe_float(getattr(_best_fit, 'max_correlation', 0)):.2f}",
                    _candidate_thesis(_best_fit),
                ),
                (
                    cc3,
                    "momentum",
                    "Trend leader",
                    _best_momentum,
                    f"Trend {safe_float(getattr(_best_momentum, 'momentum_score', 0)):.2f} | 90d {format_pct(safe_float(getattr(_best_momentum, 'return_90d', 0)) * 100)}",
                    _candidate_thesis(_best_momentum),
                ),
                (
                    cc4,
                    "risk",
                    "Biggest caution",
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

            _setup_rows = _build_discovery_setup_rows(disc.candidates)
            if _setup_rows:
                st.markdown("#### Actionable setup board")
                st.caption(
                    "Ready-to-buy names surface first, then pullback setups. "
                    "Identity warnings and weak execution plans are demoted."
                )
                st.dataframe(pd.DataFrame(_setup_rows), hide_index=True, use_container_width=True)

            st.markdown("#### Recommendation view")
            _lens = st.radio(
                "Recommendation view",
                ["Balanced growth and downside protection", "Best ideas", "Best diversifiers", "Trend leaders", "Value and quality"],
                horizontal=True,
                label_visibility="collapsed",
                key="discovery_lens",
            )
            _lens_notes = {
                "Balanced growth and downside protection": "Prioritises ready entries, calmer stocks, portfolio fit, and income support over raw excitement.",
                "Best ideas": "Balanced view of conviction, confidence, trend strength, and portfolio fit.",
                "Best diversifiers": "Highlights names that improve portfolio shape without giving up too much quality.",
                "Trend leaders": "Pulls the strongest trend-following setups to the front.",
                "Value and quality": "Pushes fundamental strength and cleaner business quality higher in the stack.",
            }
            st.markdown(f'<div class="lens-note">{_html.escape(_lens_notes[_lens])}</div>', unsafe_allow_html=True)

            _featured = _lens_sorted_candidates(disc.candidates, _lens)[:3]
            if _featured:
                st.markdown("#### Featured recommendations")
                _feature_cols = st.columns(len(_featured))
                _country_flags = {
                    "US": "US", "UK": "UK", "GB": "UK", "CA": "CA",
                    "DE": "DE", "FR": "FR", "IT": "IT", "ES": "ES",
                    "NL": "NL", "JP": "JP",
                }
                for idx, (col, cand) in enumerate(zip(_feature_cols, _featured), start=1):
                    with col:
                        _render_candidate_detail_card(cand, label=f"#{idx} in {_lens}")

            # Full results table: click a row to see the detail card.
            with st.expander("All scored candidates"):
                disc_rows = []
                for c in disc.candidates:
                    _c_entry = safe_float(getattr(c, "entry_price", None))
                    _c_stop = safe_float(getattr(c, "stop_loss", None))
                    _c_rr = safe_float(getattr(c, "r_r_ratio", None))
                    disc_rows.append({
                        "Opportunity score": c.final_rank,
                        "Top idea": "Yes" if _is_top_pick(c) else "",
                        "Ticker": c.ticker,
                        "Name": c.name,
                        "Exchange": c.exchange,
                        "Sector": c.sector,
                        "Entry view": _candidate_entry_stance(c),
                        "Base score": c.aggregate_score,
                        "Confidence": _discovery_confidence(c)[0],
                        "Portfolio fit": c.portfolio_fit_score,
                        "Market sensitivity": round(safe_float(getattr(c, "beta_90d", None), default=0.0), 2) if getattr(c, "beta_90d", None) is not None else "-",
                        "Dividend yield": f"{safe_float(getattr(c, 'dividend_yield', 0)) * 100:.1f}%" if getattr(c, "dividend_yield", None) is not None else "-",
                        "Ticker check": "Verify" if getattr(c, "ticker_identity_warning", None) else "OK",
                        "Recommendation": _plain_action(c.action),
                        "Ready now": "Yes" if getattr(c, "ready_contract_status", None) == "PASS" else "No",
                        "Next trigger": _candidate_entry_trigger(c),
                        "Price trend": round(safe_float(c.technical_score), 2),
                        "Business quality": round(safe_float(c.fundamental_score), 2),
                        "News mood": round(safe_float(c.sentiment_score), 2),
                        "Forecast": round(safe_float(c.forecast_score), 2),
                        "Buy around": f"{_c_entry:.2f}" if _c_entry else "-",
                        "Risk limit": f"{_c_stop:.2f}" if _c_stop else "-",
                        "Reward/risk": f"{_c_rr:.1f}x" if _c_rr and _c_rr > 0 else "-",
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

                    def _top_pick_style(val):
                        if val == "Yes":
                            return "background: rgba(234, 179, 8, 0.15); color: #eab308; font-weight: 700; text-align: center"
                        return ""

                    _pillar_cols = ["Price trend", "Business quality", "News mood", "Forecast"]
                    _styled = _disc_df.style.map(
                        _pillar_color, subset=_pillar_cols,
                    ).map(
                        _top_pick_style, subset=["Top idea"],
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
                            _render_candidate_detail_card(_sel_cand, label=f"{_sel_cand.ticker} detail")

            # Rejection reasons
            if disc.rejections:
                with st.expander(f"Rejected candidates ({len(disc.rejections)})"):
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
                with st.expander(f"Screener track record ({len(_perf)} evaluated, {_pending} pending)"):
                    # --- Pillar Effectiveness ---
                    if _pstats:
                        st.markdown("**Which score drivers have worked**")
                        st.caption(_help("ic"))
                        ps_rows = [{
                            "Score driver": _plain_pillar(s["pillar"]),
                            "Predictive link": f"{s['information_coefficient']:+.3f}",
                            "Hit rate": f"{s['hit_rate']:.0%}",
                            "Avg return - high score": f"{s['avg_return_high']:+.1f}%",
                            "Avg return - low score": f"{s['avg_return_low']:+.1f}%",
                            "Samples": s["sample_size"],
                        } for s in _pstats]
                        st.dataframe(pd.DataFrame(ps_rows), hide_index=True, use_container_width=True)

                    # --- Action Calibration ---
                    if _acal:
                        st.markdown("**Are recommendations calibrated?**")
                        ac_rows = [{
                            "Recommendation": _plain_action(a["action"]),
                            "Avg 90d return": f"{a['avg_return_90d']:+.1f}%",
                            "Accuracy": f"{a['hit_rate']:.0%}",
                            "Samples": a["sample_size"],
                        } for a in _acal]
                        st.dataframe(pd.DataFrame(ac_rows), hide_index=True, use_container_width=True)

                    # --- Regime Effectiveness ---
                    if _rstats:
                        st.markdown("**Which market backdrop works best?**")
                        re_rows = [{
                            "Market backdrop": r["regime"],
                            "Avg 90d return": f"{r['avg_return_90d']:+.1f}%",
                            "Best score driver": _plain_pillar(r["best_pillar"] or "-"),
                            "Samples": r["sample_size"],
                        } for r in _rstats]
                        st.dataframe(pd.DataFrame(re_rows), hide_index=True, use_container_width=True)

                    # --- Stop/Target + Forecast Stats ---
                    st_col, fc_col = st.columns(2)
                    with st_col:
                        if _st_stats and _st_stats.get("with_stops"):
                            st.markdown("**Risk limit and target hits**")
                            total_w = _st_stats["with_stops"]
                            s_hit = _st_stats.get("stops_hit") or 0
                            t_hit = _st_stats.get("targets_hit") or 0
                            st.caption(
                                f"Risk limits hit: {s_hit}/{total_w} ({s_hit/total_w:.0%})"
                                + (f" - avg day {_st_stats['avg_stop_day']:.0f}" if _st_stats.get("avg_stop_day") else "")
                            )
                            st.caption(
                                f"Targets hit: {t_hit}/{total_w} ({t_hit/total_w:.0%})"
                                + (f" - avg day {_st_stats['avg_target_day']:.0f}" if _st_stats.get("avg_target_day") else "")
                            )
                    with fc_col:
                        if _st_stats and _st_stats.get("avg_forecast_err_5d") is not None:
                            st.markdown("**Forecast accuracy**")
                            st.caption(f"5-day average miss: {_st_stats['avg_forecast_err_5d']:.1f}%")
                            if _st_stats.get("avg_forecast_err_63d") is not None:
                                st.caption(f"63-day average miss: {_st_stats['avg_forecast_err_63d']:.1f}%")

                    # --- Multi-horizon signal performance ---
                    if _perf:
                        st.markdown("**Recommendation performance across time**")
                        perf_rows = [{
                            "Date": p["run_date"][:10],
                            "Ticker": p["ticker"],
                            "Source": p.get("source", "-"),
                            "Recommendation": _plain_action(p.get("action", "-")),
                            "Score": f"{p['aggregate_score']:.3f}",
                            "30d": f"{p['return_30d']:+.1f}%" if p.get("return_30d") is not None else "-",
                            "60d": f"{p['return_60d']:+.1f}%" if p.get("return_60d") is not None else "-",
                            "90d": f"{p['return_90d']:+.1f}%" if p.get("return_90d") is not None else "-",
                            "Beat SPY": "Yes" if p.get("beat_market") else "No",
                            "Recommendation OK": "Yes" if p.get("action_correct") else "No",
                        } for p in _perf]
                        st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)

                        # Summary metrics
                        returns = [p["return_90d"] for p in _perf if p.get("return_90d") is not None]
                        if returns:
                            bc1, bc2, bc3, bc4 = st.columns(4)
                            bc1.metric("Avg return", f"{sum(returns)/len(returns):+.1f}%")
                            bc2.metric("Win rate", f"{sum(1 for r in returns if r > 0)/len(returns):.0%}")
                            bc3.metric("Best pick", f"{max(returns):+.1f}%")
                            bc4.metric("Worst pick", f"{min(returns):+.1f}%")

                        # Beat market rate
                        beat = [p for p in _perf if p.get("beat_market") is not None]
                        if beat:
                            beat_rate = sum(1 for p in beat if p["beat_market"]) / len(beat)
                            st.caption(f"Beat SPY: {beat_rate:.0%} of signals | "
                                       f"Recommendation accuracy: {sum(1 for p in _perf if p.get('action_correct'))/len(_perf):.0%}")
        except Exception:
            pass

        # Evaluation Harness — Scorecard
        try:
            from engine.evaluation_harness import compute_scorecard

            _sc = compute_scorecard(source="all", min_signals=5)
            if _sc and _sc.evaluated_signals >= 5 and _sc.sharpe_ratio is not None:
                with st.expander(f"Performance scorecard ({_sc.evaluated_signals} recommendations evaluated)"):
                    # Top-level risk/return metrics
                    ev1, ev2, ev3, ev4 = st.columns(4)
                    ev1.metric("Return per unit of risk", f"{_sc.sharpe_ratio:.2f}", help="Sharpe ratio: higher means more return for each unit of volatility.")
                    ev2.metric("Downside risk quality", f"{_sc.sortino_ratio:.2f}" if _sc.sortino_ratio else "-", help="Sortino ratio: like Sharpe, but focuses on downside moves.")
                    ev3.metric("Worst drawdown", f"{_sc.max_drawdown:+.1f}%" if _sc.max_drawdown else "-")
                    ev4.metric("Recovery quality", f"{_sc.calmar_ratio:.2f}" if _sc.calmar_ratio else "-", help="Calmar ratio: return compared with worst drawdown.")

                    ev5, ev6, ev7, ev8 = st.columns(4)
                    ev5.metric("Hit rate (90d)", f"{_sc.overall_hit_rate:.0%}" if _sc.overall_hit_rate else "-")
                    ev6.metric("Recommendation accuracy", f"{_sc.action_accuracy:.0%}" if _sc.action_accuracy else "-")
                    ev7.metric("Beat SPY rate", f"{_sc.beat_benchmark_rate:.0%}" if _sc.beat_benchmark_rate else "-")
                    ev8.metric("Rank stability", f"{_sc.ic_stability:.4f}" if _sc.ic_stability else "-", help=_help("ic"))

                    # Per-horizon table
                    if _sc.horizons:
                        st.markdown("**Returns by time window**")
                        _h_rows = pd.DataFrame([{
                            "Horizon": h.horizon,
                            "Avg return": f"{h.avg_return:+.1f}%",
                            "Median": f"{h.median_return:+.1f}%",
                            "Typical swing": f"{h.std_return:.1f}%",
                            "Hit rate": f"{h.hit_rate:.0%}",
                            "Return vs SPY": f"{h.alpha:+.1f}%" if h.horizon == "90d" else "-",
                            "Best": f"{h.best:+.1f}%",
                            "Worst": f"{h.worst:+.1f}%",
                            "N": h.sample_size,
                        } for h in _sc.horizons])
                        st.dataframe(_h_rows, hide_index=True, use_container_width=True)

                    # Per-regime table
                    if _sc.regimes:
                        st.markdown("**Performance by market backdrop**")
                        _r_rows = pd.DataFrame([{
                            "Market backdrop": r.regime,
                            "Avg 90d return": f"{r.avg_return_90d:+.1f}%",
                            "Hit rate": f"{r.hit_rate:.0%}",
                            "Best score driver": _plain_pillar(r.best_pillar or "-"),
                            "N": r.sample_size,
                        } for r in _sc.regimes])
                        st.dataframe(_r_rows, hide_index=True, use_container_width=True)

                    # Stop/target + forecast
                    st_c, fc_c = st.columns(2)
                    with st_c:
                        if _sc.stop_hit_rate is not None:
                            st.markdown("**Risk limit and target effectiveness**")
                            st.caption(f"Risk limit hit rate: {_sc.stop_hit_rate:.0%}"
                                       + (f" (avg day {_sc.avg_stop_day:.0f})" if _sc.avg_stop_day else ""))
                            st.caption(f"Target hit rate: {_sc.target_hit_rate:.0%}"
                                       + (f" (avg day {_sc.avg_target_day:.0f})" if _sc.avg_target_day else ""))
                    with fc_c:
                        if _sc.avg_forecast_error_5d is not None:
                            st.markdown("**Forecast accuracy**")
                            st.caption(f"5-day average miss: {_sc.avg_forecast_error_5d:.1f}%")
                            if _sc.avg_forecast_error_63d is not None:
                                st.caption(f"63-day average miss: {_sc.avg_forecast_error_63d:.1f}%")

        except Exception as _sc_err:
            import logging as _logging
            _logging.getLogger(__name__).warning("Performance scorecard failed: %s", _sc_err)

        # Discovery-specific evaluation — ranked by final_rank
        try:
            from engine.discovery_eval import get_discovery_scorecard

            _disc_sc = get_discovery_scorecard()
            if _disc_sc and _disc_sc.get("total_evaluated", 0) >= 5:
                with st.expander(
                    f"New-idea quality ({_disc_sc['total_evaluated']} evaluated picks)"
                ):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("Top-10 hit rate", f"{safe_float(_disc_sc.get('top10_hit_rate_90d')):.0%}")
                    dc2.metric("Top-10 avg return", format_pct(_disc_sc.get("top10_avg_return_90d")))
                    dc3.metric("Return vs SPY", format_pct(_disc_sc.get("excess_vs_spy_90d")))
                    dc4.metric("Ranking stability", f"{safe_float(_disc_sc.get('ranking_stability')):.0%}")

                    dc5, dc6 = st.columns(2)
                    dc5.metric("Top-30 hit rate", f"{safe_float(_disc_sc.get('top30_hit_rate_90d')):.0%}")
                    dc6.metric("Swap success rate", f"{safe_float(_disc_sc.get('swap_success_rate')):.0%}")

                    if _disc_sc.get("summary"):
                        st.caption(_disc_sc["summary"])
        except Exception as _disc_sc_err:
            import logging as _logging
            _logging.getLogger(__name__).warning("Discovery scorecard failed: %s", _disc_sc_err)



with tab_analytics:
    # ---------------------------------------------------------------------------
    # Trade History — Record Sales
    # ---------------------------------------------------------------------------
    st.markdown("### Trade history")

    from utils.data_fetch import load_portfolio_full, record_sale

    _portfolio_full = load_portfolio_full()
    _trade_history = _portfolio_full.get("trade_history", [])

    tab_record, tab_history = st.tabs(["Record a sale", "Past trades"])

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
                        f"Current price: {_sel_result.get('current_price', 0):.2f} | "
                        f"Avg buy: {_sel_holding['avg_buy_price']:.2f} | "
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

            if st.button("Confirm sale", type="primary"):
                trade = record_sale(
                    ticker=_sell_ticker,
                    sell_price=_sell_price,
                    quantity=_sell_qty,
                    sell_date=str(_sell_date),
                    notes=_sell_notes,
                )
                if trade:
                    st.success(
                        f"Recorded: sold {trade['quantity']} x {trade['ticker']} at {trade['sell_price']:.4f} "
                        f"- gain/loss: {trade['pnl']:+,.2f} ({trade['pnl_pct']:+.1f}%)"
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
                    "Quantity": t["quantity"],
                    "Buy price": f"{t['buy_price']:.4f}",
                    "Sell price": f"{t['sell_price']:.4f}",
                    "Gain/loss": f"{t['pnl']:+,.2f}",
                    "Return": f"{t['pnl_pct']:+.1f}%",
                    "Notes": t.get("notes", ""),
                })
                _total_realized += t.get("pnl", 0)

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total trades", len(_trade_history))
            tc2.metric("Realised gain/loss", f"{_total_realized:+,.2f}")
            _winners = sum(1 for t in _trade_history if t.get("pnl", 0) > 0)
            tc3.metric("Win rate", f"{_winners / len(_trade_history) * 100:.0f}%" if _trade_history else "-")

            st.dataframe(pd.DataFrame(_th_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No trades recorded yet. Use the 'Record a Sale' tab when you sell a stock on your broker.")


    # ---------------------------------------------------------------------------
    # Paper Trading Ledger
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("### Paper trading ledger")

    if getattr(config, "PAPER_TRADING_ENABLED", False):
        from engine.paper_trading import (
            get_all_signals, get_slippage_stats, get_pnl_summary,
            get_slippage_by_ticker, get_open_positions, get_realized_pnl,
            get_unrealized_pnl, init_db as _pt_init,
        )
        _pt_init()

        tab_overview, tab_signals, tab_positions, tab_slippage = st.tabs([
            "Overview", "Signal log", "Paper positions", "Execution drift",
        ])

        # --- Overview tab ---
        with tab_overview:
            pnl_sum = get_pnl_summary()
            slip_stats = get_slippage_stats()

            col1, col2, col3, col4 = st.columns(4)
            total_trades = pnl_sum.get("total_trades") or 0
            with col1:
                st.metric("Total closed trades", total_trades)
            with col2:
                total_pnl = pnl_sum.get("total_pnl") or 0
                st.metric("Total gain/loss", f"{'+'if total_pnl>=0 else ''}{total_pnl:,.2f}")
            with col3:
                win_rate = (
                    (pnl_sum["winners"] / total_trades * 100)
                    if total_trades > 0 and pnl_sum.get("winners") is not None else 0
                )
                st.metric("Win rate", f"{win_rate:.0f}%")
            with col4:
                avg_slip = slip_stats.get("avg_slippage_bps") or 0
                st.metric("Avg execution drift", f"{avg_slip:+.1f} bps", help="Execution drift is the difference between signal price and fill price. bps means basis points: 100 bps = 1%.")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Avg return", f"{pnl_sum.get('avg_return_pct') or 0:+.1f}%")
            with col6:
                st.metric("Best trade", f"{pnl_sum.get('best_trade_pct') or 0:+.1f}%")
            with col7:
                st.metric("Worst trade", f"{pnl_sum.get('worst_trade_pct') or 0:+.1f}%")
            with col8:
                st.metric("Avg hold", f"{pnl_sum.get('avg_hold_days') or 0:.0f} days")

            # Unrealized P&L for open positions
            unrealized = get_unrealized_pnl()
            if unrealized:
                st.markdown("#### Open paper positions - unrealised gain/loss")
                ur_rows = []
                for u in unrealized:
                    ur_rows.append({
                        "Ticker": u["ticker"],
                        "Quantity": u["quantity"],
                        "Entry price": f"{u['avg_entry_price']:.4f}",
                        "Current price": f"{u['current_price']:.4f}",
                        "Gain/loss": f"{u['unrealized_pnl']:+,.2f}",
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
                        "Signal price": f"{s['signal_price']:.4f}" if s["signal_price"] else "-",
                        "Fill Price": f"{s['fill_price']:.4f}" if s.get("fill_price") else "Pending",
                        "Execution drift (bps)": f"{s['slippage_bps']:+.1f}" if s.get("slippage_bps") is not None else "-",
                        "Score": f"{s['score']:.3f}" if s.get("score") is not None else "—",
                        "Recommendation": _plain_action(s.get("action")) if s.get("action") else "-",
                        "Swap from": s.get("swap_from") or "-",
                    })
                st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)
            else:
                st.info("No paper trade signals recorded yet. Signals are logged automatically by the daily orchestrator.")

        # --- Paper Positions tab ---
        with tab_positions:
            positions = get_open_positions()
            realized = get_realized_pnl()

            if positions:
                st.markdown("#### Open positions")
                pos_rows = [{
                    "Ticker": p["ticker"],
                    "Quantity": p["quantity"],
                    "Avg entry": f"{p['avg_entry_price']:.4f}",
                    "Opened": p["opened_at"],
                } for p in positions]
                st.dataframe(pd.DataFrame(pos_rows), hide_index=True, use_container_width=True)

            if realized:
                st.markdown("#### Closed trades")
                rl_rows = [{
                    "Ticker": r["ticker"],
                    "Entry": f"{r['entry_price']:.4f}",
                    "Exit": f"{r['exit_price']:.4f}",
                    "Quantity": r["quantity"],
                    "Gain/loss": f"{r['pnl']:+,.2f}",
                    "Return": f"{r['pnl_pct']:+.1f}%",
                    "Hold": f"{r['hold_days']}d" if r.get("hold_days") else "-",
                    "Closed": r["closed_at"],
                } for r in realized]
                st.dataframe(pd.DataFrame(rl_rows), hide_index=True, use_container_width=True)

            if not positions and not realized:
                st.info("No paper positions yet. Positions are created when pending signals are filled at next-session open.")

        # --- Execution drift tab ---
        with tab_slippage:
            slip_by_ticker = get_slippage_by_ticker()
            if slip_by_ticker:
                st.markdown("#### Execution drift by ticker")
                st.caption("Execution drift is the difference between the signal price and fill price. bps means basis points: 100 bps = 1%.")
                sl_rows = [{
                    "Ticker": t["ticker"],
                    "Fills": t["fills"],
                    "Avg drift (bps)": f"{t['avg_slippage_bps']:+.1f}",
                    "Avg absolute drift (bps)": f"{t['avg_abs_slippage_bps']:.1f}",
                    "Min (bps)": f"{t['min_slippage_bps']:+.1f}",
                    "Max (bps)": f"{t['max_slippage_bps']:+.1f}",
                } for t in slip_by_ticker]
                st.dataframe(pd.DataFrame(sl_rows), hide_index=True, use_container_width=True)

                # Overall stats
                stats = get_slippage_stats()
                if stats.get("total_fills"):
                    st.markdown("#### Overall execution drift")
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Total fills", stats["total_fills"])
                        st.metric("Buy fills", stats.get("buy_fills") or 0)
                    with mc2:
                        st.metric("Avg drift", f"{stats['avg_slippage_bps']:+.1f} bps")
                        st.metric("Avg buy drift", f"{stats.get('avg_buy_slippage') or 0:+.1f} bps")
                    with mc3:
                        st.metric("Avg absolute drift", f"{stats['avg_abs_slippage_bps']:.1f} bps")
                        st.metric("Avg sell drift", f"{stats.get('avg_sell_slippage') or 0:+.1f} bps")
            else:
                st.info("No fill data yet. Execution drift is calculated when pending signals are resolved at next-session open.")
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
