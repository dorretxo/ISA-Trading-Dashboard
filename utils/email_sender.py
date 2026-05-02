"""SMTP email sender and HTML email builder for the Autonomous Email Engine.

Pure formatting layer — receives pre-computed data from the orchestrator,
never calls any analysis functions itself.  Uses stdlib only (smtplib + email.mime).
"""

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from utils.discovery_digest import sort_trade_packets
from utils.safe_numeric import format_currency, format_pct, safe_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMTP send
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str, dry_run: bool = False) -> bool:
    """Send an HTML email via SMTP.  Returns True on success.

    - If dry_run=True, logs subject + body but does not connect to SMTP.
    - If credentials are empty, logs a warning and returns False.
    - Catches all SMTP/network errors, logs them, and returns False (never raises).
    """
    if dry_run:
        logger.info("[DRY RUN] Would send email:\n  Subject: %s\n  Body length: %d chars",
                     subject, len(html_body))
        return True

    sender = config.EMAIL_FROM
    recipient = config.EMAIL_TO
    password = config.EMAIL_PASSWORD
    host = config.EMAIL_SMTP_HOST
    port = config.EMAIL_SMTP_PORT

    if not sender or not recipient or not password:
        logger.warning("Email credentials not configured (EMAIL_FROM / EMAIL_TO / EMAIL_PASSWORD). "
                        "Skipping email send.")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        logger.info("Email sent successfully: %s", subject)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check EMAIL_PASSWORD (use App Password for Outlook).")
        return False
    except smtplib.SMTPException as e:
        logger.error("SMTP error: %s", e)
        return False
    except OSError as e:
        logger.error("Network error sending email: %s", e)
        return False


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

# Inline CSS for email clients (no external stylesheets in email)
_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; color: #1e1e1e; margin: 0; padding: 0; }
.container { max-width: 680px; margin: 0 auto; padding: 20px; }
.header { background: #1a1a2e; color: white; padding: 20px 24px; border-radius: 8px 8px 0 0; }
.header h1 { margin: 0; font-size: 22px; }
.header .regime { display: inline-block; padding: 3px 10px; border-radius: 12px;
                  font-size: 13px; font-weight: 600; margin-top: 6px; }
.regime-bull { background: #10b981; color: white; }
.regime-bear { background: #ef4444; color: white; }
.regime-neutral { background: #f59e0b; color: white; }
.section { background: #ffffff; border: 1px solid #e5e7eb; padding: 16px 20px; margin-top: -1px; }
.alert-sell { border-left: 4px solid #ef4444; background: #fef2f2; }
.alert-swap { border-left: 4px solid #3b82f6; background: #eff6ff; }
.alert-strategy { border-left: 4px solid #8b5cf6; background: #f5f3ff; }
.alert-discovery { border-left: 4px solid #06b6d4; background: #ecfeff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th { background: #f3f4f6; text-align: left; padding: 8px; font-weight: 600; border-bottom: 2px solid #d1d5db; }
td { padding: 7px 8px; border-bottom: 1px solid #e5e7eb; }
.score-pos { color: #10b981; font-weight: 600; }
.score-neg { color: #ef4444; font-weight: 600; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.pill-sell { background: #fecaca; color: #991b1b; }
.pill-strong-sell { background: #ef4444; color: white; }
.pill-keep { background: #e0e7ff; color: #3730a3; }
.pill-buy { background: #d1fae5; color: #065f46; }
.pill-strong-buy { background: #10b981; color: white; }
.pill-momentum { background: #dbeafe; color: #1e40af; }
.pill-value { background: #fef3c7; color: #92400e; }
.pill-quality { background: #d1fae5; color: #065f46; }
.pill-governance { background: #fef2f2; color: #991b1b; }
.pill-asymmetric { background: #fff7ed; color: #9a3412; }
.pill-exdiv { background: #ecfdf5; color: #065f46; }
.pill-grade-a { background: #d1fae5; color: #065f46; }
.pill-grade-b { background: #dbeafe; color: #1e40af; }
.pill-grade-c { background: #fef3c7; color: #92400e; }
.pill-grade-d { background: #fecaca; color: #991b1b; }
.pill-top-pick { background: #eab308; color: #1a1a2e; font-weight: 700; }
.pill-ready { background: #d1fae5; color: #065f46; }
.pill-pullback { background: #dbeafe; color: #1d4ed8; }
.pill-watch { background: #fef3c7; color: #92400e; }
.pill-confidence-high { background: #dcfce7; color: #166534; }
.pill-confidence-medium { background: #fef3c7; color: #92400e; }
.pill-confidence-low { background: #fee2e2; color: #991b1b; }
.pill-confidence-data { background: #e5e7eb; color: #374151; }
.top-pick-row td { background: #fffbeb !important; }
.metric-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.metric-card { flex: 1 1 140px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px;
               padding: 10px 12px; text-align: center; }
.metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 20px; font-weight: 700; margin-top: 2px; }
.meta-strip { margin-top: 10px; padding: 8px 10px; background: #f8fafc; border: 1px solid #e5e7eb;
              border-radius: 6px; font-size: 12px; color: #475569; }
.hero-setup { margin-top: 10px; padding: 14px 16px; background: #f8fafc; border: 1px solid #dbeafe;
              border-left: 4px solid #2563eb; border-radius: 8px; }
.hero-setup h4 { margin: 0 0 6px 0; font-size: 18px; color: #0f172a; }
.hero-setup .setup-meta { margin-top: 6px; font-size: 12px; color: #475569; }
.footer { background: #f9fafb; padding: 12px 20px; font-size: 11px; color: #6b7280;
          border-radius: 0 0 8px 8px; border: 1px solid #e5e7eb; border-top: none; }
.delta { font-size: 18px; font-weight: 700; color: #2563eb; }
.fx-note { font-size: 12px; color: #92400e; background: #fffbeb; padding: 4px 8px; border-radius: 4px; }
.strategy-note { font-size: 12px; color: #4b5563; line-height: 1.5; margin-top: 8px; }
"""


def _action_pill(action: str) -> str:
    """Render an action label as a coloured pill."""
    cls = {
        "STRONG SELL": "pill-strong-sell",
        "SELL": "pill-sell",
        "KEEP": "pill-keep",
        "NEUTRAL": "pill-keep",
        "MANUAL REVIEW": "pill-keep",
        "AVOID": "pill-sell",
        "BUY": "pill-buy",
        "STRONG BUY": "pill-strong-buy",
    }.get(action, "pill-keep")
    return f'<span class="pill {cls}">{action}</span>'


def _score_html(score: float) -> str:
    """Format score with colour."""
    cls = "score-pos" if score >= 0 else "score-neg"
    return f'<span class="{cls}">{score:+.3f}</span>'


def _lens_pill(lens: str) -> str:
    """Render an entry lens label as a coloured pill."""
    cls = {"momentum": "pill-momentum", "value": "pill-value", "quality": "pill-quality"}.get(lens, "pill-keep")
    return f'<span class="pill {cls}">{lens.upper()}</span>'


def _grade_pill(grade: str | None) -> str:
    """Render a balance sheet grade as a coloured pill."""
    if not grade:
        return ""
    cls = {"A": "pill-grade-a", "B": "pill-grade-b", "C": "pill-grade-c", "D": "pill-grade-d"}.get(grade, "pill-keep")
    return f'<span class="pill {cls}">{grade}</span>'


def _stance_pill(stance: str) -> str:
    cls = {
        "Ready": "pill-ready",
        "Pullback Preferred": "pill-pullback",
        "Watch Only": "pill-watch",
    }.get(stance, "pill-watch")
    return f'<span class="pill {cls}">{stance}</span>'


def _confidence_pill(label: str, tone: str) -> str:
    cls = {
        "high": "pill-confidence-high",
        "medium": "pill-confidence-medium",
        "low": "pill-confidence-low",
        "data": "pill-confidence-data",
    }.get(tone, "pill-confidence-low")
    return f'<span class="pill {cls}">{label}</span>'


def _format_price(val, currency: str = "USD") -> str:
    if val is None:
        return "&mdash;"
    if safe_float(val, default=0.0) <= 0:
        return "&mdash;"
    return format_currency(val, currency, decimals=2)


def _format_freshness(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return "missing"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except (TypeError, ValueError):
        return "unknown"
    delta = datetime.now() - dt
    minutes = int(delta.total_seconds() / 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _is_top_pick_dict(d: dict) -> bool:
    """Check if a discovery candidate dict qualifies as a Top Pick."""
    if d.get("ticker_identity_warning"):
        return False
    if d.get("action") != "STRONG BUY":
        return False
    stance = d.get("entry_stance", "")
    if stance != "Ready":
        return False
    # Replicate confidence score logic from app.py _discovery_confidence
    sent_conf = float(d.get("sentiment_score", 0) or 0)
    data_discount = float(d.get("confidence_discount", 1.0) or 1.0)
    has_data = 0.0 if d.get("action") == "INSUFFICIENT DATA" else 1.0
    score = max(0.0, min(1.0,
        0.35 * has_data + 0.30 * data_discount
        + 0.35 * (0.5 + 0.5 * max(min(sent_conf, 1.0), -1.0))))
    if score < 0.75:
        return False
    return True


def _flag_pills(data: dict) -> str:
    """Render governance/asymmetric/ex-div flag pills for a holding or candidate."""
    pills = []
    if data.get("governance_flag"):
        pills.append('<span class="pill pill-governance">GOV RISK</span>')
    if data.get("asymmetric_risk_flag"):
        pills.append('<span class="pill pill-asymmetric">ASYM RISK</span>')
    ex_div = data.get("ex_dividend_days")
    if ex_div is not None and 0 <= ex_div <= 14:
        pills.append(f'<span class="pill pill-exdiv">EX-DIV {ex_div}d</span>')
    return " ".join(pills)


def _regime_badge(vix_regime: dict) -> str:
    """Render VIX regime as a coloured badge."""
    label = vix_regime.get("regime_label", "NEUTRAL")
    level = vix_regime.get("vix_level", 0)
    cls = {"BULL": "regime-bull", "BEAR": "regime-bear"}.get(label, "regime-neutral")
    return f'<span class="regime {cls}">VIX: {label} ({level})</span>'


def build_alert_email(
    results: list[dict],
    risk_data: dict,
    position_weights: list[dict],
    vix_regime: dict,
    alerts: list[dict],
    swap_recs: list[dict],
    dry_run: bool = False,
    optimizer_alloc=None,
    discovery_candidates: list[dict] | None = None,
    exit_signals: list[dict] | None = None,
    discovery_meta: dict | None = None,
    artifact_timestamps: dict | None = None,
    discovery_ran: bool = False,
) -> tuple[str, str]:
    """Build the subject line and HTML body for an alert email.

    Args:
        results: Per-holding analysis from analyse_portfolio()
        risk_data: Portfolio risk dict (sector_weights, concentration_warnings, etc.)
        position_weights: Inverse-vol suggested allocations
        vix_regime: From get_vix_regime()
        alerts: Holdings with action in (SELL, STRONG SELL)
        swap_recs: List of dicts with keys: candidate, weakest_ticker, weakest_score, score_delta
        dry_run: If True, append [DRY RUN] to subject
        optimizer_alloc: OptimizationResult from portfolio_optimizer (optional)
        discovery_candidates: Top discovery candidates (optional, list of dicts)

    Returns:
        (subject, html_body)
    """
    discovery_meta = discovery_meta or {}
    artifact_timestamps = artifact_timestamps or {}
    discovery_packets = sort_trade_packets(discovery_candidates or [])
    lead_packet = discovery_packets[0] if discovery_packets else None

    # --- Subject line ---
    parts = []
    if alerts:
        alert_tickers = ", ".join(f"{a['action']} {a['ticker']}" for a in alerts[:3])
        parts.append(alert_tickers)
    if swap_recs:
        top = swap_recs[0]
        top_action = top["candidate"].get("action", "NEUTRAL")
        top_verb = "buy" if top_action in ("BUY", "STRONG BUY") else "review"
        parts.append(f"Swap: {top_verb} {top['candidate']['ticker']} +{top['score_delta']:.2f} delta")
    if not parts and lead_packet:
        if lead_packet.get("trade_ready"):
            parts.append(f"Discovery: {lead_packet['ticker']} ready")
        elif lead_packet.get("clean_entry"):
            parts.append(f"Discovery: {lead_packet['ticker']} {lead_packet['entry_stance'].lower()}")
        elif discovery_ran:
            parts.append(f"Discovery refresh ({len(discovery_packets)} names)")
    if not parts and exit_signals:
        parts.append(f"Exit watch {len(exit_signals)}")
    if not parts:
        parts.append("Portfolio review")

    _regime_label = vix_regime.get("regime_label", "NEUTRAL")
    subject = f"[ISA Alert] {' | '.join(parts)} | VIX: {_regime_label}"
    if dry_run:
        subject = f"[DRY RUN] {subject}"

    # --- HTML body ---
    html_parts = [
        f"<html><head><style>{_CSS}</style></head><body>",
        '<div class="container">',
    ]

    # Header
    now_str = datetime.now().strftime("%A %d %B %Y, %H:%M")
    avg_score = sum(r.get("aggregate_score", 0) for r in results) / max(len(results), 1)
    n_holdings = len(results)
    html_parts.append(f"""
    <div class="header">
        <h1>ISA Portfolio Alert</h1>
        <div style="margin-top:6px;">{_regime_badge(vix_regime)}</div>
        <div style="margin-top:8px; font-size:13px; opacity:0.85;">
            {now_str} &bull; {n_holdings} holdings &bull; Avg score: {avg_score:+.3f}
        </div>
    </div>
    """)

    freshness = [
        f"Portfolio: {_format_freshness(artifact_timestamps.get('portfolio'))}",
        f"Screener: {_format_freshness(artifact_timestamps.get('discovery'))}",
        f"Optimizer: {_format_freshness(artifact_timestamps.get('optimizer'))}",
        f"Exit: {_format_freshness(artifact_timestamps.get('exit'))}",
    ]
    html_parts.append(
        f'<div class="section"><div class="meta-strip">{" &bull; ".join(freshness)}</div></div>'
    )

    # Trading Strategy & Portfolio Metrics
    regime_label = vix_regime.get("regime_label", "NEUTRAL")
    vix_level = vix_regime.get("vix_level", 0)
    vix_pct = vix_regime.get("vix_percentile", 50)

    # Strategy narrative based on regime
    if regime_label == "BULL":
        strategy_note = (
            f"Low-volatility regime (VIX {vix_level:.0f}, {vix_pct:.0f}th percentile). "
            "Strategy: lean into momentum and quality positions. "
            "Wider position sizing acceptable. Growth and breakout plays favoured."
        )
    elif regime_label == "BEAR":
        strategy_note = (
            f"High-volatility regime (VIX {vix_level:.0f}, {vix_pct:.0f}th percentile). "
            "Strategy: reduce position sizes, tighten stops, favour quality and low-beta. "
            "Avoid aggressive new entries. Preserve capital."
        )
    else:
        strategy_note = (
            f"Neutral regime (VIX {vix_level:.0f}, {vix_pct:.0f}th percentile). "
            "Strategy: balanced approach. Favour high-conviction entries only. "
            "Monitor for regime shift signals."
        )

    html_parts.append('<div class="section alert-strategy">')
    html_parts.append('<h3 style="margin:0 0 8px 0; color:#5b21b6;">Trading Strategy</h3>')
    html_parts.append(f'<div class="strategy-note">{strategy_note}</div>')

    # Portfolio metrics grid
    n_buy = sum(1 for r in results if r.get("action") in ("BUY", "STRONG BUY"))
    n_sell = sum(1 for r in results if r.get("action") in ("SELL", "STRONG SELL"))
    n_keep = sum(1 for r in results if r.get("action") == "KEEP")

    html_parts.append('<div class="metric-grid">')
    html_parts.append(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Score</div>
            <div class="metric-value {('score-pos' if avg_score >= 0 else 'score-neg')}">{avg_score:+.3f}</div>
        </div>
    """)

    if optimizer_alloc:
        exp_ret = getattr(optimizer_alloc, "portfolio_expected_return", None)
        vol = getattr(optimizer_alloc, "portfolio_volatility", None)
        sharpe = getattr(optimizer_alloc, "portfolio_sharpe", None)
        if exp_ret is not None:
            html_parts.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Expected Return</div>
                    <div class="metric-value score-pos">{exp_ret*100:.1f}%</div>
                </div>
            """)
        if vol is not None:
            html_parts.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{vol*100:.1f}%</div>
                </div>
            """)
        if sharpe is not None:
            sharpe_cls = "score-pos" if sharpe >= 0.5 else ("score-neg" if sharpe < 0 else "")
            html_parts.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sharpe_cls}">{sharpe:.2f}</div>
                </div>
            """)

    html_parts.append(f"""
        <div class="metric-card">
            <div class="metric-label">Signal Mix</div>
            <div class="metric-value" style="font-size:14px;">
                <span class="score-pos">{n_buy} Buy</span> /
                {n_keep} Keep /
                <span class="score-neg">{n_sell} Sell</span>
            </div>
        </div>
    """)
    html_parts.append('</div>')  # metric-grid

    # Risk score
    risk_score = risk_data.get("risk_score", 0)
    if risk_score > 0.5:
        html_parts.append(f'<div style="margin-top:8px; font-size:12px; color:#991b1b;">'
                          f'Portfolio risk score: {risk_score:.2f} (elevated)</div>')

    html_parts.append('</div>')  # section

    # Discovery command center
    if discovery_packets:
        html_parts.append('<div class="section alert-discovery">')
        html_parts.append('<h3 style="margin:0 0 8px 0; color:#0e7490;">Discovery Command Center</h3>')

        if lead_packet and lead_packet.get("clean_entry"):
            lead_flags = []
            if lead_packet.get("top_pick"):
                lead_flags.append('<span class="pill pill-top-pick">&#9733; TOP PICK</span>')
            lead_flags.append(_action_pill(lead_packet["action"]))
            lead_flags.append(_stance_pill(lead_packet["entry_stance"]))
            lead_flags.append(_confidence_pill(lead_packet["confidence_label"], lead_packet["confidence_tone"]))
            plan_bits = [
                f"Rank {safe_float(lead_packet['final_rank']):.3f}",
                f"90d model {format_pct(safe_float(lead_packet['expected_return_90d']) * 100)}",
                f"Entry {_format_price(lead_packet['entry_price'], lead_packet['currency'])}",
                f"Stop {_format_price(lead_packet['stop_loss'], lead_packet['currency'])}",
                f"Target {_format_price(lead_packet['take_profit'], lead_packet['currency'])}",
                f"R/R {safe_float(lead_packet['r_r_ratio']):.1f}x" if lead_packet.get("r_r_ratio") else "R/R unavailable",
                f"Position {format_pct(safe_float(lead_packet['position_weight']) * 100, plus_sign=False)}"
                if lead_packet.get("position_weight") else "Position pending",
                f"Prior {safe_float(lead_packet.get('institutional_prior_percentile')):.0%}"
                if lead_packet.get("institutional_prior_percentile") else "Prior pending",
                f"Fit {safe_float(lead_packet['portfolio_fit_score']):.2f}",
                f"Corr {safe_float(lead_packet['max_correlation']):.2f}",
            ]
            if lead_packet.get("key_risk"):
                plan_bits.append(f"Watch: {lead_packet['key_risk']}")
            html_parts.append(f"""
            <div class="hero-setup">
                <h4>{lead_packet['ticker']} - {lead_packet['name']}</h4>
                <div>{' '.join(lead_flags)}</div>
                <div style="margin-top:8px; font-size:13px; color:#334155;">{lead_packet.get('why', '')}</div>
                <div style="margin-top:6px; font-size:13px; color:#92400e;">{lead_packet.get('readiness_summary', '')}</div>
                <div class="setup-meta">{' &bull; '.join(plan_bits)}</div>
            </div>
            """)
        else:
            html_parts.append("""
            <div class="hero-setup">
                <h4>No clean new entry</h4>
                <div style="font-size:13px; color:#334155;">
                    The highest-ranked discovery names are currently gated by timing, data quality, or identity risk.
                    Keep the list on watch rather than forcing a fresh entry.
                </div>
            </div>
            """)

        if discovery_meta:
            funnel_bits = []
            if discovery_meta.get("screened_count") is not None:
                funnel_bits.append(f"Screened {int(discovery_meta['screened_count'])}")
            if discovery_meta.get("after_momentum_screen") is not None:
                funnel_bits.append(f"Momentum {int(discovery_meta['after_momentum_screen'])}")
            if discovery_meta.get("after_quick_filter") is not None:
                funnel_bits.append(f"Filtered {int(discovery_meta['after_quick_filter'])}")
            if discovery_meta.get("after_corr_filter") is not None:
                funnel_bits.append(f"Uncorrelated {int(discovery_meta['after_corr_filter'])}")
            if discovery_meta.get("after_quick_rank") is not None:
                funnel_bits.append(f"Ranked {int(discovery_meta['after_quick_rank'])}")
            if discovery_meta.get("fully_scored") is not None:
                funnel_bits.append(f"Scored {int(discovery_meta['fully_scored'])}")
            if discovery_meta.get("run_time_seconds") is not None:
                funnel_bits.append(f"Runtime {safe_float(discovery_meta['run_time_seconds']):.0f}s")
            if funnel_bits:
                html_parts.append(f'<div class="meta-strip">{" &bull; ".join(funnel_bits)}</div>')

        html_parts.append('<h4 style="margin:14px 0 8px 0; color:#0f172a;">Actionable Discovery Setups</h4>')
        html_parts.append("""
        <table>
            <tr><th>Ticker</th><th>Status</th><th>Rank</th><th>Prior</th><th>90d Model</th><th>Entry</th><th>Stop</th><th>Target</th><th>R/R</th><th>Pos</th><th>Fit</th><th>Next trigger</th></tr>
        """)
        for packet in discovery_packets[:5]:
            row_class = ' class="top-pick-row"' if packet.get("top_pick") else ''
            status_bits = []
            if packet.get("top_pick"):
                status_bits.append('<span class="pill pill-top-pick">&#9733; TOP PICK</span>')
            status_bits.append(_action_pill(packet["action"]))
            status_bits.append(_stance_pill(packet["entry_stance"]))
            status_bits.append(_confidence_pill(packet["confidence_label"], packet["confidence_tone"]))
            html_parts.append(f"""
            <tr{row_class}>
                <td>
                    <strong>{packet['ticker']}</strong><br>
                    <span style="font-size:11px; color:#6b7280;">{packet['name'][:28]}</span>
                </td>
                <td>{' '.join(status_bits)}</td>
                <td>{safe_float(packet['final_rank']):.3f}</td>
                <td>{f"{safe_float(packet.get('institutional_prior_percentile')):.0%}" if packet.get('institutional_prior_percentile') else '&mdash;'}</td>
                <td>{format_pct(safe_float(packet['expected_return_90d']) * 100)}</td>
                <td>{_format_price(packet['entry_price'], packet['currency'])}</td>
                <td>{_format_price(packet['stop_loss'], packet['currency'])}</td>
                <td>{_format_price(packet['take_profit'], packet['currency'])}</td>
                <td>{f"{safe_float(packet['r_r_ratio']):.1f}x" if packet.get('r_r_ratio') else '&mdash;'}</td>
                <td>{format_pct(safe_float(packet['position_weight']) * 100, plus_sign=False) if packet.get('position_weight') else '&mdash;'}</td>
                <td>{safe_float(packet['portfolio_fit_score']):.2f}</td>
                <td>{packet.get('entry_trigger') or packet.get('key_risk') or '&mdash;'}</td>
            </tr>
            """)
        html_parts.append("</table>")

        lens_counts = {}
        for d in (discovery_candidates or [])[:20]:
            lens = d.get("entry_lens", "momentum")
            lens_counts[lens] = lens_counts.get(lens, 0) + 1
        if lens_counts:
            lens_summary = " &bull; ".join(f"{_lens_pill(l)} {c}" for l, c in sorted(lens_counts.items()))
            html_parts.append(
                f'<div style="margin-top:8px; font-size:11px; color:#6b7280;">Top 20 by lens: {lens_summary}</div>'
            )
        html_parts.append('</div>')

    # SELL / STRONG SELL alerts
    if alerts:
        html_parts.append('<div class="section alert-sell">')
        html_parts.append('<h3 style="margin:0 0 8px 0; color:#991b1b;">Action Required</h3>')
        for a in alerts:
            ticker = a["ticker"]
            score = a.get("aggregate_score", 0)
            action = a.get("final_action", a.get("action", "SELL"))
            base_action = a.get("base_action", action)
            why = a.get("why", "")
            structural_stop = a.get("structural_stop_loss", a.get("stop_loss"))
            trailing_stop = a.get("trailing_exit_stop")
            price = a.get("current_price")
            prior_score = a.get("aggregate_score")
            exit_score = a.get("exit_score")
            exit_penalty = a.get("_exit_penalty")
            posterior_score = a.get("_exit_posterior")

            stop_txt = ""
            if structural_stop and price:
                distance_pct = ((price - structural_stop) / price) * 100 if price > 0 else 0
                stop_txt = f" &bull; Structural stop: {structural_stop:.2f} ({distance_pct:.1f}% away)"
            trailing_txt = ""
            if trailing_stop and price:
                distance_pct = ((price - trailing_stop) / price) * 100 if price > 0 else 0
                trailing_txt = f" &bull; Trailing exit: {trailing_stop:.2f} ({distance_pct:+.1f}% vs price)"
            action_txt = f"Alpha: {base_action} &rarr; Final: {action}" if base_action != action else f"Action: {action}"
            math_txt = ""
            if posterior_score is not None and exit_penalty is not None and exit_score is not None and prior_score is not None:
                math_txt = (
                    f'<div style="font-size:12px; color:#6b7280; margin-top:4px;">'
                    f'Prior {prior_score:+.3f} &bull; Exit score {exit_score:.3f} '
                    f'&bull; Penalty {exit_penalty:+.3f} &bull; Posterior {posterior_score:+.3f}'
                    f'</div>'
                )

            html_parts.append(f"""
            <div style="margin-bottom:10px; padding:8px 0; border-bottom:1px solid #fecaca;">
                <strong>{ticker}</strong> {_action_pill(action)}
                &nbsp; Score: {_score_html(score)}{stop_txt}{trailing_txt}
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">{action_txt}</div>
                {math_txt}
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">{why}</div>
            </div>
            """)
        html_parts.append("</div>")

    # Governance / Asymmetric risk flags (holdings that need attention)
    flagged_holdings = [
        r for r in results
        if r.get("governance_flag") or r.get("asymmetric_risk_flag")
    ]
    if flagged_holdings:
        html_parts.append('<div class="section" style="border-left: 4px solid #f59e0b; background: #fffbeb;">')
        html_parts.append('<h3 style="margin:0 0 8px 0; color:#92400e;">Risk Flags</h3>')
        for r in flagged_holdings:
            ticker = r["ticker"]
            pills = _flag_pills(r)
            reasons = []
            if r.get("governance_reasons"):
                reasons.extend(r["governance_reasons"])
            if r.get("asymmetric_risk_reason"):
                reasons.append(r["asymmetric_risk_reason"])
            reasons_txt = " &bull; ".join(reasons) if reasons else ""
            html_parts.append(f"""
            <div style="margin-bottom:8px; padding:6px 0; border-bottom:1px solid #fde68a;">
                <strong>{ticker}</strong> {pills}
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">{reasons_txt}</div>
            </div>
            """)
        html_parts.append("</div>")

    if exit_signals:
        watch_signals = [
            e for e in exit_signals
            if e.get("final_action", e.get("base_action")) not in ("SELL", "STRONG SELL")
        ]
        if watch_signals:
            html_parts.append('<div class="section">')
            html_parts.append('<h3 style="margin:0 0 8px 0; color:#92400e;">Exit Watchlist</h3>')
            for e in watch_signals[:5]:
                action_txt = f"{e.get('base_action', 'KEEP')} &rarr; {e.get('final_action', e.get('base_action', 'KEEP'))}"
                extra = []
                if e.get("trailing_exit_stop") is not None:
                    extra.append(f"Trailing exit {e['trailing_exit_stop']:.2f}")
                if e.get("structural_stop_loss") is not None:
                    extra.append(f"Structural stop {e['structural_stop_loss']:.2f}")
                if e.get("posterior_score") is not None:
                    extra.append(f"Posterior {e['posterior_score']:+.3f}")
                extra_txt = " &bull; ".join(extra)
                html_parts.append(f"""
                <div style="margin-bottom:10px; padding:8px 0; border-bottom:1px solid #fde68a;">
                    <strong>{e.get('ticker', '')}</strong> {_action_pill(e.get('final_action', 'KEEP'))}
                    &nbsp; {e.get('signal_type', '').replace('_', ' ').title()}
                    <div style="font-size:12px; color:#6b7280; margin-top:4px;">{action_txt}</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:2px;">{e.get('message', '')}</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:2px;">{extra_txt}</div>
                </div>
                """)
            html_parts.append("</div>")

    # Swap recommendations
    if swap_recs:
        html_parts.append('<div class="section alert-swap">')
        html_parts.append('<h3 style="margin:0 0 8px 0; color:#1e40af;">Swap Opportunities</h3>')
        for s in swap_recs:
            cand = s["candidate"]
            delta = s["score_delta"]
            weakest = s["weakest_ticker"]
            w_score = s["weakest_score"]
            cand_action = cand.get("action", "NEUTRAL")
            trade_verb = "Buy" if cand_action in ("BUY", "STRONG BUY") else "Review"

            suggested_pct = f"{cand.get('position_weight', 0):.1%}" if cand.get("position_weight") else "N/A"

            fx_note = ""
            if cand.get("fx_penalty_applied"):
                fx_pct = cand.get("fx_penalty_pct", 0)
                fx_note = f'<div class="fx-note">FX penalty: {fx_pct:.1f}% round-trip (already deducted from score)</div>'

            html_parts.append(f"""
            <div style="margin-bottom:12px; padding:10px 0; border-bottom:1px solid #bfdbfe;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong>Sell {weakest}</strong> ({_score_html(w_score)})
                        &rarr;
                        <strong>{trade_verb} {cand['ticker']}</strong> ({_score_html(cand['aggregate_score'])})
                    </div>
                    <div class="delta">+{delta:.3f}</div>
                </div>
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">
                    {cand.get('name', '')} &bull; {cand.get('sector', '')} &bull; {_action_pill(cand_action)}
                    &bull; Suggested position: {suggested_pct}
                    {f" &bull; Yield {cand['dividend_yield']:.1%}" if cand.get('dividend_yield') else ""}
                    {f" &bull; Balance {_grade_pill(cand.get('balance_sheet_grade'))}" if cand.get('balance_sheet_grade') else ""}
                </div>
                {fx_note}
                {f'<div style="font-size:12px; margin-top:2px;">{_flag_pills(cand)}</div>' if _flag_pills(cand) else ""}
                <div style="font-size:12px; color:#6b7280; margin-top:2px;">{cand.get('why', '')}</div>
            </div>
            """)
        html_parts.append("</div>")

    # Holdings summary table
    html_parts.append('<div class="section">')
    html_parts.append('<h3 style="margin:0 0 8px 0;">All Holdings</h3>')
    html_parts.append("""
    <table>
        <tr><th>Ticker</th><th>Action</th><th>Score</th><th>Price</th><th>Daily %</th><th>Yield</th><th>B/S</th><th>Flags</th></tr>
    """)
    sorted_results = sorted(results, key=lambda r: r.get("aggregate_score", 0))
    for r in sorted_results:
        final_action = r.get("final_action", r.get("action", "KEEP"))
        base_action = r.get("base_action", final_action)
        action_html = _action_pill(final_action)
        if base_action != final_action:
            action_html += (
                f'<div style="font-size:11px; color:#6b7280; margin-top:2px;">'
                f'Alpha: {base_action}</div>'
            )
        daily = r.get("daily_change_pct")
        daily_str = f"{daily:+.1f}%" if daily is not None else "N/A"
        daily_cls = "score-pos" if daily and daily >= 0 else "score-neg"
        price = r.get("current_price")
        price_str = f"{price:.2f}" if price else "N/A"
        div_y = r.get("dividend_yield")
        div_str = f"{div_y:.1%}" if div_y else "&mdash;"
        bs_grade = r.get("balance_sheet_grade")
        flags = _flag_pills(r)
        html_parts.append(f"""
        <tr>
            <td><strong>{r['ticker']}</strong></td>
            <td>{action_html}</td>
            <td>{_score_html(r.get('aggregate_score', 0))}</td>
            <td>{price_str}</td>
            <td><span class="{daily_cls}">{daily_str}</span></td>
            <td>{div_str}</td>
            <td>{_grade_pill(bs_grade)}</td>
            <td>{flags}</td>
        </tr>
        """)
    html_parts.append("</table></div>")

    # Risk warnings
    warnings = risk_data.get("concentration_warnings", [])
    high_corrs = risk_data.get("high_correlations", [])
    risk_score = risk_data.get("risk_score", 0)

    if warnings or high_corrs:
        html_parts.append('<div class="section">')
        html_parts.append(f'<h3 style="margin:0 0 8px 0;">Risk Warnings (score: {risk_score:.2f})</h3>')
        html_parts.append("<ul>")
        for w in warnings:
            html_parts.append(f"<li>{w}</li>")
        for t1, t2, corr in high_corrs[:5]:
            html_parts.append(f"<li>{t1} / {t2} correlation: {corr:.2f}</li>")
        html_parts.append("</ul></div>")

    # Institutional risk metrics (VaR / ES / β / stress replays)
    var_es = risk_data.get("var_es") or {}
    beta_info = risk_data.get("beta") or {}
    stress_scenarios = risk_data.get("stress_scenarios") or []

    if var_es or beta_info or stress_scenarios:
        html_parts.append('<div class="section">')
        html_parts.append('<h3 style="margin:0 0 8px 0;">Risk Metrics</h3>')

        _v = var_es.get("var_1d")
        _e = var_es.get("es_1d")
        _va = var_es.get("vol_annual")
        _b = beta_info.get("beta")
        _r2 = beta_info.get("r_squared")
        _bench = beta_info.get("benchmark", "SPY")

        def _pct(v):
            return f"{v*100:+.2f}%" if v is not None else "&mdash;"

        html_parts.append("""
        <table>
            <tr><th>95% 1-day VaR</th><th>95% 1-day ES</th><th>Annual Vol</th><th>&beta; vs """ + _bench + """</th></tr>
            <tr>
                <td>""" + _pct(_v) + """</td>
                <td>""" + _pct(_e) + """</td>
                <td>""" + (f"{_va*100:.1f}%" if _va is not None else "&mdash;") + """</td>
                <td>""" + (f"{_b:.2f}" + (f" (R²={_r2:.2f})" if _r2 is not None else "") if _b is not None else "&mdash;") + """</td>
            </tr>
        </table>
        """)

        if stress_scenarios:
            html_parts.append('<h4 style="margin:10px 0 4px 0;">Historical Stress Replays</h4>')
            html_parts.append("""
            <table>
                <tr><th>Scenario</th><th>Window</th><th>Portfolio</th><th>Worst Name</th><th>Coverage</th></tr>
            """)
            for s in stress_scenarios:
                port_r = s.get("portfolio_return")
                port_cls = "score-neg" if (port_r is not None and port_r < 0) else "score-pos"
                worst = s.get("worst_name") or "&mdash;"
                worst_r = s.get("worst_return")
                worst_str = f"{worst} ({worst_r*100:+.1f}%)" if worst_r is not None else worst
                html_parts.append(f"""
                <tr>
                    <td><strong>{s.get('name','')}</strong></td>
                    <td style="font-size:11px;color:#64748b;">{s.get('window','')}</td>
                    <td><span class="{port_cls}">{port_r*100:+.1f}%</span></td>
                    <td>{worst_str}</td>
                    <td>{s.get('coverage',0)*100:.0f}%</td>
                </tr>
                """)
            html_parts.append("</table>")

        html_parts.append("</div>")

    # Rebalance suggestions (only show deltas > 5%)
    big_rebalances = [pw for pw in position_weights if abs(pw.get("rebalance_delta", 0)) > 0.05]
    if big_rebalances:
        html_parts.append('<div class="section">')
        html_parts.append('<h3 style="margin:0 0 8px 0;">Rebalance Suggestions (inv-vol)</h3>')
        html_parts.append("""
        <table>
            <tr><th>Ticker</th><th>Current</th><th>Suggested</th><th>Delta</th></tr>
        """)
        for pw in sorted(big_rebalances, key=lambda x: x["rebalance_delta"]):
            delta = pw["rebalance_delta"]
            delta_cls = "score-pos" if delta > 0 else "score-neg"
            html_parts.append(f"""
            <tr>
                <td><strong>{pw['ticker']}</strong></td>
                <td>{pw['current_weight']:.1%}</td>
                <td>{pw['suggested_weight']:.1%}</td>
                <td><span class="{delta_cls}">{delta:+.1%}</span></td>
            </tr>
            """)
        html_parts.append("</table></div>")

    # Footer
    dry_label = ' <span style="color:#ef4444; font-weight:600;">[DRY RUN]</span>' if dry_run else ""
    html_parts.append(f"""
    <div class="footer">
        ISA Dashboard v4.1 &bull; Engine ran at {now_str}{dry_label}<br>
        This is an automated analysis. Always verify before trading.
    </div>
    """)

    html_parts.append("</div></body></html>")
    html_body = "\n".join(html_parts)

    return subject, html_body
