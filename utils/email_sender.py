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
.footer { background: #f9fafb; padding: 12px 20px; font-size: 11px; color: #6b7280;
          border-radius: 0 0 8px 8px; border: 1px solid #e5e7eb; border-top: none; }
.delta { font-size: 18px; font-weight: 700; color: #2563eb; }
.fx-note { font-size: 12px; color: #92400e; background: #fffbeb; padding: 4px 8px; border-radius: 4px; }
"""


def _action_pill(action: str) -> str:
    """Render an action label as a coloured pill."""
    cls = {
        "STRONG SELL": "pill-strong-sell",
        "SELL": "pill-sell",
        "KEEP": "pill-keep",
        "BUY": "pill-buy",
        "STRONG BUY": "pill-strong-buy",
    }.get(action, "pill-keep")
    return f'<span class="pill {cls}">{action}</span>'


def _score_html(score: float) -> str:
    """Format score with colour."""
    cls = "score-pos" if score >= 0 else "score-neg"
    return f'<span class="{cls}">{score:+.3f}</span>'


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

    Returns:
        (subject, html_body)
    """
    # --- Subject line ---
    parts = []
    if alerts:
        alert_tickers = ", ".join(f"{a['action']} {a['ticker']}" for a in alerts[:3])
        parts.append(alert_tickers)
    if swap_recs:
        top = swap_recs[0]
        parts.append(f"Swap: add {top['candidate']['ticker']} +{top['score_delta']:.2f} delta")

    regime_label = vix_regime.get("regime_label", "NEUTRAL")
    vix_level = vix_regime.get("vix_level", 0)
    subject = f"[ISA Alert] {' | '.join(parts)} | VIX: {regime_label}"
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

    # SELL / STRONG SELL alerts
    if alerts:
        html_parts.append('<div class="section alert-sell">')
        html_parts.append('<h3 style="margin:0 0 8px 0; color:#991b1b;">Action Required</h3>')
        for a in alerts:
            ticker = a["ticker"]
            score = a.get("aggregate_score", 0)
            action = a.get("action", "SELL")
            why = a.get("why", "")
            stop = a.get("stop_loss")
            price = a.get("current_price")

            stop_txt = ""
            if stop and price:
                distance_pct = ((price - stop) / price) * 100 if price > 0 else 0
                stop_txt = f" &bull; Stop-loss: {stop:.2f} ({distance_pct:.1f}% away)"

            html_parts.append(f"""
            <div style="margin-bottom:10px; padding:8px 0; border-bottom:1px solid #fecaca;">
                <strong>{ticker}</strong> {_action_pill(action)}
                &nbsp; Score: {_score_html(score)}{stop_txt}
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">{why}</div>
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

            # Find suggested position weight for the candidate
            suggested_pct = "N/A"
            for pw in position_weights:
                if pw["ticker"] == weakest:
                    suggested_pct = f"{pw['suggested_weight']:.0%}"
                    break

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
                        <strong>Buy {cand['ticker']}</strong> ({_score_html(cand['aggregate_score'])})
                    </div>
                    <div class="delta">+{delta:.3f}</div>
                </div>
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">
                    {cand.get('name', '')} &bull; {cand.get('sector', '')} &bull; {cand.get('action', '')}
                    &bull; Suggested position: {suggested_pct}
                </div>
                {fx_note}
                <div style="font-size:12px; color:#6b7280; margin-top:2px;">{cand.get('why', '')}</div>
            </div>
            """)
        html_parts.append("</div>")

    # Holdings summary table
    html_parts.append('<div class="section">')
    html_parts.append('<h3 style="margin:0 0 8px 0;">All Holdings</h3>')
    html_parts.append("""
    <table>
        <tr><th>Ticker</th><th>Action</th><th>Score</th><th>Price</th><th>Daily %</th></tr>
    """)
    sorted_results = sorted(results, key=lambda r: r.get("aggregate_score", 0))
    for r in sorted_results:
        daily = r.get("daily_change_pct")
        daily_str = f"{daily:+.1f}%" if daily is not None else "N/A"
        daily_cls = "score-pos" if daily and daily >= 0 else "score-neg"
        price = r.get("current_price")
        price_str = f"{price:.2f}" if price else "N/A"
        html_parts.append(f"""
        <tr>
            <td><strong>{r['ticker']}</strong></td>
            <td>{_action_pill(r.get('action', 'KEEP'))}</td>
            <td>{_score_html(r.get('aggregate_score', 0))}</td>
            <td>{price_str}</td>
            <td><span class="{daily_cls}">{daily_str}</span></td>
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
        ISA Dashboard v4.0 &bull; Engine ran at {now_str}{dry_label}<br>
        This is an automated analysis. Always verify before trading.
    </div>
    """)

    html_parts.append("</div></body></html>")
    html_body = "\n".join(html_parts)

    return subject, html_body
