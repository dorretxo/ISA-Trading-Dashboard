"""Exit Intelligence section rendering."""

from __future__ import annotations

import html as _html

import streamlit as st

from ui.components import exit_card_tags, render_html_chips
from utils.cache_loader import format_freshness


def render_exit_intelligence(dash, results, holdings) -> None:
    exit_list = dash.cached_exit_signals
    if exit_list is None and not dash.from_cache:
        try:
            from engine.exit_engine import assess_exits

            exit_objs = assess_exits(results, holdings)
            _result_by_ticker = {r.get("ticker"): r for r in (results or [])}
            exit_list = [
                {
                    "ticker": e.ticker,
                    "name": e.name,
                    "signal_type": e.signal_type,
                    "severity": e.severity,
                    "message": e.message,
                    "current_score": e.current_score,
                    "current_price": e.current_price,
                    "smoothing_reason": _result_by_ticker.get(e.ticker, {}).get("smoothing_reason"),
                    "smoothed_action": _result_by_ticker.get(e.ticker, {}).get("smoothed_action"),
                    "cusum_override": _result_by_ticker.get(e.ticker, {}).get("_cusum_override", False),
                }
                for e in exit_objs
            ]
        except Exception:
            exit_list = []

    if not exit_list:
        return

    with st.expander(
        f"🚪 **Exit Intelligence** ({len(exit_list)} signals)"
        + (f" · {format_freshness(dash.exit_signals_timestamp)}" if dash.exit_signals_timestamp else ""),
        expanded=any(e.get("severity") == "urgent" for e in exit_list),
    ):
        st.markdown("#### Exit Command Center")
        urgent_count = sum(1 for e in exit_list if e.get("severity") == "urgent")
        action_count = sum(1 for e in exit_list if e.get("severity") == "action_needed")
        warning_count = sum(1 for e in exit_list if e.get("severity") == "warning")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Urgent", urgent_count)
        ec2.metric("Action Needed", action_count)
        ec3.metric("Watchlist", warning_count)

        sorted_exit = sorted(
            exit_list,
            key=lambda e: (
                0 if e.get("severity") == "urgent" else 1 if e.get("severity") == "action_needed" else 2,
                e.get("ticker", ""),
            ),
        )
        for card_exit in sorted_exit:
            sev = card_exit.get("severity", "warning")
            card_cls = "urgent" if sev == "urgent" else "action" if sev == "action_needed" else "warning"
            sev_label = "Urgent" if sev == "urgent" else "Action Needed" if sev == "action_needed" else "Watch"

            # Smoother diagnostic footer — auditability for why the action stuck
            # (Constantinides 1986 / Wald 1947 / Page 1954 CUSUM).
            sm_reason = card_exit.get("smoothing_reason") or ""
            sm_action = card_exit.get("smoothed_action") or ""
            sm_footer = ""
            if card_exit.get("cusum_override") or sm_reason == "cusum_override":
                sm_footer = (
                    '<div style="margin-top:6px;padding:4px 8px;border-left:3px solid #ef4444;'
                    'background:rgba(239,68,68,0.10);font-size:11px;color:#fca5a5;">'
                    "CUSUM structural-break override — bypassed dead-zone."
                    "</div>"
                )
            elif sm_reason == "persistence_satisfied":
                sm_footer = (
                    '<div style="margin-top:6px;padding:4px 8px;border-left:3px solid #f59e0b;'
                    'background:rgba(245,158,11,0.10);font-size:11px;color:#fcd34d;">'
                    f"Persistence confirmed — smoothed action {_html.escape(sm_action)}."
                    "</div>"
                )
            elif sm_reason in ("band_held_keep", "persistence_short"):
                sm_footer = (
                    '<div style="margin-top:6px;padding:4px 8px;border-left:3px solid #38bdf8;'
                    'background:rgba(56,189,248,0.10);font-size:11px;color:#7dd3fc;">'
                    "Noise filtered — smoother held prior action."
                    "</div>"
                )

            st.markdown(
                f"""
                <div class="exit-card {card_cls}">
                    <div class="exit-topline">
                        <div>
                            <div class="exit-title">{_html.escape(card_exit.get("ticker", ""))}</div>
                            <div class="exit-subtitle">{_html.escape(card_exit.get("name", ""))} · {_html.escape(card_exit.get("signal_type", "").replace("_", " ").title())}</div>
                        </div>
                        <span class="severity-pill {card_cls}">{_html.escape(sev_label)}</span>
                    </div>
                    <div class="exit-message">{_html.escape(card_exit.get("message", ""))}</div>
                    {render_html_chips(exit_card_tags(card_exit))}
                    {sm_footer}
                </div>
                """,
                unsafe_allow_html=True,
            )
