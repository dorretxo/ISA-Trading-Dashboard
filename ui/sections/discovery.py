"""Global discovery engine section rendering."""

from __future__ import annotations

import html as _html
import logging

import pandas as pd
import streamlit as st

import config
from ui.components import (
    candidate_evidence_tags,
    candidate_risk_tags,
    candidate_thesis,
    discovery_confidence,
    lens_sorted_candidates,
    render_html_chips,
    render_pillar_bars,
)
from utils.cache_loader import format_freshness
from utils.safe_numeric import safe_float, format_currency, format_pct


def _restore_cached_discovery(dash) -> None:
    if "discovery_results" in st.session_state:
        return

    try:
        cached_disc = dash.cached_discovery
        cached_disc_meta = getattr(dash, "cached_discovery_meta", {}) or {}
        last_disc_run = dash.discovery_timestamp
        if not cached_disc:
            return

        from engine.discovery import DiscoveryResult, ScoredCandidate

        restored = []
        for c in cached_disc:
            aggregate_score = c.get("aggregate_score") or 0
            technical_score = c.get("technical_score") or 0
            fundamental_score = c.get("fundamental_score") or 0
            sentiment_score = c.get("sentiment_score") or 0
            forecast_score = c.get("forecast_score") or 0
            portfolio_fit_score = c.get("portfolio_fit_score") or 0
            momentum_score = c.get("momentum_score") or 0
            final_rank = c.get("final_rank")
            if final_rank is None:
                final_rank = aggregate_score
            action = c.get("action") or "INSUFFICIENT DATA"

            pillars_all_zero = (
                abs(technical_score)
                + abs(fundamental_score)
                + abs(sentiment_score)
                + abs(forecast_score)
            ) < 0.001
            if pillars_all_zero:
                action = "INSUFFICIENT DATA"
                final_rank = (final_rank or 0) * 0.30

            restored.append(
                ScoredCandidate(
                    ticker=c.get("ticker") or "",
                    name=c.get("name") or c.get("ticker", ""),
                    exchange=c.get("exchange") or "",
                    country=c.get("country") or "",
                    sector=c.get("sector") or "",
                    industry=c.get("industry") or "",
                    market_cap=c.get("market_cap") or 0,
                    currency=c.get("currency") or "USD",
                    aggregate_score=aggregate_score,
                    technical_score=technical_score,
                    fundamental_score=fundamental_score,
                    sentiment_score=sentiment_score,
                    forecast_score=forecast_score,
                    action=action,
                    why=c.get("why") or "",
                    fx_penalty_applied=c.get("fx_penalty_applied") or False,
                    fx_penalty_pct=c.get("fx_penalty_pct") or 0,
                    max_correlation=c.get("max_correlation") or 0,
                    correlated_with=c.get("correlated_with") or "",
                    sector_weight_if_added=c.get("sector_weight_if_added") or 0,
                    portfolio_fit_score=portfolio_fit_score,
                    momentum_score=momentum_score,
                    return_90d=c.get("return_90d") or 0,
                    return_30d=c.get("return_30d") or 0,
                    volume_ratio=c.get("volume_ratio") or 1.0,
                    expected_return_90d=c.get("expected_return_90d") or 0,
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
                    final_rank=final_rank,
                )
            )

        st.session_state["discovery_results"] = DiscoveryResult(
            candidates=restored,
            screened_count=cached_disc_meta.get("screened_count", 0),
            after_momentum_screen=cached_disc_meta.get("after_momentum_screen", 0),
            after_quick_filter=cached_disc_meta.get("after_quick_filter", 0),
            after_corr_filter=cached_disc_meta.get("after_corr_filter", 0),
            after_quick_rank=cached_disc_meta.get("after_quick_rank", 0),
            fully_scored=cached_disc_meta.get("fully_scored", len(restored)),
            run_time_seconds=cached_disc_meta.get("run_time_seconds", 0.0),
            fx_penalties_applied=cached_disc_meta.get("fx_penalties_applied", 0),
        )
        st.session_state["discovery_cached_from"] = last_disc_run
    except Exception:
        return


def render_discovery_section(dash, holdings, risk_data) -> None:
    st.divider()
    st.markdown("### ?? Global Discovery Engine")
    st.caption(
        "A portfolio-aware idea engine for surfacing the best new opportunities, strongest diversifiers, "
        "and cleanest momentum setups from the daily discovery universe."
    )

    _restore_cached_discovery(dash)

    disc_col1, disc_col2 = st.columns([1, 3])
    with disc_col1:
        run_discovery_clicked = st.button("?? Re-run Screener", type="primary", use_container_width=True)
    with disc_col2:
        cached_ts = st.session_state.get("discovery_cached_from")
        if cached_ts and "discovery_results" in st.session_state:
            st.info(f"Showing cached results · {format_freshness(cached_ts)} · Click Re-run to refresh")
        else:
            st.info(
                f"Screens {len(config.DISCOVERY_EXCHANGES)} US exchanges + global universe · "
                f"Market cap = £{config.DISCOVERY_MIN_MCAP / 1e6:.0f}M (no upper cap) · "
                f"Top {config.DISCOVERY_TOP_N_FULL_SCORE} fully scored · "
                f"~60-90 min runtime"
            )

    if run_discovery_clicked:
        from engine.discovery import run_discovery

        disc_progress = st.progress(0, text="Starting discovery...")

        def disc_progress_cb(message, current, total):
            pct = min(current / max(total, 1), 1.0) if total > 0 else 0
            disc_progress.progress(pct, text=message)

        disc_result = run_discovery(
            holdings=holdings,
            risk_data=risk_data,
            progress_callback=disc_progress_cb,
        )
        disc_progress.progress(1.0, text="Discovery complete!")
        st.session_state["discovery_results"] = disc_result
        st.session_state.pop("discovery_cached_from", None)

    if "discovery_results" not in st.session_state:
        return

    disc = st.session_state["discovery_results"]

    if disc.error:
        st.warning(f"Discovery issue: {disc.error}")

    if disc.candidates:
        momentum_count = getattr(disc, "after_momentum_screen", "?")
        st.markdown(
            f"**Funnel:** {disc.screened_count} screened ? "
            f"{momentum_count} momentum ? "
            f"{disc.after_quick_filter} filtered ? "
            f"{disc.after_corr_filter} uncorrelated ? "
            f"{disc.after_quick_rank} ranked ? "
            f"{disc.fully_scored} scored · "
            f"?? {disc.run_time_seconds:.0f}s"
        )

        best_idea = max(disc.candidates, key=lambda c: safe_float(getattr(c, "final_rank", 0)))
        best_fit = max(
            disc.candidates,
            key=lambda c: (
                safe_float(getattr(c, "portfolio_fit_score", 0)),
                safe_float(getattr(c, "final_rank", 0)),
            ),
        )
        best_momentum = max(
            disc.candidates,
            key=lambda c: (
                safe_float(getattr(c, "momentum_score", 0)),
                safe_float(getattr(c, "return_90d", 0)),
            ),
        )
        watch_candidate = max(
            disc.candidates[: min(10, len(disc.candidates))],
            key=lambda c: (
                len(candidate_risk_tags(c)),
                safe_float(getattr(c, "parabolic_penalty", 0)),
                safe_float(getattr(c, "fx_penalty_pct", 0)),
            ),
        )

        st.markdown("#### Discovery Command Center")
        cc1, cc2, cc3, cc4 = st.columns(4)
        command_cards = [
            (
                cc1,
                "best",
                "Best New Opportunity",
                best_idea,
                f"Final rank {safe_float(getattr(best_idea, 'final_rank', 0)):.3f} · {getattr(best_idea, 'action', 'NEUTRAL')}",
            ),
            (
                cc2,
                "fit",
                "Best Diversifier",
                best_fit,
                f"Portfolio fit {safe_float(getattr(best_fit, 'portfolio_fit_score', 0)):.2f} · Corr {safe_float(getattr(best_fit, 'max_correlation', 0)):.2f}",
            ),
            (
                cc3,
                "momentum",
                "Momentum Leader",
                best_momentum,
                f"Momentum {safe_float(getattr(best_momentum, 'momentum_score', 0)):.2f} · 90d {format_pct(safe_float(getattr(best_momentum, 'return_90d', 0)) * 100)}",
            ),
            (
                cc4,
                "risk",
                "Biggest Watchout",
                watch_candidate,
                candidate_risk_tags(watch_candidate)[0] if candidate_risk_tags(watch_candidate) else "No major red-flag overlays in the current top tier",
            ),
        ]
        for col, tone, label, cand, meta in command_cards:
            with col:
                st.markdown(
                    f"""
                    <div class="insight-card {tone}">
                        <div class="insight-label">{_html.escape(label)}</div>
                        <div class="insight-title">{_html.escape(getattr(cand, "ticker", ""))}</div>
                        <div class="insight-sub">{_html.escape(candidate_thesis(cand))}</div>
                        <div class="insight-meta">{_html.escape(meta)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### Recommendation Lens")
        lens = st.radio(
            "Recommendation lens",
            ["Best Ideas", "Best Diversifiers", "Momentum Leaders", "Value / Quality"],
            horizontal=True,
            label_visibility="collapsed",
            key="discovery_lens",
        )
        lens_notes = {
            "Best Ideas": "Balanced view of conviction, confidence, momentum, and portfolio fit.",
            "Best Diversifiers": "Highlights names that improve portfolio shape without giving up too much quality.",
            "Momentum Leaders": "Pulls the strongest trend-following setups to the front.",
            "Value / Quality": "Pushes fundamental strength and cleaner business quality higher in the stack.",
        }
        st.markdown(f'<div class="lens-note">{_html.escape(lens_notes[lens])}</div>', unsafe_allow_html=True)

        featured = lens_sorted_candidates(disc.candidates, lens)[:3]
        if featured:
            st.markdown("#### Featured Recommendations")
            feature_cols = st.columns(len(featured))
            country_flags = {"US": "US", "UK": "UK", "GB": "UK", "CA": "CA", "DE": "DE", "FR": "FR", "IT": "IT", "ES": "ES", "NL": "NL", "JP": "JP"}
            for idx, (col, cand) in enumerate(zip(feature_cols, featured), start=1):
                with col:
                    conf_label, conf_tone, _ = discovery_confidence(cand)
                    action = getattr(cand, "action", "NEUTRAL")
                    if action in ("STRONG BUY", "BUY"):
                        action_tone = "buy"
                    elif action == "INSUFFICIENT DATA":
                        action_tone = "data"
                    elif action in ("AVOID", "SELL", "STRONG SELL"):
                        action_tone = "avoid"
                    else:
                        action_tone = "neutral"
                    mcap = safe_float(cand.market_cap)
                    geo = country_flags.get(cand.country, "Global")
                    subtitle = f"{geo} · {cand.exchange} · {cand.sector}" + (f" · {format_currency(mcap / 1e9, 'GBP', decimals=1)}B mcap" if mcap > 0 else "")
                    badge_html = (
                        '<div class="badge-row">'
                        f'<span class="signal-badge {action_tone}">{_html.escape(action)}</span>'
                        f'<span class="confidence-chip {conf_tone}">{_html.escape(conf_label)}</span>'
                        "</div>"
                    )
                    st.markdown(
                        f"""
                        <div class="rec-hero">
                            <div class="rec-rankline">
                                <div class="rec-rank">#{idx} in {_html.escape(lens)}</div>
                                <div class="rec-rank">Final Rank {safe_float(cand.final_rank):.3f}</div>
                            </div>
                            <div class="rec-title">{_html.escape(cand.ticker)}</div>
                            <div class="rec-subtitle">{_html.escape(cand.name)}<br>{_html.escape(subtitle)}</div>
                            {badge_html}
                            <div class="rec-thesis">{_html.escape(candidate_thesis(cand))}</div>
                            {render_html_chips(candidate_evidence_tags(cand))}
                            {render_html_chips([(tag, "risk") for tag in candidate_risk_tags(cand)])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Momentum", f"{safe_float(cand.momentum_score):.2f}")
                    mc2.metric("Fit", f"{safe_float(cand.portfolio_fit_score):.2f}")
                    mc3.metric("90d Model", format_pct(safe_float(getattr(cand, "expected_return_90d", 0)) * 100))
                    st.markdown(
                        render_pillar_bars(
                            safe_float(cand.technical_score),
                            safe_float(cand.fundamental_score),
                            safe_float(cand.sentiment_score),
                            safe_float(cand.forecast_score),
                        ),
                        unsafe_allow_html=True,
                    )
                    with st.expander("Why this ranked here"):
                        st.markdown(f"**Thesis:** {candidate_thesis(cand)}")
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

        with st.expander("?? All Scored Candidates"):
            disc_rows = []
            for c in disc.candidates:
                disc_rows.append(
                    {
                        "Rank": c.final_rank,
                        "Ticker": c.ticker,
                        "Name": c.name,
                        "Exchange": c.exchange,
                        "Sector": c.sector,
                        "Score": c.aggregate_score,
                        "Confidence": discovery_confidence(c)[0],
                        "FX Penalty": f"-{c.fx_penalty_pct:.1f}%" if c.fx_penalty_applied else "—",
                        "Fit": c.portfolio_fit_score,
                        "Max Corr": c.max_correlation,
                        "Action": c.action,
                        "Currency": c.currency,
                        "Cap": getattr(c, "cap_tier", "—"),
                        "Parabolic": f"-{c.parabolic_penalty:.2f}" if getattr(c, "is_parabolic", False) else "—",
                        "Earnings": f"{c.earnings_days}d" if getattr(c, "earnings_days", None) else "—",
                    }
                )
            if disc_rows:
                st.dataframe(pd.DataFrame(disc_rows), hide_index=True, use_container_width=True)

        if disc.rejections:
            with st.expander(f"?? Rejected Candidates ({len(disc.rejections)})"):
                rej_rows = [
                    {"Ticker": r.ticker, "Name": r.name, "Exchange": r.exchange, "Stage": r.stage, "Reason": r.reason}
                    for r in disc.rejections[:100]
                ]
                st.dataframe(pd.DataFrame(rej_rows), hide_index=True, use_container_width=True)
    elif not disc.error:
        st.info("No candidates found meeting the criteria. Try adjusting discovery parameters in config.py.")

    try:
        from engine.discovery_backtest import (
            get_action_calibration,
            get_pending_picks_count,
            get_pick_performance,
            get_pillar_stats,
            get_regime_stats,
            get_stop_target_stats,
        )

        perf = get_pick_performance(limit=50)
        pstats = get_pillar_stats()
        pending = get_pending_picks_count()
        acal = get_action_calibration()
        rstats = get_regime_stats()
        st_stats = get_stop_target_stats()

        if perf or pending:
            with st.expander(f"?? Signal Track Record ({len(perf)} evaluated, {pending} pending)"):
                if pstats:
                    ps_rows = [{"Pillar": s["pillar"].title(), "IC": f"{s['information_coefficient']:+.3f}", "Hit Rate": f"{s['hit_rate']:.0%}", "Avg Return (High)": f"{s['avg_return_high']:+.1f}%", "Avg Return (Low)": f"{s['avg_return_low']:+.1f}%", "Samples": s["sample_size"]} for s in pstats]
                    st.markdown("**Pillar Effectiveness (which signals predict 90-day returns)**")
                    st.dataframe(pd.DataFrame(ps_rows), hide_index=True, use_container_width=True)

                if acal:
                    ac_rows = [{"Action": a["action"], "Avg 90d Return": f"{a['avg_return_90d']:+.1f}%", "Accuracy": f"{a['hit_rate']:.0%}", "Samples": a["sample_size"]} for a in acal]
                    st.markdown("**Action Calibration (are action labels accurate?)**")
                    st.dataframe(pd.DataFrame(ac_rows), hide_index=True, use_container_width=True)

                if rstats:
                    re_rows = [{"Regime": r["regime"], "Avg 90d Return": f"{r['avg_return_90d']:+.1f}%", "Best Pillar": (r["best_pillar"] or "—").title(), "Samples": r["sample_size"]} for r in rstats]
                    st.markdown("**Regime Effectiveness (which regime works best?)**")
                    st.dataframe(pd.DataFrame(re_rows), hide_index=True, use_container_width=True)

                st_col, fc_col = st.columns(2)
                with st_col:
                    if st_stats and st_stats.get("with_stops"):
                        total_w = st_stats["with_stops"]
                        s_hit = st_stats.get("stops_hit") or 0
                        t_hit = st_stats.get("targets_hit") or 0
                        st.markdown("**Stop-Loss / Take-Profit Hits**")
                        st.caption(f"Stops hit: {s_hit}/{total_w} ({s_hit/total_w:.0%})" + (f" — avg day {st_stats['avg_stop_day']:.0f}" if st_stats.get("avg_stop_day") else ""))
                        st.caption(f"Targets hit: {t_hit}/{total_w} ({t_hit/total_w:.0%})" + (f" — avg day {st_stats['avg_target_day']:.0f}" if st_stats.get("avg_target_day") else ""))
                with fc_col:
                    if st_stats and st_stats.get("avg_forecast_err_5d") is not None:
                        st.markdown("**Forecast Accuracy**")
                        st.caption(f"5-day avg error: {st_stats['avg_forecast_err_5d']:.1f}%")
                        if st_stats.get("avg_forecast_err_63d") is not None:
                            st.caption(f"63-day avg error: {st_stats['avg_forecast_err_63d']:.1f}%")

                if perf:
                    perf_rows = [{"Date": p["run_date"][:10], "Ticker": p["ticker"], "Source": p.get("source", "—"), "Action": p.get("action", "—"), "Score": f"{p['aggregate_score']:.3f}", "30d": f"{p['return_30d']:+.1f}%" if p.get("return_30d") is not None else "—", "60d": f"{p['return_60d']:+.1f}%" if p.get("return_60d") is not None else "—", "90d": f"{p['return_90d']:+.1f}%" if p.get("return_90d") is not None else "—", "Beat SPY": "Yes" if p.get("beat_market") else "No", "Action OK": "Yes" if p.get("action_correct") else "No"} for p in perf]
                    st.markdown("**Signal Performance — Multi-Horizon Returns**")
                    st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)
                    returns = [p["return_90d"] for p in perf if p.get("return_90d") is not None]
                    if returns:
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        bc1.metric("Avg Return", f"{sum(returns)/len(returns):+.1f}%")
                        bc2.metric("Win Rate", f"{sum(1 for r in returns if r > 0)/len(returns):.0%}")
                        bc3.metric("Best Pick", f"{max(returns):+.1f}%")
                        bc4.metric("Worst Pick", f"{min(returns):+.1f}%")
                    beat = [p for p in perf if p.get("beat_market") is not None]
                    if beat:
                        beat_rate = sum(1 for p in beat if p["beat_market"]) / len(beat)
                        st.caption(f"Beat SPY: {beat_rate:.0%} of signals | Action accuracy: {sum(1 for p in perf if p.get('action_correct'))/len(perf):.0%}")
    except Exception:
        pass

    try:
        from engine.evaluation_harness import compute_scorecard

        sc = compute_scorecard(source="all", min_signals=5)
        if sc and sc.evaluated_signals >= 5 and sc.sharpe_ratio is not None:
            with st.expander(f"?? **Performance Scorecard** ({sc.evaluated_signals} signals evaluated)"):
                ev1, ev2, ev3, ev4 = st.columns(4)
                ev1.metric("Sharpe Ratio", f"{sc.sharpe_ratio:.2f}")
                ev2.metric("Sortino Ratio", f"{sc.sortino_ratio:.2f}" if sc.sortino_ratio else "—")
                ev3.metric("Max Drawdown", f"{sc.max_drawdown:+.1f}%" if sc.max_drawdown else "—")
                ev4.metric("Calmar Ratio", f"{sc.calmar_ratio:.2f}" if sc.calmar_ratio else "—")

                ev5, ev6, ev7, ev8 = st.columns(4)
                ev5.metric("Hit Rate (90d)", f"{sc.overall_hit_rate:.0%}" if sc.overall_hit_rate else "—")
                ev6.metric("Action Accuracy", f"{sc.action_accuracy:.0%}" if sc.action_accuracy else "—")
                ev7.metric("Beat SPY Rate", f"{sc.beat_benchmark_rate:.0%}" if sc.beat_benchmark_rate else "—")
                ev8.metric("IC Stability", f"{sc.ic_stability:.4f}" if sc.ic_stability else "—")

                if sc.horizons:
                    h_rows = pd.DataFrame([{"Horizon": h.horizon, "Avg Return": f"{h.avg_return:+.1f}%", "Median": f"{h.median_return:+.1f}%", "Std Dev": f"{h.std_return:.1f}%", "Hit Rate": f"{h.hit_rate:.0%}", "Alpha vs SPY": f"{h.alpha:+.1f}%" if h.horizon == "90d" else "—", "Best": f"{h.best:+.1f}%", "Worst": f"{h.worst:+.1f}%", "N": h.sample_size} for h in sc.horizons])
                    st.markdown("**Returns by Horizon**")
                    st.dataframe(h_rows, hide_index=True, use_container_width=True)

                if sc.regimes:
                    r_rows = pd.DataFrame([{"Regime": r.regime, "Avg 90d Return": f"{r.avg_return_90d:+.1f}%", "Hit Rate": f"{r.hit_rate:.0%}", "Best Pillar": (r.best_pillar or "—").title(), "N": r.sample_size} for r in sc.regimes])
                    st.markdown("**Performance by Market Regime**")
                    st.dataframe(r_rows, hide_index=True, use_container_width=True)

                st_c, fc_c = st.columns(2)
                with st_c:
                    if sc.stop_hit_rate is not None:
                        st.markdown("**Stop/Target Effectiveness**")
                        st.caption(f"Stop-loss hit rate: {sc.stop_hit_rate:.0%}" + (f" (avg day {sc.avg_stop_day:.0f})" if sc.avg_stop_day else ""))
                        st.caption(f"Take-profit hit rate: {sc.target_hit_rate:.0%}" + (f" (avg day {sc.avg_target_day:.0f})" if sc.avg_target_day else ""))
                with fc_c:
                    if sc.avg_forecast_error_5d is not None:
                        st.markdown("**Forecast Accuracy**")
                        st.caption(f"5-day avg error: {sc.avg_forecast_error_5d:.1f}%")
                        if sc.avg_forecast_error_63d is not None:
                            st.caption(f"63-day avg error: {sc.avg_forecast_error_63d:.1f}%")
    except Exception as sc_err:
        logging.getLogger(__name__).warning("Performance scorecard failed: %s", sc_err)

    try:
        from engine.discovery_eval import get_discovery_scorecard

        disc_sc = get_discovery_scorecard()
        if disc_sc and disc_sc.get("total_evaluated", 0) >= 5:
            with st.expander(f"?? **Discovery Quality** ({disc_sc['total_evaluated']} evaluated picks)"):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Top-10 Hit Rate", f"{safe_float(disc_sc.get('top10_hit_rate_90d')):.0%}")
                dc2.metric("Top-10 Avg Return", format_pct(disc_sc.get("top10_avg_return_90d")))
                dc3.metric("Excess vs SPY", format_pct(disc_sc.get("excess_vs_spy_90d")))
                dc4.metric("Ranking Stability", f"{safe_float(disc_sc.get('ranking_stability')):.0%}")

                dc5, dc6 = st.columns(2)
                dc5.metric("Top-30 Hit Rate", f"{safe_float(disc_sc.get('top30_hit_rate_90d')):.0%}")
                dc6.metric("Swap Success Rate", f"{safe_float(disc_sc.get('swap_success_rate')):.0%}")

                if disc_sc.get("summary"):
                    st.caption(disc_sc["summary"])
    except Exception as disc_sc_err:
        logging.getLogger(__name__).warning("Discovery scorecard failed: %s", disc_sc_err)
