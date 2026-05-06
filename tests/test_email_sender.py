import unittest

from utils.email_sender import build_alert_email


class EmailSenderTests(unittest.TestCase):
    def test_discovery_only_email_uses_actionable_subject_and_setup_table(self):
        results = [
            {
                "ticker": "MSFT",
                "action": "KEEP",
                "aggregate_score": 0.25,
            }
        ]
        risk_data = {"risk_score": 0.22}
        position_weights = []
        vix_regime = {"regime_label": "NEUTRAL", "vix_level": 18, "vix_percentile": 45}
        discovery_candidates = [
            {
                "ticker": "NVDA",
                "name": "NVIDIA Corp",
                "action": "STRONG BUY",
                "entry_stance": "Ready",
                "sentiment_score": 0.7,
                "confidence_discount": 1.0,
                "final_rank": 1.437,
                "aggregate_score": 0.82,
                "expected_return_90d": 0.14,
                "entry_price": 910.0,
                "stop_loss": 860.0,
                "take_profit": 1035.0,
                "r_r_ratio": 2.5,
                "position_weight": 0.07,
                "portfolio_fit_score": 0.78,
                "max_correlation": 0.34,
                "currency": "USD",
                "why": "Momentum leadership with clean forward upside.",
            }
        ]

        subject, html = build_alert_email(
            results=results,
            risk_data=risk_data,
            position_weights=position_weights,
            vix_regime=vix_regime,
            alerts=[],
            swap_recs=[],
            discovery_candidates=discovery_candidates,
            discovery_meta={"screened_count": 1000, "fully_scored": 120, "run_time_seconds": 88},
            artifact_timestamps={
                "portfolio": "2026-04-22T07:00:00",
                "discovery": "2026-04-22T07:05:00",
                "optimizer": "2026-04-22T07:00:00",
                "exit": "2026-04-22T07:00:00",
            },
            discovery_ran=True,
            dry_run=True,
        )

        self.assertIn("Discovery: NVDA STRONG BUY ready", subject)
        self.assertIn("Actionable Discovery Setups", html)
        self.assertIn("Momentum leadership with clean forward upside.", html)
        self.assertIn("Portfolio:", html)
        self.assertIn("Rank 1.437", html)
        self.assertIn("TOP PICK", html)

    def test_discovery_strong_buy_stays_in_subject_when_portfolio_alerts_exist(self):
        results = [{"ticker": "MSFT", "action": "SELL", "aggregate_score": -0.4}]
        risk_data = {"risk_score": 0.22}
        vix_regime = {"regime_label": "NEUTRAL", "vix_level": 18, "vix_percentile": 45}
        alerts = [{"ticker": "MSFT", "action": "SELL", "aggregate_score": -0.4}]
        discovery_candidates = [
            {
                "ticker": "KLR.L",
                "name": "Keller Group",
                "action": "STRONG BUY",
                "entry_stance": "Ready",
                "sentiment_score": 0.6,
                "confidence_discount": 1.0,
                "final_rank": 0.355,
                "aggregate_score": 0.415,
                "expected_return_90d": 0.08,
                "entry_price": 12.0,
                "stop_loss": 10.8,
                "take_profit": 15.0,
                "r_r_ratio": 2.5,
                "position_weight": 0.04,
                "portfolio_fit_score": 0.72,
                "max_correlation": 0.2,
                "currency": "GBP",
                "ready_contract_status": "PASS",
                "strong_buy_eligible": True,
            }
        ]

        subject, html = build_alert_email(
            results=results,
            risk_data=risk_data,
            position_weights=[],
            vix_regime=vix_regime,
            alerts=alerts,
            swap_recs=[],
            discovery_candidates=discovery_candidates,
            discovery_ran=True,
        )

        self.assertIn("SELL MSFT", subject)
        self.assertIn("Discovery: KLR.L STRONG BUY ready", subject)
        self.assertIn("KLR.L", html)


    def test_subject_names_up_to_three_strong_buys(self):
        """Multi-emission days should name the top 3 STRONG BUY tickers, not just one."""
        risk_data = {"risk_score": 0.22}
        vix_regime = {"regime_label": "NEUTRAL", "vix_level": 18, "vix_percentile": 45}

        def _sb(t, fr):
            return {
                "ticker": t, "name": t, "action": "STRONG BUY",
                "entry_stance": "Ready", "sentiment_score": 0.6,
                "confidence_discount": 1.0, "final_rank": fr,
                "aggregate_score": 0.55, "expected_return_90d": 0.10,
                "entry_price": 100.0, "stop_loss": 90.0, "take_profit": 120.0,
                "r_r_ratio": 2.5, "position_weight": 0.04,
                "portfolio_fit_score": 0.7, "max_correlation": 0.2,
                "currency": "USD", "ready_contract_status": "PASS",
                "strong_buy_eligible": True,
            }

        # Four STRONG BUY candidates ordered by sort key.
        discovery_candidates = [
            _sb("AAA", 1.50), _sb("BBB", 1.20), _sb("CCC", 1.00), _sb("DDD", 0.95),
        ]

        subject, _html = build_alert_email(
            results=[],
            risk_data=risk_data,
            position_weights=[],
            vix_regime=vix_regime,
            alerts=[],
            swap_recs=[],
            discovery_candidates=discovery_candidates,
            discovery_ran=True,
        )

        # Top 3 named, with "+1" overflow for the 4th.
        self.assertIn("AAA", subject)
        self.assertIn("BBB", subject)
        self.assertIn("CCC", subject)
        self.assertIn("+1", subject)


if __name__ == "__main__":
    unittest.main()
