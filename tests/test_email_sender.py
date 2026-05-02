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

        self.assertIn("Discovery: NVDA ready", subject)
        self.assertIn("Actionable Discovery Setups", html)
        self.assertIn("Momentum leadership with clean forward upside.", html)
        self.assertIn("Portfolio:", html)
        self.assertIn("Rank 1.437", html)
        self.assertIn("TOP PICK", html)


if __name__ == "__main__":
    unittest.main()
