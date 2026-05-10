import unittest

from engine.factor_momentum import compute_factor_returns
from engine.factors import (
    adjust_alpha_for_confidence,
    compute_downside_vol,
    compute_factor_scores_from_result,
    compute_fundamental_quality_metrics,
    compute_max_drawdown,
)
from engine.sentiment import extract_qa_section


class QuantFactorTests(unittest.TestCase):
    def test_fundamental_quality_metrics_follow_prompt_components(self):
        info = {
            "grossProfits": 420.0,
            "totalAssets": 1000.0,
            "returnOnEquity": 0.24,
            "freeCashflow": 95.0,
        }
        annual_income = [
            {"date": "2020-12-31", "eps": 1.00},
            {"date": "2021-12-31", "eps": 1.18},
            {"date": "2022-12-31", "eps": 1.33},
            {"date": "2023-12-31", "eps": 1.55},
            {"date": "2024-12-31", "eps": 1.74},
        ]

        metrics = compute_fundamental_quality_metrics(info, annual_income)

        self.assertAlmostEqual(metrics["gross_profitability"], 0.42)
        self.assertAlmostEqual(metrics["fcf_to_assets"], 0.095)
        self.assertIsNotNone(metrics["eps_growth_variance_5y"])
        self.assertGreater(metrics["quality_score"], 0.0)

    def test_fundamental_quality_metrics_empty_is_unknown_not_zero(self):
        metrics = compute_fundamental_quality_metrics({})

        self.assertIsNone(metrics["quality_score"])
        self.assertEqual(metrics["component_count"], 0)

    def test_confidence_adjustment_uses_floor_and_cross_sectional_mean(self):
        adjusted, effective = adjust_alpha_for_confidence(
            alpha=1.0,
            confidence=0.2,
            cross_sectional_mean=0.1,
        )

        self.assertAlmostEqual(effective, 0.6)
        self.assertAlmostEqual(adjusted, 0.64)

    def test_factor_scores_bundle_is_bounded(self):
        scores = compute_factor_scores_from_result(
            {
                "quality_score_fundamental": 0.55,
                "pe_ratio": 11.0,
                "peg_ratio": 0.9,
                "fcf_yield": 0.08,
                "momentum_score": 0.80,
                "vol_20d": 0.18,
            }
        )

        self.assertGreater(scores["quality_factor_score"], 0.0)
        self.assertGreater(scores["value_factor_score"], 0.0)
        self.assertGreater(scores["momentum_factor_score"], 0.0)
        self.assertGreater(scores["volatility_factor_score"], 0.0)
        self.assertIn("qmj_factor_score", scores)
        self.assertIn("bab_factor_score", scores)
        self.assertIn("turnover_cost_score", scores)

    def test_qmj_lite_requires_multiple_non_overlapping_components(self):
        sparse = compute_factor_scores_from_result({"gpa_score": 0.7})
        self.assertIsNone(sparse["qmj_factor_score"])
        self.assertEqual(sparse["qmj_component_count"], 1)

        composite_only = compute_factor_scores_from_result({"quality_score_fundamental": 0.9})
        self.assertIsNone(composite_only["qmj_factor_score"])
        self.assertEqual(composite_only["qmj_component_count"], 0)

        enough = compute_factor_scores_from_result({
            "gpa_score": 0.7,
            "f_score_score": 0.5,
            "earnings_stability": 0.3,
        })
        self.assertIsNotNone(enough["qmj_factor_score"])
        self.assertGreaterEqual(enough["qmj_component_count"], 2)

    def test_bab_and_turnover_scores_reward_lower_risk_liquid_names(self):
        scores = compute_factor_scores_from_result(
            {
                "quality_score_fundamental": 0.6,
                "momentum_score": 0.7,
                "beta_90d": 0.65,
                "vol_20d": 0.14,
                "avg_dollar_volume": 80_000_000,
                "market_cap": 15_000_000_000,
                "return_10d_prior": 0.02,
                "return_90d_prior": 0.12,
            }
        )

        self.assertGreater(scores["bab_factor_score"], 0.0)
        self.assertGreater(scores["turnover_cost_score"], 0.0)

    def test_downside_risk_helpers_capture_bad_volatility(self):
        downside_vol = compute_downside_vol([0.01, -0.02, 0.03, -0.04, -0.01])
        max_dd = compute_max_drawdown([100, 110, 90, 95, 80, 120])

        self.assertIsNotNone(downside_vol)
        self.assertGreater(downside_vol, 0)
        self.assertAlmostEqual(max_dd, 0.2727, places=3)

    def test_factor_momentum_prefers_explicit_factor_scores(self):
        feature_rows = {
            "AAA": {
                "quality_factor_score": 0.9,
                "value_factor_score": 0.1,
                "momentum_factor_score": 0.8,
                "returns_90d": [0.01] * 63,
            },
            "BBB": {
                "quality_factor_score": -0.9,
                "value_factor_score": 0.8,
                "momentum_factor_score": -0.7,
                "returns_90d": [-0.01] * 63,
            },
            "CCC": {
                "quality_factor_score": 0.7,
                "value_factor_score": -0.6,
                "momentum_factor_score": 0.5,
                "returns_90d": [0.008] * 63,
            },
            "DDD": {
                "quality_factor_score": -0.4,
                "value_factor_score": 0.7,
                "momentum_factor_score": -0.5,
                "returns_90d": [-0.007] * 63,
            },
            "EEE": {
                "quality_factor_score": 0.6,
                "value_factor_score": -0.3,
                "momentum_factor_score": 0.4,
                "returns_90d": [0.006] * 63,
            },
            "FFF": {
                "quality_factor_score": -0.6,
                "value_factor_score": 0.9,
                "momentum_factor_score": -0.8,
                "returns_90d": [-0.009] * 63,
            },
            "GGG": {
                "quality_factor_score": 0.4,
                "value_factor_score": -0.2,
                "momentum_factor_score": 0.3,
                "returns_90d": [0.004] * 63,
            },
            "HHH": {
                "quality_factor_score": -0.5,
                "value_factor_score": 0.6,
                "momentum_factor_score": -0.4,
                "returns_90d": [-0.005] * 63,
            },
            "III": {
                "quality_factor_score": 0.3,
                "value_factor_score": -0.1,
                "momentum_factor_score": 0.2,
                "returns_90d": [0.003] * 63,
            },
            "JJJ": {
                "quality_factor_score": -0.3,
                "value_factor_score": 0.4,
                "momentum_factor_score": -0.2,
                "returns_90d": [-0.003] * 63,
            },
        }

        factor_returns = compute_factor_returns(feature_rows)

        self.assertIn("quality_tilt", factor_returns)
        self.assertIn("value_tilt", factor_returns)
        self.assertIn("momentum_tilt", factor_returns)
        self.assertGreater(factor_returns["quality_tilt"], 0.0)
        self.assertGreater(factor_returns["momentum_tilt"], 0.0)
        self.assertLessEqual(abs(factor_returns["quality_tilt"]), 0.35)
        self.assertLessEqual(abs(factor_returns["momentum_tilt"]), 0.35)

    def test_extract_qa_section_finds_unscripted_portion(self):
        transcript = """
        Management Presentation
        Revenue was strong.

        Question-and-Answer Session
        Analyst: How should we think about margins?
        CEO: We expect stable improvement.

        This concludes today's call.
        """

        qa_section = extract_qa_section(transcript)

        self.assertIsNotNone(qa_section)
        self.assertIn("Analyst:", qa_section)
        self.assertNotIn("Management Presentation", qa_section)


if __name__ == "__main__":
    unittest.main()
