import unittest
from unittest import mock

import numpy as np
import pandas as pd

import engine.portfolio_optimizer as portfolio_optimizer


class PortfolioOptimizerTests(unittest.TestCase):
    def test_method_registry_contains_all_ensemble_methods(self):
        expected = {
            "mean_variance",
            "min_variance",
            "risk_parity",
            "black_litterman",
            "hrp",
        }
        self.assertEqual(set(portfolio_optimizer.OPTIMIZER_METHODS), expected)
        self.assertTrue(all(callable(fn) for fn in portfolio_optimizer.OPTIMIZER_METHODS.values()))

    @mock.patch("engine.portfolio_optimizer.yf.download")
    def test_gerber_covariance_path_returns_psd_matrix(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        close = pd.DataFrame(
            {
                "AAA": np.linspace(100.0, 121.0, len(dates)),
                "BBB": np.linspace(80.0, 92.0, len(dates)) + np.sin(np.arange(len(dates))) * 0.8,
                "CCC": np.linspace(60.0, 69.0, len(dates)) + np.cos(np.arange(len(dates))) * 0.5,
                "DDD": np.linspace(40.0, 49.0, len(dates)) + np.sin(np.arange(len(dates)) / 3.0) * 0.6,
            },
            index=dates,
        )
        mock_download.return_value = pd.concat({"Close": close}, axis=1)

        cov = portfolio_optimizer._estimate_covariance(
            ["AAA", "BBB", "CCC", "DDD"],
            regime_label="NEUTRAL",
            method="gerber",
        )

        self.assertEqual(cov.shape, (4, 4))
        self.assertTrue(np.allclose(cov, cov.T))
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(cov))), -1e-8)

    @mock.patch("engine.portfolio_optimizer.yf.download")
    def test_covariance_resolves_portfolio_alias_before_download(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        close = pd.DataFrame(
            {
                "AAA": np.linspace(100.0, 121.0, len(dates)),
                "GFRD.L": np.linspace(500.0, 540.0, len(dates)),
            },
            index=dates,
        )
        mock_download.return_value = pd.concat({"Close": close}, axis=1)

        cov = portfolio_optimizer._estimate_covariance(
            ["AAA", "GFRD"],
            regime_label="NEUTRAL",
            method="gerber",
        )

        self.assertEqual(mock_download.call_args.args[0], ["AAA", "GFRD.L"])
        self.assertEqual(cov.shape, (2, 2))

    def test_black_litterman_posterior_reverts_to_prior_with_wide_omega(self):
        cov = np.array(
            [
                [0.0400, 0.0100, 0.0080],
                [0.0100, 0.0625, 0.0120],
                [0.0080, 0.0120, 0.0324],
            ],
            dtype=float,
        )
        mu_views = np.array([0.30, -0.15, 0.12], dtype=float)
        tickers = ["AAA", "BBB", "CCC"]
        results = [{"ticker": t, "confidence_discount": 1.0} for t in tickers]

        mu_bl, pi, omega_diag = portfolio_optimizer._black_litterman_posterior(
            mu_views,
            cov,
            tickers,
            results,
            risk_aversion=2.0,
            tau=0.05,
            confidence_override=np.full(3, 1e-12),
        )

        self.assertTrue(np.all(omega_diag > 0))
        self.assertTrue(np.allclose(mu_bl, pi, atol=1e-6))

    @mock.patch("engine.portfolio_optimizer._record_optimizer_run")
    @mock.patch("engine.portfolio_optimizer._load_optimizer_history")
    @mock.patch("engine.portfolio_optimizer._estimate_covariance")
    @mock.patch("engine.portfolio_optimizer._get_fx_rate")
    @mock.patch("engine.portfolio_optimizer._load_score_to_return_scale")
    @mock.patch("engine.portfolio_optimizer.yf.download")
    def test_optimize_portfolio_keeps_common_constraints(
        self,
        mock_download,
        mock_score_scale,
        mock_fx,
        mock_covariance,
        mock_history,
        mock_record_run,
    ):
        mock_download.return_value = pd.DataFrame({"Close": [4.0]})
        mock_score_scale.return_value = 0.10
        mock_fx.return_value = 1.0
        mock_record_run.return_value = None
        mock_history.return_value = {"version": 1, "method_stats": {}, "runs": []}

        vols = np.array([0.22, 0.27, 0.18, 0.21, 0.19, 0.20, 0.17], dtype=float)
        corr = np.full((7, 7), 0.20, dtype=float)
        np.fill_diagonal(corr, 1.0)
        mock_covariance.return_value = np.outer(vols, vols) * corr

        holdings = [
            {"ticker": "AAA", "name": "AAA", "quantity": 10, "currency": "GBP"},
            {"ticker": "BBB", "name": "BBB", "quantity": 10, "currency": "GBP"},
            {"ticker": "CCC", "name": "CCC", "quantity": 10, "currency": "GBP"},
            {"ticker": "DDD", "name": "DDD", "quantity": 10, "currency": "GBP"},
            {"ticker": "EEE", "name": "EEE", "quantity": 10, "currency": "GBP"},
            {"ticker": "FFF", "name": "FFF", "quantity": 10, "currency": "GBP"},
            {"ticker": "GGG", "name": "GGG", "quantity": 10, "currency": "GBP"},
        ]
        results = [
            {
                "ticker": "AAA",
                "current_price": 100.0,
                "expected_return_90d": -0.12,
                "aggregate_score": -0.8,
                "confidence_discount": 0.8,
                "max_weight_scale": 1.0,
                "sector": "Tech",
                "currency": "GBP",
                "action": "STRONG SELL",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "BBB",
                "current_price": 100.0,
                "expected_return_90d": -0.06,
                "aggregate_score": -0.4,
                "confidence_discount": 0.9,
                "max_weight_scale": 1.0,
                "sector": "Tech",
                "currency": "GBP",
                "action": "SELL",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "CCC",
                "current_price": 100.0,
                "expected_return_90d": 0.11,
                "aggregate_score": 0.6,
                "confidence_discount": 1.0,
                "max_weight_scale": 1.0,
                "sector": "Finance",
                "currency": "GBP",
                "action": "BUY",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "DDD",
                "current_price": 100.0,
                "expected_return_90d": 0.08,
                "aggregate_score": 0.4,
                "confidence_discount": 1.0,
                "max_weight_scale": 1.0,
                "sector": "Finance",
                "currency": "GBP",
                "action": "KEEP",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "EEE",
                "current_price": 100.0,
                "expected_return_90d": 0.10,
                "aggregate_score": 0.5,
                "confidence_discount": 1.0,
                "max_weight_scale": 1.0,
                "sector": "Healthcare",
                "currency": "GBP",
                "action": "BUY",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "FFF",
                "current_price": 100.0,
                "expected_return_90d": 0.09,
                "aggregate_score": 0.45,
                "confidence_discount": 1.0,
                "max_weight_scale": 1.0,
                "sector": "Industrials",
                "currency": "GBP",
                "action": "BUY",
                "avg_dollar_volume": 5_000_000,
            },
            {
                "ticker": "GGG",
                "current_price": 100.0,
                "expected_return_90d": 0.07,
                "aggregate_score": 0.35,
                "confidence_discount": 1.0,
                "max_weight_scale": 1.0,
                "sector": "Utilities",
                "currency": "GBP",
                "action": "BUY",
                "avg_dollar_volume": 5_000_000,
            },
        ]

        alloc = portfolio_optimizer.optimize_portfolio(
            results,
            holdings,
            regime={"regime_label": "NEUTRAL"},
        )

        weights = np.array([h.optimal_weight for h in alloc.holdings], dtype=float)
        weight_map = {h.ticker: h.optimal_weight for h in alloc.holdings}

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=3)
        self.assertLessEqual(weight_map["AAA"], portfolio_optimizer.MIN_WEIGHT + 1e-3)
        self.assertLessEqual(weight_map["BBB"], portfolio_optimizer.MIN_WEIGHT * 2 + 1e-3)
        self.assertTrue(
            all(weight <= portfolio_optimizer.SECTOR_CAP + 1e-3 for weight in alloc.sector_weights.values())
        )
        self.assertAlmostEqual(sum(alloc.method_weights.values()), 1.0, places=3)
        self.assertEqual(set(alloc.method_weights), set(portfolio_optimizer.OPTIMIZER_METHODS))
        self.assertEqual(set(alloc.per_method_weights), {h["ticker"] for h in holdings})


if __name__ == "__main__":
    unittest.main()
