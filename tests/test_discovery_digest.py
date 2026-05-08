import unittest

from utils.discovery_digest import build_trade_packet, discovery_confidence


class DiscoveryDigestTests(unittest.TestCase):
    def test_trade_packet_marks_top_pick_only_when_actionable_and_planned(self):
        candidate = {
            "ticker": "ABC",
            "name": "Alpha Beta Co",
            "action": "STRONG BUY",
            "entry_stance": "Ready",
            "sentiment_score": 0.6,
            "confidence_discount": 1.0,
            "final_rank": 1.234,
            "aggregate_score": 0.72,
            "expected_return_90d": 0.11,
            "entry_price": 100.0,
            "stop_loss": 92.0,
            "take_profit": 125.0,
            "r_r_ratio": 3.1,
            "position_weight": 0.08,
            "portfolio_fit_score": 0.84,
            "max_correlation": 0.24,
            "currency": "USD",
        }

        packet = build_trade_packet(candidate)

        self.assertTrue(packet["buy_eligible"])
        self.assertTrue(packet["clean_entry"])
        self.assertTrue(packet["trade_ready"])
        self.assertTrue(packet["top_pick"])
        self.assertEqual(packet["status_rank"], 4)
        self.assertEqual(packet["confidence_label"], "High Confidence")

    def test_trade_packet_demotes_identity_and_watch_risk(self):
        candidate = {
            "ticker": "XYZ",
            "name": "Risky Co",
            "action": "BUY",
            "entry_stance": "Watch Only",
            "ticker_identity_warning": "Possible ADR mismatch",
            "sentiment_score": 0.2,
            "confidence_discount": 1.0,
            "final_rank": 0.95,
            "entry_price": 50.0,
            "stop_loss": 46.0,
            "take_profit": 60.0,
            "currency": "USD",
        }

        packet = build_trade_packet(candidate)

        self.assertTrue(packet["buy_eligible"])
        self.assertFalse(packet["clean_entry"])
        self.assertFalse(packet["trade_ready"])
        self.assertFalse(packet["top_pick"])
        self.assertEqual(packet["status_rank"], 0)
        self.assertEqual(packet["key_risk"], "Ticker identity")

    def test_discovery_confidence_flags_data_gap(self):
        candidate = {
            "action": "INSUFFICIENT DATA",
            "sentiment_score": 0.9,
            "confidence_discount": 1.0,
        }

        label, tone, score = discovery_confidence(candidate)

        self.assertEqual(label, "Data Gap")
        self.assertEqual(tone, "data")
        self.assertLess(score, 1.0)

    def test_trade_packet_requires_ready_contract_for_top_pick(self):
        candidate = {
            "ticker": "ABC",
            "name": "Alpha Beta Co",
            "action": "STRONG BUY",
            "entry_stance": "Ready",
            "sentiment_score": 0.8,
            "confidence_discount": 1.0,
            "final_rank": 1.4,
            "entry_price": 100.0,
            "stop_loss": 94.0,
            "take_profit": 116.0,
            "r_r_ratio": 2.6,
            "position_weight": 0.05,
            "ready_contract_status": "FAIL",
            "ready_contract_reasons": ["institutional prior below bar"],
            "strong_buy_eligible": False,
        }

        packet = build_trade_packet(candidate)

        self.assertFalse(packet["top_pick"])
        self.assertFalse(packet["entry_ready"])
        self.assertFalse(packet["trade_ready"])
        self.assertEqual(packet["key_risk"], "institutional prior below bar")


if __name__ == "__main__":
    unittest.main()
