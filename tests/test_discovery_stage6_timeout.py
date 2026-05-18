import time

import config
from engine import discovery
from engine import scoring


def test_stage6_uses_partial_result_after_hard_timeout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "DISCOVERY_PER_TICKER_TIMEOUT", 1)
    monkeypatch.setattr(config, "DISCOVERY_SCORING_WORKERS", 1, raising=False)

    def slow_analyse(_holding):
        time.sleep(1.5)
        return {"ticker": "SLOW", "aggregate_score": 1.0}

    monkeypatch.setattr(scoring, "analyse_holding", slow_analyse)

    result = discovery._stage_full_scoring(
        [
            {
                "symbol": "SLOW",
                "companyName": "Slow Co",
                "price": 10.0,
                "_momentum_score": 0.5,
                "_quick_score": 0.5,
            }
        ]
    )

    assert len(result) == 1
    assert result[0]["ticker"] == "SLOW"
    assert result[0]["analysis_degraded"] is True
    assert result[0]["_stage6_timeout"] is True
    assert "Timeout after 1s" in result[0]["analysis_degraded_reason"]
