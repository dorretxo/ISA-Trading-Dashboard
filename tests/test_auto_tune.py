from __future__ import annotations

import pytest

import config
from engine import auto_tune


def test_dynamic_meta_threshold_uses_floor_for_compressed_distribution(monkeypatch):
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", True, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_FLOOR", 0.55, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_PCTILE", 0.90, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MULTIPLIER", 0.92, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 50, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60, raising=False)

    candidates = [{"meta_label_proba": 0.43 + i * 0.0014} for i in range(100)]

    assert auto_tune.get_meta_strong_buy_min_prob(candidates) == pytest.approx(0.55)


def test_dynamic_meta_threshold_uses_tail_when_all_live_probs_below_floor(monkeypatch):
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", True, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_FLOOR", 0.55, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_FLOOR", 0.45, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_PCTILE", 0.95, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_PCTILE", 0.90, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MULTIPLIER", 0.92, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 20, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60, raising=False)

    candidates = [{"meta_label_proba": 0.477} for _ in range(18)] + [{"meta_label_proba": 0.507} for _ in range(2)]

    assert auto_tune.get_meta_strong_buy_min_prob(candidates) == pytest.approx(0.507)


def test_dynamic_meta_threshold_falls_back_when_sample_is_small(monkeypatch):
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", True, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 50, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60, raising=False)

    candidates = [{"meta_label_proba": 0.80} for _ in range(10)]

    assert auto_tune.get_meta_strong_buy_min_prob(candidates) == pytest.approx(0.60)


def test_core_ready_meta_threshold_uses_core_ready_distribution(monkeypatch):
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", True, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 20, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_FLOOR", 0.55, raising=False)
    monkeypatch.setattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_THRESHOLD", True, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_MIN_CANDIDATES", 2, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_FLOOR", 0.45, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_PCTILE", 0.50, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_MULTIPLIER", 0.98, raising=False)
    monkeypatch.setattr(config, "META_LABEL_CORE_READY_DYNAMIC_MAX_THRESHOLD", 0.55, raising=False)

    candidates = [
        {"meta_label_proba": 0.477, "ready_contract_core_status": "PASS"},
        {"meta_label_proba": 0.507, "ready_contract_core_status": "PASS"},
        {"meta_label_proba": 0.564, "ready_contract_core_status": "FAIL"},
    ]

    assert auto_tune.get_core_ready_meta_strong_buy_min_prob(candidates) == pytest.approx(0.4675)
