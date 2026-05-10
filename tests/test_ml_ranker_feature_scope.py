from __future__ import annotations

import config
from engine import ml_ranker


def test_execution_readiness_fields_are_not_alpha_ranker_features():
    readiness_fields = set(getattr(config, "REPLAY_LIVE_PARITY_READINESS_FIELDS", []))

    assert {"fill_probability", "r_r_ratio", "strong_buy_eligible"} <= readiness_fields
    assert readiness_fields.isdisjoint(ml_ranker.FEATURE_COLS)


def test_ml_parity_critical_fields_exclude_readiness_contract_fields():
    readiness_fields = set(getattr(config, "REPLAY_LIVE_PARITY_READINESS_FIELDS", []))
    critical_fields = set(getattr(config, "ML_RANKER_PARITY_CRITICAL_FIELDS", []))

    assert readiness_fields.isdisjoint(critical_fields)
    assert {"quality_factor_score", "value_factor_score", "f_score", "gpa"} <= critical_fields
