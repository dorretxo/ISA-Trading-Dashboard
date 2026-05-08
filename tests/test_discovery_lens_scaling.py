from engine.discovery import (
    _adaptive_promote_candidates,
    _gate_aware_ready_lane_score,
    _promote_ready_lane,
    _scaled_lens_min_counts,
)


def test_scaled_lens_min_counts_preserves_mix_for_small_target():
    scaled = _scaled_lens_min_counts(
        {"quality": 60, "value": 55, "composite": 40, "momentum": 60},
        20,
    )

    assert sum(scaled.values()) == 20
    assert scaled["quality"] > 0
    assert scaled["value"] > 0
    assert scaled["composite"] > 0
    assert scaled["momentum"] > 0


def test_adaptive_promote_scales_lens_quotas_in_bounded_runs():
    candidates = []
    for lens in ("quality", "value", "composite", "momentum"):
        for i in range(10):
            candidates.append(
                {
                    "symbol": f"{lens[:2].upper()}{i}",
                    "_entry_lens": lens,
                    "_quick_score": 1.0 if lens == "quality" else 0.5,
                    "_region": "US",
                    "sector": "Technology",
                    "_avg_dollar_volume": 10_000_000,
                }
            )

    selected = _adaptive_promote_candidates(
        candidates,
        target_n=8,
        score_key="_quick_score",
        preselect_pct=0.55,
        lens_min_counts={"quality": 60, "value": 55, "composite": 40, "momentum": 60},
    )

    counts = {}
    for candidate in selected:
        lens = candidate["_entry_lens"]
        counts[lens] = counts.get(lens, 0) + 1

    assert len(selected) == 8
    assert counts == {"quality": 2, "value": 2, "composite": 2, "momentum": 2}


def test_ready_lane_score_penalizes_stretched_missing_names():
    clean = {
        "symbol": "READY",
        "_above_sma50": True,
        "_above_sma200": True,
        "_pct_from_high": 0.93,
        "_ret_30d": 0.05,
        "_ret_90d": 0.18,
        "_vol_20d": 0.22,
        "_avg_dollar_volume": 20_000_000,
        "_f_score": 7,
        "_f_score_coverage": 1.0,
        "_gpa": 0.18,
        "_ev_ebit": 15,
    }
    stretched = {
        "symbol": "WATCH",
        "_above_sma50": True,
        "_above_sma200": True,
        "_pct_from_high": 1.02,
        "_ret_30d": 0.24,
        "_vol_20d": 0.45,
        "_avg_dollar_volume": 20_000_000,
    }

    assert _gate_aware_ready_lane_score(clean) >= 0.58
    assert _gate_aware_ready_lane_score(stretched) < _gate_aware_ready_lane_score(clean)


def test_ready_lane_promotes_clean_candidates_without_expanding_shortlist():
    selected = [
        {"symbol": f"SEL{i}", "_entry_lens": "quality", "_quick_score": 1.0 - i * 0.01, "_ready_lane_score": 0.2}
        for i in range(6)
    ]
    pool = selected + [
        {"symbol": "READY1", "_entry_lens": "value", "_quick_score": 0.4, "_ready_lane_score": 0.7},
        {"symbol": "READY2", "_entry_lens": "momentum", "_quick_score": 0.39, "_ready_lane_score": 0.68},
    ]

    promoted = _promote_ready_lane(
        pool,
        selected,
        target_n=6,
        reserve_n=2,
        min_ready_score=0.58,
        score_key="_quick_score",
    )

    symbols = {candidate["symbol"] for candidate in promoted}
    assert len(promoted) == 6
    assert {"READY1", "READY2"} <= symbols
