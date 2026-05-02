from engine.discovery import _evaluate_discovery_gates_v2, _evaluate_trap_safeguard


def test_trap_safeguard_flags_stretched_commodity_with_missing_health():
    triggered, reason = _evaluate_trap_safeguard(
        {"name": "PLS Group", "f_score": None, "gpa": None},
        sector="Basic Materials",
        industry="Other Industrial Metals & Mining",
        stretch=0.68,
    )

    assert triggered is True
    assert "Commodity-cycle trap safeguard" in reason


def test_trap_safeguard_allows_healthy_stretched_commodity():
    triggered, _ = _evaluate_trap_safeguard(
        {"name": "Healthy Miner", "f_score": 7, "f_score_coverage": 1.0, "gpa": 0.34},
        sector="Materials",
        industry="Copper Mining",
        stretch=0.68,
    )

    assert triggered is False


def test_trap_safeguard_ignores_non_commodity_names():
    triggered, _ = _evaluate_trap_safeguard(
        {"name": "Software Winner", "f_score": None, "gpa": None},
        sector="Technology",
        industry="Software",
        stretch=0.80,
    )

    assert triggered is False


def test_discovery_gates_v2_marks_low_quality_as_reject():
    status, reasons = _evaluate_discovery_gates_v2(
        {
            "name": "Weak Industrial",
            "f_score": 2,
            "f_score_coverage": 1.0,
            "gpa": 0.04,
            "ev_ebit": 40,
            "revenue_growth": 0.02,
        },
        sector="Industrials",
        industry="Machinery",
        stretch=0.10,
    )

    assert status == "REJECT"
    assert any("F-score" in reason for reason in reasons)
    assert any("GPA" in reason for reason in reasons)
