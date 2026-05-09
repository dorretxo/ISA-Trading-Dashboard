"""Runtime audit for pillar-weight governance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from engine.pillar_weighting import normalise_weights, pillar_parity_gate
from utils.atomic_io import atomic_write_json


def _pillar_effectiveness_rows(source: str) -> list[dict]:
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            rows = conn.execute(
                """SELECT source, pillar, horizon, regime, information_coefficient,
                          sample_size, updated_at
                   FROM pillar_effectiveness
                   WHERE source IN (?, 'all')
                     AND regime IS NULL
                   ORDER BY source, horizon, pillar""",
                (source,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        return [{"error": str(exc)}]


def _safe_call(label: str, func):
    try:
        return {"ok": True, "value": func()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_pillar_weight_audit(*, source: str = "discovery", horizon: str = "90d") -> dict:
    default_weights = normalise_weights(config.WEIGHTS)

    adaptive_result = _safe_call(
        "adaptive",
        lambda: __import__("engine.discovery_backtest", fromlist=["get_adaptive_weights"]).get_adaptive_weights(
            source=source,
            horizon=horizon,
        ),
    )
    adaptive_weights = adaptive_result.get("value") if adaptive_result.get("ok") else None

    bayesian_result = None
    bayesian_weights = None
    if source == "discovery":
        base = adaptive_weights or default_weights
        bayesian_result = _safe_call(
            "bayesian",
            lambda: __import__("engine.bayesian_learning", fromlist=["get_bayesian_pillar_weights"]).get_bayesian_pillar_weights(
                base,
                source=source,
            ),
        )
        bayesian_weights = bayesian_result.get("value") if bayesian_result.get("ok") else None

    regime_base = bayesian_weights or adaptive_weights or default_weights
    regime_result = _safe_call(
        "regime",
        lambda: __import__("engine.regime", fromlist=["get_regime_adjusted_weights"]).get_regime_adjusted_weights(regime_base),
    )
    regime_weights = regime_result.get("value") if regime_result.get("ok") else None

    gate_ok, gate_reasons, gate_summary = pillar_parity_gate()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "horizon": horizon,
        "parity_gate": {
            "ok": gate_ok,
            "reasons": gate_reasons,
            "summary": gate_summary,
        },
        "layers": {
            "default": default_weights,
            "adaptive": adaptive_result,
            "bayesian": bayesian_result,
            "regime": regime_result,
            "effective": normalise_weights(regime_weights or regime_base),
        },
        "pillar_effectiveness": _pillar_effectiveness_rows(source),
    }


def write_pillar_weight_audit(*, source: str = "discovery", horizon: str = "90d", path: str | Path | None = None) -> dict:
    payload = build_pillar_weight_audit(source=source, horizon=horizon)
    out_path = Path(path or getattr(config, "PILLAR_WEIGHT_AUDIT_PATH", "feature_cache/pillar_weight_audit.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, payload, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit runtime pillar weights.")
    parser.add_argument("--source", default="discovery", choices=["portfolio", "discovery", "all"])
    parser.add_argument("--horizon", default="90d")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    payload = write_pillar_weight_audit(source=args.source, horizon=args.horizon, path=args.path)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
