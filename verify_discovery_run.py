"""Post-discovery-run verification harness.

Validates that the Phase 1 + 2 + 3 cold-start fixes are firing in production:

  Phase 1: action_calibration has STRONG BUY rows (backfill effect)
  Phase 2: scorecard + override + conformal active in cached candidates
  Phase 3: multi-horizon IC weighting active in adaptive weights;
           active-labelling rows recorded today

Designed to run after every ``daily_orchestrator.py --force-discovery`` to
catch regressions.  Exits 0 (all pass) or 1 (any fail).  WARN-level checks
do not fail the run.

Usage:
    python verify_discovery_run.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _colour(text: str, name: str) -> str:
    if not sys.stdout.isatty():
        return text
    codes = {"red": "31", "green": "32", "yellow": "33"}
    if name not in codes:
        return text
    return f"\033[{codes[name]}m{text}\033[0m"


_FAILS: list[str] = []
_WARNS: list[str] = []


def _emit(check: str, status: str, message: str, details: str = "") -> None:
    tone = {PASS: "green", FAIL: "red", WARN: "yellow"}.get(status, "")
    coloured = _colour(status, tone)
    print(f"  [{coloured:^16}] {check:40} {message}")
    if details:
        for ln in str(details).splitlines():
            print(f"      {ln}")
    if status == FAIL:
        _FAILS.append(check)
    elif status == WARN:
        _WARNS.append(check)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_cached_discovery() -> list:
    state_path = ROOT / "orchestrator_state.json"
    if not state_path.exists():
        _emit("orchestrator_state.json", FAIL, "missing")
        return []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:
        _emit("orchestrator_state.json", FAIL, f"unreadable: {e}")
        return []
    cands = state.get("cached_discovery") or []
    if not cands:
        _emit("cached_discovery", FAIL, "empty")
        return []
    last_run = state.get("last_discovery_run", "?")
    _emit("cached_discovery", PASS, f"{len(cands)} candidates (last run {last_run})")
    return cands


def check_strong_buy_count(cands: list) -> list:
    sb = [c for c in cands if c.get("action") == "STRONG BUY"]
    n = len(sb)
    tickers = ", ".join(c.get("ticker", "?") for c in sb) or "(none)"
    if n >= 2:
        _emit("STRONG BUY count", PASS, f"{n} emitted (target >= 2)", tickers)
    elif n == 1:
        _emit("STRONG BUY count", WARN, "only 1 (target >= 2)", tickers)
    else:
        _emit("STRONG BUY count", FAIL, "0 emitted")
    return sb


def check_strong_buy_quality(sb: list) -> None:
    if not sb:
        return
    weak: list[str] = []
    for c in sb:
        fs = c.get("f_score")
        gpa = c.get("gpa")
        agg = c.get("aggregate_score")
        ok_f = isinstance(fs, (int, float)) and fs >= 6
        ok_gpa = isinstance(gpa, (int, float)) and gpa >= 0.10
        ok_agg = isinstance(agg, (int, float)) and agg >= 0.20
        if not (ok_f or ok_gpa or ok_agg):
            weak.append(f"{c.get('ticker')} (F={fs}, GPA={gpa}, agg={agg})")
    if not weak:
        _emit("STRONG BUY quality", PASS, "all pass F>=6 OR GPA>=0.10 OR agg>=0.20")
    else:
        _emit("STRONG BUY quality", WARN, f"{len(weak)} weak", "; ".join(weak))


def check_scorecard_distribution(cands: list) -> None:
    scores = [c.get("sb_score") for c in cands if isinstance(c.get("sb_score"), (int, float))]
    if not scores:
        _emit("sb_score populated", FAIL, "no candidates have sb_score")
        return
    top = max(scores)
    above_07 = sum(1 for s in scores if s >= 0.7)
    above_10 = sum(1 for s in scores if s >= 1.0)
    msg = f"top={top:+.2f}  >=0.7: {above_07}  >=1.0: {above_10}"
    if top >= 1.0 and above_07 >= 5:
        _emit("sb_score distribution", PASS, msg)
    elif top >= 0.5:
        _emit("sb_score distribution", WARN, msg)
    else:
        _emit("sb_score distribution", FAIL, msg)


def check_conformal_present(sb: list) -> None:
    if not sb:
        return
    populated = sum(1 for c in sb if c.get("conformal_p") is not None)
    if populated == len(sb):
        _emit("conformal_p on STRONG BUY", PASS, f"all {len(sb)} populated")
    elif populated > 0:
        _emit("conformal_p on STRONG BUY", WARN, f"{populated}/{len(sb)} populated")
    else:
        _emit("conformal_p on STRONG BUY", FAIL, "none populated")


def check_action_calibration() -> None:
    try:
        with sqlite3.connect(str(ROOT / "paper_trading.db")) as con:
            cur = con.execute(
                """SELECT source, sample_size, hit_rate, avg_return_90d
                   FROM action_calibration
                   WHERE action='STRONG BUY'
                   ORDER BY sample_size DESC""",
            )
            rows = cur.fetchall()
    except Exception as e:
        _emit("action_calibration STRONG BUY", FAIL, f"query failed: {e}")
        return
    if not rows:
        _emit("action_calibration STRONG BUY", FAIL, "no rows (Phase 1 backfill regressed)")
        return
    top = rows[0]
    src, n, hit, avg = top
    summary = f"{src} n={n} hit={hit:.3f} avg_90d={avg}"
    if n >= 6000:
        _emit("action_calibration STRONG BUY", PASS, summary)
    elif n >= 1000:
        _emit("action_calibration STRONG BUY", WARN, summary)
    else:
        _emit("action_calibration STRONG BUY", FAIL, summary)


def check_active_labelling_today() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(str(ROOT / "paper_trading.db")) as con:
            cur = con.execute(
                """SELECT COUNT(*), COUNT(DISTINCT sector)
                   FROM signal_backtest
                   WHERE source='active_label' AND run_date LIKE ?""",
                (today + "%",),
            )
            n, n_sectors = cur.fetchone() or (0, 0)
    except Exception as e:
        _emit("active_label rows today", FAIL, f"query failed: {e}")
        return
    if n >= 3:
        _emit("active_label rows today", PASS, f"{n} recorded across {n_sectors} sectors")
    elif n > 0:
        _emit("active_label rows today", WARN, f"only {n} recorded (target >= 3)")
    else:
        _emit("active_label rows today", FAIL, "0 recorded")


def check_log_signals() -> None:
    log_path = ROOT / "orchestrator_output.log"
    if not log_path.exists():
        _emit("orchestrator_output.log", FAIL, "missing")
        return
    try:
        log = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        _emit("orchestrator_output.log", FAIL, f"unreadable: {e}")
        return

    # Multi-horizon adaptive weights
    if "multi-horizon blend active" in log.lower():
        m = re.search(r"Adaptive weights \([^/]+/multi[^)]*\):\s*(\{[^}]+\})", log)
        details = m.group(1) if m else ""
        # Extra check: fundamental should outweight technical
        if m:
            try:
                weights = json.loads(m.group(1).replace("'", '"'))
                fund = float(weights.get("fundamental", 0))
                tech = float(weights.get("technical", 0))
                if fund > tech:
                    _emit("multi-horizon blend", PASS, "fundamental > technical", details)
                else:
                    _emit("multi-horizon blend", WARN, f"fund={fund} ≤ tech={tech}", details)
            except Exception:
                _emit("multi-horizon blend", PASS, "active (weights unparseable)", details)
        else:
            _emit("multi-horizon blend", PASS, "active (weights line not found)")
    else:
        _emit("multi-horizon blend", FAIL, "not found in log")

    # ML ranker active
    m = re.search(r"ML ranker active for final ranking[^\n]*IC=([\-\d.]+)", log)
    if m:
        ic = float(m.group(1))
        if ic >= 0.05:
            _emit("ML ranker active", PASS, f"rank-IC={ic:+.4f}")
        else:
            _emit("ML ranker active", WARN, f"rank-IC={ic:+.4f} below 0.05 threshold")
    elif "ML ranker active for final ranking" in log:
        _emit("ML ranker active", PASS, "active (IC not parsed)")
    else:
        _emit("ML ranker active", FAIL, "not active this run")

    # Scorecard override
    override_lines = [ln for ln in log.splitlines() if "Scorecard override:" in ln and "STRONG BUY (was" in ln]
    if override_lines:
        _emit("scorecard override fired", PASS,
              f"{len(override_lines)} candidates restored",
              "\n".join(override_lines[:5]))
    else:
        _emit("scorecard override fired", WARN, "no candidates met override threshold")

    # Conformal overlay
    if "Conformal overlay:" in log:
        m = re.search(r"Conformal overlay:[^\n]*", log)
        _emit("conformal overlay logged", PASS, "fired", m.group(0) if m else "")
    else:
        _emit("conformal overlay logged", WARN, "not in log")

    # Errors / exceptions
    n_err = sum(1 for ln in log.splitlines() if "[ERROR]" in ln)
    n_exc = sum(1 for ln in log.splitlines() if "Traceback" in ln)
    if n_err == 0 and n_exc == 0:
        _emit("no errors in log", PASS, "clean run")
    elif n_err <= 2 and n_exc == 0:
        _emit("no errors in log", WARN, f"{n_err} ERROR lines, 0 exceptions")
    else:
        _emit("no errors in log", FAIL, f"{n_err} ERROR lines, {n_exc} exceptions")


def main() -> int:
    print("=" * 80)
    print("Discovery run verification harness")
    print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    cands = check_cached_discovery()
    if not cands:
        # No further checks possible without state.
        print()
        print(_colour("RESULT: NO CACHED DISCOVERY — abort", "red"))
        return 1

    sb = check_strong_buy_count(cands)
    check_strong_buy_quality(sb)
    check_scorecard_distribution(cands)
    check_conformal_present(sb)
    check_action_calibration()
    check_active_labelling_today()
    check_log_signals()

    print()
    print("=" * 80)
    if not _FAILS:
        if not _WARNS:
            print(_colour("RESULT: ALL CHECKS PASSED", "green"))
        else:
            print(_colour(f"RESULT: PASS with {len(_WARNS)} warning(s)", "yellow"))
            for w in _WARNS:
                print(f"  - {w}")
        return 0
    else:
        print(_colour(f"RESULT: {len(_FAILS)} FAILURE(S), {len(_WARNS)} warning(s)", "red"))
        for f in _FAILS:
            print(f"  FAIL: {f}")
        for w in _WARNS:
            print(f"  WARN: {w}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
