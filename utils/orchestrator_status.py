"""Check whether the batch orchestrator is running and estimate completion.

Used by the Streamlit UI to prevent double-runs and show progress.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_PID_FILE = _ROOT / "orchestrator.pid"
_CHECKPOINT_FILE = _ROOT / "feature_cache" / "discovery_checkpoint.json"


def _is_process_alive(pid: int) -> bool:
    """Check if a process is running (Windows-compatible)."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


def get_orchestrator_status() -> dict:
    """Check if the batch orchestrator is running and estimate completion.

    Returns::

        {
            "running": bool,
            "pid": int | None,
            "stage": str,           # "discovery_scoring", "screening", "idle"
            "checkpoint": {         # None if no checkpoint
                "scored_count": int,
                "total": int,
            },
            "eta_minutes": float | None,
        }
    """
    result = {
        "running": False,
        "pid": None,
        "stage": "idle",
        "checkpoint": None,
        "eta_minutes": None,
    }

    # --- Check PID file ---
    if not _PID_FILE.exists():
        return result

    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return result

    if not _is_process_alive(pid):
        return result

    result["running"] = True
    result["pid"] = pid

    # --- Read checkpoint for scoring progress ---
    if not _CHECKPOINT_FILE.exists():
        result["stage"] = "screening"
        return result

    try:
        with open(_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            cp = json.load(f)
        scored = cp.get("scored_count", 0)
        total = cp.get("total", 0)

        if total <= 0:
            result["stage"] = "screening"
            return result

        result["stage"] = "discovery_scoring"
        result["checkpoint"] = {"scored_count": scored, "total": total}

        # Estimate ETA from PID file creation time + progress ratio
        if scored > 0:
            pid_start = datetime.fromtimestamp(_PID_FILE.stat().st_mtime)
            elapsed_s = (datetime.now() - pid_start).total_seconds()
            progress = scored / total
            if progress < 1.0:
                remaining_s = elapsed_s * (1.0 - progress) / progress
                result["eta_minutes"] = round(remaining_s / 60, 1)
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.debug("Could not read checkpoint: %s", e)
        result["stage"] = "screening"

    return result
