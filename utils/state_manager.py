"""Orchestrator state persistence — cooldowns, cached discovery, run timestamps.

Follows the same JSON load/save pattern as engine/forecasting.py.
State file lives next to portfolio.json in the project root.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


def _default_state() -> dict:
    """Return a fresh default state structure."""
    return {
        "version": 2,
        "last_portfolio_run": None,
        "last_discovery_run": None,
        "last_email_sent": None,
        "cooldowns": {},
        "cached_discovery": [],
        "cached_portfolio": None,
        "cached_optimizer": None,
        "cached_exit_signals": None,
    }


def _state_path() -> Path:
    return _ROOT / config.ORCHESTRATOR_STATE_FILE


def load_state() -> dict:
    """Load state from disk.  Returns defaults if file is missing or corrupt."""
    path = _state_path()
    if not path.exists():
        return _default_state()

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Ensure all expected keys exist (forward-compatible)
        defaults = _default_state()
        for key, value in defaults.items():
            if key not in state:
                state[key] = value
        return state
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("State file corrupt, resetting: %s", e)
        return _default_state()


def save_state(state: dict) -> None:
    """Persist state to disk."""
    path = _state_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(path)  # Atomic rename — prevents corruption from OneDrive sync
    except OSError as e:
        logger.error("Failed to save state: %s", e)


# ---------------------------------------------------------------------------
# Cooldown management
# ---------------------------------------------------------------------------

def is_on_cooldown(state: dict, ticker: str) -> bool:
    """True if ticker is within the cooldown window."""
    cooldowns = state.get("cooldowns", {})
    expiry_str = cooldowns.get(ticker)
    if not expiry_str:
        return False
    try:
        expiry = date.fromisoformat(expiry_str)
        return date.today() <= expiry
    except (ValueError, TypeError):
        return False


def set_cooldown(state: dict, ticker: str) -> None:
    """Set cooldown expiry for ticker to today + COOLDOWN_DAYS."""
    days = getattr(config, "COOLDOWN_DAYS", 14)
    expiry = date.today() + timedelta(days=days)
    state.setdefault("cooldowns", {})[ticker] = expiry.isoformat()


def prune_expired_cooldowns(state: dict) -> None:
    """Remove expired cooldown entries."""
    cooldowns = state.get("cooldowns", {})
    today = date.today()
    expired = [
        ticker for ticker, expiry_str in cooldowns.items()
        if _parse_date(expiry_str) is not None and _parse_date(expiry_str) < today
    ]
    for ticker in expired:
        del cooldowns[ticker]


def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def should_run_discovery(state: dict) -> bool:
    """True if discovery has never run or last run was >= FREQ_DAYS ago."""
    last = state.get("last_discovery_run")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        freq = getattr(config, "ORCHESTRATOR_DISCOVERY_FREQ_DAYS", 7)
        return datetime.now() - last_dt >= timedelta(days=freq)
    except (ValueError, TypeError):
        return True


def get_cached_discovery(state: dict) -> list[dict]:
    """Return cached discovery candidates, or empty if stale or missing.

    Candidates are considered stale if the last discovery run was more than
    2x the configured frequency ago (e.g., >14 days for weekly discovery).
    """
    cached = state.get("cached_discovery", [])
    if not cached:
        return []

    last = state.get("last_discovery_run")
    if not last:
        return []

    try:
        last_dt = datetime.fromisoformat(last)
        freq = getattr(config, "ORCHESTRATOR_DISCOVERY_FREQ_DAYS", 7)
        max_age = timedelta(days=freq * 2)
        if datetime.now() - last_dt > max_age:
            logger.info("Cached discovery is stale (%s), ignoring", last)
            return []
    except (ValueError, TypeError):
        return []

    return cached
