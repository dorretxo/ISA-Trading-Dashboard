"""Tier 5 — Self-learning threshold ensemble.

Closes the loop the screener was missing.  We maintain three named threshold
profiles (Conservative / Moderate / Aggressive) — each is just a dict of
config-flag overrides — and a Beta-distribution posterior over each profile's
30-day STRONG-BUY hit-rate.  Once outcomes mature, the posterior updates and
the next discovery run picks the active profile by Thompson sampling.

References:

- Russo & Van Roy (2014, *Math. of OR*) — Thompson-sampling regret bounds.
- Helmbold, Schapire, Singer & Warmuth (1998) — Exponentiated-gradient
  online portfolio updates; used here for sleeve-weight learning.
- López de Prado & Bailey (2014, *J. Risk*) — Probabilistic Sharpe Ratio;
  controls when a tightened threshold is "trusted enough" to deploy.
- Page (1954) — Page-Hinkley; already implemented in utils/drift_monitor.py
  and consumed here as the "force conservative" override.

State lives in ``feature_cache/threshold_learner_state.json`` (path is
configurable).  All updates are atomic — the file is written via temp-rename
so a crashed run never corrupts the posteriors.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile definitions — these are *deltas* applied on top of the base config
# at runtime.  Keep them small and additive.
# ---------------------------------------------------------------------------

CONSERVATIVE_PROFILE: dict[str, Any] = {
    "PERCENTILE_STRONG_BUY_PCT": 0.03,
    "PERCENTILE_STRONG_BUY_MIN_AGG": 0.20,
    "ALTMAN_Z_STRONG_BUY_MIN": 3.0,
    "F_SCORE_STRONG_BUY_MIN": 7,
    "EV_EBIT_STRONG_BUY_MAX": 25.0,
    "QMJ_PCTILE_STRONG_BUY_MIN": 0.60,
    "RSI_STRONG_BUY_MAX": 70,
    "STRETCH_200DMA_STRONG_BUY_MAX": 0.25,
}

MODERATE_PROFILE: dict[str, Any] = {
    # Empty == use the base config defaults
}

AGGRESSIVE_PROFILE: dict[str, Any] = {
    "PERCENTILE_STRONG_BUY_PCT": 0.07,
    "PERCENTILE_STRONG_BUY_MIN_AGG": -0.05,
    "ALTMAN_Z_STRONG_BUY_MIN": 2.2,
    "F_SCORE_STRONG_BUY_MIN": 5,
    "EV_EBIT_STRONG_BUY_MAX": 35.0,
    "RSI_STRONG_BUY_MAX": 78,
    "STRETCH_200DMA_STRONG_BUY_MAX": 0.45,
}

PROFILES: dict[str, dict[str, Any]] = {
    "conservative": CONSERVATIVE_PROFILE,
    "moderate": MODERATE_PROFILE,
    "aggressive": AGGRESSIVE_PROFILE,
}


# ---------------------------------------------------------------------------
# Beta posterior — minimal implementation, no scipy dependency
# ---------------------------------------------------------------------------

@dataclass
class BetaPosterior:
    alpha: float = 1.0    # Prior: Beta(1, 1) == uniform
    beta: float = 1.0

    def update(self, successes: int, failures: int) -> None:
        self.alpha += float(max(0, successes))
        self.beta += float(max(0, failures))

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random | None = None) -> float:
        rng = rng or random
        # Marsaglia-Tsang via gammavariate; identical distribution to Beta(α, β)
        a = rng.gammavariate(self.alpha, 1.0)
        b = rng.gammavariate(self.beta, 1.0)
        denom = a + b
        return 0.0 if denom <= 0 else a / denom


# ---------------------------------------------------------------------------
# Exponentiated-gradient sleeve weights
# ---------------------------------------------------------------------------

DEFAULT_SLEEVE_WEIGHTS: dict[str, float] = {
    "quality": 0.24, "momentum": 0.20, "value": 0.18,
    "low_risk": 0.15, "ready": 0.15, "pead": 0.08,
}


def eg_update_sleeve_weights(
    weights: dict[str, float],
    realised_ic: dict[str, float],
    *,
    learning_rate: float = 0.05,
) -> dict[str, float]:
    """Exponentiated-gradient update — multiplicative weight adjustment.

    Each sleeve's weight is multiplied by ``exp(eta * IC_sleeve)``, then the
    vector is renormalised to sum to 1.  Sleeves whose realised IC is positive
    grow; negative-IC sleeves shrink.  The renormalisation step keeps the
    weights on the simplex, the update being equivalent to gradient descent
    in the dual (Cover 1991; Helmbold et al. 1998).
    """
    keys = list(weights.keys())
    raw = []
    for k in keys:
        w = max(1e-6, float(weights.get(k, 0.0)))
        ic = float(realised_ic.get(k, 0.0) or 0.0)
        if not math.isfinite(ic):
            ic = 0.0
        raw.append(w * math.exp(learning_rate * ic))
    total = sum(raw) or 1.0
    return {k: round(r / total, 4) for k, r in zip(keys, raw)}


# ---------------------------------------------------------------------------
# Threshold ensemble — Thompson sampling over named profiles
# ---------------------------------------------------------------------------

@dataclass
class ThresholdLearnerState:
    posteriors: dict[str, BetaPosterior] = field(default_factory=lambda: {
        name: BetaPosterior() for name in PROFILES
    })
    sleeve_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SLEEVE_WEIGHTS))
    last_active_profile: str = "moderate"
    drift_active: bool = False
    n_updates: int = 0

    def to_dict(self) -> dict:
        return {
            "posteriors": {k: asdict(p) for k, p in self.posteriors.items()},
            "sleeve_weights": dict(self.sleeve_weights),
            "last_active_profile": self.last_active_profile,
            "drift_active": self.drift_active,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThresholdLearnerState":
        if not isinstance(data, dict):
            return cls()
        posteriors = {}
        for name in PROFILES:
            d = (data.get("posteriors") or {}).get(name) or {}
            posteriors[name] = BetaPosterior(
                alpha=float(d.get("alpha", 1.0)), beta=float(d.get("beta", 1.0))
            )
        return cls(
            posteriors=posteriors,
            sleeve_weights={**DEFAULT_SLEEVE_WEIGHTS, **dict(data.get("sleeve_weights") or {})},
            last_active_profile=str(data.get("last_active_profile", "moderate")),
            drift_active=bool(data.get("drift_active", False)),
            n_updates=int(data.get("n_updates", 0)),
        )


_LOCK = threading.Lock()


def _state_path(config_module=None) -> Path:
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]
    return Path(getattr(cfg, "THRESHOLD_LEARNER_STATE_FILE",
                        "feature_cache/threshold_learner_state.json"))


def load_state(config_module=None) -> ThresholdLearnerState:
    path = _state_path(config_module)
    try:
        with path.open("r", encoding="utf-8") as f:
            return ThresholdLearnerState.from_dict(json.load(f))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return ThresholdLearnerState()


def save_state(state: ThresholdLearnerState, config_module=None) -> None:
    path = _state_path(config_module)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _LOCK:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        tmp.replace(path)


def select_profile(
    state: ThresholdLearnerState,
    *,
    drift_alert: bool = False,
    rng: random.Random | None = None,
    config_module=None,
) -> str:
    """Choose the active profile via Thompson sampling, with drift override.

    When ``drift_alert`` is True and the config flag
    ``DRIFT_FORCE_CONSERVATIVE`` is set, the conservative profile wins
    unconditionally — this is the safety mechanism that ratchets thresholds
    in when the live model degrades.
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    if drift_alert and bool(getattr(cfg, "DRIFT_FORCE_CONSERVATIVE", True)):
        return "conservative"

    rng = rng or random.Random()
    samples = {name: post.sample(rng) for name, post in state.posteriors.items()}
    return max(samples, key=samples.get)


def overlay_profile(name: str, base_config: Any) -> dict[str, Any]:
    """Return a dict of attribute overrides for the given profile name.

    Use the returned dict as ``getattr(profile_overlay, key, getattr(config, key))``
    in the hot path — see :func:`apply_overlay`.
    """
    return dict(PROFILES.get(name, {}))


@dataclass
class _Overlay:
    """Lightweight thin wrapper used as a config-module replacement."""
    _base: Any
    _overrides: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        if item in self._overrides:
            return self._overrides[item]
        return getattr(self._base, item)


def apply_overlay(base_config: Any, name: str) -> _Overlay:
    """Return a config-like object: the base config with the profile applied."""
    return _Overlay(_base=base_config, _overrides=dict(PROFILES.get(name, {})))


# ---------------------------------------------------------------------------
# Posterior update — call once per evaluation window
# ---------------------------------------------------------------------------

@dataclass
class OutcomeBatch:
    """Outcomes attributed to one profile in one evaluation window."""
    profile: str
    successes: int = 0
    failures: int = 0


def update_posteriors(
    state: ThresholdLearnerState,
    outcomes: list[OutcomeBatch],
) -> ThresholdLearnerState:
    for ob in outcomes:
        post = state.posteriors.get(ob.profile)
        if post is not None:
            post.update(successes=ob.successes, failures=ob.failures)
    state.n_updates += 1
    return state


def update_sleeve_weights_state(
    state: ThresholdLearnerState,
    realised_ic: dict[str, float],
    *,
    learning_rate: float | None = None,
    config_module=None,
) -> ThresholdLearnerState:
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]
    lr = float(learning_rate if learning_rate is not None
               else getattr(cfg, "SLEEVE_WEIGHTS_LEARNING_RATE", 0.05))
    state.sleeve_weights = eg_update_sleeve_weights(
        state.sleeve_weights, realised_ic, learning_rate=lr,
    )
    return state


# ---------------------------------------------------------------------------
# Convenience entry point used by the orchestrator
# ---------------------------------------------------------------------------

def get_active_profile_overlay(
    *,
    drift_alert: bool = False,
    rng: random.Random | None = None,
    config_module=None,
) -> tuple[str, _Overlay, ThresholdLearnerState]:
    """Load the persisted state, sample a profile, and return the overlay.

    The caller should hand the returned overlay to functions that read
    config (e.g. :mod:`engine.action_gates`) so the active thresholds
    flow through naturally.  The state is also returned so the caller can
    persist any updates after the run.
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    state = load_state(cfg)
    profile = select_profile(state, drift_alert=drift_alert, rng=rng, config_module=cfg)
    state.last_active_profile = profile
    if drift_alert:
        state.drift_active = True
    overlay = apply_overlay(cfg, profile)
    return profile, overlay, state
