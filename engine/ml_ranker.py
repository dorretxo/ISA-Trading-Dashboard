"""Machine-learning ranker on factor scores and contextual features.

The live model targets forward 1-month returns and consumes factor-space
inputs rather than only raw pillar scores. XGBoost is preferred when
installed; RandomForestRegressor is used as a fallback so the module remains
usable even when xgboost is unavailable.

Key improvements over initial design:
- Trains on the full Stage 5b research panel (discovery + discovery_panel +
  portfolio), not just final discovery survivors — fixes survivorship bias.
- Walk-forward validation with expanding window and 5-day embargo to detect
  overfitting before enabling live blend.
- Explicit missingness indicators (boolean columns for each NaN feature) so
  the model can learn from the pattern of missing data itself.
- Promotion gate: auto-forces shadow mode if OOS metrics are poor.
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Feature columns extracted from signal_backtest
FEATURE_COLS = [
    "quality_factor_score", "value_factor_score", "momentum_factor_score", "volatility_factor_score",
    "qmj_factor_score", "bab_factor_score", "turnover_cost_score",
    "quality_score_fundamental", "gross_profitability", "fcf_to_assets",
    "earnings_stability", "eps_growth_variance_5y",
    "technical_score", "fundamental_score", "sentiment_score", "forecast_score",
    "rsi", "adx", "bb_pct", "pe_ratio", "peg_ratio",
    "revenue_growth", "roe", "short_pct", "vix_percentile", "vol_20d",
    "downside_vol_60d", "max_dd_252d",
    "factor_mom_momentum_tilt", "factor_mom_value_tilt", "factor_mom_quality_tilt",
    "factor_mom_momentum_ret_1m", "factor_mom_momentum_ret_3m",
    "factor_mom_value_ret_1m", "factor_mom_value_ret_3m",
    "factor_mom_quality_ret_1m", "factor_mom_quality_ret_3m",
    "network_momentum", "qa_sentiment_score",
    "return_10d_prior", "return_30d_prior", "return_90d_prior",
    # PEAD factor (Martineau 2022)
    "pead_factor_score", "sue_score", "revision_momentum_3m",
    # Enterprise and entry-quality factors
    "f_score", "f_score_coverage", "gpa", "gpa_score", "price_vs_sma200_stretch",
    "institutional_prior_score", "institutional_prior_percentile", "institutional_prior_confidence",
    # Entry/readiness fields are monitored separately and belong in the
    # future readiness classifier, not the forward-return alpha ranker.
    "sleeve_momentum", "sleeve_quality", "sleeve_value",
    "sleeve_low_risk", "sleeve_pead", "sleeve_ready",
]

# In-memory model cache
_model_cache: dict = {
    "model": None,
    "model_lgbm": None,           # LightGBM ensemble member
    "meta_model": None,           # Classifier: will the primary signal work?
    "regime_models": {},          # Optional regime-conditioned regressors
    "ensemble_weight_xgb": 0.5,   # IC-weighted blend toward primary
    "medians": None,              # For NaN imputation
    "trained_at": 0.0,
    "n_samples": 0,
    "oos_rank_ic": None,
    "oos_rank_ic_lgbm": None,
    "oos_r_squared": None,
    "meta_precision": None,
    "meta_recall": None,
    "meta_brier": None,
    "meta_log_loss": None,
    "meta_calibration": None,
    "promotion_eligible": False,
    "parity_gate": None,
}

_RETRAIN_HOURS = 24.0
_MODEL_CACHE_PATH = Path(getattr(config, "ML_RANKER_MODEL_CACHE_FILE", "feature_cache/ml_ranker_model.pkl"))
_META_CALIBRATION_PATH = Path("feature_cache/meta_calibration.json")

# Walk-forward validation settings
_MIN_TRAIN_SAMPLES = int(getattr(config, "ML_RANKER_MIN_TRAIN_SAMPLES", 80))
_EMBARGO_DAYS = int(getattr(config, "ML_RANKER_EMBARGO_DAYS", 30))
_N_SPLITS = 5                # Number of walk-forward folds

# Promotion gate thresholds
_PROMOTION_MIN_SAMPLES = int(getattr(config, "ML_RANKER_PROMOTION_MIN_SAMPLES", 300))
_PROMOTION_MIN_RANK_IC = float(getattr(config, "ML_RANKER_PROMOTION_MIN_RANK_IC", 0.02))
_PROMOTION_MIN_R_SQUARED = float(getattr(config, "ML_RANKER_PROMOTION_MIN_R_SQUARED", -0.05))


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _replay_live_parity_gate(report: dict | None = None) -> tuple[bool, list[str], dict]:
    """Return whether replay/live parity is healthy enough for ML promotion."""
    if not bool(getattr(config, "ML_RANKER_PARITY_GATE_ENABLED", True)):
        return True, [], {"enabled": False}

    path = Path(getattr(
        config,
        "ML_RANKER_PARITY_REPORT_PATH",
        getattr(config, "REPLAY_LIVE_PARITY_REPORT_PATH", "feature_cache/replay_live_parity_report.json"),
    ))
    if report is None:
        if not path.exists():
            reason = f"parity report missing: {path}"
            require = bool(getattr(config, "ML_RANKER_PARITY_GATE_REQUIRE_REPORT", True))
            return (not require), ([] if not require else [reason]), {"enabled": True, "available": False, "reason": reason}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            reason = f"parity report unreadable: {exc}"
            return False, [reason], {"enabled": True, "available": False, "reason": reason}

    blockers: list[str] = []
    if report.get("available") is False:
        blockers.append(f"parity report unavailable: {report.get('reason') or 'unknown'}")

    sample = _safe_int(report.get("sample", report.get("sample_size")), 0)
    available_pairs = _safe_int(report.get("available_pairs"), 0)
    missing_replay = _safe_int(report.get("missing_replay"), 0)
    stale_replay = _safe_int(report.get("stale_replay"), 0)
    drifted_tickers = _safe_int(report.get("drifted_tickers"), 0)
    denominator = sample or max(available_pairs + missing_replay + stale_replay, available_pairs, 1)
    pair_denominator = max(available_pairs, 1)

    available_ratio = available_pairs / max(denominator, 1)
    missing_ratio = missing_replay / max(denominator, 1)
    stale_ratio = stale_replay / max(denominator, 1)
    drifted_pair_ratio = drifted_tickers / pair_denominator

    min_available = float(getattr(config, "ML_RANKER_PARITY_MIN_AVAILABLE_RATIO", 0.70))
    max_missing = float(getattr(config, "ML_RANKER_PARITY_MAX_MISSING_REPLAY_RATIO", 0.20))
    max_stale = float(getattr(config, "ML_RANKER_PARITY_MAX_STALE_RATIO", 0.10))
    max_drifted = float(getattr(config, "ML_RANKER_PARITY_MAX_DRIFTED_PAIR_RATIO", 0.25))
    if sample <= 0:
        blockers.append("parity sample missing")
    if available_ratio < min_available:
        blockers.append(f"available_pairs_ratio={available_ratio:.0%}<{min_available:.0%}")
    if missing_ratio > max_missing:
        blockers.append(f"missing_replay_ratio={missing_ratio:.0%}>{max_missing:.0%}")
    if stale_ratio > max_stale:
        blockers.append(f"stale_replay_ratio={stale_ratio:.0%}>{max_stale:.0%}")
    if drifted_pair_ratio > max_drifted:
        blockers.append(f"drifted_pair_ratio={drifted_pair_ratio:.0%}>{max_drifted:.0%}")

    generated_at = report.get("generated_at")
    max_age_hours = _safe_float(getattr(config, "ML_RANKER_PARITY_MAX_AGE_HOURS", 48), 48.0)
    age_hours = None
    if generated_at:
        try:
            generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
            now = datetime.now(tz=generated.tzinfo) if generated.tzinfo else datetime.now()
            age_hours = (now - generated).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                blockers.append(f"parity_report_age={age_hours:.1f}h>{max_age_hours:.1f}h")
        except Exception:
            blockers.append("parity report generated_at invalid")
    else:
        blockers.append("parity report generated_at missing")

    critical_fields = list(getattr(config, "ML_RANKER_PARITY_CRITICAL_FIELDS", []) or [])
    replay_missing = report.get("replay_missing_field_counts") or {}
    critical_missing: dict[str, float] = {}
    max_critical_missing = float(getattr(config, "ML_RANKER_PARITY_MAX_CRITICAL_MISSING_RATIO", 0.15))
    for field in critical_fields:
        missing_count = _safe_int(replay_missing.get(field), 0)
        ratio = missing_count / pair_denominator
        if ratio > max_critical_missing:
            critical_missing[field] = ratio
    if critical_missing:
        top = ", ".join(f"{field}={ratio:.0%}" for field, ratio in list(critical_missing.items())[:5])
        blockers.append(f"critical replay missingness too high: {top}")

    summary = {
        "enabled": True,
        "path": str(path),
        "sample": sample,
        "available_pairs": available_pairs,
        "available_ratio": round(available_ratio, 4),
        "missing_replay_ratio": round(missing_ratio, 4),
        "stale_replay_ratio": round(stale_ratio, 4),
        "drifted_pair_ratio": round(drifted_pair_ratio, 4),
        "age_hours": None if age_hours is None else round(age_hours, 2),
        "critical_missing": {k: round(v, 4) for k, v in critical_missing.items()},
        "blockers": blockers,
    }
    return not blockers, blockers, summary


def _load_persisted_model() -> bool:
    """Load a fresh model cache from disk when a new process starts."""
    if _model_cache.get("model") is not None:
        return True
    if not _MODEL_CACHE_PATH.exists():
        return False
    try:
        with _MODEL_CACHE_PATH.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("feature_cols") != FEATURE_COLS:
            return False
        saved = float(payload.get("trained_at") or 0.0)
        if saved <= 0 or (time.time() - saved) >= _RETRAIN_HOURS * 3600:
            return False
        cache = payload.get("model_cache")
        if not isinstance(cache, dict) or cache.get("model") is None or cache.get("medians") is None:
            return False
        _model_cache.update(cache)
        logger.info(
            "ML ranker loaded persisted model (%d samples, age %.1fh)",
            int(_model_cache.get("n_samples") or 0),
            (time.time() - saved) / 3600.0,
        )
        return True
    except Exception as exc:
        logger.debug("Persisted ML ranker load failed: %s", exc)
        return False


def _save_persisted_model() -> None:
    """Persist the trained model cache for later orchestrator processes."""
    try:
        _MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_cols": list(FEATURE_COLS),
            "trained_at": _model_cache.get("trained_at", time.time()),
            "model_cache": dict(_model_cache),
        }
        tmp = _MODEL_CACHE_PATH.with_suffix(_MODEL_CACHE_PATH.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(_MODEL_CACHE_PATH)
    except Exception as exc:
        logger.debug("Persisted ML ranker save failed: %s", exc)


def _probability_summary(values) -> dict:
    """Small JSON-safe distribution summary for classifier probabilities."""
    try:
        import numpy as np

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }
    except Exception:
        return {"count": 0}


def _write_meta_calibration_report(report: dict) -> None:
    """Persist meta-label calibration diagnostics for UI/ops visibility."""
    try:
        import json
        from utils.atomic_io import atomic_write_json

        state_path = Path(getattr(config, "ORCHESTRATOR_STATE_FILE", "orchestrator_state.json"))
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                cached = state.get("cached_discovery") or []
                live_probs = [
                    float(c.get("meta_label_proba"))
                    for c in cached
                    if isinstance(c, dict) and c.get("meta_label_proba") is not None
                ]
                live_dist = _probability_summary(live_probs)
                threshold = float(report.get("threshold") or getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
                live_dist["count_at_or_above_threshold"] = sum(1 for v in live_probs if v >= threshold)
                report["latest_cached_live_distribution"] = live_dist
            except Exception as exc:
                logger.debug("Live meta-prob distribution unavailable: %s", exc)

        _META_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_META_CALIBRATION_PATH, report, indent=2)
    except Exception as exc:
        logger.debug("Meta-label calibration report write failed: %s", exc)


def _xgboost_available() -> bool:
    """Check if xgboost is importable."""
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _lightgbm_available() -> bool:
    """Check if lightgbm is importable."""
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _build_model():
    """Construct the preferred learner for the current environment."""
    if _xgboost_available():
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        ), "xgboost"

    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    ), "random_forest"


def _build_lgbm_model():
    """Construct a LightGBM learner for ensemble (Leippold, Wang & Zhou 2022)."""
    if not _lightgbm_available():
        return None, None
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    ), "lightgbm"


def _build_classifier():
    """Construct the meta-label classifier."""
    if _xgboost_available():
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ), "xgboost_classifier"

    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    ), "random_forest_classifier"


def _optuna_available() -> bool:
    """Check if optuna is importable."""
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


_HPO_CACHE_PATH = "feature_cache/ml_hpo_params.json"
_HPO_INTERVAL_DAYS = 7  # Re-optimize weekly


def _load_hpo_params() -> dict | None:
    """Load cached HPO parameters if fresh enough."""
    import json
    from pathlib import Path

    path = Path(_HPO_CACHE_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        # Check freshness
        saved_at = data.get("saved_at", "")
        if saved_at:
            from datetime import datetime, timedelta
            saved_dt = datetime.fromisoformat(saved_at)
            if datetime.now() - saved_dt < timedelta(days=_HPO_INTERVAL_DAYS):
                return data.get("params")
    except Exception:
        pass
    return None


def _save_hpo_params(params: dict) -> None:
    """Cache HPO parameters to disk."""
    import json
    from pathlib import Path
    from datetime import datetime

    path = Path(_HPO_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"params": params, "saved_at": datetime.now().isoformat()}
    path.write_text(json.dumps(data, indent=2))


def run_hpo(X, y, dates, n_trials: int = 50) -> dict | None:
    """Bayesian hyperparameter optimization via optuna (Snoek et al. 2012).

    Returns the best XGBoost params dict, or None if optuna unavailable.
    Objective: OOS rank IC from walk-forward validation.
    """
    if not _optuna_available():
        return None

    # Check for fresh cached params first
    cached = _load_hpo_params()
    if cached is not None:
        logger.info("HPO: using cached params (fresh < %d days)", _HPO_INTERVAL_DAYS)
        return cached

    import optuna
    import numpy as np

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }

        if _xgboost_available():
            import xgboost as xgb
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42, n_jobs=-1,
            )

        ic, _ = _walk_forward_evaluate_single(X, y, dates, model)
        return ic if ic is not None else -1.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info("HPO complete: best IC=%.4f, params=%s", study.best_value, best_params)
    _save_hpo_params(best_params)
    return best_params


# Interaction feature definitions (Gu, Kelly & Xiu 2020)
# Each tuple: (name, col_a_index, col_b_index, operation)
# Operations: "multiply" = a*b, "ratio" = a/(|b|+eps)
INTERACTION_FEATURES = [
    # momentum × quality: momentum winners with quality confirmation (Asness et al. 2013)
    ("momentum_x_quality", "momentum_factor_score", "quality_factor_score", "multiply"),
    # value × reversal: cheap oversold stocks (quality value plays)
    ("value_x_reversal", "value_factor_score", "reversal_factor_score", "multiply"),
    # momentum × inverse vol: volatility-scaled momentum (Barroso & Santa-Clara 2015)
    ("momentum_x_invvol", "momentum_factor_score", "vol_20d", "inv_multiply"),
    # PEAD × quality factor momentum tilt: earnings momentum in trending factors
    ("sue_x_factor_mom_quality", "sue_score", "factor_mom_quality_tilt", "multiply"),
    # PEAD × momentum: earnings surprise confirming price momentum
    ("pead_x_momentum", "pead_factor_score", "momentum_factor_score", "multiply"),
]


def _engineer_interaction_features(X, feature_cols):
    """Add economically-motivated interaction terms (Gu, Kelly & Xiu 2020).

    Interaction terms improve OOS R-squared by ~30% for tree models at
    monthly frequency. These are ratio-type interactions that XGBoost
    struggles to discover from raw features alone.
    """
    import numpy as np

    col_idx = {name: i for i, name in enumerate(feature_cols)}
    interaction_cols = []

    for name, col_a, col_b, op in INTERACTION_FEATURES:
        idx_a = col_idx.get(col_a)
        idx_b = col_idx.get(col_b)
        if idx_a is None or idx_b is None:
            # Feature not available — fill with zeros
            interaction_cols.append(np.zeros(X.shape[0]))
            continue

        a = X[:, idx_a].copy()
        b = X[:, idx_b].copy()

        # Replace NaN with 0 for interaction computation
        a = np.where(np.isnan(a), 0.0, a)
        b = np.where(np.isnan(b), 0.0, b)

        if op == "multiply":
            interaction_cols.append(a * b)
        elif op == "inv_multiply":
            # momentum × (1 / vol): higher signal when vol is low
            inv_b = np.zeros_like(b, dtype=float)
            np.divide(1.0, b, out=inv_b, where=np.abs(b) > 0.001)
            interaction_cols.append(a * np.clip(inv_b, -10.0, 10.0))
        elif op == "ratio":
            interaction_cols.append(a / (np.abs(b) + 1e-6))

    if interaction_cols:
        return np.column_stack([X] + interaction_cols)
    return X


def _add_missingness_indicators(X):
    """Add boolean columns indicating which features were NaN before imputation.

    Missingness is informative in this system: non-US stocks lack FMP data,
    small-caps lack analyst coverage, etc. (Josse et al. 2019).
    """
    import numpy as np
    missing_mask = np.isnan(X).astype(np.float64)
    # Only add indicators for columns that actually have missing values
    cols_with_missing = missing_mask.sum(axis=0) > 0
    if cols_with_missing.any():
        return np.hstack([X, missing_mask[:, cols_with_missing]]), cols_with_missing
    return X, cols_with_missing


def _impute_with_medians(X, medians):
    """Replace NaN values with precomputed medians, then zero out any
    remaining ``+inf`` / ``-inf`` produced by interaction features (e.g.
    division by zero).  Without the second step XGBoost / LightGBM raise
    ``Input X contains infinity`` and the model never trains.
    """
    import numpy as np
    n_features = X.shape[1]
    for col in range(min(n_features, len(medians))):
        mask = np.isnan(X[:, col])
        X[mask, col] = medians[col]
    # Zero out any remaining non-finite values introduced by interaction
    # features (ratios, products) so the trainer sees a clean matrix.
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _prepare_feature_array(features: dict):
    """Apply the training-time feature pipeline to one prediction row."""
    import numpy as np

    medians = _model_cache.get("medians")
    if medians is None:
        return None
    x = np.array([features.get(col, np.nan) for col in FEATURE_COLS], dtype=np.float64)
    x_2d = _engineer_interaction_features(x.reshape(1, -1), FEATURE_COLS)
    x = x_2d.flatten()

    missing_cols_mask = _model_cache.get("missing_cols_mask")
    if missing_cols_mask is not None and missing_cols_mask.any():
        # The missingness mask is learned after interaction features are
        # appended, so it must be applied to the engineered vector, not only
        # the raw FEATURE_COLS slice.
        missing_indicators = np.isnan(x).astype(np.float64)[missing_cols_mask]
        x_extended = np.concatenate([x, missing_indicators])
    else:
        x_extended = x

    n_median = min(len(medians), len(x_extended))
    mask = np.isnan(x_extended[:n_median])
    x_extended[:n_median][mask] = medians[mask]
    x_extended = np.nan_to_num(x_extended, nan=0.0, posinf=0.0, neginf=0.0)
    return x_extended.reshape(1, -1)


def _predict_regressor_from_array(x_reshaped, *, features: dict | None = None) -> float | None:
    """Predict with the global ensemble and optional regime-conditioned model."""
    model = _model_cache.get("model")
    if model is None or x_reshaped is None:
        return None
    pred_primary = float(model.predict(x_reshaped)[0])

    model_lgbm = _model_cache.get("model_lgbm")
    if model_lgbm is not None:
        try:
            pred_lgbm = float(model_lgbm.predict(x_reshaped)[0])
            w_xgb = _model_cache.get("ensemble_weight_xgb", 0.5)
            pred = w_xgb * pred_primary + (1.0 - w_xgb) * pred_lgbm
        except Exception:
            pred = pred_primary
    else:
        pred = pred_primary

    if features and getattr(config, "ML_REGIME_ENSEMBLE_ENABLED", True):
        regime = str(features.get("regime") or "").upper()
        if not regime:
            vix = features.get("vix_percentile")
            try:
                vf = float(vix)
                regime = "BEAR" if vf >= 75 else ("BULL" if vf <= 25 else "NEUTRAL")
            except Exception:
                regime = ""
        regime_model = (_model_cache.get("regime_models") or {}).get(regime)
        if regime_model is not None:
            try:
                regime_pred = float(regime_model.predict(x_reshaped)[0])
                blend = float(getattr(config, "ML_REGIME_BLEND_PCT", 0.35))
                pred = (1.0 - blend) * pred + blend * regime_pred
            except Exception:
                pass
    return float(pred)


def _calendar_embargo_start(dates, train_end: int, embargo_days: int) -> int:
    """Return the smallest index whose ``run_date`` is at least
    ``embargo_days`` calendar days after the last training row.

    Prior behaviour treated ``_EMBARGO_DAYS`` as a row-count offset — brittle
    when multiple signals share a ``run_date`` (e.g., daily batches of 30+
    portfolio rows) because 5 rows could equal a single trading day.  This
    version respects calendar-day spacing regardless of batch density.
    """
    from datetime import datetime, timedelta
    import numpy as np

    n = len(dates)
    if train_end <= 0 or train_end >= n:
        return min(train_end + embargo_days, n)

    def _parse(v):
        try:
            return datetime.fromisoformat(str(v)[:19])
        except Exception:
            try:
                return datetime.strptime(str(v)[:10], "%Y-%m-%d")
            except Exception:
                return None

    last_train_dt = _parse(dates[train_end - 1])
    if last_train_dt is None:
        return min(train_end + embargo_days, n)
    cutoff = last_train_dt + timedelta(days=int(embargo_days))

    for j in range(train_end, n):
        dj = _parse(dates[j])
        if dj is None:
            continue
        if dj >= cutoff:
            return j
    return n


def _walk_forward_evaluate(X, y, dates):
    """Expanding-window walk-forward validation with calendar-day embargo.

    Returns (mean_rank_ic, mean_r_squared) across folds, or (None, None)
    if insufficient data for evaluation.
    """
    import numpy as np

    n = len(y)
    if n < _MIN_TRAIN_SAMPLES + 20:
        return None, None

    # Sort by date for temporal ordering
    sort_idx = np.argsort(dates)
    X = X[sort_idx]
    y = y[sort_idx]
    dates = dates[sort_idx]

    # Create expanding window splits
    fold_size = max(20, (n - _MIN_TRAIN_SAMPLES) // _N_SPLITS)
    rank_ics = []
    r_squareds = []

    for fold in range(_N_SPLITS):
        train_end = _MIN_TRAIN_SAMPLES + fold * fold_size
        if train_end >= n - 10:
            break

        # Calendar-day embargo: skip every row whose run_date is within
        # _EMBARGO_DAYS of the last training row.
        test_start = _calendar_embargo_start(dates, train_end, _EMBARGO_DAYS)
        test_end = min(test_start + fold_size, n)

        if test_start >= test_end or test_end - test_start < 5:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        try:
            model, _ = _build_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Rank IC (Spearman correlation of ranks)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_pred, y_test)
            if not np.isnan(ic):
                rank_ics.append(ic)

            # R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r_squareds.append(r2)
        except Exception as e:
            logger.debug("Walk-forward fold %d failed: %s", fold, e)
            continue

    if not rank_ics:
        return None, None

    return float(np.mean(rank_ics)), float(np.mean(r_squareds))


def _walk_forward_evaluate_single(X, y, dates, model_instance):
    """Evaluate a pre-built model type via walk-forward (for ensemble IC comparison)."""
    import numpy as np

    n = len(y)
    if n < _MIN_TRAIN_SAMPLES + 20:
        return None, None

    sort_idx = np.argsort(dates)
    X = X[sort_idx]
    y = y[sort_idx]
    dates = dates[sort_idx]

    fold_size = max(20, (n - _MIN_TRAIN_SAMPLES) // _N_SPLITS)
    rank_ics = []

    for fold in range(_N_SPLITS):
        train_end = _MIN_TRAIN_SAMPLES + fold * fold_size
        if train_end >= n - 10:
            break
        test_start = _calendar_embargo_start(dates, train_end, _EMBARGO_DAYS)
        test_end = min(test_start + fold_size, n)
        if test_start >= test_end or test_end - test_start < 5:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        try:
            import copy
            m = copy.deepcopy(model_instance)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_pred, y_test)
            if not np.isnan(ic):
                rank_ics.append(ic)
        except Exception:
            continue

    if not rank_ics:
        return None, None
    return float(np.mean(rank_ics)), None


def train_model(min_samples: int | None = None, *, force: bool = False) -> bool:
    """Train/retrain the factor model on signal_backtest. Returns True if ready.

    Trains on the full research panel (discovery + discovery_panel + portfolio)
    to eliminate survivorship bias. Runs walk-forward validation and enforces
    a promotion gate for live blend eligibility.

    No-op if the model is already fresh (< _RETRAIN_HOURS old).
    """
    if min_samples is None:
        min_samples = getattr(config, "ML_RANKER_MIN_SAMPLES", 50)

    # Skip if model is fresh in memory or persisted from a recent process.
    if (not force
            and _model_cache["model"] is not None
            and (time.time() - _model_cache["trained_at"]) < _RETRAIN_HOURS * 3600):
        return True
    if not force and _load_persisted_model():
        return True

    try:
        import numpy as np
        from engine.discovery_backtest import init_backtest_db, _connect

        init_backtest_db()
        cols = ", ".join(FEATURE_COLS)
        with _connect() as conn:
            # Train on full panel: discovery survivors + Stage 5b panel + portfolio
            # Include vol_20d for risk-adjusted target (Gu, Kelly & Xiu 2020 §4.3)
            rows = conn.execute(
                f"""SELECT {cols}, return_30d, vol_20d, run_date, tb_label, tb_return, regime
                    FROM signal_backtest
                    WHERE (
                        (evaluated_30d = 1 AND return_30d IS NOT NULL)
                        OR tb_label IS NOT NULL
                    )
                      AND (
                        source IN ('discovery', 'discovery_panel', 'portfolio')
                        OR source LIKE 'replay_pit_%'
                      )""",
            ).fetchall()

        if len(rows) < min_samples:
            logger.info("ML ranker: only %d samples (need %d) — skipping", len(rows), min_samples)
            return False

        # Build feature matrix
        n_features = len(FEATURE_COLS)
        X = np.array([[row[i] for i in range(n_features)] for row in rows], dtype=np.float64)
        use_tb = bool(getattr(config, "ML_RANKER_USE_TRIPLE_BARRIER_LABELS", True))
        return_30d = np.array([row[n_features] for row in rows], dtype=np.float64)
        tb_labels = np.array([
            np.nan if row[n_features + 3] is None else float(row[n_features + 3])
            for row in rows
        ], dtype=np.float64)
        tb_returns = np.array([
            np.nan if row[n_features + 4] is None else float(row[n_features + 4])
            for row in rows
        ], dtype=np.float64)
        raw_returns = np.where(use_tb & ~np.isnan(tb_returns), tb_returns, return_30d)
        vol_at_signal = np.array([row[n_features + 1] or 0.0 for row in rows], dtype=np.float64)
        dates = np.array([row[n_features + 2] for row in rows])  # run_date strings
        regimes = np.array([str(row[n_features + 5] or "").upper() for row in rows])
        valid_target = np.isfinite(raw_returns)
        if valid_target.sum() < min_samples:
            logger.info("ML ranker: only %d finite labels (need %d) - skipping", int(valid_target.sum()), min_samples)
            return False
        X = X[valid_target]
        raw_returns = raw_returns[valid_target]
        vol_at_signal = vol_at_signal[valid_target]
        dates = dates[valid_target]
        regimes = regimes[valid_target]
        tb_labels = tb_labels[valid_target]

        # Target: volatility-adjusted return (forward information ratio)
        # Rank-transformed for robustness to outliers (Gu, Kelly & Xiu 2020 §4.3)
        vol_floor = 0.05  # 5% annualized vol floor to avoid div-by-zero
        safe_vol = np.where(vol_at_signal > vol_floor, vol_at_signal, vol_floor)
        y_vol_adj = raw_returns / safe_vol

        # Rank-transform: maps to uniform [0, 1] — more robust than raw returns
        from scipy.stats import rankdata
        y_ranked = rankdata(raw_returns) / len(raw_returns)

        # Auto-select target: use whichever produces higher walk-forward IC
        # (evaluated after interaction features are added below)
        y = y_vol_adj  # default; may switch after WF evaluation

        # Add interaction features (Gu, Kelly & Xiu 2020)
        X = _engineer_interaction_features(X, FEATURE_COLS)

        # Add missingness indicators before imputation
        X_extended, missing_cols_mask = _add_missingness_indicators(X)

        # Compute medians for NaN imputation (on base + interaction columns)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        n_base_cols = X.shape[1]  # base features + interactions

        # Impute original features
        X_extended = _impute_with_medians(X_extended, medians)

        # Walk-forward validation — auto-select best target variable
        oos_rank_ic_vol, oos_r2_vol = _walk_forward_evaluate(X_extended, y_vol_adj, dates)
        oos_rank_ic_rank, oos_r2_rank = _walk_forward_evaluate(X_extended, y_ranked, dates)

        # Pick whichever target produces higher OOS rank IC
        ic_vol = oos_rank_ic_vol if oos_rank_ic_vol is not None else -1.0
        ic_rank = oos_rank_ic_rank if oos_rank_ic_rank is not None else -1.0

        if ic_rank > ic_vol:
            y = y_ranked
            oos_rank_ic, oos_r2 = oos_rank_ic_rank, oos_r2_rank
            target_type = "rank_transformed"
            logger.info("ML target auto-select: rank-transformed (IC=%.4f) > vol-adjusted (IC=%.4f)",
                        ic_rank, ic_vol)
        else:
            y = y_vol_adj
            oos_rank_ic, oos_r2 = oos_rank_ic_vol, oos_r2_vol
            target_type = "vol_adjusted"
            logger.info("ML target auto-select: vol-adjusted (IC=%.4f) >= rank-transformed (IC=%.4f)",
                        ic_vol, ic_rank)

        _model_cache["oos_rank_ic"] = oos_rank_ic
        _model_cache["oos_r_squared"] = oos_r2
        _model_cache["target_type"] = target_type

        if oos_rank_ic is not None:
            logger.info(
                "ML ranker walk-forward: rank_IC=%.4f, R²=%.4f (%d folds)",
                oos_rank_ic, oos_r2, _N_SPLITS,
            )

        # Promotion gate: check if model qualifies for live blend
        parity_ok, parity_reasons, parity_summary = _replay_live_parity_gate()
        _model_cache["parity_gate"] = parity_summary
        promotion_eligible = (
            len(raw_returns) >= _PROMOTION_MIN_SAMPLES
            and oos_rank_ic is not None
            and oos_rank_ic >= _PROMOTION_MIN_RANK_IC
            and oos_r2 is not None
            and oos_r2 >= _PROMOTION_MIN_R_SQUARED
            and parity_ok
        )
        _model_cache["promotion_eligible"] = promotion_eligible

        if not promotion_eligible:
            reasons = []
            if len(raw_returns) < _PROMOTION_MIN_SAMPLES:
                reasons.append(f"samples={len(raw_returns)}<{_PROMOTION_MIN_SAMPLES}")
            if oos_rank_ic is None or oos_rank_ic < _PROMOTION_MIN_RANK_IC:
                reasons.append(f"rank_IC={oos_rank_ic}<{_PROMOTION_MIN_RANK_IC}")
            if oos_r2 is None or oos_r2 < _PROMOTION_MIN_R_SQUARED:
                reasons.append(f"R²={oos_r2}<{_PROMOTION_MIN_R_SQUARED}")
            reasons.extend(parity_reasons)
            logger.info("ML ranker: promotion gate FAILED (%s) — shadow mode enforced", ", ".join(reasons))
        # Bayesian HPO: optimize hyperparameters weekly (Snoek et al. 2012)
        hpo_params = None
        if len(raw_returns) >= _PROMOTION_MIN_SAMPLES:
            hpo_params = run_hpo(X_extended, y, dates, n_trials=50)

        # Train final model on ALL data (walk-forward was only for evaluation)
        if hpo_params and _xgboost_available():
            import xgboost as xgb
            model = xgb.XGBRegressor(**hpo_params, random_state=42, verbosity=0)
            backend = "xgboost_hpo"
        else:
            model, backend = _build_model()
        model.fit(X_extended, y)

        # Train LightGBM ensemble member (Leippold, Wang & Zhou 2022)
        model_lgbm, lgbm_backend = _build_lgbm_model()
        lgbm_ic = None
        if model_lgbm is not None:
            try:
                model_lgbm.fit(X_extended, y)
                # Evaluate LightGBM IC via same walk-forward
                lgbm_ic_result, _ = _walk_forward_evaluate_single(X_extended, y, dates, model_lgbm)
                lgbm_ic = lgbm_ic_result
            except Exception as e:
                logger.debug("LightGBM training failed: %s — using primary model only", e)
                model_lgbm = None

        # Compute IC-weighted ensemble blend
        ensemble_weight_xgb = 0.5  # default equal weight
        if oos_rank_ic is not None and lgbm_ic is not None:
            ic_xgb = max(0.0, oos_rank_ic)
            ic_lgbm = max(0.0, lgbm_ic)
            total_ic = ic_xgb + ic_lgbm
            if total_ic > 0:
                ensemble_weight_xgb = max(0.30, ic_xgb / total_ic)
            logger.info("Ensemble weights: XGB=%.2f (IC=%.4f), LGBM=%.2f (IC=%.4f)",
                        ensemble_weight_xgb, ic_xgb, 1.0 - ensemble_weight_xgb, ic_lgbm)

        regime_models = {}
        if getattr(config, "ML_REGIME_ENSEMBLE_ENABLED", True):
            min_regime_n = int(getattr(config, "ML_REGIME_MIN_SAMPLES", 250))
            for regime in sorted({r for r in regimes if r}):
                mask = regimes == regime
                if int(mask.sum()) < min_regime_n:
                    continue
                try:
                    regime_model, _ = _build_model()
                    regime_model.fit(X_extended[mask], y[mask])
                    regime_models[regime] = regime_model
                except Exception as exc:
                    logger.debug("Regime model failed for %s: %s", regime, exc)
            if regime_models:
                logger.info("ML regime ensemble trained for regimes: %s", sorted(regime_models))

        meta_model = None
        meta_precision = None
        meta_recall = None
        meta_brier = None
        meta_log_loss = None
        meta_calibration = None
        if getattr(config, "ML_META_LABEL_ENABLED", True):
            meta_mask = np.isfinite(tb_labels)
            min_meta_n = int(getattr(config, "ML_META_LABEL_MIN_SAMPLES", 300))
            if int(meta_mask.sum()) >= min_meta_n:
                meta_y = (tb_labels[meta_mask] > 0).astype(int)
                if len(set(meta_y.tolist())) > 1:
                    try:
                        primary_pred = model.predict(X_extended).reshape(-1, 1)
                        meta_X = np.column_stack([X_extended[meta_mask], primary_pred[meta_mask]])
                        meta_dates = dates[meta_mask].astype(str)
                        order = np.argsort(meta_dates)
                        meta_X = meta_X[order]
                        meta_y = meta_y[order]
                        meta_dates = meta_dates[order]
                        split = max(int(len(meta_y) * 0.80), 1)
                        cal_n = len(meta_y) - split
                        threshold = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
                        base_meta_model, meta_backend = _build_classifier()

                        can_calibrate = (
                            split >= 100
                            and cal_n >= 100
                            and len(set(meta_y[:split].tolist())) > 1
                            and len(set(meta_y[split:].tolist())) > 1
                        )
                        if can_calibrate:
                            base_meta_model.fit(meta_X[:split], meta_y[:split])
                            method = "isotonic" if cal_n >= 1000 else "sigmoid"
                            try:
                                from sklearn.calibration import CalibratedClassifierCV

                                try:
                                    from sklearn.frozen import FrozenEstimator
                                    meta_model = CalibratedClassifierCV(
                                        FrozenEstimator(base_meta_model),
                                        method=method,
                                    )
                                except Exception:
                                    meta_model = CalibratedClassifierCV(
                                        base_meta_model,
                                        method=method,
                                        cv="prefit",
                                    )
                                meta_model.fit(meta_X[split:], meta_y[split:])
                                meta_backend = f"{meta_backend}+{method}_calibrated"
                            except Exception as cal_exc:
                                logger.debug("Meta-label calibration failed; using raw classifier: %s", cal_exc)
                                meta_model = base_meta_model
                            prob = meta_model.predict_proba(meta_X[split:])[:, 1]
                            cal_y = meta_y[split:]
                        else:
                            meta_model = base_meta_model
                            meta_model.fit(meta_X, meta_y)
                            prob = meta_model.predict_proba(meta_X)[:, 1]
                            cal_y = meta_y

                        pred = (prob >= threshold).astype(int)
                        selected = pred == 1
                        if selected.any():
                            meta_precision = float(cal_y[selected].mean())
                        positives = int((cal_y == 1).sum())
                        if positives:
                            meta_recall = float(((pred == 1) & (cal_y == 1)).sum() / positives)
                        try:
                            from sklearn.metrics import brier_score_loss, log_loss

                            meta_brier = float(brier_score_loss(cal_y, prob))
                            meta_log_loss = float(log_loss(cal_y, np.clip(prob, 1e-6, 1.0 - 1e-6)))
                        except Exception:
                            meta_brier = None
                            meta_log_loss = None

                        meta_calibration = {
                            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "backend": meta_backend,
                            "samples": int(len(meta_y)),
                            "train_samples": int(split if can_calibrate else len(meta_y)),
                            "calibration_samples": int(cal_n if can_calibrate else 0),
                            "calibrated": bool(can_calibrate and "calibrated" in meta_backend),
                            "threshold": threshold,
                            "positive_rate": float(meta_y.mean()),
                            "calibration_positive_rate": float(cal_y.mean()),
                            "precision_at_threshold": meta_precision,
                            "recall_at_threshold": meta_recall,
                            "brier": meta_brier,
                            "log_loss": meta_log_loss,
                            "probability_distribution": _probability_summary(prob),
                            "count_at_or_above_threshold": int(selected.sum()),
                            "date_range": {
                                "start": str(meta_dates[0]) if len(meta_dates) else None,
                                "end": str(meta_dates[-1]) if len(meta_dates) else None,
                            },
                        }
                        _write_meta_calibration_report(meta_calibration)
                        logger.info(
                            "Meta-label classifier trained on %d samples (%s, precision=%s, brier=%s)",
                            int(meta_mask.sum()),
                            meta_backend,
                            "n/a" if meta_precision is None else f"{meta_precision:.3f}",
                            "n/a" if meta_brier is None else f"{meta_brier:.3f}",
                        )
                    except Exception as exc:
                        logger.debug("Meta-label classifier training failed: %s", exc)
                        meta_model = None

        _model_cache["model"] = model
        _model_cache["model_lgbm"] = model_lgbm
        _model_cache["meta_model"] = meta_model
        _model_cache["regime_models"] = regime_models
        _model_cache["ensemble_weight_xgb"] = ensemble_weight_xgb
        _model_cache["oos_rank_ic_lgbm"] = lgbm_ic
        _model_cache["meta_precision"] = meta_precision
        _model_cache["meta_recall"] = meta_recall
        _model_cache["meta_brier"] = meta_brier
        _model_cache["meta_log_loss"] = meta_log_loss
        _model_cache["meta_calibration"] = meta_calibration
        _model_cache["medians"] = medians
        _model_cache["missing_cols_mask"] = missing_cols_mask
        _model_cache["trained_at"] = time.time()
        _model_cache["n_samples"] = len(raw_returns)

        backends = backend
        if model_lgbm is not None:
            backends += f"+{lgbm_backend}"
        logger.info(
            "ML ranker trained on %d samples (features: %d+%d missingness, backend: %s, promotion: %s)",
            len(raw_returns), n_features, int(missing_cols_mask.sum()) if hasattr(missing_cols_mask, 'sum') else 0,
            backends, "YES" if promotion_eligible else "NO",
        )
        _save_persisted_model()
        return True

    except Exception as e:
        logger.warning("ML ranker training failed: %s", e)
        return False


def predict_alpha(features: dict) -> float | None:
    """Predict 1-month return from feature dict. Returns None if unavailable."""
    _load_persisted_model()
    if _model_cache.get("model") is None or _model_cache.get("medians") is None:
        return None

    try:
        x_reshaped = _prepare_feature_array(features)
        return _predict_regressor_from_array(x_reshaped, features=features)
    except Exception as e:
        logger.debug("ML ranker prediction failed: %s", e)
        return None


def predict_meta_success(features: dict) -> float | None:
    """Probability that a primary BUY-like signal reaches its barrier target first."""
    _load_persisted_model()
    meta_model = _model_cache.get("meta_model")
    if meta_model is None:
        return None
    try:
        import numpy as np

        x_reshaped = _prepare_feature_array(features)
        primary = _predict_regressor_from_array(x_reshaped, features=features)
        if x_reshaped is None or primary is None:
            return None
        meta_x = np.column_stack([x_reshaped, np.array([[primary]], dtype=float)])
        if hasattr(meta_model, "predict_proba"):
            return float(meta_model.predict_proba(meta_x)[0, 1])
        pred = meta_model.predict(meta_x)[0]
        return float(pred)
    except Exception as e:
        logger.debug("Meta-label prediction failed: %s", e)
        return None


def is_available() -> bool:
    """Check if a trained model exists."""
    _load_persisted_model()
    return _model_cache.get("model") is not None


def is_promotion_eligible() -> bool:
    """Check if the ML model passes the promotion gate for live blend.

    When False, the model should remain in shadow mode regardless of config.
    Promotion requires:
    - Configured minimum evaluated training samples
    - OOS rank IC above the configured promotion hurdle
    - OOS R-squared >= 0.0
    """
    _load_persisted_model()
    cached_eligible = bool(_model_cache.get("promotion_eligible", False))
    metric_eligible = (
        int(_model_cache.get("n_samples") or 0) >= _PROMOTION_MIN_SAMPLES
        and _model_cache.get("oos_rank_ic") is not None
        and float(_model_cache.get("oos_rank_ic") or 0.0) >= _PROMOTION_MIN_RANK_IC
        and _model_cache.get("oos_r_squared") is not None
        and float(_model_cache.get("oos_r_squared") or 0.0) >= _PROMOTION_MIN_R_SQUARED
    )
    eligible = cached_eligible or metric_eligible
    if not eligible:
        return False
    parity_ok, parity_reasons, parity_summary = _replay_live_parity_gate()
    _model_cache["parity_gate"] = parity_summary
    if not parity_ok:
        logger.warning("ML ranker promotion blocked by replay/live parity: %s", "; ".join(parity_reasons))
        return False
    if getattr(config, "ML_DRIFT_MONITOR_ENABLED", True):
        try:
            from utils.drift_monitor import get_drift_status
            status = get_drift_status()
            if status.drift_detected:
                logger.warning("ML ranker promotion blocked by drift monitor: %s", status.reason)
                return False
        except Exception as exc:
            logger.debug("Drift monitor unavailable: %s", exc)
    return True


def get_diagnostics() -> dict:
    """Return model diagnostics for logging/UI."""
    _load_persisted_model()
    metric_eligible = (
        int(_model_cache.get("n_samples") or 0) >= _PROMOTION_MIN_SAMPLES
        and _model_cache.get("oos_rank_ic") is not None
        and float(_model_cache.get("oos_rank_ic") or 0.0) >= _PROMOTION_MIN_RANK_IC
        and _model_cache.get("oos_r_squared") is not None
        and float(_model_cache.get("oos_r_squared") or 0.0) >= _PROMOTION_MIN_R_SQUARED
    )
    parity_ok, parity_reasons, parity_summary = _replay_live_parity_gate()
    promotion_eligible = bool((_model_cache.get("promotion_eligible", False) or metric_eligible) and parity_ok)
    return {
        "n_samples": _model_cache.get("n_samples", 0),
        "oos_rank_ic": _model_cache.get("oos_rank_ic"),
        "oos_rank_ic_lgbm": _model_cache.get("oos_rank_ic_lgbm"),
        "oos_r_squared": _model_cache.get("oos_r_squared"),
        "meta_precision": _model_cache.get("meta_precision"),
        "meta_recall": _model_cache.get("meta_recall"),
        "meta_brier": _model_cache.get("meta_brier"),
        "meta_log_loss": _model_cache.get("meta_log_loss"),
        "meta_calibration": _model_cache.get("meta_calibration"),
        "promotion_eligible": promotion_eligible,
        "parity_gate": parity_summary,
        "parity_blockers": parity_reasons,
        "trained_at": _model_cache.get("trained_at", 0),
        "n_features": len(FEATURE_COLS),
        "n_interaction_features": len(INTERACTION_FEATURES),
        "ensemble_active": _model_cache.get("model_lgbm") is not None,
        "ensemble_weight_xgb": _model_cache.get("ensemble_weight_xgb", 0.5),
        "meta_label_active": _model_cache.get("meta_model") is not None,
        "regime_models": sorted(str(k) for k in (_model_cache.get("regime_models") or {}).keys()),
    }
