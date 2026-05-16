"""Configuration for the trading dashboard."""

import os
from pathlib import Path

# Load .env file if present (secrets live there, not in this file)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

# Scoring weights (must sum to 1.0) — forecast capped at 30% per Gemini review
WEIGHTS = {
    "technical": 0.30,       # Trend/momentum — IC 0.03-0.05 at 90d (Moskowitz et al. 2012)
    "fundamental": 0.40,     # Value+quality — IC 0.04-0.07 at 90d (Asness et al. 2013)
    "sentiment": 0.08,       # News — IC ~0.005 at 90d (Tetlock 2007); confirmatory only
    "forecast": 0.22,        # Statistical ensemble — IC 0.01-0.03 (Rapach & Zhou 2013)
}

# Forecast-to-score conversion: maps predicted % change to a -1..+1 score
# A ±10% predicted move maps to ±1.0 score (linear, capped)
FORECAST_SCORE_SCALE = 10.0  # % change that maps to score of 1.0

# Technical thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_SIGNAL_THRESHOLD = 0  # Bearish when MACD < signal line

# Stop-loss / take-profit settings
TRAILING_STOP_PCT = 0.08  # 8% trailing stop from recent high (fallback default)
ATR_MULTIPLIER = 2.0  # ATR-based stop-loss multiplier (fallback default)
RISK_REWARD_RATIO = 2.0  # Target profit = risk * this ratio

# Volatility-adjusted stops — dynamic multipliers based on realized vol percentile
VOL_LOOKBACK = 30          # Days for realized volatility calculation
ATR_MULT_LOW_VOL = 1.5     # ATR multiplier when vol < 20th percentile
ATR_MULT_HIGH_VOL = 3.0    # ATR multiplier when vol > 80th percentile
TRAIL_PCT_LOW_VOL = 0.06   # 6% trailing when calm
TRAIL_PCT_HIGH_VOL = 0.12  # 12% trailing when volatile

# Scoring thresholds for actions (from highest to lowest)
SCORE_STRONG_BUY_THRESHOLD = 0.40  # Above this = STRONG BUY (top decile)
SCORE_BUY_THRESHOLD = 0.20         # Above this = BUY (top quartile)
SCORE_KEEP_THRESHOLD = -0.25       # Above this = KEEP
SCORE_SELL_THRESHOLD = -0.50       # Above this = SELL, below = STRONG SELL

# --- Exit-action smoother: hysteresis bands + persistence gate (EXIT side only) ---
# Replaces scalar threshold crossings on the KEEP↔SELL↔STRONG SELL axis with a
# two-threshold no-trade region (Constantinides 1986 JPE; Davis-Norman 1990
# SIAM J. Control), confirmed by a fixed-window persistence rule (Wald 1947
# SPRT, simplified count form), with band width scaled by realised σ_score and
# VIX percentile (Kaminski-Lo 2014).  CUSUM-confirmed urgent signals from
# exit_engine override the band (Page 1954, Lorden 1971-optimal).  BUY /
# STRONG BUY pass through untouched — STRONG BUY has its own scorecard above.
EXIT_SMOOTHER_ENABLED = True                  # Master toggle (set False to bypass)
EXIT_SMOOTHER_BAND_MIN = 0.05                 # δ_min — floor on dead-zone half-width
EXIT_SMOOTHER_BAND_K_VOL = 1.5                # band = K_VOL · σ_score (capped by min)
EXIT_SMOOTHER_VIX_SCALE = 0.50                # +50% widening at VIX_pct=100
EXIT_SMOOTHER_VOL_LOOKBACK = 30               # Days of score history for σ_score
EXIT_SMOOTHER_VOL_FALLBACK = 0.10             # σ_score default when <10 obs
EXIT_SMOOTHER_PERSISTENCE_N = 5               # Window size for N-of-M test
EXIT_SMOOTHER_PERSISTENCE_M = 3               # SELL needs ≥M of N down-days
EXIT_SMOOTHER_STRONG_SELL_M = 4               # STRONG SELL needs ≥M_strong of N
EXIT_SMOOTHER_RECOVERY_RATIO = 0.5            # δ_up = δ_down · ratio (recovery boundary)
EXIT_SMOOTHER_CUSUM_OVERRIDE_MIN = 0.60       # exit_score floor for CUSUM override

# Discovery action labels. Percentile mode turns the day's best risk-screened
# names into BUY/STRONG BUY candidates while absolute floors prevent weak days
# from being promoted.
USE_PERCENTILE_ACTIONS = True
PERCENTILE_STRONG_BUY_PCT = 0.05
PERCENTILE_BUY_PCT = 0.15
PERCENTILE_NEUTRAL_PCT = 0.50
PERCENTILE_BEAR_STRONG_BUY_PCT = 0.03
PERCENTILE_BEAR_BUY_PCT = 0.10
PERCENTILE_STRONG_BUY_MIN_AGG = 0.0
PERCENTILE_BUY_MIN_AGG = -0.05
PERCENTILE_STRONG_BUY_MIN_PRIOR_PCT = 0.85

# Data settings
PRICE_HISTORY_DAYS = 730  # 2 years of history for technicals + backtest training
NEWS_HEADLINE_COUNT = 5

# Portfolio file path
PORTFOLIO_FILE = "portfolio.json"

# Forecast settings
FORECAST_HORIZON_DAYS = 5  # Trading days ahead to predict
FORECAST_ROLLING_WINDOW = 100  # Past predictions for rolling MAE (larger window for 24mo backtest)
FORECAST_MIN_HISTORY = 5  # Min evaluated predictions before adapting weights
FORECAST_STORE_FILE = "forecast_store.json"

# Expert model parameters
EXPERT_LR_LOOKBACK = 30  # Linear regression lookback days
EXPERT_REVERSION_SPEED = 0.5  # Mean reversion speed toward SMA-50
EXPERT_MOMENTUM_WINDOW = 10  # Rate of change window
EXPERT_ATR_CONFIDENCE_MULT = 1.5  # ATR band multiplier for volatility expert
EXPERT_CONFIDENCE_Z = 1.28  # Z-score for ~80% confidence interval

# Multi-horizon forecast — short horizon for technical, long for fundamental/macro
FORECAST_HORIZON_LONG = 63           # Trading days for fundamental/macro (~3 months)
FORECAST_SCORE_SCALE_LONG = 15.0     # % change that maps to score of 1.0 at long horizon

# VIX regime detection — tilts pillar weights based on market conditions
VIX_HISTORY_DAYS = 365               # 1 year of VIX data for percentile ranking
VIX_PERCENTILE_BULL = 25             # Below this percentile = BULL regime
VIX_PERCENTILE_BEAR = 75             # Above this percentile = BEAR regime
REGIME_TILT_PCT = 0.05               # ±5% weight tilt per regime

# Position sizing — inverse-volatility weighting
MAX_POSITION_WEIGHT = 0.25           # 25% max per position

# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio optimiser — multi-method ensemble
# Markowitz 1952; Jorion 1986 (James-Stein); Jagannathan & Ma 2003 (MinVar);
# Maillard-Roncalli-Teïletche 2010 (Risk Parity); Black & Litterman 1992;
# He & Litterman 1999; López de Prado 2016 (HRP); Gerber et al. 2022;
# Longin & Solnik 2001 (bear-regime correlation inflation).
# ═══════════════════════════════════════════════════════════════════════════════
OPTIMIZER_MU_V2 = True                            # Use calibrated μ with James-Stein shrinkage
OPTIMIZER_MU_WEIGHTS = (0.70, 0.30)               # (w_moe, w_score) blend for μ
OPTIMIZER_MU_SHRINKAGE_MAX = 0.40                 # Cap on James-Stein α
OPTIMIZER_COV_METHOD = "blend"                    # "lw_ewma" | "gerber" | "blend"
OPTIMIZER_BEAR_CORR_INFLATE = 0.20                # Off-diagonal corr inflate in BEAR regime
OPTIMIZER_ENSEMBLE_METHODS = [
    "mean_variance", "min_variance", "risk_parity", "black_litterman", "hrp",
]
OPTIMIZER_ENSEMBLE_TEMPERATURE = 0.5              # Softmax τ for history-weighted combiner
OPTIMIZER_METHOD_WEIGHT_MIN = 0.05                # Floor per method in the ensemble
OPTIMIZER_METHOD_WEIGHT_MAX = 0.50                # Cap per method in the ensemble
OPTIMIZER_BL_TAU = 0.05                           # Black-Litterman prior scaling

# --- Cold-start STRONG BUY scorecard (Hull 2018 / Patton-Timmermann 2009) ---
# Replaces the conjunctive 10-predicate AND-gate with a continuous
# cross-sectional weighted score.  Trap-safeguard and V2-reject vetoes still
# apply (they exclude candidates from the scoring cohort entirely).
STRONG_BUY_SCORECARD_ENABLED = True                # Master toggle for parallel promotion path
STRONG_BUY_SCORECARD_TOP_PCT = 0.05                # Top-N% by sb_score eligible for STRONG BUY
STRONG_BUY_SCORECARD_MIN_Z = 0.50                  # Minimum sb_score (z-score scale) to promote
SB_SCORECARD_W_AGG = 0.40                          # weight: aggregate_score (z)
SB_SCORECARD_W_FSCORE = 0.15                       # weight: f_score / 9 (z, coverage-gated)
SB_SCORECARD_W_GPA = 0.10                          # weight: gross profitability (z)
SB_SCORECARD_W_META = 0.10                         # weight: meta_success_prob (raw)
SB_SCORECARD_W_PRIOR = 0.10                        # weight: institutional prior percentile (z)
SB_SCORECARD_W_MOM = 0.05                          # weight: momentum_factor_score (z)
SB_SCORECARD_W_QUAL = 0.05                         # weight: quality_factor_score (z)
SB_SCORECARD_W_VAL = 0.05                          # weight: value_factor_score (z)
SB_SCORECARD_STRETCH_PENALTY = 0.20                # penalty per unit above stretch threshold
SB_SCORECARD_STRETCH_THRESHOLD = 0.50              # SMA200 stretch above which penalty kicks in
# Post-action-gates "high-conviction" override: restore STRONG BUY when the
# additive scorecard evidence is overwhelming, even if a single Tier gate
# flagged a single-axis problem (F-score, EV/EBIT, entry-stance).
STRONG_BUY_SCORECARD_OVERRIDE_ENABLED = True
STRONG_BUY_SCORECARD_OVERRIDE_Z = 1.0              # min sb_score (z) to override gate cap
STRONG_BUY_SCORECARD_OVERRIDE_TOP_N = 5            # max candidates restored per run

# --- Conformal prediction (Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2021)
CONFORMAL_ENABLED = True                           # Compute conformal p-value per candidate
CONFORMAL_CALIBRATION_DAYS = 365                   # Look-back window for calibration set
CONFORMAL_MIN_CALIBRATION_N = 50                   # Min mature 90d returns to enable
CONFORMAL_K_IC_FALLBACK = 0.10                     # Fallback k_ic when Fama-MacBeth unavailable

# --- Multi-horizon IC weighting (Lo & MacKinlay 1990, Hou-Xue-Zhang 2017) ---
# Combines 5d/10d/30d/60d/90d IC via information half-life so the system
# responds to regime change in days (not quarters) while anchoring on 90d.
ADAPTIVE_WEIGHTS_MULTI_HORIZON_ENABLED = True
MULTI_HORIZON_IC_WEIGHTS = {                       # weight per horizon
    "5d": 0.20,
    "10d": 0.30,
    "30d": 0.30,
    "60d": 0.10,
    "90d": 0.10,
}
MULTI_HORIZON_IC_MIN_SAMPLES = 20                  # Per-horizon sample floor
# Pillar-IC floor: 0.05 (legacy) flattened weights to equal because realistic
# equity-factor ICs are 0.01-0.03 (Grinold-Kahn 2000).  Lower floor preserves
# differentiation while still preventing single-pillar zero-out.
ADAPTIVE_WEIGHTS_IC_FLOOR = 0.01

# --- Active labelling (Settles 2010, Cohn-Atlas-Ladner 1994) ---
# Records candidates whose sb_score lies just-below STRONG BUY override so
# their realised 90d returns become high-information-gain training samples
# for boundary refinement.  Uncertainty sampling near the decision boundary
# is 3-10× more label-efficient than random.
ACTIVE_LABEL_ENABLED = True
ACTIVE_LABEL_SB_SCORE_LOW = 0.70                   # lower bound of borderline band
ACTIVE_LABEL_SB_SCORE_HIGH = 1.00                  # equals STRONG_BUY_SCORECARD_OVERRIDE_Z
ACTIVE_LABEL_MAX_RECORDS = 25                      # cap per run (sector-stratified)

# --- Enterprise factor bundle (roadmap items #1-#10) -----------------------
ENTERPRISE_FACTORS_ENABLED = True                  # Master toggle
EV_EBIT_YIELD_ANCHOR = 0.10                        # 10% EBIT/EV is median
F_SCORE_GATE_ENABLED = True                       # Hard gate on F-score ≥ 6
F_SCORE_GATE_MIN = 6                               # Piotroski 2000 original cut
F_SCORE_GATE_MIN_COVERAGE = 6 / 9                  # Must have computed >=6 of 9 components
FACTOR_SECTOR_NEUTRAL = True                       # Apply sector-neutral z-scoring
FACTOR_WINSOR_SIGMA = 3.0                          # ±3σ winsorisation
FACTOR_JS_SHRINKAGE_MAX = 0.40                     # James-Stein max shrinkage
REGIME_FACTOR_TILT_ENABLED = True                  # Apply BULL/NEUTRAL/BEAR tilts
ORTHOGONALISE_SENTIMENT = True                     # Residualise sentiment vs factors
PIT_FUNDAMENTAL_LAG_DAYS = 45                      # SEC 10-Q reporting lag
RESIDUAL_MOMENTUM_TARGET_VOL = 0.12                # Barroso-Santa-Clara target

# Discovery gates v2.  Broad gate decisions are shadowed until the diagnostics
# table has enough matured signals to calibrate thresholds.  The narrow trap
# safeguard is active because it only blocks commodity-cycle chase setups.
DISCOVERY_GATES_V2_ENABLED = False
DISCOVERY_GATES_V2_SHADOW = True
DISCOVERY_GATES_V2_F_SCORE_MIN = 5
DISCOVERY_GATES_V2_F_SCORE_MIN_COVERAGE = F_SCORE_GATE_MIN_COVERAGE
DISCOVERY_GATES_V2_GPA_MIN = 0.15
DISCOVERY_GATES_V2_MAX_STRETCH = 0.50
DISCOVERY_GATES_V2_REJECT_RANK_MULTIPLIER = 0.20
DISCOVERY_TRAP_SAFEGUARD_ENABLED = True
DISCOVERY_TRAP_SAFEGUARD_MIN_STRETCH = 0.50
DISCOVERY_TRAP_SAFEGUARD_F_SCORE_MAX = 3
DISCOVERY_TRAP_SAFEGUARD_F_SCORE_MIN_COVERAGE = F_SCORE_GATE_MIN_COVERAGE
DISCOVERY_TRAP_SAFEGUARD_GPA_MAX = 0.15
DISCOVERY_TRAP_SAFEGUARD_RANK_MULTIPLIER = 0.25

# Institutional-prior cold start.  These are literature-backed priors used
# before the local paper-trading sample is large enough to trust as a primary
# selector.  Local learning may only blend toward these values, not replace them.
INSTITUTIONAL_PRIOR_ENABLED = True
INSTITUTIONAL_PRIOR_ALPHA_WEIGHT = 0.0
INSTITUTIONAL_PRIOR_RANK_WEIGHT = 0.15
INSTITUTIONAL_PRIOR_MAX_TOTAL_WEIGHT = 0.20
INSTITUTIONAL_PRIOR_STRONG_BUY_PERCENTILE = 0.90
INSTITUTIONAL_PRIOR_MIN_CONFIDENCE = 0.55
INSTITUTIONAL_PRIOR_MIN_COVERAGE = 0.45
INSTITUTIONAL_PRIOR_DYNAMIC_COMPONENTS = True
INSTITUTIONAL_PRIOR_DYNAMIC_BLEND = 0.50
INSTITUTIONAL_PRIOR_SECTOR_COMPONENT_MIN_N = 8
INSTITUTIONAL_PRIOR_TURNOVER_COST_ENABLED = True

# Signal-backtest evaluation. The batched path downloads each ticker window
# once and slices all matured signal horizons locally, avoiding thousands of
# per-signal yfinance calls.
SIGNAL_BACKTEST_BATCHED_EVALUATOR_ENABLED = True
SIGNAL_BACKTEST_BATCH_SIZE = 75
SIGNAL_BACKTEST_BATCH_HORIZONS = [5, 10, 30, 60, 90]

# PIT warm-start and self-learning data pipeline.
PIT_BACKFILL_DEFAULT_QUARTERS = 40                 # 10 years of quarterly statements
HISTORICAL_PRICE_CACHE_DIR = "feature_cache/price_history"
PRICE_CACHE_BATCH_SIZE = 75
HISTORICAL_REPLAY_SOURCE = "replay_pit_v1"
HISTORICAL_REPLAY_FRESH_REPORT_PATH = "feature_cache/replay_parity_refresh.json"
HISTORICAL_REPLAY_FRESH_DEFAULT_TOP_N = 250
HISTORICAL_REPLAY_FRESH_DEFAULT_DAYS = 7
HISTORICAL_REPLAY_CROSS_SECTIONAL_MOMENTUM = False  # Shadow-only; latest parity run showed worse momentum alignment.
HISTORICAL_REPLAY_TTM_MIN_QUARTER_GAP_DAYS = 45
HISTORICAL_REPLAY_TTM_MAX_QUARTER_GAP_DAYS = 130
HISTORICAL_REPLAY_TTM_GPA_MAX = 1.50

# Trade-aware labels (Lopez de Prado triple-barrier method). These are used by
# the learner before enough live 30d returns mature, and are also persisted for
# diagnostics.
TRIPLE_BARRIER_HORIZON_DAYS = 30
TRIPLE_BARRIER_TARGET_ATR_MULT = 2.0
TRIPLE_BARRIER_STOP_ATR_MULT = 1.0

# ML ranker upgrades: use triple-barrier labels when available, train an
# optional meta-label classifier for Strong Buy gating, and blend regime-
# conditioned models only after they have enough samples.
ML_RANKER_USE_TRIPLE_BARRIER_LABELS = True
ML_META_LABEL_ENABLED = True
ML_META_LABEL_MIN_SAMPLES = 300
META_LABEL_STRONG_BUY_GATE_ENABLED = True
META_LABEL_STRONG_BUY_MIN_PROB = 0.60
META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD = True
META_LABEL_STRONG_BUY_DYNAMIC_FLOOR = 0.55
META_LABEL_STRONG_BUY_DYNAMIC_PCTILE = 0.90
META_LABEL_STRONG_BUY_DYNAMIC_MULTIPLIER = 0.92
META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES = 20
META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_FLOOR = 0.45
META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_PCTILE = 0.95
META_LABEL_RANK_MULTIPLIER_ON_FAIL = 0.90
META_LABEL_CORE_READY_DYNAMIC_THRESHOLD = True
META_LABEL_CORE_READY_DYNAMIC_MIN_CANDIDATES = 2
META_LABEL_CORE_READY_DYNAMIC_FLOOR = 0.45
META_LABEL_CORE_READY_DYNAMIC_PCTILE = 0.50
META_LABEL_CORE_READY_DYNAMIC_MULTIPLIER = 0.98
META_LABEL_CORE_READY_DYNAMIC_MAX_THRESHOLD = 0.55
ML_REGIME_ENSEMBLE_ENABLED = True
ML_REGIME_MIN_SAMPLES = 250
ML_REGIME_BLEND_PCT = 0.35

# Drift monitor: if live realised rank-IC breaks down, keep ML in shadow mode
# until the model is retrained/reviewed.
ML_DRIFT_MONITOR_ENABLED = True
DRIFT_MONITOR_HORIZON_COL = "return_10d"
DRIFT_MONITOR_MIN_DAYS = 20
DRIFT_MONITOR_DELTA = 0.002
DRIFT_MONITOR_THRESHOLD = 0.08
DRIFT_MONITOR_ALPHA = 0.99
DRIFT_KS_P_THRESHOLD = 0.01
DRIFT_KS_MIN_SAMPLE = 30
DRIFT_KS_COLUMNS = [
    "quality_factor_score", "value_factor_score", "momentum_factor_score",
    "bab_factor_score", "sleeve_quality", "sleeve_value",
    "sleeve_momentum", "sleeve_low_risk", "sleeve_ready",
]

# Trading-cost objective: expected cost is both stored as an ML feature and
# deducted from alpha in ranking. This scale is intentionally conservative.
DISCOVERY_COST_PENALTY_SCALE = 0.15

# Gate-first Strong Buy contract.  A weighted aggregate can create BUY/Watch
# ideas, but STRONG BUY must also pass quality, gate, and entry-readiness checks.
DISCOVERY_GATE_FIRST_ENABLED = True
READY_STRONG_BUY_ALLOW_MISSING_DATA_REVIEW = True
READY_STRONG_BUY_MIN_RR = 1.50
READY_STRONG_BUY_MIN_CONFIDENCE = 0.70
READY_STRONG_BUY_MIN_POSITION_WEIGHT = 0.005
READY_STRONG_BUY_PRIOR_SOFT_PASS_ENABLED = True
READY_STRONG_BUY_PRIOR_SOFT_PERCENTILE = 0.80
READY_STRONG_BUY_PRIOR_SOFT_CONFIDENCE = 0.55
READY_STRONG_BUY_PRIOR_SOFT_COVERAGE = 0.40
READY_STRONG_BUY_PRIOR_SOFT_MIN_SCORE = 0.00
READY_STRONG_BUY_NEAR_READY_MIN_SCORE = 0.70
READY_STRONG_BUY_CALIBRATION_REPORT_PATH = "feature_cache/ready_contract_calibration.json"
READY_STRONG_BUY_VETO_REPORT_PATH = "feature_cache/final_strong_buy_veto_report.json"
DISCOVERY_LIVE_SANITY_REPORT_PATH = "feature_cache/live_run_sanity.json"
DISCOVERY_SWAPS_REQUIRE_ENTRY_READY = True
REPLAY_LIVE_PARITY_REPORT_PATH = "feature_cache/replay_live_parity_report.json"
# When True (default), the daily orchestrator chains a replay refresh on the
# fresh discovery cohort before writing the replay/live parity report.  This
# keeps the parity comparison cohort-aligned (same day on both sides) and
# avoids phantom drift on price-derived fields caused by live and replay using
# different price-cache endpoints.  See daily_orchestrator.py around the call
# to _write_replay_live_parity_report for the rationale.  Flip to False to
# disable the refresh (e.g. if it adds unacceptable wall time).
ORCHESTRATOR_AUTO_REPLAY_REFRESH = True
REPLAY_LIVE_PARITY_TOLERANCE = 0.15
REPLAY_LIVE_PARITY_MAX_DATE_GAP_DAYS = 7
REPLAY_LIVE_PARITY_EXCLUDED_FIELDS = [
    # Live-only or not yet PIT-replayable; keep out of drift counts so the
    # report measures comparable replay/live features rather than roadmap gaps.
    "sentiment_score", "forecast_score", "rsi", "adx", "bb_pct",
    "peg_ratio", "roe", "short_pct", "vix_percentile",
    "downside_vol_60d", "max_dd_252d",
    "factor_mom_momentum_tilt", "factor_mom_value_tilt", "factor_mom_quality_tilt",
    "factor_mom_momentum_ret_1m", "factor_mom_momentum_ret_3m",
    "factor_mom_value_ret_1m", "factor_mom_value_ret_3m",
    "factor_mom_quality_ret_1m", "factor_mom_quality_ret_3m",
    "network_momentum", "qa_sentiment_score",
    "pead_factor_score", "sue_score", "revision_momentum_3m",
    "sleeve_momentum", "sleeve_quality", "sleeve_value",
    "sleeve_low_risk", "sleeve_pead", "sleeve_ready",
]
REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = ["yfinance_balance_only", "no_data"]
REPLAY_LIVE_PARITY_ZERO_AS_MISSING_FIELDS = ["pe_ratio", "peg_ratio", "roe", "short_pct"]
# Inputs that are live-only and not yet PIT-replayable.  The canonicalizer
# nulls these before recomputing derived factor scores so live and replay
# rows feed `compute_factor_scores_from_result` on the same component basis.
# They remain in REPLAY_LIVE_PARITY_EXCLUDED_FIELDS so they are also never
# field-on-field compared.  Remove an entry only once replay can supply it.
REPLAY_LIVE_PARITY_LIVE_ONLY_INPUTS_FOR_DERIVED = [
    # PEG ratio is yfinance-.info-only; no PIT analog (needs forward analyst growth).
    "peg_ratio",
    # Earnings stability requires 5y annual EPS series via FMP; replay's PIT
    # store only carries quarterly snapshots so it stays NULL on replay.
    # Leaving it in derived recompute creates systematic qmj_factor_score and
    # fundamental_score drift on fmp_full rows (observed on CBOE/STX/MYRG/REPX
    # 2026-05-13: live qmj component_count=3 incl. safety, replay=2 missing it).
    "earnings_stability",
    # Return-on-equity on the live path is yfinance .info trailing-TTM.  Naive
    # PIT-derivation (net_income / equity_proxy) would use single-quarter net
    # income — same TTM-vs-quarterly basis trap that bit pe_ratio.  Until a
    # TTM helper exists, replay leaves roe NULL and this nulls it on live too
    # for canonicalizer recompute.
    "roe",
]
# Stored aggregate fields whose live computation incorporates inputs that
# replay cannot supply structurally (earnings_stability needs 5y annual EPS;
# roe needs TTM aggregation).  The canonicalizer does NOT recompute these
# (they are not factor scores derived inside compute_factor_scores_from_result),
# so the drift cannot be neutralised by nulling inputs.  Reported separately
# from actionable drift via structural_drift_summary so the gate signal stays
# clean while transparency is preserved.  Remove an entry once replay can
# supply the underlying inputs.  Verified 2026-05-13: 45/45 fmp_full drifters
# on these fields have the live-only-input pattern; zero non-structural cases.
REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS = [
    "fundamental_score",
    "institutional_prior_score",
    "institutional_prior_percentile",
    "quality_score_fundamental",
    # revenue_growth lives on two different bases between live and replay:
    #   live (engine.fundamental.analyse): info.get("revenueGrowth") from
    #     yfinance .info, which is yfinance's own TTM-YoY calculation.
    #   replay (engine.historical_replay._fundamental_features): single-quarter
    #     over single-quarter via revenue / prior_revenue - 1.0 from PIT
    #     quarterly snapshots.
    # Same TTM-vs-quarterly basis mismatch class as the pe_ratio bug.  Drift
    # is structural until PIT TTM aggregation lands.  14 actionable drifters
    # observed 2026-05-13 with values diverging in both directions (LOGN.SW
    # 0.06 vs 2.56, LQDA 30.55 vs 10.31, BATRK 0.18 vs 1.27, JHG 0.61 vs 0.25)
    # — direction-mixed pattern confirms it's basis mismatch, not noise.
    "revenue_growth",
]
REPLAY_LIVE_PARITY_FIELD_TOLERANCES = {
    "quality_factor_score": 0.35,
    "value_factor_score": 0.35,
    "momentum_factor_score": 0.30,
    "volatility_factor_score": 0.25,
    "qmj_factor_score": 0.35,
    "bab_factor_score": 0.35,
    "turnover_cost_score": 0.25,
    "quality_score_fundamental": 0.35,
    "technical_score": 0.30,
    "fundamental_score": 0.35,
    "pe_ratio": 5.0,
    "revenue_growth": 0.25,
    "return_10d_prior": 0.20,
    "return_30d_prior": 0.20,
    "return_90d_prior": 0.25,
    "vol_20d": 0.15,
    "f_score": 1.25,
    "f_score_coverage": 0.35,
    "gpa": 0.12,
    "gpa_score": 0.50,
    "price_vs_sma200_stretch": 0.20,
    "institutional_prior_score": 0.35,
    "institutional_prior_percentile": 0.35,
    "institutional_prior_confidence": 0.35,
    "fill_probability": 0.36,
    "r_r_ratio": 0.75,
}
REPLAY_LIVE_PARITY_ADAPTIVE_FIELDS = [
    "technical_score", "return_10d_prior", "return_30d_prior", "return_90d_prior", "vol_20d",
]
REPLAY_LIVE_PARITY_READINESS_FIELDS = [
    # Execution-timing fields are not alpha-ranker features.  Keep them in
    # the report so readiness-model work remains measurable without blocking
    # the current forward-return model on a separate contract.
    "fill_probability", "r_r_ratio", "strong_buy_eligible",
]
READY_ENTRY_NEAR_HIGH_PULLBACK_RET30 = 0.12
READY_ENTRY_NEAR_HIGH_MIN_UPSIDE = 8.0

# ═══════════════════════════════════════════════════════════════════════════════
# Action-label hard gates (calibrated against SEB SA / Trustpilot 2026 review).
# Each gate cites the academic reference behind its threshold.  When a gate
# cannot be evaluated (missing data) the candidate is NOT blocked — but the
# coverage flag is recorded so the threshold-learner can debias over time.
# ═══════════════════════════════════════════════════════════════════════════════
ACTION_GATES_ENABLED = True            # Master toggle for Tier 1-4 action gates
ACTION_GATES_SHADOW = False            # When True, log gate failures but don't downgrade
ACTION_GATES_SECTOR_RELATIVE_QUALITY_PCTILES = True
ACTION_GATES_SECTOR_RELATIVE_MIN_N = 8

# Tier 1 — distress / quality-of-earnings (Altman 1968; Beneish 1999;
# Piotroski 2000; Sloan 1996; Cooper-Gulen-Schill 2008).
ALTMAN_Z_GATE_ENABLED = True
ALTMAN_Z_STRONG_BUY_MIN = 2.6           # Above grey-zone — safe for STRONG BUY
ALTMAN_Z_BUY_MIN = 1.81                  # Below this = formal distress zone

BENEISH_M_GATE_ENABLED = True
BENEISH_M_STRONG_BUY_MAX = -2.22         # Conservative non-manipulator cut
BENEISH_M_BUY_MAX = -1.78                # Beneish (1999) original threshold

F_SCORE_STRONG_BUY_MIN = 6               # Piotroski recommended long cut (was: only blocked F<=3)
F_SCORE_BUY_MIN = 5
FUNDAMENTAL_COVERAGE_GATE_ENABLED = True
FUNDAMENTAL_COVERAGE_FAIL_CAP = "BUY"     # All quality gates unevaluable cannot keep STRONG BUY
SOURCE_AWARE_COVERAGE_ENABLED = True
SOURCE_AWARE_NON_US_MIN_QMJ_COMPONENTS = 2      # yfinance-backed global names need at least two QMJ dimensions
SOURCE_AWARE_NON_US_THIN_EVIDENCE_CAP = "BUY"   # "We do not know enough" -> investable, not execution-grade
SOURCE_AWARE_US_MISSING_CAP = "NEUTRAL"          # FMP-style missing fundamentals are unusual enough to cap harder

ACCRUALS_GATE_ENABLED = True
ACCRUALS_FACTOR_STRONG_BUY_MIN = -0.60   # accruals_factor_score in [-1,1]; very-negative = high accruals
ASSET_GROWTH_FACTOR_STRONG_BUY_MIN = -0.60   # investment_factor_score in [-1,1]

NET_DEBT_EBITDA_STRONG_BUY_MAX = 3.0     # Above this = balance-sheet stress

# Tier 2 — value-discipline caps (Greenblatt 2006; Loughran-Wellman 2011;
# Asness-Liew-Pedersen-Thapar 2020).
EV_EBIT_GATE_ENABLED = True
EV_EBIT_STRONG_BUY_MAX = 30.0
EV_EBIT_BUY_MAX = 50.0
EV_SALES_STRONG_BUY_MAX = 8.0            # Loss-making/sub-scale ceiling
EV_SALES_QMJ_OVERRIDE_PCTILE = 0.80      # QMJ percentile that lifts EV/Sales cap
PE_FORWARD_STRONG_BUY_MAX = 30.0
PE_FORWARD_GROWTH_OVERRIDE_CAGR = 0.25   # Override P/E cap if 3y EPS CAGR > this
DEEP_VALUE_SPREAD_PCTILE = 0.80          # Up-weight value when E/P spread > this
VALUE_CAP_SHADOW_REPORT_ENABLED = True
VALUE_CAP_SHADOW_REPORT_PATH = "feature_cache/value_cap_shadow_report.json"
SIGNAL_QUALITY_BACKFILL_REPORT_PATH = "feature_cache/signal_quality_backfill_report.json"
VALUE_CAP_SHADOW_EVENT_LEDGER_ENABLED = True
VALUE_CAP_SHADOW_ANALOGUE_HISTORY_LIMIT = 5000
VALUE_CAP_SHADOW_PROMOTION_MIN_LIVE_N = 25
STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED = False  # Live-off; evaluated only by the shadow report unless flipped.
STRONG_BUY_VALUATION_QUALITY_OVERRIDE_SHADOW = True
STRONG_BUY_VALUATION_OVERRIDE_QMJ_FLOOR = 0.80          # Top-quintile QMJ required to override rich multiples.
STRONG_BUY_VALUATION_OVERRIDE_MIN_F_SCORE = 7
STRONG_BUY_VALUATION_OVERRIDE_MIN_REVENUE_GROWTH = 0.0
STRONG_BUY_VALUATION_OVERRIDE_REQUIRE_SECTOR_GROWTH = True
STRONG_BUY_VALUATION_OVERRIDE_SECTOR_GROWTH_MULTIPLIER = 1.0
STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT = 35.0
STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD = 40.0
INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW = 0.42

# Tier 3 — quality floors (Novy-Marx 2013; Damodaran cost-of-capital;
# Asness-Frazzini-Pedersen 2019 QMJ; Bartov-Givoly-Hayn 2002).
GPA_GATE_ENABLED = True
GPA_PCTILE_STRONG_BUY_MIN = 0.25         # Bottom-quartile gross-profitability blocked

ROIC_WACC_GATE_ENABLED = True
ROIC_VS_WACC_STRONG_BUY_MIN_BPS = 200    # Need 200 bps spread for STRONG BUY

QMJ_GATE_ENABLED = True
QMJ_PCTILE_STRONG_BUY_MIN = 0.50         # QMJ below median blocked
QMJ_LITE_ENABLED = True
QMJ_LITE_MIN_COMPONENTS = 2              # Require at least two non-overlapping QMJ dimensions
ENTERPRISE_FACTORS_MIN_COMPONENTS = 2    # Avoid one-signal dominance in holding fundamentals
OP_MARGIN_YOY_DELTA_FLOOR_BPS = -150     # Margins compressed > 150 bps demote one label
SB_SCORECARD_MISSING_IMPUTATION_ENABLED = True
SB_SCORECARD_MISSING_FILL_Z = -0.25      # Mild penalty for unavailable scorecard components
SB_SCORECARD_IMPUTED_WEIGHT = 0.50       # Imputed dimensions contribute at half confidence

# Tier 4 — momentum sanity (Wilder 1978; Faber 2007; George-Hwang 2004;
# Daniel-Moskowitz 2016).
RSI_STRONG_BUY_MAX = 75                  # RSI above this caps action at BUY
RSI_NEUTRAL_CAP = 80                     # RSI above this caps action at NEUTRAL
STRETCH_200DMA_STRONG_BUY_MAX = 0.35     # > +35% above 200-DMA caps at BUY-with-limit
ENTRY_STANCE_BLOCKS_STRONG_BUY = True    # Honour "Pullback Preferred" as a hard block
MOMENTUM_VOL_SCALING_ENABLED = True
ACTION_GATES_CORE_READY_STRETCH_OVERRIDE_ENABLED = True
ACTION_GATES_CORE_READY_STRETCH_MAX = 0.40
ACTION_GATES_CORE_READY_STRETCH_MIN_RR = 2.0
ACTION_GATES_CORE_READY_STRETCH_MIN_SCORE = 0.95

# Tier 5 — self-learning threshold ensemble (Russo-Van Roy 2014;
# Helmbold-Schapire-Singer-Warmuth 1998; López de Prado-Bailey 2014).
THRESHOLD_LEARNER_ENABLED = True
THRESHOLD_LEARNER_PROFILE = "moderate"   # "conservative" | "moderate" | "aggressive"
THRESHOLD_LEARNER_STATE_FILE = "feature_cache/threshold_learner_state.json"
THRESHOLD_LEARNER_UPDATE_REPORT_PATH = "feature_cache/threshold_learner_update_report.json"
THRESHOLD_LEARNER_INFER_MISSING_PROFILE = True
THRESHOLD_LEARNER_MISSING_PROFILE_FALLBACK = "moderate"
THRESHOLD_PSR_MIN_TO_DEPLOY = 0.95
SLEEVE_WEIGHTS_LEARNER = "exponentiated_gradient"
SLEEVE_WEIGHTS_LEARNING_RATE = 0.05
DRIFT_FORCE_CONSERVATIVE = True          # Drift alert => force conservative profile

# Tier 6 — UI surfacing
DISCOVERY_DIGEST_HIDE_GATE_FAILURES = True   # Hide STRONG BUYs that fail any Tier-1 gate
DISCOVERY_LIMIT_PRICE_BUFFER = 0.05          # Limit price = max(200-DMA*1.05, support*1.02)

# Insider / institutional thresholds
SHORT_INTEREST_HIGH = 0.20  # 20% float short = crowded
SHORT_INTEREST_LOW = 0.02  # <2% = no short pressure
INST_OWNERSHIP_HIGH = 0.70  # >70% = smart money holds

# Macro expert settings
MACRO_LOOKBACK = 365  # Days of macro data for correlation / VIX percentile reuse
MACRO_CORRELATION_MIN = 0.3  # Min |r| to use a macro expert
MACRO_TICKERS = {
    "vix": "^VIX",
    "bonds_10y": "^TNX",
    "bonds_2y": "^IRX",       # 2-year proxy (13-week T-bill × ~4 for short end)
    "oil": "CL=F",
}

# Extended macro tickers for factor timing (Arnott et al. 2019)
MACRO_TICKERS_EXTENDED = {
    "bonds_10y": "^TNX",      # 10-year yield
    "bonds_2y": "^IRX",       # Short-term yield proxy (13-week T-bill)
    "hy_spread": "HYG",       # High-yield corporate bond ETF (credit spread proxy)
    "ig_spread": "LQD",       # Investment-grade bond ETF
    "spy": "SPY",             # S&P 500 for equity risk premium
}

# Macro regime thresholds for factor timing
MACRO_TERM_SPREAD_EXPANSION = 1.0     # 10Y-2Y spread > 1.0% = expansion (tilt momentum)
MACRO_TERM_SPREAD_CONTRACTION = 0.0   # 10Y-2Y spread < 0% = inversion (tilt quality)
MACRO_CREDIT_SPREAD_TIGHT = 0.02      # HY-IG < 2% = risk-on (tilt momentum)
MACRO_CREDIT_SPREAD_WIDE = 0.05       # HY-IG > 5% = risk-off (tilt quality + low-vol)

# Insider transaction settings
INSIDER_LOOKBACK_DAYS = 90  # Consider transactions within last 90 days
INSIDER_BUY_BOOST = 0.15  # Score boost for recent insider buying
INSIDER_SELL_PENALTY = -0.1  # Score penalty for recent insider selling

# Reddit sentiment settings (updated when FMP active: news 0.45, reddit 0.30, fmp 0.25)
SENTIMENT_WEIGHTS = {"news": 0.6, "reddit": 0.4}
REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]
REDDIT_POST_LIMIT = 10
REDDIT_CACHE_TTL = 1800  # 30 minutes
SENTIMENT_CACHE_TTL = 3600  # 1 hour in-process cache
SENTIMENT_PERSISTENT_CACHE_TTL = 172800  # 48 hours across runs (avoids rate-limiting in long batches)

# Persistent deep-analysis cache (repeat discovery runs)
FORECAST_PERSISTENT_CACHE_TTL = 21600  # 6 hours across runs

# Weight optimization settings
BACKTEST_WEIGHT_STEP = 0.05    # Grid search step size for weight optimization
BACKTEST_KEEP_THRESHOLD = 2.0  # Forward return % below which KEEP is "correct"
WEIGHT_SHRINKAGE = 0.40        # Blend 40% toward equal weights to prevent overfitting
WEIGHT_MIN_FLOOR = 0.10        # Minimum 10% weight per pillar (signal diversification)
PILLAR_WEIGHT_AUDIT_PATH = "feature_cache/pillar_weight_audit.json"
PILLAR_WEIGHT_MIN_FLOOR = 0.03
PILLAR_WEIGHT_MAX_SINGLE = 0.55
PILLAR_WEIGHT_NONPOSITIVE_IC_MAX = 0.12
PILLAR_WEIGHT_SENTIMENT_LONG_MAX = 0.08
PILLAR_WEIGHT_FORECAST_MAX = 0.30
PILLAR_WEIGHT_FORECAST_MAX_WITH_NEGATIVE_IC = 0.03
PILLAR_WEIGHT_LONG_HORIZONS = ("30d", "60d", "90d", "multi")
ADAPTIVE_WEIGHTS_GATE_ENABLED = True
ADAPTIVE_WEIGHTS_GATE_REQUIRE_PARITY = True
ADAPTIVE_WEIGHTS_GATE_MIN_POSITIVE_PILLARS = 2
ADAPTIVE_WEIGHTS_LIVE_BLEND = 0.60
ADAPTIVE_WEIGHTS_PARITY_CRITICAL_FIELDS = [
    "technical_score", "return_10d_prior", "return_30d_prior", "return_90d_prior", "vol_20d",
]

# FMP (Financial Modeling Prep) API — PRIMARY data source (Starter plan)
# Sign up at https://site.financialmodelingprep.com/developer/docs
# Set env var FMP_API_KEY or leave empty to disable FMP features
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_PLAN = "starter"              # Plan tier for UI display
FMP_RATE_LIMIT_PER_MIN = 300      # Starter plan: 300 calls/minute
FMP_CACHE_TTL_QUARTERLY = 86400   # 24h — fundamentals change quarterly

# Hugging Face token used by Stage 6 FinBERT sentiment.
# Prefer HF_TOKEN; accept older/common aliases so local shells keep working.
HF_TOKEN = (
    os.environ.get("HF_TOKEN", "")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
)
FMP_CACHE_TTL_DAILY = 3600        # 1h — technicals, news
FMP_CACHE_TTL_CALENDAR = 43200    # 12h — earnings calendar

# Global Discovery Engine (v4 — multi-lens + wider funnel)
DISCOVERY_EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]  # FMP screener (US only on Starter)
DISCOVERY_FMP_LIMIT = 1000             # Per-exchange FMP screener limit (was 200)
DISCOVERY_MIN_MCAP = 50_000_000        # £50M floor (liquidity only, no upper cap)
DISCOVERY_VOLUME_MIN = 50_000          # Minimum daily volume (legacy raw-share floor; retained for FMP screener input)
# Region-aware average dollar-volume floors (USD-equivalent) used at the
# momentum prescreen. Replaces the raw share-volume gate which over-filters
# expensive high-quality names and under-filters cheap low-quality ones.
DISCOVERY_DOLLAR_VOLUME_FLOORS = {
    "US": 3_000_000,        # NYSE/NASDAQ/AMEX
    "DEVELOPED": 1_000_000, # UK, EU, JP, CA, AU, HK, SG
    "OTHER": 500_000,       # Everything else
}
DISCOVERY_TOP_N_LIGHTWEIGHT = 600       # Stage 5a: lightweight scoring (tech + momentum)
DISCOVERY_TOP_N_FULL_SCORE = 250        # Stage 5b: full 4-pillar analysis on top N
DISCOVERY_INFO_WORKERS = 4              # yfinance .info workers for medium-cost fundamentals
DISCOVERY_INFO_TIMEOUT_SECONDS = 8      # Per-ticker .info timeout before falling back to cache/empty
DISCOVERY_INFO_PREFETCH_TIMEOUT_SECONDS = 150  # Total .info prefetch budget before fail-fast
DISCOVERY_INFO_CIRCUIT_MIN_SAMPLE = 50  # Minimum responses before empty-rate circuit breaker can fire
DISCOVERY_INFO_EMPTY_RATE_CIRCUIT = 0.85  # Treat broad empty .info responses as provider block
DISCOVERY_YAHOO_METADATA_POLICY = "cache_only"  # cache_only | live_fallback; avoid Yahoo quoteSummary crumb blocks in live discovery
DISCOVERY_YAHOO_METADATA_HEALTH_PATH = "feature_cache/yahoo_metadata_health.json"
DISCOVERY_IDENTITY_WARNING_ALLOW_YAHOO_NETWORK = False
DISCOVERY_RISK_OVERLAY_ALLOW_YAHOO_EARNINGS = False
DISCOVERY_STAGE5B_FMP_TOPUP_MAX_CALLS = 60  # Bound sequential FMP top-up calls in quick rank
DISCOVERY_STAGE5B_FMP_TOPUP_TIME_BUDGET = 90  # Bound slow sequential FMP top-ups in quick rank
DISCOVERY_STAGE5B_SCORING_TIMEOUT = 900  # Internal budget for Stage 5b scoring loop
DISCOVERY_STAGE5B_MIN_SCORED_FOR_TIMEOUT = 100  # Minimum scored before deadline can return partial panel
DISCOVERY_STAGE5B_PIT_BATCH_RECORD = True  # Batch live PIT writes instead of rewriting JSON per ticker
DISCOVERY_USE_ETF_DECOMPOSITION = True  # Enable ETF holdings decomposition for universe expansion
DISCOVERY_RECONSTITUTE_UNIVERSE_ENABLED = True  # Refresh dynamic supplement before discovery
DISCOVERY_RECONSTITUTE_TTL_HOURS = 24   # Skip expensive universe validation when cache is fresh
DISCOVERY_USE_BENCHMARK_REBUILD = True  # Feed Wikipedia-sourced index constituents into reconstitute_universe()
DISCOVERY_BENCHMARK_TTL_DAYS = 30        # Index rebalances are quarterly; 30d keeps us fresh w/o hammering Wikipedia
ETF_HOLDINGS_CACHE_TTL = 604800        # 7 days — holdings change quarterly
DISCOVERY_GLOBAL_MAX_TIER = 2          # Daily snapshot includes all configured tiers
DISCOVERY_GLOBAL_INCLUDE_TIER2_DAILY = True  # Tier 2 is no longer rotated in/out by weekday
DISCOVERY_BETA_MAX = 2.5               # Maximum beta (soft penalty above 2.0)
DISCOVERY_CORRELATION_THRESHOLD = 0.70 # Max correlation with existing holdings
DISCOVERY_SECTOR_CONCENTRATION_MAX = 0.30  # Legacy alias for sector weight cap
DISCOVERY_USE_GLOBAL_UNIVERSE = True   # Enable yfinance-based global screening
DISCOVERY_TIER2_DAYS = [0, 3]          # Days to screen mid-caps (0=Mon, 3=Thu)

# Hard universe exclusions. These are applied before the screener spends
# feature or deep-scoring budget, and are intentionally broader than a static
# ticker delete so excluded markets cannot leak back through ETF decomposition,
# dynamic reconstitution, ADR aliases, or forced coverage.
DISCOVERY_EXCLUDED_COUNTRIES = {
    "KR",
    "KOR",
    "Korea",
    "South Korea",
    "Republic of Korea",
}
DISCOVERY_EXCLUDED_TICKER_SUFFIXES = (".KS", ".KQ")
DISCOVERY_EXCLUDED_TICKERS = {
    # Korean local listings / common ADRs and OTC symbols.
    "005930.KS", "000660.KS", "003670.KS", "006400.KS", "028260.KS",
    "032830.KS", "034730.KS", "051910.KS", "055550.KS", "066570.KS",
    "096770.KS", "105560.KS", "SSNLF", "SMSN.IL", "SMSD.IL",
    "PKX", "KB", "WF", "SHG", "KT", "SKM", "LPL",
}
DISCOVERY_TICKER_ALIASES = {
    # Static benchmark constituents can use exchange tickers that Yahoo does
    # not price directly. Resolve them before universe membership and pricing.
    "BAE.L": "BA.L",        # BAE Systems
    "CMC.L": "CMCX.L",      # CMC Markets
    "DSM.AS": "DSFIR.AS",   # dsm-firmenich after DSM/Firmenich merger
}
DISCOVERY_TICKER_QUARANTINE = {
    # Current Yahoo no-data symbols from `python -m utils.validate_universe`
    # on 2026-05-03. Aliasable names are handled above instead of quarantined.
    "0011.HK", "1COV.DE", "AHT.L", "ARMN", "BDEV.L", "BPSO.MI", "BVIC.L",
    "CINE.L", "CRH.L", "CSGN.SW", "DARK.L", "GFI.PA",
    "GPAY.L", "ICP.L", "ILD.PA", "MGGT.L", "NCM.AX", "PHNX.L",
    "SGC.L", "SMDS.L", "SNDR.L", "STM.MI", "STM.PA", "TKWY.AS",
    "URW.AS", "WIN.L",
}

# Stage 2 institutional cheap-screen sleeves.  These reserve part of the
# prescreen budget for cached point-in-time factor evidence (quality, value,
# low-risk/BAB) instead of letting price momentum consume every seat.
DISCOVERY_STAGE2_FACTOR_SLEEVES_ENABLED = True
DISCOVERY_STAGE2_FACTOR_RESERVE_PCT = 0.20
DISCOVERY_STAGE2_FACTOR_QUALITY_PCT = 0.35
DISCOVERY_STAGE2_FACTOR_VALUE_PCT = 0.25
DISCOVERY_STAGE2_FACTOR_LOW_RISK_PCT = 0.30
DISCOVERY_STAGE2_FACTOR_PEAD_PCT = 0.10
DISCOVERY_STAGE2_FACTOR_MIN_COVERAGE = 0.25
DISCOVERY_STAGE2_LOW_RISK_MIN_SCORE = 0.55
DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_LANE_ENABLED = True
DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_RESERVE_PCT = 0.12
DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MAX_PER_GROUP = 3
DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MIN_SCORE = 0.55
DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MIN_DOLLAR_VOLUME = 10_000_000
DISCOVERY_STAGE2_DEFENSIVE_SECTORS = (
    "Healthcare",
    "Consumer Defensive",
    "Consumer Staples",
    "Utilities",
    "Communication Services",
    "Communication",
    "Real Estate",
)

# Cohort-aware promotion: keep a strong global core, then backfill missing
# region / sector / liquidity buckets so the funnel is not dominated by one
# market regime or one crowded cluster.
DISCOVERY_LIGHTWEIGHT_PRESELECT_PCT = 0.55
DISCOVERY_LIGHTWEIGHT_REGION_FLOOR = 12
DISCOVERY_LIGHTWEIGHT_SECTOR_FLOOR = 6
DISCOVERY_LIGHTWEIGHT_LIQUIDITY_FLOOR = 8
DISCOVERY_LIGHTWEIGHT_LENS_FLOOR = 30   # Min candidates per entry lens at Stage 5a (Asness et al. 2013)
DISCOVERY_FULL_SCORE_PRESELECT_PCT = 0.55
DISCOVERY_FULL_SCORE_REGION_FLOOR = 5
DISCOVERY_FULL_SCORE_SECTOR_FLOOR = 3
DISCOVERY_FULL_SCORE_LIQUIDITY_FLOOR = 4
DISCOVERY_FULL_SCORE_LENS_FLOOR = 15    # Min candidates per entry lens at Stage 5c
DISCOVERY_FULL_SCORE_LENS_MIN_COUNTS = {
    "quality": 60,
    "value": 55,
    "composite": 40,
    "momentum": 60,
}
DISCOVERY_FULL_SCORE_READY_LANE_ENABLED = True
DISCOVERY_FULL_SCORE_READY_LANE_PCT = 0.30
DISCOVERY_FULL_SCORE_READY_LANE_MIN_SCORE = 0.58
DISCOVERY_FULL_SCORE_READY_LANE_MAX_REPLACE_PCT = 0.45
DISCOVERY_FULL_SCORE_READY_RESERVE = 25
DISCOVERY_FULL_SCORE_READY_MIN_SCORE = 0.55
DISCOVERY_FULL_SCORE_CHEAP_QUALITY_RESERVE_ENABLED = True
DISCOVERY_FULL_SCORE_CHEAP_QUALITY_RESERVE = 20
DISCOVERY_FULL_SCORE_CHEAP_QUALITY_MIN_SCORE = 0.55
DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_MAX = 200
DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_MIN_PRIORITY = 1.0
DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_PATH = "feature_cache/fundamental_refresh_queue.json"
DISCOVERY_FUNDAMENTAL_REFRESH_RESULTS_PATH = "feature_cache/fundamental_refresh_results.json"
DISCOVERY_FUNDAMENTAL_REFRESH_MAX_TICKERS = 40
HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_QUEUE_PATH = "feature_cache/replay_fundamental_refresh_queue.json"
HISTORICAL_REPLAY_FUNDAMENTAL_ATTEMPT_LEDGER_PATH = "feature_cache/replay_fundamental_attempt_ledger.json"
HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_MAX_TICKERS = 40
HISTORICAL_REPLAY_FUNDAMENTAL_ATTEMPT_COOLDOWN_HOURS = 168
HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_FIELDS = [
    "quality_factor_score", "value_factor_score", "qmj_factor_score",
    "quality_score_fundamental", "gross_profitability", "fcf_to_assets",
    "fundamental_score", "pe_ratio", "revenue_growth",
    "f_score", "f_score_coverage", "gpa", "gpa_score",
]
DISCOVERY_CHALLENGE_RESERVE_ENABLED = True
DISCOVERY_CHALLENGE_RESERVE_N = 30
DISCOVERY_CHALLENGE_AUTO_ENABLED = True
DISCOVERY_CHALLENGE_TARGET_N = 60
DISCOVERY_CHALLENGE_NEAR_MISS_PCT = 0.50
DISCOVERY_CHALLENGE_FACTOR_PCT = 0.35
DISCOVERY_CHALLENGE_MANUAL_PCT = 0.15
DISCOVERY_CHALLENGE_MAX_PER_SECTOR = 8
DISCOVERY_CHALLENGE_MAX_PER_COUNTRY = 10
DISCOVERY_CHALLENGE_EXPIRY_DAYS = 30
DISCOVERY_CHALLENGE_LOOKBACK_DAYS = 21
DISCOVERY_CHALLENGE_MIN_DOLLAR_VOLUME = 5_000_000
DISCOVERY_CHALLENGE_MIN_FACTOR_GROUPS = 2
DISCOVERY_CHALLENGE_CACHE_PATH = "feature_cache/challenge_candidates.json"
DISCOVERY_CHALLENGE_CACHE_MAX_AGE_HOURS = 12
DISCOVERY_CHALLENGE_MANUAL_OVERRIDES = []
DISCOVERY_CHALLENGE_TICKERS = []  # Legacy/manual fallback; auto generator is preferred.

# Momentum screening (90-day cycle optimisation)
DISCOVERY_MODE = "momentum_90d"        # "balanced" or "momentum_90d"
MOMENTUM_WEIGHTS = {                   # Pillar weights in momentum mode
    "technical": 0.40,                 # Trend-following signals dominate
    "fundamental": 0.25,              # Quality filter prevents momentum traps
    "sentiment": 0.10,                # News/social momentum (short-lived)
    "forecast": 0.25,                 # MoE price prediction
}
MOMENTUM_TOP_N_PRESCREEN = 1000        # Keep top N by momentum score before filtering
MOMENTUM_MIN_AVG_VOLUME = DISCOVERY_VOLUME_MIN  # Legacy alias, kept in sync

# Multi-lens entry (each lens gets a quota within MOMENTUM_TOP_N_PRESCREEN)
DISCOVERY_LENS_MOMENTUM_PCT = 0.35     # 35% momentum (reduced from 50% — Barroso & Santa-Clara 2015)
DISCOVERY_LENS_VALUE_PCT = 0.25        # 25% value/turnaround plays
DISCOVERY_LENS_QUALITY_PCT = 0.30      # 30% quality/steady outperformers (Novy-Marx 2013)
DISCOVERY_LENS_COMPOSITE_PCT = 0.10    # 10% composite (momentum+quality intersection)

# Region-balanced sampling (legacy; geographic minimums are no longer enforced)
DISCOVERY_REGION_MIN_PCT = 0.15

# Diversified final selector
DISCOVERY_MAX_PER_SECTOR = 6           # Max candidates from any single sector (Kacperczyk et al. 2005)
DISCOVERY_USE_INDUSTRY_DIVERSIFICATION = True
DISCOVERY_MAX_PER_INDUSTRY = 3          # Sub-industry cap before falling back to sector labels
DISCOVERY_MIN_REGIONS = 2              # Legacy; geographic minimums no longer enforced
DISCOVERY_SECTOR_PCT_CAP = 0.30        # No GICS sector > 30% of portfolio weight

# Timeout protection (prevents stuck tickers from blocking the whole run)
DISCOVERY_PER_TICKER_TIMEOUT = 120     # Max seconds per ticker in deep analysis (Stage 6)
SCORING_COMPONENT_TIMEOUT = 45         # Max seconds per scoring sub-component (tech/fund/sent)
SCORING_FORECAST_TIMEOUT = 60          # Max seconds for forecast model per ticker
SCORING_PARALLEL_COMPONENTS = True     # Run independent Stage 6 pillars in parallel per ticker
SCORING_ADAPTIVE_WEIGHT_CACHE_TTL = 900 # Cache expensive adaptive-weight DB reads per process
DISCOVERY_PIT_FIRST_STAGE5B = True      # Prefer PIT/FMP cached fundamentals before yfinance .info
DISCOVERY_YFINANCE_INFO_FALLBACK = True # Use yfinance .info only when PIT/FMP data is thin
DISCOVERY_STAGE5_READY_BOOST_ENABLED = True
DISCOVERY_STAGE5_READY_BOOST_STRENGTH = 0.30  # Multiplier range roughly 0.85x..1.15x
DISCOVERY_SLEEVES_ENABLED = True       # Cache-only multi-sleeve Stage 5a ranker
DISCOVERY_SLEEVE_SECTOR_MAX = None     # Optional per-sector cap inside adaptive promotion
DISCOVERY_SLEEVE_SECTOR_NEUTRAL_BLEND = 0.35
DISCOVERY_SLEEVE_MIN_COVERAGE = 0.50
DISCOVERY_SLEEVE_WEIGHTS = {
    "quality": 0.24,
    "momentum": 0.20,
    "value": 0.18,
    "low_risk": 0.15,
    "ready": 0.15,
    "pead": 0.08,
}
DISCOVERY_FORCE_INCLUDE_TICKERS = [
    {"symbol": "CI", "companyName": "The Cigna Group", "sector": "Healthcare", "industry": "Healthcare Plans", "country": "US", "exchange": "NYSE", "index_source": "FORCED_HEALTHCARE_CORE"},
    {"symbol": "ELV", "companyName": "Elevance Health, Inc.", "sector": "Healthcare", "industry": "Healthcare Plans", "country": "US", "exchange": "NYSE", "index_source": "FORCED_HEALTHCARE_CORE"},
    {"symbol": "HUM", "companyName": "Humana Inc.", "sector": "Healthcare", "industry": "Healthcare Plans", "country": "US", "exchange": "NYSE", "index_source": "FORCED_HEALTHCARE_CORE"},
    {"symbol": "CNC", "companyName": "Centene Corporation", "sector": "Healthcare", "industry": "Healthcare Plans", "country": "US", "exchange": "NYSE", "index_source": "FORCED_HEALTHCARE_CORE"},
    {"symbol": "MOH", "companyName": "Molina Healthcare, Inc.", "sector": "Healthcare", "industry": "Healthcare Plans", "country": "US", "exchange": "NYSE", "index_source": "FORCED_HEALTHCARE_CORE"},
]
ORCHESTRATOR_MAX_RUNTIME = 57600       # Max total orchestrator runtime in seconds (16 hours)
ORCHESTRATOR_LOCK_FILE = "feature_cache/orchestrator.lock"
ORCHESTRATOR_LOCK_STALE_SECONDS = ORCHESTRATOR_MAX_RUNTIME + 1800

# Multi-swap evaluation
DISCOVERY_MAX_SWAPS_PER_RUN = 3        # Allow up to N swap recommendations per run
SWAP_CANDIDATE_THRESHOLD = -0.10       # Holdings with score below this are swap-eligible

# FX fees for ISA — typical platform charges ~0.75% per leg
FX_FEE_TIER = 0.0075                  # 0.75% per currency conversion

# FMP-sourced technical indicator thresholds
ADX_STRONG_TREND = 25     # ADX > 25 = trending market
ADX_VERY_STRONG_TREND = 40  # ADX > 40 = very strong trend
WILLIAMS_OVERBOUGHT = -20  # Williams %R > -20 = overbought
WILLIAMS_OVERSOLD = -80    # Williams %R < -80 = oversold

# Earnings calendar
EARNINGS_PROXIMITY_DAYS = 7  # Warn when earnings within N days
RISK_OVERLAY_LIVE_TARGET_CHECK = False # Use cached analyst target info only during overlay
RISK_OVERLAY_INFO_TIMEOUT_SECONDS = 3
RISK_OVERLAY_POST_EARNINGS_CACHE_TTL = 43200

# ═══════════════════════════════════════════════════════════════════════════════
# Autonomous Email Engine (v4.0)
# ═══════════════════════════════════════════════════════════════════════════════

# SMTP — Gmail default. Secrets via environment variables.
EMAIL_SMTP_HOST = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")           # sender address
EMAIL_TO = os.environ.get("EMAIL_TO", "")               # recipient address
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")   # app password

# Orchestrator scheduling
ORCHESTRATOR_DISCOVERY_FREQ_DAYS = 1   # Fallback: max days between runs (overridden by day-of-week)
ORCHESTRATOR_DISCOVERY_DAYS = [0, 1, 2, 3, 4]  # Mon-Fri (0=Mon ... 4=Fri)
DISCOVERY_REQUIRED_BENCHMARKS = ("SP500", "SPMIDCAP400")

# Decision logic — swap hurdle rates
HURDLE_RATE = 0.20             # candidate.aggregate_score must beat weakest by this margin
PORTFOLIO_FIT_MIN = 0.50       # minimum portfolio_fit_score to qualify as swap candidate
COOLDOWN_DAYS = 7             # suppress re-alerting same ticker for N days

# Self-monitoring — "propose, don't apply". Reads telemetry from the
# decision log and writes a recommendation JSON. Config is never mutated.
AUTO_TUNE_ENABLED = True
AUTO_TUNE_WINDOW_DAYS = 30

# State + decision log
ORCHESTRATOR_STATE_FILE = "orchestrator_state.json"
ORCHESTRATOR_LOG_FILE = "orchestrator_log.jsonl"

# Paper Trading Ledger (SQLite)
PAPER_TRADING_DB = "paper_trading.db"
PAPER_TRADING_ENABLED = True                    # Log all signals to paper ledger
PAPER_TRADING_LOG_DRY_RUN = False               # Keep dry runs read-only for paper ledger

# Timeouts (seconds)
DISCOVERY_TIMEOUT = 7200       # 2 hours max for expanded discovery pipeline

# ═══════════════════════════════════════════════════════════════════════════════
# Algorithmic Upgrades
# ═══════════════════════════════════════════════════════════════════════════════

# Sentiment recency decay — exponential decay by article age
SENTIMENT_RECENCY_DECAY = True
SENTIMENT_DECAY_HALF_LIFE_HOURS = 48.0     # Half-life in hours (older articles count less)

# Cross-sectional z-scoring — normalize pillar scores within discovery batch
DISCOVERY_CROSS_SECTIONAL_ZSCORE = True

# Position sizing / volatility management
POSITION_RISK_BUDGET_PCT = 0.01             # Risk 1% of portfolio per trade before caps
POSITION_ADV_PARTICIPATION_CAP = 0.10       # ≤10% of 20-day ADV per position (Almgren-Chriss)
VOL_MANAGED_TARGET_ANN = 0.20               # 20% annualized target vol for alpha scaling
VOL_MANAGED_FLOOR = 0.50                    # Never scale alpha below 50%
VOL_MANAGED_CAP = 1.25                      # Never scale alpha above 125%
PEAD_MAX_OVERLAY = 0.15                     # Cap PEAD / revision overlay magnitude (raised from 0.10)
PEAD_FACTOR_WEIGHT = 0.12                   # Weight of PEAD factor in alpha blend (Martineau 2022)
PEAD_SUE_WINDOW_QUARTERS = 8               # Quarters for SUE std dev calculation
PEAD_ENABLED = True                         # Enable first-class PEAD factor
DISCOVERY_MAX_RISK_PENALTY = 0.30           # Cap total risk overlay deduction per candidate
DISCOVERY_MAX_PILLAR_WEIGHT = 0.70          # Cap any single pillar after adaptive redistribution

# Dividend safety thresholds
DIVIDEND_PAYOUT_HEALTHY = 0.40              # Payout ratio below this = healthy (+score)
DIVIDEND_PAYOUT_STRETCHED = 0.60            # Payout ratio above this = stretched (-score)
DIVIDEND_PAYOUT_UNSUSTAINABLE = 0.80        # Payout ratio above this = unsustainable (red flag)
DIVIDEND_YIELD_TRAP_THRESHOLD = 0.08        # Yield above 8% = potential yield trap
EX_DIVIDEND_PROXIMITY_DAYS = 14             # Flag ex-dividend within N days

# Balance sheet strength thresholds
NET_DEBT_EBITDA_FORTRESS = 0.0              # Negative net debt = fortress balance sheet
NET_DEBT_EBITDA_HIGH = 3.0                  # Above this = leveraged
NET_DEBT_EBITDA_DANGER = 5.0                # Above this = dangerously leveraged
CURRENT_RATIO_MIN = 1.0                     # Below this = liquidity risk

# Governance red flag composite — flag when N+ signals align
GOVERNANCE_FLAG_THRESHOLD = 3               # Number of warning signals to trigger flag

# ML ranker guardrails — keep conservative until walk-forward sample is larger
ML_RANKER_MIN_SAMPLES = 300                 # Minimum evaluated signals before ML can train live
ML_RANKER_BLEND_PCT = 0.30                  # Increased 0.15→0.30 after IC study confirmed
                                             # ML rank-IC=0.16 vs aggregate_score IC≈-0.01.
                                             # ML signal has 16x better IC; 30% blend is the
                                             # mid-point between conservative 15% and 50%.
ML_RANKER_SHADOW_ONLY = False               # Live when enough data exists; still degrades gracefully
ML_RANKER_SURFACE_SHADOW = True             # Cache ML scores even when live blend is disabled/blocked
ML_RANKER_META_PROBA_FOR_ALL = True         # Cache meta-label probability for every deep-scored candidate
ML_RANKER_PROMOTION_MIN_SAMPLES = 300       # Promotion gate after purged validation
ML_RANKER_PROMOTION_MIN_RANK_IC = 0.02      # Positive, but not top-decile-only
ML_RANKER_PROMOTION_MIN_R_SQUARED = -0.05   # Ranker is judged mainly on OOS rank IC
ML_RANKER_PARITY_GATE_ENABLED = True        # Block live ML blend when replay/live features diverge
ML_RANKER_PARITY_GATE_REQUIRE_REPORT = True
ML_RANKER_PARITY_REPORT_PATH = REPLAY_LIVE_PARITY_REPORT_PATH
ML_RANKER_PARITY_MAX_AGE_HOURS = 48
ML_RANKER_PARITY_MIN_AVAILABLE_RATIO = 0.70
ML_RANKER_PARITY_MAX_MISSING_REPLAY_RATIO = 0.20
ML_RANKER_PARITY_MAX_STALE_RATIO = 0.10
ML_RANKER_PARITY_MAX_DRIFTED_PAIR_RATIO = 0.25
ML_RANKER_PARITY_MAX_CRITICAL_MISSING_RATIO = 0.15
ML_RANKER_PARITY_CRITICAL_FIELDS = [
    "quality_factor_score", "value_factor_score", "momentum_factor_score",
    "return_10d_prior", "return_30d_prior", "return_90d_prior", "vol_20d",
    "f_score", "gpa", "institutional_prior_score",
]
# When True (default), the parity gate prefers the actionable-segmented drift
# counts (actionable_drifted_tickers / actionable_available_pairs) over the
# raw drifted_tickers count.  Actionable excludes yfinance_balance_only and
# no_data evidence classes (noise we already gate downstream via source-aware
# caps) and ignores drift on REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS
# (drift explained by live-only inputs replay cannot supply structurally).
# Falls back to raw counts when the report lacks the actionable fields, so
# stale parity reports do not crash the gate.  Flip to False to force raw
# legacy behaviour for rollback.
ML_RANKER_PARITY_USE_ACTIONABLE_DRIFT = True
ML_RANKER_EMBARGO_DAYS = 30                 # Purge overlapping 30d target windows
ML_RANKER_MIN_TRAIN_SAMPLES = 80
ML_RANKER_MODEL_CACHE_FILE = "feature_cache/ml_ranker_model.pkl"

# Data confidence floor (Hou, Xue & Zhang 2020 — prevent data sparsity bias)
CONFIDENCE_FLOOR = 0.60                     # Min confidence multiplier (was 0.30, crushed mid-caps)

# Factor momentum (Ehsani & Linnainmaa 2022)
FACTOR_MOMENTUM_MIN_UNIVERSE = 10         # Smallest cross-section that still permits quintile spreads
FACTOR_MOMENTUM_LOOKBACK_1M = 21            # 1-month rolling window (trading days)
FACTOR_MOMENTUM_LOOKBACK_3M = 63            # 3-month rolling window (trading days)
FACTOR_MOMENTUM_CACHE_TTL = 86400           # 24h cache for factor returns
FACTOR_MOMENTUM_REQUIRE_3M = True           # Need 3m confirmation before live tilt
FACTOR_MOMENTUM_TILT_CAP = 0.35             # Conservative cap; no +/-1 one-month swings
FACTOR_MOMENTUM_CONFIRMATION_MULT = 0.25    # Damp tilts when 1m and 3m disagree

# Bayesian self-learning.  Starts from academic/commercial priors and only
# blends toward realised local IC when there is enough evaluated history.
BAYESIAN_SELF_LEARNING_ENABLED = True
BAYESIAN_PRIOR_STRENGTH = 125               # Virtual observations backing the prior
BAYESIAN_MAX_LIVE_BLEND = 0.25              # Local data cannot dominate cold-start priors
BAYESIAN_MIN_SAMPLES = 30
BAYESIAN_HORIZON_CHAIN = ["30d", "10d", "5d"]
BAYESIAN_MIN_SAMPLES_BY_HORIZON = {"30d": 100, "10d": 250, "5d": 500}
BAYESIAN_HORIZON_BLEND_MULT = {"30d": 1.0, "10d": 0.60, "5d": 0.35}
BAYESIAN_EFFECTIVENESS_MIN_SAMPLES = 200
BAYESIAN_EFFECTIVENESS_PRIOR_STRENGTH = 200
BAYESIAN_DAILY_MAX_REL_DELTA = 0.20
BAYESIAN_REGIME_CONDITIONAL = True

# Calibration can use Lopez de Prado triple-barrier labels from PIT replay
# instead of waiting for months of live 90d action outcomes.
ACTION_CALIBRATION_USE_TRIPLE_BARRIER = True
ACTION_CALIBRATION_MIN_SAMPLE_SIZE = 20

# Broad universe for cross-sectional weight optimization (diverse sectors + geographies)
BACKTEST_UNIVERSE = [
    # US Large Cap — diversified sectors
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",   # Tech
    "JPM", "BAC", "GS",                          # Financials
    "JNJ", "UNH", "PFE",                         # Healthcare
    "XOM", "CVX",                                 # Energy
    "PG", "KO", "WMT",                            # Consumer staples
    "TSLA", "HD", "NKE",                           # Consumer discretionary
    "CAT", "BA", "UPS",                            # Industrials
    "NEE", "DUK",                                  # Utilities
    "AMT", "PLD",                                  # REITs
    # UK / Europe
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L",         # FTSE 100
    "GSK.L", "RIO.L", "ULVR.L",                  # FTSE 100
    "SAP.DE", "SIE.DE",                           # Germany
    # Broad ETFs
    "SPY", "QQQ", "IWM", "EFA", "EEM",            # US + Intl ETFs
]
