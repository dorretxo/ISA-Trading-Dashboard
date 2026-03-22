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
    "technical": 0.3709,
    "fundamental": 0.2076,
    "sentiment": 0.1593,
    "forecast": 0.2623,
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
SCORE_STRONG_BUY_THRESHOLD = 0.50  # Above this = STRONG BUY
SCORE_BUY_THRESHOLD = 0.25         # Above this = BUY
SCORE_KEEP_THRESHOLD = -0.3        # Above this = KEEP
SCORE_SELL_THRESHOLD = -0.6        # Above this = SELL, below = STRONG SELL

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

# Insider / institutional thresholds
SHORT_INTEREST_HIGH = 0.20  # 20% float short = crowded
SHORT_INTEREST_LOW = 0.02  # <2% = no short pressure
INST_OWNERSHIP_HIGH = 0.70  # >70% = smart money holds

# Macro expert settings
MACRO_LOOKBACK = 90  # Days of macro data for correlation
MACRO_CORRELATION_MIN = 0.3  # Min |r| to use a macro expert
MACRO_TICKERS = {
    "vix": "^VIX",
    "bonds_10y": "^TNX",
    "oil": "CL=F",
}

# Insider transaction settings
INSIDER_LOOKBACK_DAYS = 90  # Consider transactions within last 90 days
INSIDER_BUY_BOOST = 0.15  # Score boost for recent insider buying
INSIDER_SELL_PENALTY = -0.1  # Score penalty for recent insider selling

# Reddit sentiment settings (updated when FMP active: news 0.45, reddit 0.30, fmp 0.25)
SENTIMENT_WEIGHTS = {"news": 0.6, "reddit": 0.4}
REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]
REDDIT_POST_LIMIT = 10
REDDIT_CACHE_TTL = 1800  # 30 minutes

# Weight optimization settings
BACKTEST_WEIGHT_STEP = 0.05    # Grid search step size for weight optimization
BACKTEST_KEEP_THRESHOLD = 2.0  # Forward return % below which KEEP is "correct"
WEIGHT_SHRINKAGE = 0.40        # Blend 40% toward equal weights to prevent overfitting
WEIGHT_MIN_FLOOR = 0.10        # Minimum 10% weight per pillar (signal diversification)

# FMP (Financial Modeling Prep) API — PRIMARY data source (Starter plan)
# Sign up at https://site.financialmodelingprep.com/developer/docs
# Set env var FMP_API_KEY or leave empty to disable FMP features
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_PLAN = "starter"              # Plan tier for UI display
FMP_RATE_LIMIT_PER_MIN = 300      # Starter plan: 300 calls/minute
FMP_CACHE_TTL_QUARTERLY = 86400   # 24h — fundamentals change quarterly
FMP_CACHE_TTL_DAILY = 3600        # 1h — technicals, news
FMP_CACHE_TTL_CALENDAR = 43200    # 12h — earnings calendar

# Global Discovery Engine (v2 — expanded global + momentum)
DISCOVERY_EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]  # FMP screener (US only on Starter)
DISCOVERY_FMP_LIMIT = 1000             # Per-exchange FMP screener limit (was 200)
DISCOVERY_MIN_MCAP = 50_000_000        # £50M floor (liquidity only, no upper cap)
DISCOVERY_VOLUME_MIN = 50_000          # Minimum daily volume (liquidity floor)
DISCOVERY_TOP_N_LIGHTWEIGHT = 100       # Stage 5a: lightweight scoring (tech + momentum)
DISCOVERY_TOP_N_FULL_SCORE = 30         # Stage 5b: full 4-pillar analysis on top N
DISCOVERY_BETA_MAX = 2.5               # Maximum beta (relaxed for momentum plays)
DISCOVERY_CORRELATION_THRESHOLD = 0.70 # Max correlation with existing holdings
DISCOVERY_SECTOR_CONCENTRATION_MAX = 0.40  # Max sector weight before penalty (relaxed)
DISCOVERY_USE_GLOBAL_UNIVERSE = True   # Enable yfinance-based global screening

# Momentum screening (90-day cycle optimisation)
DISCOVERY_MODE = "momentum_90d"        # "balanced" or "momentum_90d"
MOMENTUM_WEIGHTS = {                   # Pillar weights in momentum mode
    "technical": 0.45,                 # RSI, MACD, breakouts dominate
    "fundamental": 0.10,              # Light fundamental check
    "sentiment": 0.20,                # News/social momentum
    "forecast": 0.25,                 # MoE price prediction
}
MOMENTUM_TOP_N_PRESCREEN = 150         # Keep top N by momentum score before filtering
MOMENTUM_MIN_AVG_VOLUME = 100_000      # Minimum 20-day average volume for momentum

# Multi-swap evaluation
DISCOVERY_MAX_SWAPS_PER_RUN = 3        # Allow up to N swap recommendations per run
SWAP_CANDIDATE_THRESHOLD = -0.10       # Holdings with score below this are swap-eligible

# FX fees for ISA — Interactive Investor charges 0.75% per leg
FX_FEE_TIER = 0.0075                  # 0.75% per currency conversion

# FMP-sourced technical indicator thresholds
ADX_STRONG_TREND = 25     # ADX > 25 = trending market
ADX_VERY_STRONG_TREND = 40  # ADX > 40 = very strong trend
WILLIAMS_OVERBOUGHT = -20  # Williams %R > -20 = overbought
WILLIAMS_OVERSOLD = -80    # Williams %R < -80 = oversold

# Earnings calendar
EARNINGS_PROXIMITY_DAYS = 7  # Warn when earnings within N days

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
ORCHESTRATOR_DISCOVERY_FREQ_DAYS = 1   # Run discovery every N days (weekly default)

# Decision logic — swap hurdle rates
HURDLE_RATE = 0.20             # candidate.aggregate_score must beat weakest by this margin
PORTFOLIO_FIT_MIN = 0.50       # minimum portfolio_fit_score to qualify as swap candidate
COOLDOWN_DAYS = 14             # suppress re-alerting same ticker for N days

# State + decision log
ORCHESTRATOR_STATE_FILE = "orchestrator_state.json"
ORCHESTRATOR_LOG_FILE = "orchestrator_log.jsonl"

# Paper Trading Ledger (SQLite)
PAPER_TRADING_DB = "paper_trading.db"
PAPER_TRADING_ENABLED = True                    # Log all signals to paper ledger

# Timeouts (seconds)
DISCOVERY_TIMEOUT = 7200       # 2 hours max for expanded discovery pipeline

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
