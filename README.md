# ISA Portfolio Dashboard

A data-driven stock analysis and portfolio decision system for self-directed investors.

Built around a 90-day review cycle, the dashboard combines multi-factor scoring, risk management, global discovery, and performance tracking into a single Streamlit interface. Trades are executed manually &mdash; this is a decision-support tool, not an auto-trading bot.

---

## Dashboard Tabs

### Dashboard

Portfolio-level summary with total value (FX-converted to GBP), P&L, sector allocation, and at-a-glance action cards for every holding. The portfolio optimizer suggests weight adjustments using mean-variance optimisation with turnover, FX, sector concentration, and max-weight constraints. A Monte Carlo projection models 90-day forward return distributions.

### Holdings

Per-holding deep dive with five sub-tabs:

| Sub-tab | Content |
|---------|---------|
| **Overview** | Price chart, action label, aggregate score, stop/target levels |
| **Scores** | Pillar breakdown (technical, fundamental, sentiment, forecast) with reasoning |
| **Fundamentals** | P/E, EPS growth, margins, dividend safety, balance sheet grade, governance flags |
| **Sentiment** | News headlines, FinBERT scores, Reddit mentions, recency-weighted composite |
| **Forecast** | Multi-horizon price prediction, expert confidence bands, rolling MAE |

Risk chips surface key warnings inline: earnings proximity, parabolic moves, governance concerns, asymmetric risk, and ex-dividend dates.

### Discovery

Global screening pipeline that finds replacement candidates across NYSE, NASDAQ, and AMEX. Multi-stage funnel:

1. Universe assembly from FMP screener + yfinance global tickers
2. Multi-lens entry (momentum, value, quality quotas)
3. Lightweight momentum prescreen
4. Portfolio correlation and sector fit filtering
5. Full four-pillar deep scoring on top candidates
6. Final ranking with diversified sector/region selection

Swap recommendations compare discovery candidates against the weakest current holdings, with hurdle rates and cooldown logic to prevent churn.

### Analytics

Signal quality and historical effectiveness:

- **Accuracy** &mdash; hit rates by action label and pillar
- **Weights** &mdash; adaptive pillar weight optimisation from backtest results
- **Impact** &mdash; alpha vs benchmark, drawdown analysis
- **Backtest** &mdash; walk-forward weight optimisation on a broad universe
- **Performance** &mdash; paper trading ledger with P&L, slippage, and signal evaluation

---

## Scoring Engine

Each holding is scored across four pillars with configurable weights:

| Pillar | Default Weight | Signals |
|--------|---------------|---------|
| **Technical** (30%) | Trend/momentum | RSI, MACD, ADX, Williams %R, moving averages, ATR |
| **Fundamental** (40%) | Value + quality | P/E, PEG, margins, EPS growth, insider activity, dividend safety, balance sheet strength, analyst consensus |
| **Sentiment** (8%) | News flow | Google News + FinBERT, Reddit retail sentiment, FMP news, recency decay |
| **Forecast** (22%) | Statistical ensemble | Mixture-of-experts (linear regression, mean reversion, momentum, volatility, macro correlation), multi-horizon |

Scores map to action labels: **STRONG BUY** > 0.40, **BUY** > 0.20, **KEEP** > -0.25, **SELL** > -0.50, **STRONG SELL** below.

A risk overlay adjusts scores for parabolic price moves, earnings proximity, and market-cap-aware confidence scaling.

### Additional Analytics

- **Dividend safety** &mdash; yield vs 5-year average, payout ratio sustainability, ex-dividend proximity
- **Balance sheet strength** &mdash; net debt/EBITDA, current ratio, cash-to-debt, letter grade (A/B/C/D)
- **Governance red flag** &mdash; composite proxy from insider selling patterns, short interest, analyst divergence, margin trends, and earnings execution (metadata only, no double-counting)
- **Asymmetric risk flag** &mdash; detects binary outcome patterns (above consensus + overbought, or parabolic + high short interest)
- **VIX regime detection** &mdash; tilts pillar weights based on bull/neutral/bear market conditions
- **Volatility-adjusted stops** &mdash; dynamic ATR multipliers and trailing percentages based on realised volatility percentile

---

## Architecture

```
ISA-Trading-Dashboard/
├── app.py                          # Streamlit dashboard (UI composition)
├── daily_orchestrator.py           # Background runner (analysis + discovery + email)
├── config.py                       # Central configuration and feature toggles
├── run_orchestrator.bat            # Windows Task Scheduler launcher
├── run_orchestrator_wrapper.py     # Signal-protected wrapper (survives logoff)
│
├── engine/                         # Analysis and ranking logic
│   ├── scoring.py                  #   Per-holding aggregate scoring
│   ├── technical.py                #   Technical indicators and signals
│   ├── fundamental.py              #   Fundamental analysis + dividend/balance sheet
│   ├── sentiment.py                #   News + Reddit + FinBERT sentiment
│   ├── forecasting.py              #   Mixture-of-experts price prediction
│   ├── discovery.py                #   Global screening and candidate ranking
│   ├── discovery_backtest.py       #   Discovery signal recording
│   ├── discovery_eval.py           #   Discovery-specific evaluation reporting
│   ├── portfolio_optimizer.py      #   Mean-variance portfolio optimisation
│   ├── portfolio_projection.py     #   Monte Carlo return projection
│   ├── portfolio_risk.py           #   Portfolio-level risk metrics
│   ├── position_sizing.py          #   Inverse-volatility position weighting
│   ├── exit_engine.py              #   Exit signals and stop proximity
│   ├── stops.py                    #   ATR + trailing stop-loss / take-profit
│   ├── risk_overlay.py             #   Parabolic, earnings, cap-aware overlays
│   ├── regime.py                   #   VIX-based market regime detection
│   ├── evaluation_harness.py       #   Signal accuracy and pillar evaluation
│   ├── paper_trading.py            #   Paper trading ledger (SQLite)
│   ├── ml_ranker.py                #   ML ranking model (shadow mode)
│   ├── backtest.py                 #   Walk-forward weight optimisation
│   └── performance.py              #   Performance attribution
│
├── utils/                          # Runtime and state helpers
│   ├── data_fetch.py               #   yfinance data with session caching
│   ├── fmp_client.py               #   Financial Modeling Prep API client
│   ├── cache_loader.py             #   Dashboard cache restore path
│   ├── state_manager.py            #   Orchestrator state schema and defaults
│   ├── email_sender.py             #   HTML email report generation
│   ├── feature_store.py            #   Daily cached batch price factors
│   ├── analysis_cache.py           #   Persistent deep-analysis cache
│   ├── safe_numeric.py             #   Safe formatting for money/pct/NaN
│   ├── global_universe.py          #   Tiered global stock universe
│   ├── orchestrator_status.py      #   Orchestrator status reporting
│   └── validate_universe.py        #   Universe data validation
│
├── ui/                             # Extracted UI components
│   ├── components.py               #   Shared Streamlit widgets
│   └── sections/
│       ├── discovery.py            #   Discovery tab rendering
│       └── exit_intelligence.py    #   Exit signals tab rendering
│
├── portfolio.example.json          # Template for holdings file
├── .env.example                    # Template for environment variables
└── requirements.txt                # Python dependencies
```

### How It Runs

1. **`daily_orchestrator.py`** runs on a schedule (or manually), performing portfolio analysis, discovery screening, and email report generation.
2. Results are persisted to local state files, SQLite, and JSON caches.
3. **`app.py`** loads cached artifacts first for instant dashboard loads. Live recomputation happens only on explicit refresh.
4. The email engine sends an HTML report with action recommendations, risk flags, swap opportunities, and exit watchlist items.

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/dorretxo/ISA-Trading-Dashboard.git
cd ISA-Trading-Dashboard
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy the template files:

```bash
cp .env.example .env
cp portfolio.example.json portfolio.json
```

Edit **`.env`** with your credentials:

```env
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_FROM=your-email@gmail.com
EMAIL_TO=recipient@example.com
EMAIL_PASSWORD=your-app-password
FMP_API_KEY=              # optional, enables enhanced data
```

Edit **`portfolio.json`** with your holdings:

```json
{
    "currency": "GBP",
    "account_type": "ISA",
    "platform": "Your Broker",
    "holdings": [
        {
            "ticker": "AAPL",
            "name": "Apple Inc",
            "avg_buy_price": 150.00,
            "quantity": 50,
            "currency": "USD"
        }
    ]
}
```

Supported currencies: `GBP`, `GBX` (pence), `USD`, `EUR`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

### 4. Run the orchestrator

```bash
# Dry run (no email, no state changes)
python daily_orchestrator.py --dry-run

# Full run
python daily_orchestrator.py

# Portfolio analysis only (skip discovery)
python daily_orchestrator.py --portfolio-only

# Force discovery even if not scheduled
python daily_orchestrator.py --force-discovery
```

### 5. Schedule automatic runs (Windows)

The included `run_orchestrator.bat` is designed for Windows Task Scheduler. It launches the orchestrator as a detached background process that survives logoff and console closure.

---

## Data Sources

| Source | Data | Required |
|--------|------|----------|
| **yfinance** | Price history, quotes, fundamentals, dividends | Yes (free) |
| **Google News RSS** | Headline flow for sentiment | Yes (free) |
| **FinBERT** | NLP sentiment scoring | Yes (bundled via transformers) |
| **VADER** | Fallback sentiment scoring | Yes (bundled) |
| **Reddit** | Retail/social sentiment from r/stocks, r/investing, r/wallstreetbets | Yes (free) |
| **Financial Modeling Prep** | Enhanced fundamentals, technicals, analyst data, news, screening | Optional (API key) |

The dashboard works without FMP, but discovery screening, news enrichment, and some fundamental indicators are reduced.

---

## Configuration

All tuneable parameters live in [`config.py`](config.py):

- **Scoring weights** &mdash; pillar weights for portfolio analysis and momentum mode
- **Action thresholds** &mdash; score boundaries for STRONG BUY through STRONG SELL
- **Stop-loss settings** &mdash; ATR multipliers, trailing percentages, volatility-adjusted bands
- **Discovery parameters** &mdash; exchange filters, market cap floors, sector concentration limits, funnel sizes
- **Forecast settings** &mdash; horizon days, expert parameters, multi-horizon configuration
- **VIX regime** &mdash; percentile boundaries and weight tilt magnitudes
- **Cache TTLs** &mdash; per data type (quarterly fundamentals, daily technicals, sentiment)
- **Email settings** &mdash; SMTP configuration (credentials via `.env`)
- **Swap logic** &mdash; hurdle rates, portfolio fit minimums, cooldown periods

---

## Privacy and Security

The repository is configured to keep personal data and secrets out of version control.

**`.gitignore` excludes:**
- `.env` (credentials)
- `portfolio.json` (holdings)
- `*.db` (paper trading database)
- `forecast_store.json`, `orchestrator_state.json`, `orchestrator_log.jsonl`
- `feature_cache/` (runtime caches)
- Virtual environments (`.venv/`, `venv/`)

**If you fork this project:**
- Never commit real credentials or portfolio data
- Rotate any credential immediately if accidentally pushed
- Use the `.example` files as templates

---

## Disclaimer

This project is for research and decision support only. It is not financial advice. Always do your own research and validate any trade before execution.
