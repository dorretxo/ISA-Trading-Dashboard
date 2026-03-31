# ISA Portfolio Dashboard

A cache-first stock analysis and portfolio decision system for manually managed portfolios.

This project combines:

- a Streamlit dashboard for reviewing holdings, risks, exits, optimizer output, discovery candidates, and performance
- a scheduled orchestrator for precomputing heavy analysis in the background
- a multi-factor scoring engine built from technical, fundamental, sentiment, and forecasting signals
- a portfolio-aware optimizer designed around a 90-day decision cycle

The app is designed for investors who still execute trades manually, but want a disciplined, data-driven workflow for deciding what to hold, trim, replace, or monitor next.

## What It Does

- Scores each holding across four pillars:
  - technical
  - fundamental
  - sentiment
  - forecasting
- Produces action labels such as `STRONG BUY`, `BUY`, `KEEP`, `SELL`, and `STRONG SELL`
- Applies a risk overlay for:
  - parabolic price moves
  - earnings-event proximity
  - market-cap-aware confidence and sizing constraints
- Optimizes the current portfolio jointly instead of treating positions independently
- Runs a global discovery pipeline to find potential replacement candidates
- Tracks signal outcomes and computes scorecards from the paper-trading/evaluation data
- Uses a cache-first dashboard model so the UI loads from precomputed artifacts by default

## Key Features

- `Portfolio analysis`
  - per-holding scoring, reasoning, P&L, stop/target logic, and detailed drill-downs
- `Portfolio optimizer`
  - mean-variance allocation with turnover, FX, sector concentration, and max-weight constraints
- `Exit intelligence`
  - stop proximity, score deterioration, and tactical exit signals
- `Global discovery engine`
  - multi-stage screening, portfolio-fit ranking, and swap candidate generation
- `90-day projection`
  - interactive Monte Carlo portfolio return projection
- `Evaluation harness`
  - scorecards, hit rates, alpha vs benchmark, drawdown, and forecast error reporting
- `Background orchestration`
  - scheduled recomputation and cached artifacts for fast dashboard loads

## Architecture

At a high level:

1. `daily_orchestrator.py` runs the heavy analysis jobs.
2. Results are persisted into local state files and SQLite.
3. `app.py` loads cached artifacts first and only recomputes live on explicit refresh.
4. The dashboard presents analysis, optimizer output, discovery candidates, and historical effectiveness in one place.

Important entry points:

- [`app.py`](app.py): Streamlit dashboard
- [`daily_orchestrator.py`](daily_orchestrator.py): scheduled/background runner
- [`config.py`](config.py): central configuration
- [`engine/`](engine/): scoring, forecasting, optimizer, discovery, exits, evaluation
- [`utils/`](utils/): cache/state/data/email helpers

## Project Layout

```text
.
├── app.py
├── daily_orchestrator.py
├── config.py
├── engine/
│   ├── scoring.py
│   ├── discovery.py
│   ├── forecasting.py
│   ├── portfolio_optimizer.py
│   ├── portfolio_projection.py
│   ├── evaluation_harness.py
│   └── ...
├── utils/
│   ├── cache_loader.py
│   ├── data_fetch.py
│   ├── fmp_client.py
│   ├── state_manager.py
│   └── ...
├── portfolio.example.json
├── .env.example
└── requirements.txt
```

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add your local configuration

Copy the example files:

```powershell
Copy-Item .env.example .env
Copy-Item portfolio.example.json portfolio.json
```

Then edit:

- `.env`
  - `EMAIL_SMTP_HOST`
  - `EMAIL_SMTP_PORT`
  - `EMAIL_FROM`
  - `EMAIL_TO`
  - `EMAIL_PASSWORD`
  - optional `FMP_API_KEY`
- `portfolio.json`
  - your holdings, quantities, buy prices, and currencies

### 3. Launch the dashboard

```powershell
streamlit run app.py
```

### 4. Run the orchestrator manually

Dry run:

```powershell
python daily_orchestrator.py --dry-run
```

Normal run:

```powershell
python daily_orchestrator.py
```

Optional flags:

```powershell
python daily_orchestrator.py --force-discovery
python daily_orchestrator.py --portfolio-only
```

## Data Sources

- `yfinance`
  - price history, quotes, and broad market coverage
- `Financial Modeling Prep (FMP)`
  - enhanced fundamentals, technicals, analyst data, news, and screening
- `Google News RSS`
  - headline flow
- `Reddit`
  - retail/social sentiment context
- `FinBERT / NLP sentiment pipeline`
  - headline sentiment scoring

The app still works without FMP, but some discovery, news, and fundamental enrichment features will be reduced.

## Typical Workflow

1. Update `portfolio.json` locally with current holdings.
2. Run the orchestrator on a schedule or manually.
3. Open the dashboard to review cached analysis immediately.
4. Use `Refresh Analysis` only when you want a live recomputation.
5. Review:
   - current holdings
   - optimizer suggestions
   - exit signals
   - discovery candidates
   - performance scorecard
6. Execute trades manually with your broker.
7. Record completed sales in the dashboard to keep history and learning loops current.

## Privacy And Security

This repository is set up to keep personal data and secrets out of version control.

Ignored local-only files include:

- `.env`
- `portfolio.json`
- orchestrator state/log artifacts
- forecast stores and local databases
- generated architecture documents
- local workspace config

If you fork or publish this project:

- never commit real credentials
- never commit your live portfolio
- rotate any credential immediately if it is ever accidentally pushed
- use the example files as templates instead of committing personal data

## Notes

- Trades are executed manually. This is a decision-support tool, not an auto-execution bot.
- The system is tuned around a practical 90-day review/rebalance cycle rather than high-frequency trading.
- Discovery and forecasting are probabilistic decision aids, not guarantees.

## Disclaimer

This project is for research and decision support only. It is not financial advice. Always do your own research and validate any trade before execution.
