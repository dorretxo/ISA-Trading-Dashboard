# Trading Dashboard

## Purpose
- Streamlit app for reviewing portfolio holdings, optimizer output, exits, discovery candidates, and historical signal quality.
- `daily_orchestrator.py` precomputes heavy analysis and writes cached artifacts so the dashboard stays fast.
- Trades are executed manually. This repo is a decision-support system, not an auto-trading bot.

## Core Commands
- `streamlit run app.py`
- `python daily_orchestrator.py`
- `python daily_orchestrator.py --dry-run`
- `python daily_orchestrator.py --force-discovery`
- `python daily_orchestrator.py --portfolio-only`
- `python -m py_compile app.py`

## Architecture
- `app.py`
  Streamlit entry point and UI composition. High-risk file because many sections live here.
- `daily_orchestrator.py`
  Background runner that refreshes portfolio analysis, discovery, and cached artifacts.
- `config.py`
  Central configuration, API keys, feature toggles, and cache TTLs.
- `engine/`
  Analysis and ranking logic.
  Important modules:
  - `scoring.py`: full per-holding analysis
  - `discovery.py`: screening, ranking, portfolio-fit logic
  - `discovery_backtest.py`: discovery signal recording and evaluation storage
  - `discovery_eval.py`: discovery-specific reporting
  - `forecasting.py`: forecast engine and persistent forecast history
  - `sentiment.py`: sentiment analysis and caching
  - `technical.py`, `fundamental.py`, `risk_overlay.py`
- `utils/`
  Runtime/state helpers.
  Important modules:
  - `cache_loader.py`: dashboard cache restore path
  - `state_manager.py`: persisted state schema/defaults
  - `feature_store.py`: daily cached batch price factors
  - `analysis_cache.py`: persistent deep-analysis cache
  - `safe_numeric.py`: safe formatting and NaN handling
- `feature_cache/`
  Runtime cache directory for feature factors and persistent sentiment/forecast outputs.

## Data And Runtime Artifacts
- Local-only runtime files include:
  - `.env`
  - `portfolio.json`
  - `paper_trading.db`
  - `forecast_store.json`
  - `orchestrator_state.json`
  - `orchestrator_log.jsonl`
  - `orchestrator_output.log`
  - `feature_cache/`
- Treat runtime files as user data. Do not delete or reset them unless explicitly asked.

## Working Conventions
- Prefer backend changes in `engine/` and `utils/` over embedding business logic directly in `app.py`.
- Keep cache-first behavior intact. The UI should load restored artifacts quickly and only recompute on explicit refresh paths.
- Use `final_rank` for discovery recommendation ordering. Any evaluation or reporting path should use `final_rank` with safe fallback for older rows.
- When changing discovery or scoring outputs, update all dependent restore/reporting paths together:
  - producer
  - persisted schema/cache
  - loader
  - UI consumer
- Preserve backward compatibility for older cached payloads where practical.
- Prefer small helper extraction over long inline blocks, especially in `app.py`.

## UI Conventions
- Use card-based layouts for decision-heavy sections.
- Keep dense scanning surfaces table-first.
- Use safe numeric formatting helpers from `utils/safe_numeric.py`.
- Do not format money or percentages directly with raw f-strings when values may be missing or NaN.
- Confidence and data quality should be visible in the UI when recommendations depend on incomplete data.

## Discovery And Scoring Notes
- The discovery pipeline is multi-stage:
  - universe assembly
  - momentum prescreen
  - quick filter
  - portfolio correlation/fit logic
  - quick rank
  - deep scoring
  - final ranking
- Feature-store caching speeds repeated same-day discovery runs.
- Correlation is now a soft penalty, not a hard rejection.
- Discovery evaluation uses `paper_trading.db`.
- Older discovery rows may not have `final_rank`; fallback logic must remain intact.

## Cache And State Watchouts
- If you add a new field to cached discovery or orchestrator state, update:
  - writer path
  - `utils/state_manager.py` defaults
  - `utils/cache_loader.py`
  - any UI restore logic in `app.py`
- Persistent deep-analysis caches live under `feature_cache/`.
- `Refresh Analysis` should not silently invalidate persistent caches unless the user expects a cold refresh.

## Safety
- Never read or expose secrets from `.env`.
- Never commit personal data or credentials.
- Avoid destructive commands such as deleting caches, DBs, or state unless explicitly requested.

## Preferred Validation
- After touching `app.py`, run `python -m py_compile app.py`.
- After touching discovery/scoring modules, run a focused import or smoke test when feasible.
- If a runtime test cannot be executed, say so clearly and explain why.
