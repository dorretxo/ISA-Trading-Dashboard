---
paths:
  - "engine/discovery.py"
  - "engine/discovery_backtest.py"
  - "engine/discovery_eval.py"
  - "engine/scoring.py"
  - "engine/fundamental.py"
  - "engine/sentiment.py"
  - "engine/forecasting.py"
---
# Discovery Pipeline Rules

- The pipeline is multi-stage with specific ordering. Do not reorder stages without understanding downstream effects.
- `final_rank` uses alpha×confidence+fit: `0.55 * (alpha * confidence) + 0.15 * momentum + 0.30 * portfolio_fit`
- Correlation is a soft graduated penalty, not a hard rejection threshold.
- Feature store is the cheap batch layer — never add per-ticker API calls to the momentum prescreen stage.
- Sentiment confidence metadata (`sentiment_confidence`, `article_count`, `active_sources`) must flow through to the UI.
- Sector-relative P/E uses a local cache fallback when FMP sector_pe is unavailable (non-US stocks).
- Older discovery rows may lack `final_rank` — all reporting paths must handle this with safe defaults.
- When modifying scoring weights or thresholds, verify the evaluation harness still measures meaningful signal.
