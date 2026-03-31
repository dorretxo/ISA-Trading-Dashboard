---
paths:
  - "daily_orchestrator.py"
  - "utils/cache_loader.py"
  - "utils/state_manager.py"
  - "utils/feature_store.py"
  - "utils/analysis_cache.py"
  - "engine/discovery_backtest.py"
  - "engine/discovery_eval.py"
---
# State And Cache Rules

- Treat cache and state files as user-facing data contracts.
- When adding fields, update writer, loader, default schema, and UI reader paths together.
- Prefer additive migrations over breaking rewrites.
- Keep restore paths tolerant of older payloads and missing keys.
- Distinguish between:
  - source-of-truth runtime state
  - derived caches
  - historical evaluation storage
- Use atomic writes or existing save helpers when persisting JSON/state.
- Discovery evaluation should remain aligned with live recommendation ordering.
- Avoid silently deleting runtime artifacts unless the user explicitly requests it.
