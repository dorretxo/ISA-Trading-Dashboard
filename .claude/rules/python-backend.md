---
paths:
  - "engine/**/*.py"
  - "utils/**/*.py"
---
# Python Backend Rules

- Preserve contracts consumed by `app.py`, `daily_orchestrator.py`, and persisted caches.
- Prefer extending existing helpers over duplicating scoring or cache logic in new modules.
- When changing ranking semantics, verify all downstream consumers:
  - cache/state writers
  - cache/state loaders
  - UI labels and explanations
  - evaluation/reporting paths
- Handle missing data explicitly. `None` and `NaN` are not interchangeable.
- Prefer cache-safe, backward-compatible changes to runtime schemas when practical.
- Add small helper functions for complex logic rather than long nested conditionals.
- Run a focused syntax/import/smoke check after meaningful backend edits when possible.
