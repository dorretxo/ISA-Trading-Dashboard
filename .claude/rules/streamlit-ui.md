---
paths:
  - "app.py"
---
# Streamlit UI Rules

- Optimize first for decision clarity, then for density.
- Use cards for summary, triage, and recommendation surfaces.
- Keep tables for large scan-heavy datasets such as holdings lists, scored-candidate tables, and history logs.
- Preserve the existing visual language before introducing new patterns.
- Use `utils.safe_numeric` formatting helpers for money, percentages, and nullable values.
- Avoid raw currency or percent f-strings on values that may be missing, cached, or derived from external data.
- Keep expensive work out of the render path when cached data already exists.
- If UI logic depends on cached discovery or orchestrator state, verify the restore path after changes.
