# Testing And Smoke Rules

- Prefer a small real smoke test over no validation when dependencies are available.
- After editing `app.py`, at minimum run `python -m py_compile app.py`.
- After editing discovery, scoring, or cache modules, run a focused import or narrow smoke test if feasible.
- If a command cannot run because dependencies, credentials, or the runtime are unavailable, state that clearly in the final response.
- Do not claim end-to-end verification unless the app path actually ran.
