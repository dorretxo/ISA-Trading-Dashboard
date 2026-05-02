---
description: Review discovery-related changes for ranking, cache, and UI regressions
---
## Changed Files

!`git diff --name-only -- engine/discovery.py engine/discovery_backtest.py engine/discovery_eval.py utils/feature_store.py utils/analysis_cache.py app.py`

## Discovery Diff

!`git diff -- engine/discovery.py engine/discovery_backtest.py engine/discovery_eval.py utils/feature_store.py utils/analysis_cache.py app.py`

Review the above changes with a code-review mindset.

Focus on:
1. Ranking correctness and `final_rank` usage
2. Cache/state backward compatibility
3. UI restore-path regressions
4. Missing validation or smoke-test coverage
