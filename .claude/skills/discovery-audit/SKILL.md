---
name: discovery-audit
description: Comprehensive discovery pipeline audit. Use when reviewing discovery
  output quality, checking recommendation accuracy, or when the user mentions
  discovery results look wrong, scores are zero, or candidates seem low quality.
allowed-tools: Read, Grep, Glob, Bash(python -c *)
---

Perform a discovery pipeline quality audit:

1. **Feature Store Freshness**
   Check `feature_cache/` for today's features file. Report ticker count and staleness.

2. **Funnel Drop-off**
   Trace the last run's numbers through each stage:
   - Universe size → momentum survivors → quick filter → correlation → deep scored → final ranked

3. **Score Distribution**
   Check if final_rank values cluster near zero (indicates data quality issues).
   Check sentiment_confidence distribution (low = rate limiting problem).

4. **Known Failure Patterns**
   - All metrics zero → feature store empty or stale
   - Identical sentiment scores → Google News rate limited
   - NaN in rankings → missing price data propagation
   - forecast_store.json > 50MB → needs pruning

5. **Evaluation Quality**
   Reference @../../engine/discovery_eval.py for forward return tracking.
   Check if paper_trading.db has recent entries.

Report with specific data and actionable recommendations.
