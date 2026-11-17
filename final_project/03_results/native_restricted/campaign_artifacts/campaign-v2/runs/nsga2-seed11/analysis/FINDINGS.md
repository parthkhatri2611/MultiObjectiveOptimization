# Measured Syftr local-model baseline findings

Stage: **baseline**. Dataset: **hotpotqa**, partition **test**.
Quality: `f1`; resource: `total_tokens_mean`.

Attempted 25 / 25 candidates; 25 eligible complete evaluations; 6 Pareto configurations.
Normalized hypervolume: 0.371658, fixed resource reference 50000.

This report contains only saved measurements. Failed/partial candidates are excluded from the final frontier but their known calls/tokens remain in search-cost accounting.

## Supported observations

- `pareto.csv`: complete trade-offs and exact configurations.
- `convergence.csv`: hypervolume versus attempted candidates, known calls/tokens and candidate wall time.
- `parameter_observations.csv`: conditional observed component frequencies and average objectives; these are associations, not causal improvements.
- `failures.json`: infrastructure/model/measurement failures, retained in the budget.
- Each candidate contains raw answers, judge outcomes, truncation, per-query timing and nested model request/response usage.

## Interpretation limits

- The generator and local judge share weights: judge bias remains; inspect calibration and human-audit examples.
- Scores and resource use are local adaptations, not reproductions of the paper's hosted-model dollar figures.
- Search-set outcomes require independent holdout confirmation.
- A baseline alone cannot establish whether evolutionary approaches are better.
- Comparison runs must share this exact data/model/search-space/objective protocol and hardware block.

Highest observed quality: 0.375238 at resource 1673.800 (trial 2).
Lowest observed resource: 379.800 at quality 0.105882 (trial 19).
