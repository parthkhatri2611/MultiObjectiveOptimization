# Audited four-method search comparison

Checkpoint: 15 attempted candidates per arm; declared final budget 25.


One pinned Qwen model; restricted Syftr subset; seed 11. Search metrics are not independent test evidence.

| Method | Attempts | Eligible | Failed | HV | Best F1 | Cheapest tokens/QA |
|---|---:|---:|---:|---:|---:|---:|
| syftr_motpe | 15 | 14 | 1 | 0.367165 | 0.37523809523809526 | 377.2 |
| nsga2 | 15 | 15 | 0 | 0.371062 | 0.37523809523809526 | 381.7 |
| hybrid_motpe | 15 | 15 | 0 | 0.441597 | 0.44857142857142857 | 381.7 |
| adaptive_nsga2 | 15 | 15 | 0 | 0.441805 | 0.44857142857142857 | 381.7 |

All eligible objectives were recomputed from raw answers and generation request/completion journals.

Use methods.csv, pareto_fronts.csv, pairwise_front_coverage.csv, per-run convergence.csv, module outcomes/associations, failure logs, generation logs and raw per-question records for findings.

Coverage counts weak dominance, including identical points. More sampled good configurations is not a causal module-effect claim.

Timing is descriptive under a warm shared index cache; no unbiased wall-clock speedup claim. No across-seed significance claim is supported. Holdout confirmation is separate.
