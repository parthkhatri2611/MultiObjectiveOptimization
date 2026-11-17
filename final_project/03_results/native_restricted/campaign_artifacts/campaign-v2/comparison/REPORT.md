# Audited four-method search comparison

Checkpoint: 25 attempted candidates per arm; declared final budget 25.


One pinned Qwen model; restricted Syftr subset; seed 11. Search metrics are not independent test evidence.

| Method | Attempts | Eligible | Failed | HV | Best F1 | Cheapest tokens/QA |
|---|---:|---:|---:|---:|---:|---:|
| syftr_motpe | 25 | 24 | 1 | 0.392754 | 0.3984313725490196 | 377.2 |
| nsga2 | 25 | 25 | 0 | 0.371658 | 0.37523809523809526 | 379.8 |
| hybrid_motpe | 25 | 24 | 1 | 0.444376 | 0.44857142857142857 | 379.8 |
| adaptive_nsga2 | 25 | 25 | 0 | 0.442513 | 0.44857142857142857 | 380.3 |

All eligible objectives were recomputed from raw answers and generation request/completion journals.

Use methods.csv, pareto_fronts.csv, pairwise_front_coverage.csv, per-run convergence.csv, module outcomes/associations, failure logs, generation logs and raw per-question records for findings.

Coverage counts weak dominance, including identical points. More sampled good configurations is not a causal module-effect claim.

Timing is descriptive under a warm shared index cache; no unbiased wall-clock speedup claim. No across-seed significance claim is supported. Holdout confirmation is separate.
