# Controlled k / c / Bc RAG experiment

Search questions per configuration: 1.
Optimizer results are replays on a fixed table of actual measured answers and token usage. No independent GPU-speedup claim.
At one question this is a smoke result only. Three questions are preliminary; five are the planned small-sample study.

| Method | Mean HV | HV std | Mean unique queries |
|---|---:|---:|---:|
| syftr_style_motpe | 0.751520 | 0.0 | 22.60 |
| nsga2 | 0.751520 | 0.0 | 17.60 |
| hybrid_motpe | 0.751520 | 0.0 | 20.00 |
| adaptive_nsga2 | 0.751520 | 0.0 | 18.10 |

Measured eligible grid cells: 27/27.
The finite-grid frontier is exact only if all 27 cells are eligible; otherwise it is the frontier of valid measured cells.
Optimizer-seed variability is conditional on this small fixed question set and measured table.
This is a Syftr-derived controlled design with an added context budget, not the native/full Syftr search space.
