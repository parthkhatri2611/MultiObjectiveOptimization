# Audited compact k / c / Bc experiment

Received and audited on 16 November 2026. This is the controlled Syftr-derived finite-grid RAG experiment, with measured fitness replay. It is separate from the native restricted/full Syftr campaigns.

## Main finding

All four methods recovered the same measured finite-grid Pareto front by the 25-query budget in all ten optimizer seeds. Final normalized hypervolume was **0.5229264**, with zero across-seed standard deviation, zero hypervolume gap and zero IGD to this measured front. This experiment supplies **no evidence that our evolutionary approaches outperform MOTPE**. It also does not establish equivalence in larger search spaces or unseen questions.

Seven of the ten shared initial pools had already recovered the complete front by query 10, before method-specific search began. The remaining three had only very small hypervolume gaps. This small, redundant landscape is an easy optimizer task and therefore a limited stress test.

## Evidence and preservation

The returned ZIP is 3,525,628 bytes, SHA-256 `85e0ef0edda8bd3b285a20af4c45e4e5161cf61942bcd5fb1bc2eecc8883c611`. It matches the supplied receipt. All 234 content-manifest file hashes verify. Original ZIP and receipt are saved alongside this report; extracted artifacts are under `artifacts/compact_kcb_results/`.

The independent audit recomputed F1/EM, objective means, Pareto fronts, hypervolume trajectories, shared initial configurations, proposal-to-table lookup, charged repeats and equivalent token costs. It checked 258 QA record references across progressive search stages and holdout, linking them to raw endpoint responses and exact request contexts. Search/holdout IDs and question texts are disjoint. Frozen representatives match the prespecified search-only selection rule. All 77 physical model responses terminated with `stop`; no failed model calls were recorded. Recorded context counts obey Bc, and context hashes/request bodies agree. The pinned tokenizer was not independently rerun in the local return audit; this is a remaining verification limit.

The three software tests are software checks, not research measurements. The actual research measurements are the archived answers and endpoint usage from Kaggle.

## Experimental scope

- Model: pinned Qwen2.5-7B-Instruct Q4_K_M; generator on GPU0, BGE-small embeddings on GPU1.
- Data: pinned prepared DataRobot HotpotQA `train_hard`; five fixed test questions for search, five distinct holdout questions for confirmation. All selected questions are marked hard.
- Whole prepared grounding splits used: 4,972 test documents and 78,102 holdout documents. No question-specific gold evidence was supplied to retrieval. Corpus size differs considerably across splits, so the search/holdout gap is not a clean same-corpus comparison.
- Variables: k ∈ {2,4,6}, c ∈ {256,512,1024} Qwen tokens, Bc ∈ {512,1024,2048} Qwen context tokens. 27 cells; fixed dense single-call RAG, 10% chunk overlap, temperature zero, output cap 256.
- The context budget is an added controlled design choice. BGE truncates each chunk representation at 512 embedding-tokenizer tokens; c uses generator tokens. These units are not interchangeable.
- Objectives: maximize raw-answer token F1, minimize mean generator input+output tokens per QA. Hypervolume reference is fixed at (quality 0, resource 5,000).
- Each optimizer: 25 charged queries, including ten identical distinct initial configurations for a given seed; ten optimizer seeds. NSGA-II population five, three offspring generations; hybrid has six MOTPE injection slots. Repeated configuration queries consume budget.
- Once measured, the table supplies deterministic fitness to replay. These are not independent GPU inference campaigns for each method/seed. Ten optimizer seeds describe search randomness conditional on one small question table.

## Measured Pareto front

| Operating point | Search F1 | Mean generator tokens/QA | Settings |
|---|---:|---:|---|
| Lower resource | 0.4600 | 289.8 | k=2, all tested c and Bc values |
| Higher quality | 0.5600 | 520.6 | k=4, Bc=512, all tested c values |

There are **two distinct objective points**, represented by **12 nondominated configurations** because several settings tie. Moving from the lower-resource point to the higher-quality point adds 0.10 absolute F1 and 230.8 tokens/QA, approximately 79.6% more tokens on these five questions. The low-resource point saves approximately 44.3% of the high-quality point's tokens. These statements are restricted to the observed search set.

All 27 configurations are eligible. However, the grid has only 11 distinct objective pairs and 18 distinct five-question prompt sequences. Parameter redundancy and context truncation reduce effective complexity. A native-space performance claim cannot be inferred from this small grid.

## Optimizer comparison

| Method | Final HV, mean ± SD | Unique cells, mean | Repeated queries, mean | First full-front HV query, mean |
|---|---:|---:|---:|---:|
| Syftr-style MOTPE | 0.5229264 ± 0 | 20.5 | 4.5 | 8.5 |
| NSGA-II | 0.5229264 ± 0 | 17.0 | 8.0 | 9.9 |
| Hybrid MOTPE + NSGA-II | 0.5229264 ± 0 | 18.0 | 7.0 | 9.9 |
| Adaptive NSGA-II | 0.5229264 ± 0 | 18.3 | 6.7 | 10.1 |

The first-hit statistic includes initialization. Only three seeds needed method-specific work to recover the last missing Pareto objective. For those seeds (11,37,97), first-hit query indices were MOTPE (16,12,14), NSGA-II (16,24,16), hybrid (19,20,17), adaptive (16,22,20). MOTPE tied or reached the target earlier than every evolutionary method in these three cases. This is a descriptive convergence finding, not a significance/general superiority claim.

MOTPE explored more unique configurations. Adaptive mutation reduced repeats relative to fixed NSGA-II (6.7 vs 8.0 mean), but did not improve final hypervolume or target attainment in this experiment. Hybrid injection also reduced repeats relative to NSGA-II (7.0 vs 8.0), without final-front improvement.

Equivalent generator-token expenditures across the 25-query episodes averaged 63,376.5 (MOTPE), 62,151.9 (NSGA-II), 62,251.5 (hybrid), and 65,193.7 (adaptive). These are **logical replay charges**, not additional physically spent GPU tokens; lower cost alone is not evidence of better optimization.

## Holdout confirmation

Representatives were chosen using the declared maximum single-point search-HV rule, with quality, cost and earliest discovery tie breakers. Because all methods reached the same search-quality point, the selected k=4, Bc=512 settings differ mainly in c. Deduplicating the 40 method/seed selections left three actual holdout configurations.

| Selected setting | Search F1 | Holdout F1 | Holdout EM | Holdout tokens/QA |
|---|---:|---:|---:|---:|
| k=4, c=256, Bc=512 | 0.5600 | 0.2571 | 0.20 | 535.6 |
| k=4, c=512, Bc=512 | 0.5600 | 0.2571 | 0.20 | 535.6 |
| k=4, c=1024, Bc=512 | 0.5600 | 0.0571 | 0.00 | 535.6 |

The substantial holdout drop shows that strong search-set quality did not reliably generalize. One extra correct answer accounts for the 0.20 F1 difference between the first two settings and c=1024 on only five questions. All three predicted Istanbul against the longer reference “European side of the city of Istanbul,” receiving partial F1 0.2857 for that question. Several other holdout answers were `unknown`; another predicted Bristol Rovers against Watford. Attribution to retrieval, truncation or generation requires further analysis; this archive alone does not isolate a cause.

Across the ten selected representatives per method, mean holdout F1 is 0.1371 MOTPE, 0.1571 NSGA-II, 0.1371 hybrid, 0.1371 adaptive. These are repeated selections from only three configurations on the **same five questions**, not 50 independent observations per method. The small 0.02 difference does not justify claiming NSGA-II superiority. Do not select c=256 after seeing holdout and then describe that choice as independently validated: that would use the holdout for tuning.

Raw-answer F1 also has limitations. For example, `Kevin Wilson` against `Kevin Reece Wilson` receives 0.8; `six weeks` against `six-week television series` receives zero under the frozen normalization/scoring rule. Retain the predefined primary metric; any semantic/manual assessment should be a separately labelled secondary analysis applied consistently.

## Parameter findings

The marginal search-set means, averaged over the other two factors, are:

| Factor | Value | Mean F1 | Mean tokens/QA |
|---|---:|---:|---:|
| k | 2 | 0.4600 | 289.8 |
| k | 4 | 0.5600 | 550.6 |
| k | 6 | 0.5156 | 797.8 |
| c | 256 | 0.5044 | 538.2 |
| c | 512 | 0.5119 | 550.0 |
| c | 1024 | 0.5193 | 550.0 |
| Bc | 512 | 0.5193 | 459.8 |
| Bc | 1024 | 0.5044 | 585.7 |
| Bc | 2048 | 0.5119 | 592.7 |

Increasing retrieved context did not monotonically improve quality. k=4 provided the strongest marginal F1, while k=6 added resource without matching that quality. The smallest context budget had the best marginal F1 and lowest cost here. c=1024 had slightly higher marginal search F1 but the weakest selected holdout setting. These are descriptive finite-sample effects, not universal recommended parameter settings.

## Cost, progress and deliverables

135 logical search configuration-question evaluations plus 15 selected holdout evaluations were completed. Exact-request caching reduced this to **77 physical inference calls**, **47,228 total generator tokens**, zero failed calls. Embedding calls, setup preflight and software tests are not included in this generator-call/token count. Optimizer replay itself added no physical generator calls.

Progressive stage HV values were 0.75152 (one question), 0.72588 (three), and 0.5229264 (five). These use different question sets and cannot be treated as a single optimizer convergence trajectory. Their variation highlights the instability of tiny QA samples.

Available artifacts include raw requests/responses, all configuration QA rows, three stage reports, per-query trajectories, generation/mutation histories, hypervolume/IGD gaps, Pareto and convergence PNG/PDF files, frozen holdout selection, actual holdout outcomes and environment/protocol metadata. See `AUDIT.json` for machine-readable independent checks and `audit_return.py` in the deployment directory for reproducibility.

## How to present this honestly

Suggested statement: “We implemented three evolutionary search strategies and compared them with a Syftr-style MOTPE baseline on a controlled 27-configuration RAG landscape measured with a local Qwen model. All methods recovered the same finite-grid Pareto front; evolutionary search did not show an advantage at this budget. MOTPE explored more unique settings and matched or improved convergence in the seeds that required search beyond initialization. A separate five-question holdout exposed a large search-to-holdout performance drop.”

Do not call this a reproduction of the original paper's scores, claim statistically significant gains, use replay timings as inference speedups, or pool it with full/restricted-space runs. The restricted native Syftr experiment remains a separate source of evidence and should be audited when its ZIP is available.

Potential follow-up designs: predeclare a larger QA sample, use a less redundant grid with smaller initial fraction, and evaluate duplicate-aware evolutionary proposals under a fair charged budget. These are proposed next experiments, not results from this run. Preserve this completed experiment unchanged.
