# Completed GPU pilot review — 15 November 2026

## Decision: ready for the baseline

The completed pilot passed integrity and measurement checks. Proceed with seed 11 of the frozen 30-attempt, 20-question F1/token baseline. Run seeds 23 and 37 separately after preserving each result. The Kaggle source remains the validated 115-file GPU snapshot; no source upload, capability rerun, or model rebuild is required for the launcher.

## Audit

- Archive `gpu1_pilot_completed_review.zip`: 794 files, SHA-256 `c39a59442a0e750337c71aa5a3a1576119f2734b3af9a56b6fa9d02fbce00f27`. It includes the actual completed GPU pilot, unlike the earlier renamed capability ZIP.
- Verified all 115 source files and the source identity against the GPU capability and readiness records. Source/model/data revisions and the declared GPU policy match.
- All ten trials completed the same ten test QA IDs: 100 evaluations, zero failed candidates, zero generation/judge parsing errors, zero truncated question outputs.
- Matched request/completion IDs, checked duplicate IDs and successful HTTP usage records, recomputed answer F1/EM and candidate token means.
- Verified 499 calls: 399 generator-workflow calls and 100 diagnostic judge calls. Generator tokens: 245,743; judge tokens: 18,490; zero unknown-usage calls.
- Every candidate device record confirms physical GPU 1 with a common UUID. The constructor enforces actual CUDA placement. GPU 0 serves the unchanged generator.
- Recomputed Pareto membership: four configurations, four unique objective pairs. Recomputed normalized hypervolume: 0.7049886372 at the fixed 50,000-token reference.
- A prompt spot-check found no exact repeated `Question: <current query>` across the candidate prompts. Source inspection confirms few-shot examples are drawn from the train partition. This does not prove absence of all semantic duplicates in the upstream dataset.

## Measurements

| Trial | Workflow | Raw F1 | Tokens / QA | Construction s | Mean query s | Candidate wall s |
|---|---|---:|---:|---:|---:|---:|
| 0 | rag | 0.10588 | 373.1 | 33.82 | 0.99 | 74.17 |
| 1 | sub_question_rag | 0.05148 | 1641.1 | 1.95 | 5.95 | 91.42 |
| 2 | critique_rag_agent | 0.05644 | 3205.3 | 2.01 | 12.59 | 162.66 |
| 3 | react_rag_agent | 0.09204 | 3799.8 | 1.93 | 12.86 | 158.39 |
| 4 | lats_rag_agent | 0.06609 | 8064.7 | 1.97 | 31.34 | 343.56 |
| 5 | rag | 0.14491 | 1489.8 | 274.76 | 2.27 | 323.97 |
| 6 | rag | 0.52187 | 1312.4 | 304.53 | 2.54 | 355.25 |
| 7 | rag | 0.04383 | 1084.8 | 21.92 | 3.17 | 87.05 |
| 8 | rag | 0.52524 | 1687.4 | 310.48 | 7.56 | 412.02 |
| 9 | rag | 0.72444 | 1915.9 | 393.77 | 3.97 | 459.51 |

Candidate wall time summed to 2,467.99 seconds, about 41.13 minutes. Construction accounted for 1,347.13 seconds, about 22.45 minutes. The longest complete candidate took 459.51 seconds. These are observed pilot costs, not guaranteed baseline completion times. The first five were standard initialization workflows; trials 5–9 were TPE-proposed under the pilot startup setting of five.

Trial 5, the BGE-large hybrid configuration previously interrupted on CPU, completed in 323.97 seconds; construction took 274.76 seconds. The CPU version was interrupted before completion, so a precise full-candidate CPU/GPU speedup cannot be computed.

The observed pilot frontier consists of trials 0, 6, 8 and 9. Trial 9 attained raw F1 0.72444 at 1,915.9 generator tokens/QA; trial 0 used 373.1 tokens/QA at F1 0.10588. Several parameters change together, particularly few-shot settings and answer formatting, so higher F1 is not a causal claim about an embedding model. These ten search questions are not holdout evidence.

## Conditional sampler warnings

The controller log contains 10 unique parameter/trial independent-sampling warnings. They arise when conditional dimensions are outside the current multivariate relative search space. They did not stop evaluations. Preserve the validated sampler: Optuna 4.2.1, multivariate=True, constant_liar=True, group=False. Describe the baseline as multivariate MOTPE with conditional independent sampling, not as a joint model of every conditional dimension. Do not silently enable group=True, suppress issues, or switch sampler behavior halfway through the comparison. The warnings and exact settings remain archived.

## Baseline next step

Use `RUN_SYFTR_BASELINE_GPU1.py` in one new Kaggle cell. Leave SEED=11 for the first run. The launcher copies the existing baseline F1 profile, adds the GPU-1 and cache policies, checks source/server/capability identity, creates an isolated index workspace for this seed, and starts a NEW 30-attempt/20-question study. It shares only downloaded model/data files with the pilot; it does not inherit the pilot vector indexes or study observations. No Kaggle source file is changed.

It prints per-minute controller and active-worker progress, records the exact command, and automatically exports uniquely named review/checkpoint archives on normal completion or interruption. Review the budget_exhausted field; a four-hour time-window stop is not automatically a completed baseline. Resume the same seed with unchanged source/profile when needed. Do not run experiments concurrently.

## Validation and preservation

- All 22 baseline/GPU-routing unit tests passed after synchronizing the reviewed GPU support into the local project.
- A mocked launcher fixture verified profile values, fresh isolated indexes/shared download cache, resume flag selection, unique review/checkpoint exports, active-source recovery pointer, and unchanged original server metadata. The baseline itself has not been executed here.
- Original completed pilot ZIP and its extracted artifacts are retained under `ourwork/kaggle_returns/gpu1-pilot-completed`. The frozen executed Kaggle source remains separately captured there; newer local serving/packaging helpers are not silently deployed.
- `BASELINE_CAMPAIGN_PROTOCOL.md` and the JSON files in `ourwork/experiment_protocols` record the comparison plan. Use the same evaluation and storage policy for later algorithms; do not pool CPU/GPU timings.
