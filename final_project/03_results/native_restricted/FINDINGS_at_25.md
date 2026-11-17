# Restricted native Syftr comparison: final 25-attempt audit

Received and independently audited 16 November 2026. This is the restricted native Syftr experiment, distinct from the compact grid replay and the unfinished full-space study.

## Verified evidence

All four downloaded parts match their receipt. The reconstructed ZIP is 14,582,370 bytes, SHA-256 `b131713b8dcb20df388eafe3f737160a6d3ab751014dd0ef4744a4839b4dc4ca`. All 177 outer bundle members and 1,338 capsule members passed size/hash checks. Original parts, receipt, assembled ZIP, source snapshot, Kaggle inventory and raw capsule are retained here. The inventory describes files on Kaggle; it does not contain all those files or model weights.

The trusted local comparison implementation independently recomputed every eligible candidate objective from raw answers/references and generator request/completion journals: 98 candidates, 980 QA rows. Common protocol, data/source/model pins, questions, package environment and ten shared initial parameter configurations matched across arms. No unknown-usage calls were reported.

## Final search results

| Method | Attempts | Completed | Charged duplicates | Hypervolume | Best F1 | Generator search tokens |
|---|---:|---:|---:|---:|---:|---:|
| syftr_motpe | 25 | 24 | 1 | 0.392753796 | 0.398431373 | 322040 |
| nsga2 | 25 | 25 | 0 | 0.371658409 | 0.375238095 | 292225 |
| hybrid_motpe | 25 | 24 | 1 | 0.444376247 | 0.448571429 | 314785 |
| adaptive_nsga2 | 25 | 25 | 0 | 0.442513250 | 0.448571429 | 316327 |

Hybrid hypervolume is 13.14% higher than the local MOTPE reference; adaptive is 12.67% higher, and standard NSGA-II is -5.37% relative to it. These are descriptive observed differences, not significance estimates. Hybrid and adaptive tie on best observed F1.

## Protocol and scope

Pinned Qwen2.5-7B-Instruct Q4_K_M; GPU0 generation and GPU1 embeddings; HotPotQA; ten fixed search questions; optimizer seed 11; 25 attempted configurations per method including ten common initial configurations; population five and three offspring generations for evolutionary arms. Restricted workflows: RAG and sub-question RAG; BGE-small and BGE-base embeddings, with the remaining declared conditional module choices. Primary objectives maximize raw-answer F1 and minimize mean total generator prompt plus completion tokens per QA. Judge tokens are recorded separately; judge correctness is diagnostic. Hypervolume uses the common 50,000-token reference. Failures and duplicates consume budget. See raw protocol.json files for every field.

## Interpretation and mechanisms

The search completed at the same 25-attempt budget for all four arms. The two failures are duplicate_candidate events, one in MOTPE and one in hybrid; they are not missing attempts or generation crashes.

Adaptive generation logs show mutation_multiplier_used = mutation_multiplier_next = 1.0 in all three generations. The adaptation trigger did not activate, so these results do not establish a benefit from changing mutation strength.

Hybrid best F1 came from trial 11, before its first TPE injection. Its final front also includes trial 16 (F1 0.4252380952, 402.1 tokens/QA) and trial 21 (F1 0.1058823529, 379.8 tokens/QA). The observed lead must not automatically be attributed to TPE injections.

Previously audited checkpoint-15 evidence showed the identical trial-11 configuration scored differently between standard NSGA-II and hybrid/adaptive, including a yes/no answer reversal on one question. Evaluation variation can affect this small-set ranking. Its cause has not been isolated. The final result does not remove that caveat.

## Remaining work

**Independent holdout confirmation was not run or recorded.** Frozen finalist selections are preserved: MOTPE trial 15, NSGA-II trial 2, hybrid trial 11, adaptive trial 19. They were chosen by the declared single-point hypervolume rule before holdout outcomes. Search results are not independent generalization evidence; the partition named test was used for optimization.

One seed and ten questions support an exploratory comparison only. No across-seed significance, global optimum, exact paper reproduction, or causal module advantage is established. Cache warming and fixed method order also limit timing comparisons. Compact27 findings remain separate; no medium60 measured return has been received.

The original generated comparison/REPORT.md contains a hardcoded full-space phrase. The protocol explicitly identifies restricted-v2; interpret this experiment as restricted throughout. Original evidence has been preserved without rewriting it.

## Files

- AUDIT_at_25.json: independent audit and full per-method metrics.
- campaign_artifacts/campaign-v2/comparison/: final Pareto fronts, convergence figures, per-question table, pairwise coverage and frozen selections.
- campaign_artifacts/campaign-v2/runs/: raw answers, calls, requests, SQLite snapshots, parameter and optimizer records.
- artifacts/executed_project/: exact returned project source; artifacts/recovery_launchers/: recovery scripts.
- Original four parts and receipt plus reconstructed ZIP: immutable returned evidence.
