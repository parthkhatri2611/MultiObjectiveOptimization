# Research Methodology Project context and continuity record

Last consolidated: 16 November 2026, India time. Project root: `/media/parth/New Volume/Ongoing Projects/Research_Methodology_Project`.

This is the main continuity document for the project. Read it before running code, interpreting results, or following older handoff documents. It records the history and the latest evidence available locally. Historical plans are not automatically current instructions. A running Kaggle session can advance beyond the most recent downloaded checkpoint; remote progress is not known unless fresh output is returned.

## 1 Current position at a glance

| Workstream | Latest locally verified state | Where to resume |
|---|---|---|
| Base paper and project analysis | Complete detailed analysis of supplied Syftr paper, repository and proposal | `ourwork/COMPLETE_PROJECT_AND_SYFTR_ANALYSIS.md` |
| Hardware and model integration | Local PC assessed; Kaggle dual T4 pipeline validated with pinned Qwen | Hardware plan and GPU pilot audit |
| Original 30 by 20 baseline | Historical partial summary only; unexported raw records lost with old session | Preserve history; do not reconstruct or resume from summary |
| Native full space v2 | Runnable deployment prepared; no completed full space return audited locally | `ourwork/campaign_v2_deployment/` |
| Native restricted v2 | **Final matched 25 attempts independently audited**; 98 completed candidates, two charged duplicates; holdout pending | See final25 findings and preserved frozen holdout selections |
| Compact 27 cell study | **Complete**, including four optimizer replays, holdout, audit, PDF and Word report | `ourwork/experiment_archive/compact_kcb_2026-11-16/` |
| Medium 60 cell study | **Prepared and software tested; no actual medium result received yet** | `ourwork/medium_kcb_deployment/` |
| Overall project | Real findings exist, but native holdout, medium experiment, consolidated final report and presentation remain outstanding | Follow the outstanding work list below |

The most recent user request is to preserve all work and findings in a folder and create this complete context record. A fresh verified consolidated snapshot is being created under `ourwork/research_checkpoints/`; `LATEST.json` identifies the final archive and checksum once complete.

## 2 Original aim and the user's decisions

The project uses the Syftr paper as its base research paper. The downloaded GitHub implementation is present in this workspace. The proposal is `PPTnall/RM.pptx` (the user initially called it RM.ppt). The supplied base paper is `PPTnall/syftr-BasePaper.pdf`; RAGGED and LLMLingua supporting PDFs are in the same folder.

The research question is whether alternative evolutionary optimization methods discover better quality and resource tradeoffs than a Syftr style multiobjective TPE reference under a common evaluation budget. The three proposed approaches are NSGA-II, hybrid NSGA-II with MOTPE injections, and adaptive mutation NSGA-II. They are search algorithms over RAG configurations, not three new language models.

Initially the user asked for a complete project/paper analysis before implementation. The user then clarified that the three-variable k, c, Bc formulation should be set aside in favor of working on Syftr's hierarchical space. Later, because native agentic runs were slow and session failures created deadline risk, the user explicitly requested a small three-variable fallback. That is why two distinct kinds of experiments now exist. Neither should be passed off as the other.

The original Syftr implementation uses multiple hosted models/APIs unavailable to the team. We substituted a pinned self-hosted open model shared across methods and chose a token resource proxy. This adapts the methodology; it does not reproduce the paper's original absolute scores, model catalog, API dollar costs or all datasets.

The user authorized local code work, hardware inspection, debugging, packaging, preservation and analysis. The user prefers to operate Kaggle manually and return output archives while the assistant builds/debugs/analyzes locally. The user initially reported about 48 hours, later about 12 hours remaining; these were planning constraints, not reliable runtime forecasts. Do not promise completion or algorithm superiority based on them.

## 3 Initial review and hardware assessment

The analysis on 14 November covered the 44-page supplied Syftr PDF, its appendix material, the proposal's 26 slides, the initial project inventory, Python source structure, study configurations and existing notes. The original read-only review recorded 275 non-Git files and parsed 128 Python files. Repository HEAD observed locally is `7595f1d`; this is a checkout identity, not a commit containing all later uncommitted work.

The original PC audit found an Acer Swift SFG14-73, Intel Core Ultra 7 155H, about 15.3 GiB usable RAM, integrated Intel Arc/Meteor Lake GPU and NPU, and no NVIDIA GPU. Available memory/disk were time-specific observations. Local work therefore focused on implementation, tests, analysis and reporting; GPU model inference moved to Kaggle. Do not assume that old available-RAM/disk figures still describe the current machine.

Kaggle runs recorded two Tesla T4 GPUs, roughly 15 GiB each, and about 30 GiB system RAM. GPU 0 serves generation; physical GPU 1 handles embeddings. Two GPUs are not automatically one pooled VRAM device. UUID-based CUDA isolation exposes the second physical device as logical cuda:0 in the embedding process.

Read `ourwork/HARDWARE_AND_KAGGLE_EXECUTION_PLAN.md`, `hardware_audit_2026-11-14.json`, and `pc_local_hardware_and_baseline_pipeline.md` for the original evidence. Current quota/session availability must be checked in Kaggle, not inferred from historical documentation.

## 4 Frozen local model and source identities

These identities are reused where the declared protocol says so. Each experiment's saved manifest remains authoritative.

| Component | Pinned identity |
|---|---|
| Generator | Qwen/Qwen2.5-7B-Instruct-GGUF, Q4_K_M |
| Model revision | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` |
| Model alias | `rm-qwen25-7b-q4` |
| Tokenizer | Qwen/Qwen2.5-7B-Instruct |
| Tokenizer revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| llama.cpp revision | `32bdc3a89e6528d3e419024c833c5dabf32c7541` |
| llama-server binary SHA-256 | `e2b0d885b7711af2a424c6ce684eaa0b28cf92ea3264a19edb1fd522bdb4e864` |
| Prepared dataset | DataRobot-Research/hotpotqa, train_hard |
| Dataset revision | `9f0ca285022ab1d7741ec681bf8add538f87db33` |
| BGE-small embedding revision for compact/medium | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Server context | 16,384 tokens |

GGUF shard hashes:

- `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`: `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`.
- `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`: `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`.

Native environment includes Python 3.12, Torch 2.7.0, Optuna 4.2.1 and the locked Syftr dependencies. Kaggle's notebook kernel itself may be Python 3.13; the isolated research interpreter is separate. Compact/medium explicitly pin Torch 2.7.0, Transformers 4.48.3, sentence-transformers 4.1.0, datasets 3.6.0, Optuna 4.2.1, NumPy 2.2.6, Matplotlib 3.10.3 and huggingface-hub 0.33.0, plus transitive constraints.

The reviewed GPU common evaluator source snapshot has content identity `0a57d95b7e2af78749c85e5f91e1f8da74a8731fa3b1b1916206ad39db5b11e5`. The exact snapshot is in the completed GPU pilot's `executed_source_gpu1` artifacts. Local packaging/serving helpers have evolved since then; never substitute the entire current local baseline directory for the reviewed snapshot without versioning and verification.

## 5 Measurements and fair comparison rules

Primary quality is raw-answer token F1, with exact match as a secondary metric. Normalization lowercases, removes punctuation and English articles, and collapses whitespace. Yes/no/noanswer mismatches score zero. Raw F1 can penalize semantically acceptable wording; preserve the predefined metric and label any semantic/manual secondary assessment separately.

The same-weight local LLM judge passed mechanical execution/parsing checks but failed semantic reliability/calibration checks. Its correctness outcomes are diagnostic only. Do not call the primary objective validated judge accuracy.

Primary resource is mean generator workflow input plus output tokens per QA. Native workflows can make multiple nested generation calls. Judge tokens are recorded separately and do not enter the generator resource objective. Tokens are not API dollars, FLOPs, measured energy or unbiased elapsed inference time.

Quality is maximized and resource minimized. Hypervolume uses a fixed quality-zero/resource reference: 50,000 native, 5,000 compact/medium. These numerical HVs cannot be compared directly across different references, spaces or question sets. Native empirical-front IGD is distance to a pooled observed front, not an unknown global optimum. A complete finite grid allows an exact reference only for its measured table.

Match methods within an experiment on model/data/source pins, concrete QA IDs, grounding corpus, common evaluator, starting configurations and attempted-query budget. Failures and repeats consume budget according to that experiment's declared policy. Compare only a prefix finished by all arms. Later work by one method is not evidence at an equal budget.

Search uses the prepared dataset partition named test, but its role is optimization/development. It is not independent final test evidence. Freeze selection before evaluating disjoint holdout questions. Shared retrieval/download caches confound wall-time comparisons; use timing descriptively.

Software fixtures and synthetic integration tests verify code behavior. They are never benchmark findings. Avoid loading received sampler pickle files merely to analyze results; raw records and SQLite are sufficient.

## 6 Smoke capability and early pilot history

The smoke completed 2/2 candidates. Capability measurements completed 14 representative workflow/module cases on three sample questions each (42 QA), with all eleven embedding models loading and producing finite vectors. Capability raw usage was 176 model calls: 134 generator workflow calls and 42 diagnostic judge calls, 58,046 generator tokens and 8,096 judge tokens, zero unknown-usage calls and zero recorded truncated question outputs.

The full embedding catalog includes BGE small/base/large, thenlper/gte-large, mixedbread-ai/mxbai-embed-large-v1, WhereIsAI/UAE-Large-V1, avsolatorio/GIST-large-Embedding-v0, w601sxs/b1ade-embed, Labib11/MUG-B-1.6, sentence-transformers/all-MiniLM-L12-v2 and sentence-transformers/paraphrase-multilingual-mpnet-base-v2. A warning about mean pooling for b1ade was observed; the finite-vector check still passed. Capability support is not proof of high task accuracy.

The CPU-era pilot was slow, especially embedding/index construction and agent workflows. GPU1 embedding readiness later verified actual CUDA placement, with the generator server unchanged on GPU0. The synthetic readiness batch was not a CPU/GPU benchmark. Earlier interrupted CPU candidates cannot support a full-candidate speedup claim.

The completed GPU pilot, audited on 15 November, had 10/10 complete candidates and 100 QA, zero failed candidates and zero generation/judge parsing errors. It recorded 499 calls (399 generator workflow, 100 judge), 245,743 generator tokens and 18,490 judge tokens, with no unknown usage or truncated question outputs. Four distinct Pareto objective pairs gave HV 0.7049886371888242 at reference 50,000. Summed candidate wall time was 2,467.99 seconds (41.13 minutes); construction accounted for 1,347.13 seconds.

The pilot sampled RAG, sub-question RAG, critique, ReAct and LATS families before further RAG proposals. These are search observations, not independent held-out results. See `ourwork/GPU_PILOT_REVIEW.md` and `ourwork/SYFTR_CAPABILITY_REVIEW.md`.

## 7 Original baseline and session loss

The initial frozen plan was 30 attempted candidates × 20 fixed questions per method/seed, optimizer seeds 11/23/37, startup target ten, full conditional Syftr space, no pruning. It was more expensive than the pilot, especially agent branches and index construction.

A downloaded original seed-11 baseline summary/log preserved 14 completed candidates, zero failed, three Pareto points and HV 0.5997301251478685, with 1,474,549 known generator tokens, 54,287 judge tokens and 2,278 model calls. The full raw candidate/SQLite/sampler capsule for that stage was not received before its Kaggle session was closed and the user confirmed it could not be retrieved.

Those unexported raw files are lost. The saved summary is historical evidence only. Do not reconstruct configurations/answers/front plots from it, do not call it a final audited baseline, and do not claim it can resume from an older capability checkpoint. The source, completed pilot and runtime exports survived locally.

Recovery/backup instructions were then strengthened: draft notebook saving does not secure working files; a checkpoint preserves data/state, not a live process. Explicitly export, download, verify and preserve original archives. A refresh can reconnect UI but is not a reliable preservation operation. Do not repeat unproductive Cancel Run clicks or reset a session before securing output.

## 8 Algorithm implementation

`ourwork/optimizers/nsga2.py` implements nondominated sorting, crowding distance, tournament parent choice, elitist environmental selection, generation state, mutation adaptation and hybrid proposal slots. `space.py` represents actual conditional distributions and activates native parameters through the SearchSpace sampler. `bridge.py` integrates the controller with the common Optuna runner; `run.py` supports the earlier deployment design.

NSGA-II population size is five. Crossover probability is 0.9. Per-active-gene mutation probability is min(0.5, multiplier / active mutable gene count), with multiplier starting at one. A no-op parent copy forces a gene change, but global duplicates remain possible. Conditional genotypes are materialized through the actual search space; inactive genes do not define arbitrary invalid flows.

The hybrid uses genetic proposals for the first three of each five child slots and MOTPE for the last two. Its Bayesian component uses pinned Optuna, not a from-scratch Bayesian implementation. Injection TPE uses ten startup observations; sufficiently early injections may therefore be startup samples.

Adaptive NSGA-II updates mutation strength after a generation: if HV gain is at most 1e-6 or diversity is below 0.15, multiply strength by 1.5 up to five; otherwise divide by 1.2 down to one. Per-gene probability is capped at 0.5. The adaptive rule cannot explain a lead before it changes behavior.

Native arms re-evaluate their common starting parameter settings and charge that work; they do not borrow the baseline's fitness. Repeated measurements can differ even with the same configuration, so matched parameter lists are not guaranteed identical observed values.

## 9 Native campaign v2 and restricted campaign

To obtain useful fair-budget checkpoints sooner, v2 declares 25 attempts per arm, ten search questions, seed 11, startup ten and population five. The schedule is round-robin at checkpoints 10, 15, 20 and 25: baseline then NSGA-II then hybrid then adaptive at each checkpoint. Checkpoint 10 contains shared initialization only; later checkpoints include one, two and three offspring generations.

Code is in `ourwork/campaign_v2/`; the controller is a background process, not a notebook cell blocked inside inference. SQLite backups and raw journals are exported into `campaign_v2_exports/campaign_v2_review.zip`. The export includes sampler/version state and provenance, not weights or scratch indexes. Search invocation window is eight hours; final confirmation has a separate one-hour window. These are stopping bounds, not completion guarantees. Candidate boundaries can delay a stop.

Full v2 retains five workflow families and eleven embeddings. No completed full-space v2 export has been audited locally. Its remote state is unverified here.

Restricted v2 is a separately versioned native Syftr subset with RAG and sub-question RAG, BGE-small/base, retrieval top k 2–6, few-shot top k 2–4, query decomposition count 2–3, chunk exponent 8–10 and overlap 0–0.2, reranker top k 2–4 and additional context count 2–4. Optional module choices remain conditional. Critique, ReAct and LATS are excluded. A fresh eleven-case restricted capability suite and both embeddings gate search. See the restricted deployment README for the authoritative bounds.

Shared retrieval indexes/downloads are permitted in v2; answers/fitness are not shared. Duplicates, failed evaluations and interrupted attempts consume budget. Final representatives use maximum single-point search HV, then F1, lower tokens and earliest trial; confirmation is separate and is not yet complete in the received restricted checkpoint.

Full and restricted packages both export the same campaign filenames. Always label downloaded files by experiment and verify protocol/source identity rather than assuming filename implies scope.

## 10 Setup and recovery problems already diagnosed

1. **Missing shared-library aliases after restoration.** Early restoration could miss names such as libggml.so.1. Runtime manifests identify 23 aliases with nine unique binary contents. Current flat compact/medium packaging restores verified local aliases from deduplicated blobs.
2. **Kaggle recursively expanded nested ZIP/TAR inputs.** frozen_common.zip and gpu1_checkpoint.tar.gz were sometimes presented as extracted folders. `archive_loader_fix.py` and `build_fixed_loaders.py` reconstruct container archives only after verifying logical per-file content. Original manifests are retained and derived-container hashes recorded. Do not disable integrity checks.
3. **Software fixture saw live setup status.** A notebook-path test expected a fixture but read the real setup JSON. A narrow test-only isolation overlay fixed the path; assertions still run. See `kaggle_test_isolation_fix.py` and deployment test-isolation records.
4. **Duplicate setup processes and flock failure.** A prior setup owner retained the lock. The launcher now waits/reuses validated completed setup and holds the actual setup mutex. It does not blindly start another installer or overwrite a live owner's status.
5. **Restricted source conflicted with full source snapshot.** The restricted installer originally reused executed_source_gpu1. A namespace overlay uses executed_source_restricted_v2 and preserves the original full snapshot. Source differences are not accepted by editing manifests.
6. **TPESampler replacement happened before Syftr subclass import.** Evolutionary sampler factory patching caused TypeError while defining HierarchicalTPESampler. The recovery launcher configures pins/workspace and pre-imports syftr.hierarchical_tpe before applying the temporary factory patch. Research source bytes remain unchanged; recovery provenance is archived.
7. **First failure left an empty NSGA-II initialization folder.** It had protocol.json but no sampler checkpoint, causing a false resume. The updated recovery checks for zero existing SQLite trials, no measured attempts/results, and moves the empty folder into initialization_failures. It refuses to reset any folder containing actual trials. The completed baseline is never reset.
8. **Kaggle inherited an unavailable notebook Matplotlib backend.** Compact setup installed correctly but plotting tests failed on module://matplotlib_inline.backend_inline. The corrected notebook and all medium launch/setup paths explicitly set MPLBACKEND=Agg and a writable MPLCONFIGDIR. No dataset ZIP replacement was needed for the compact environment fix.
9. **Output tree spinner prevented downloads.** `DOWNLOAD_RUNNING_CAMPAIGN.py` reads a completed atomically replaced capsule, verifies every internal file, makes a uniquely named immutable copy and creates data-URL download links for ZIP/receipt. It does not stop the controller or concurrently create a new snapshot. Old error text in append-only logs is not itself a current error.

Relevant files: `ourwork/RECOVER_OPTIMIZER_IMPORT_ORDER.py`, `DOWNLOAD_RUNNING_CAMPAIGN.py`, `build_ready_notebooks.py`, `test_setup_coordinator.py`, full/restricted `RUN_BATCH_SAFE.py` and `*_BATCH_SAFE.ipynb`. Keep existing deployed ZIP bytes immutable; prefer the already documented launcher overlays to untracked edits.

## 11 Latest restricted native result received

Original files: `restricted_progress_20261116_124506.zip` and its receipt. Size 7,277,020 bytes; SHA-256 `e7f8e3e9bf9577e379302992e55d6a90c547655dac06b0d9a88c5e60bcd98ec3`. All 1,000 internal file hashes verify.

Saved under `ourwork/kaggle_returns/restricted-progress-20261116-124506/`, with a safety copy in `ourwork/experiment_archive/restricted_checkpoint15_2026-11-16/`.

The snapshot contains baseline 20 attempts (19 eligible, one duplicate), NSGA-II 18 eligible attempts, hybrid 15 and adaptive 15. The current controller heartbeat in the archive was advancing NSGA-II toward 20; its progress field can lag the separately exported summary. **Only the first 15 attempts are the fair shared comparison currently available.**

| Method | Matched attempts | Eligible | Charged failures | HV at reference 50000 | Best search F1 |
|---|---:|---:|---:|---:|---:|
| Syftr MOTPE | 15 | 14 | 1 | 0.3671648203441973 | 0.37523809523809526 |
| NSGA-II | 15 | 15 | 0 | 0.3710624082539683 | 0.37523809523809526 |
| Hybrid | 15 | 15 | 0 | 0.4415972692063492 | 0.44857142857142857 |
| Adaptive | 15 | 15 | 0 | 0.4418045549206349 | 0.44857142857142857 |

Independent audit on a temporary copy validated 59 eligible candidates/590 QA, common protocols/questions/grounding/packages/pins/initial parameter pools, raw F1/EM, generation call coverage/usage and objective aggregates. No sampler pickle was executed. Raw records and original capsule were not altered.

Hybrid/adaptive show about 20% higher observed HV than baseline at this prefix; this is an interim descriptive result, not established mechanism superiority. Important qualifications:

- Trial number 11 is the same concrete genetic configuration in NSGA-II/hybrid/adaptive: concise RAG, BGE-base, retrieval k 5, decomposition two, recursive chunk exponent 9, overlap zero, rerank k 4, HyDE, few-shot k 3, additional context four.
- That identical configuration scored F1 0.348571 in NSGA-II versus 0.448571 in hybrid/adaptive. One yes/no QA (`5ae2cd3b5542996483e64a6d`) changed from no to yes, explaining the entire 0.10 difference.
- The archive does not isolate whether the variation came from nested generation, context/retrieval or backend behavior. Do not assert a specific source without further checks.
- The adaptive arm's first generation used multiplier 1.0 and retained next multiplier 1.0. It had not changed mutation strength; its lead cannot be attributed to adaptive strengthening.
- The hybrid's best setting came from its shared genetic proposal before its first BO injection slots. Its lead cannot be causally attributed to those injections.
- One optimizer seed, ten search questions, one charged baseline duplicate and no final holdout limit interpretation.

At shared initialization checkpoint 10 all methods had nearly identical HV near 0.36623–0.36627. The archived automatically generated comparison REPORT.md incorrectly inherited wording about full conditional space; the frozen protocol, independent findings and this context correctly label the restricted subset. The original report is preserved as evidence, not silently edited.

Next: let the friend's current campaign continue unchanged to matched 20 and 25, export/download fresh immutable capsules, audit each, and complete the predeclared holdout confirmation. A later separately labelled repeated-finalist analysis could quantify evaluation variability, but must not replace original observations retroactively.

## 12 Compact 27 configuration experiment completed

This is a controlled Syftr-derived single-call RAG grid and measured-fitness replay. It is not the native hierarchical campaign. Source: `ourwork/compact_kcb/`; deployment: `ourwork/compact_kcb_deployment/`.

Grid: k = 2/4/6; c = 256/512/1024 Qwen tokens; Bc = 512/1024/2048 Qwen context tokens. Exactly 27 cells. Chunk overlap 10%, BGE-small 384 dimensions, embedding cap 512 BGE tokenizer tokens, Qwen output cap 256, temperature zero, no agents/few-shot/local judge. Generator and embedding token units differ.

Data: five fixed search QA plus five disjoint holdout QA from the prepared test/holdout partitions, seed 20261114. Grounding corpora have 4,972 and 78,102 documents respectively. Whole prepared split corpora are used, not per-question gold supporting evidence.

Progressive stages use 1/3/5 search QA and reuse prior answers: 135 final logical search configuration-question evaluations. At each complete table, each of four optimizers performs 25 charged requests, ten shared distinct starting settings and population five; ten optimizer seeds 11/23/37/41/53/67/79/83/97/101. Repeats consume attempts/equivalent token charges. Optimizers see only requested fitness. Replays make no physical inference call. Three distinct selected settings then add 15 logical holdout evaluations.

Exact identical partition/QA/model/request keys cache physical responses, retaining authoritative logical usage. Final generation ledger: **77 physical calls, 47,228 known generator tokens, zero failed calls**, all finish_reason stop. This excludes embedding compute, setup preflight and software fixtures.

All four methods recovered the same finite-grid front in every seed: HV **0.5229264**, zero final standard deviation, zero HV gap and zero IGD. Mean unique explored configurations were MOTPE 20.5, NSGA-II 17.0, hybrid 18.0, adaptive 18.3. First-full-HV query means were 8.5/9.9/9.9/10.1, including initialization. Seven of ten common initial pools already contained the full front. Only three seeds needed post-initialization search; MOTPE tied or reached the target sooner than every evolutionary method in those three.

The front contains two distinct objective points represented by 12 tied configurations:

| Operating point | Search F1 | Generator tokens per QA | Settings |
|---|---:|---:|---|
| Lower resource | 0.46 | 289.8 | k=2, any tested c/Bc |
| Higher quality | 0.56 | 520.6 | k=4, Bc=512, any tested c |

There are only eleven distinct objective pairs and eighteen distinct five-question prompt sequences; parameter redundancy limits the challenge. Adaptive/hybrid reduced repeats relative to NSGA-II but did not improve final front quality. No evolutionary superiority is supported by this experiment.

Holdout selected settings all use k=4/Bc=512. c=256 and 512 yield F1 0.2571428571, EM 0.20, tokens 535.6 per QA. c=1024 yields F1 0.0571428571, EM 0.00, tokens 535.6. One extra nationality answer explains the 0.20 F1 difference. Mean selected-representative holdout F1 is 0.1371 MOTPE/hybrid/adaptive and 0.1571 NSGA-II, but these reuse three settings on the same five questions and are not independent 50-sample tests. Do not claim significance or select a new setting using this holdout and then call it untouched validation.

Marginal means suggest k=4 outperformed k=2/6 in this sample; k=6 spent more tokens without higher mean F1. Bc=512 had the strongest marginal quality and lowest cost. These are conditional small-sample findings, not universal configuration recommendations. The much larger holdout grounding corpus complicates attribution of the search-to-holdout drop.

Received ZIP: 3,525,628 bytes, SHA-256 `85e0ef0edda8bd3b285a20af4c45e4e5161cf61942bcd5fb1bc2eecc8883c611`. All 234 archived hashes verified; independent audit checked 258 QA row references across stages/holdout, raw answers, prompts, usage, objective aggregates, proposals, repeat charging, fronts and HV trajectories. Recorded context bounds/request hashes verified; the pinned tokenizer was not independently rerun during the local return audit.

Saved files:

- `ourwork/kaggle_returns/compact-kcb-completed/FINDINGS.md` and `AUDIT.json`.
- All raw/extracted results and original review/receipt beside those files.
- `ourwork/experiment_archive/compact_kcb_2026-11-16/04_report/Compact_RAG_Experiment_Report.pdf` and `.docx`: visually checked 16-page report covering all 27 rows, methods, data, metrics, figures, QA references, caveats and provenance.
- `ourwork/experiment_archive/Compact_KCB_Experiment_Safety_Backup.zip`: 68,089,787 bytes, SHA-256 `ae855fbd42a45ef6906c2208fac21df7b4132ae5a215e70589ab216a07fbff9b`, 252 files, CRC/per-file hashes verified.

## 13 Medium 60 configuration experiment ready

The user asked to build research incrementally after the compact success. We prepared a larger, separately versioned study without altering completed compact source/results/packages.

Grid: k = 1/2/4/6/8; c = 128/256/512; Bc = 256/512/1024/2048. Exactly **60 cells**. It is not a strict superset: c=1024 is removed and finer chunks/context limits added. Removal does not establish universal inferiority from the old holdout.

Ten fresh search questions and ten fresh holdout questions are selected with seed 20261117. Both IDs and stripped/casefolded question text from all ten previous compact QA are excluded across both partitions. Selection does not rank candidates by reference answers or measured quality. No old answer/fitness table is imported.

Progressive stages 2/5/10 QA total 600 logical search requests before exact-prompt reuse. Four methods × ten optimizer seeds × 30 charged queries, only five shared distinct initial settings, population five and five offspring generations. Hybrid has ten injection slots; its ten-observation injection startup rule is explicitly declared. Up to forty selected representatives yield at most 400 additional logical holdout calls, often fewer after deduplication. Total pre-cache upper bound is 1000 logical calls.

The initial fraction falls from 10/27 to 5/60, and budget fraction from 25/27 to 30/60. This intends a less saturated comparison but does not guarantee any method advantage. Questions, budget and factor values also change, so do not treat differences versus compact v1 as a controlled grid-size-only ablation or pool their scores.

Model, embeddings, tokenizer/dataset/runtime pins, overlap, output cap, F1/tokens and reference 5000 are unchanged from compact. Medium code reuses verified pinned model/Python where available, memory-maps retrieval vectors and reuses per-query ranking across k/Bc variants. Old scratch indexes can be shared when identities match; old answers are not imported.

New outputs include paired seed differences against MOTPE, budget checkpoints 5/10/15/20/25/30, post-initialization HV gain/mean, 99%-reference attainment, HV gap plots and explicit active question/holdout progress. Primary endpoint is final HV at 30 queries on the ten-question table. Four hours is a measurement stopping window, not an expected completion time; setup/embedding and boundary effects add time.

Five focused tests passed, plus a complete synthetic 60-cell progression/replay/selection/holdout/report/export integration check. Standalone extracted-package tests, hashes, syntax, embedded notebook/package identities, Agg fix and unchanged original compact package were verified. These are software tests only; **no actual medium GPU result has yet been received**.

Artifacts:

- `ourwork/medium_kcb/`.
- `ourwork/medium_kcb_deployment/syftr_medium_kcb_v2.zip`, 60,982,650 bytes, SHA-256 `a8cff807534db11cbf8c35706023ad4320facc2561afd239d532f0e271e8143c`.
- `RUN_MEDIUM_BATCH.py`, `SYFTR_MEDIUM_KCB_BATCH.ipynb`, `README.md`, `BUILD_RECEIPT.json`, `VALIDATION.json`, `validate_offline_pipeline.py`.

The user said the compact Kaggle session is still open. If its GPU session is actually alive and idle, attach the new private Dataset and run only the new medium launch cell at the bottom of the old notebook. **Do not Run All in the old compact notebook.** A browser tab alone does not prove a live GPU allocation; the launcher can cold-start if needed. For a clean new notebook, import the medium notebook, attach the medium Dataset, select T4 x2 and Internet ON, and Save Version then Save and Run All once. Return `medium_kcb_review.zip` and `medium_kcb_receipt.json`. Its data goes to a separate medium_kcb_results folder.

## 14 Important packages and recovery archives

| File | Purpose and identity |
|---|---|
| `ourwork/kaggle_returns/capabilities-seed11/capabilities_checkpoint.tar.gz` | Original recovery, 62,023,582 bytes; SHA `411b2346dda16d83a210b8161e6feec3dda28f190951e514e83013b0ab3bc08e` |
| `ourwork/kaggle_returns/gpu1-recovery/gpu1_checkpoint.tar.gz` | Later GPU runtime recovery; 23 aliases/nine unique binary blobs; weights not included |
| `ourwork/kaggle_returns/gpu1-pilot-completed/gpu1_pilot_completed_review.zip` | Completed pilot raw source/measurement evidence; SHA `c39a59442a0e750337c71aa5a3a1576119f2734b3af9a56b6fa9d02fbce00f27` |
| `ourwork/campaign_v2_deployment/syftr_campaign_v2.zip` | Original full native v2 package; SHA `9c2cb054066048147580c669f6f1b2ff6c89ef20485a196ca05f982ff55b1003` |
| `ourwork/restricted_v2_deployment/syftr_restricted_v2.zip` | Original restricted native package; SHA `03672d8b049fa20aa5818b970faef791fe31ef8b9f666e021fdcd7a455eabfb3` |
| `ourwork/compact_kcb_deployment/syftr_compact_kcb_v1.zip` | Exact compact uploaded package; SHA `79d05303acc0d846e2c9f605c100407d4ea655fdb9b2783175b989777d978f65` |
| `ourwork/medium_kcb_deployment/syftr_medium_kcb_v2.zip` | Medium package, identity above |

Compact/medium packages use flat hashed `.bin` blobs instead of nested recovery archives. This avoids Kaggle's nested extraction problem. The exact compact uploaded ZIP predates the Agg environment fix, but its corrected notebook sets that environment before launch. Do not rebuild an old deployed package silently.

## 15 Workspace map and document precedence

- `PPTnall/`: base paper, proposal and supporting paper material.
- `syftr/`, `studies/`, `tests/`, `docs/`, `notebooks/`, `pyproject.toml`, `uv.lock`: base project.
- `ourwork/baseline/`: common measurement/evaluation and source/device/token-accounting helpers.
- `ourwork/optimizers/`: proposal methods and native integration.
- `ourwork/campaign_v2/`: round-robin controller, cross-arm audit/report, snapshots and confirmation.
- `ourwork/campaign_v2_deployment/` and `restricted_v2_deployment/`: separate original packages and fixed/SAFE launchers.
- `ourwork/compact_kcb/` and `medium_kcb/`: separate controlled grid studies.
- `ourwork/kaggle_returns/`: original received ZIPs/receipts, extracted artifacts, audits and findings. These are the source of measured claims.
- `ourwork/experiment_archive/`: named experiment safety folders and report/backup exports.
- `ourwork/report_build/compact_kcb/build_report.py`: reproducible report authoring; rendered QA intermediates are not final deliverables.
- `ourwork/research_checkpoints/LATEST.json`: newest consolidated snapshot receipt.

Read this context first, then the latest experiment-specific protocol, raw result receipt/audit, findings and deployment README. `MASTER_RESEARCH_HANDOFF.md`, `START_HERE_12H.md`, the first section of `CURRENT_STATUS_2026-11-16.md`, and original campaign notes contain historical plans. They may still describe no cross-arm results or an old 30 by 20 campaign. Those statements are superseded by this chronology; keep them as history, do not erase evidence or interpret them as live status.

## 16 Outstanding work and safe continuation

1. Restricted search is complete and its final matched25 export is locally verified. Preserve the final evidence; no further search rerun is needed for this declared budget.
2. Complete the restricted predeclared final holdout stage once all final budgets finish. The last received checkpoint contains no completed final holdout evidence.
3. Run the prepared medium experiment in an idle appropriate session; return its ZIP/receipt. No actual medium result has been audited yet.
4. If repeated evaluation variation remains important, design a separately labelled finalist repeatability analysis; do not modify running experiments or overwrite observed values.
5. Keep compact, medium, restricted native and full native findings separate. Do not directly compare HV across references/protocols, and do not conflate replay with independent GPU campaigns.
6. Build the final project report and presentation around what actually finished. The compact report is complete; the overall final report/presentation still need consolidation with any later native/medium evidence.
7. Keep zero/negative findings, limitations, failure charges and uncertain mechanisms visible. No claim of significance, universal superiority, exact paper reproduction or guaranteed runtime is supported yet.
8. Download result files before closing Kaggle. Preserve originals, check receipt/content hashes, and copy the local safety snapshot to an independent drive/cloud location.

## 17 Preservation boundaries

The consolidated backup contains locally available project source, paper/proposal material, records, original result capsules, extracted measurements, reports, audited findings, runnable deployment packages, protocol files and recovery helpers. It cannot include progress still only on Kaggle or recover the lost original raw baseline.

Do not back up secrets, private credentials, environment variables, live virtual environments, Git internal state, downloaded model-weight caches or recursively nested prior consolidated snapshots. Model weights remain reproducible from pinned revisions; saved runtime binaries are preserved in the recovery/deployment archives. Excluded caches/rendered review images are not research measurements. The new snapshot manifest records its exact inclusion/exclusion policy and file hashes.

No project findings or old checkpoints are deleted by preservation. A same-disk folder and ZIP protect against accidental local edits, not failure of the entire disk. An off-device copy remains necessary for that protection.

## 18 Final restricted native export received and independently audited

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

Evidence directory: `ourwork/kaggle_returns/restricted-final-20261116-142224`. Earlier matched15 results remain historical checkpoints, superseded for final-budget ranking by this matched25 audit.

## 19 Optional native finalist holdout cell prepared

`ourwork/holdout_confirmation/RUN_RESTRICTED_HOLDOUT.py` is a single-cell warm-session launcher for the original, unchanged confirmation evaluator. It checks completed restricted25 state, exact frozen finalists and confirm.py hash, server health and campaign implementation identity, then evaluates up to four finalists on ten disjoint holdout QA each. The original one-hour evaluation deadline is retained; preparation/export add time. It prints progress every30seconds and exports a small SHA-verified holdout review ZIP and receipt even after an evaluator exception. Original search evidence is not rerun. Local syntax and simulated completed/partial-error export checks passed; no actual GPU holdout result has been received yet. User should paste this whole file into one new cell in the existing live restricted session, not restart or Run All. Returned holdout data must be audited before claiming independent confirmation. The prior consolidated backup predates this optional launcher; completed search data remain in that verified backup.

## 20 Final project dossier and curated implementation handover

The user elected to proceed with completed findings because optional native holdout did not run successfully; no usable native holdout data have been returned. Further Kaggle execution is not required for the current dossier.

`FINAL_RESEARCH_PROJECT/` contains the curated implementation and evidence handover. `01_documentation/Local_Syftr_Project_Implementation_and_Findings.docx` and its rendered PDF explain fundamentals, proposal, paper adaptations, decisions and reasons, exact local protocol, common evaluator, conditional representation, every optimizer, final native metrics, compact replay and holdout, research questions, limits, raw answer examples and gene/revision appendices. The document is51 pages with about11900 words. Fifteen figure exports in both PNG/PDF cover native Pareto/convergence/histograms/question scores/module counts/variation/operator diagnostics and separate compact landscape/replay findings.

`02_implementation/native_exact` is the executed source copied from the final export, including its restricted spec override. `03_results` preserves native and compact original capsules and extracted records. `04_reproducibility` contains portable CPU-only raw audit/plots/document builders, package/environment records and original deployment packages with safe notebooks and recovery helpers. `05_reference_material` holds the supplied proposal/papers and historical context. No native holdout claim is added, adaptive multiplier remained1, and the compact negative result remains visible.

Validation: the packaged offline audit reproduces all four final scores and audits98 eligible candidates/980 QA rows; standalone plot rebuild succeeds;19 existing optimizer and storage bridge unit tests pass. Software tests are not extra research measurements. The PDF is rendered for visual review. The final project ZIP and MANIFEST.json are generated and read-back verified at handover; see adjacent receipt for its exact bytes and SHA256. This curated final archive is separate from earlier consolidated full-workspace snapshots, which remain intact. Presentation and institution-specific final report preparation remain the next tasks.

## 21 Judges presentation completed on 17 November 2026

`FINAL_PRESENTATION/Local_Syftr_Research_Presentation.pptx` is the editable38-slide deck. Main talk slides1–28 follow introduction, research decisions, common protocol, local MOTPE baseline, NSGA-II implementation and comparison, hybrid mechanism and results, adaptive mechanism and diagnostics, common Pareto/convergence findings, research answers and conclusion. Slides29–35 provide technical backup;36–38 contain literature, data/software and project evidence references. All38 slides contain explanation and source notes. `Speaker_Guide.md` provides the notes separately. A PDF review copy is included.

The deck includes8 native editable charts with literal-data workbook snapshots and13 native editable tables. Chart values use12 significant figures for workbook portability while raw results remain unchanged. All slides were rendered and visually inspected. Package integrity, geometry/fonts, native chart data and first-party reimport checks pass. Convergence smoothing is explicitly disabled. The Pareto scatter shows discrete measured non-dominated points without interpolation, keeping its meaning consistent across PowerPoint/PDF viewers. No native PowerPoint UI execution is claimed.

A supplementary copy is in `FINAL_RESEARCH_PROJECT/06_presentation`. The earlier287MB FINAL_RESEARCH_PROJECT.zip and its original manifest remain unchanged, preserving the earlier handover. `FINAL_PRESENTATION.zip` separately archives the deck PDF notes and receipt. The presentation retains one-seed/small-QA/fitness-variation caveats and the inactive adaptive trigger, and keeps compact replay separate. Native holdout is still unavailable.
