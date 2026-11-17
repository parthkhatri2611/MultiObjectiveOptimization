# Independent restricted-space backup experiment

Run this package on your friend's own Kaggle account with **T4 x2 and Internet ON**. The laptop is an interface; Kaggle supplies the GPUs. The full-space package on your account remains unchanged. Use a separate private dataset and notebook for this study.

## Files and execution

Upload **syftr_restricted_v2.zip** as a private dataset. Import **SYFTR_RESTRICTED_V2_BATCH.ipynb**, attach that dataset, select T4 x2 and Internet ON, then choose **Save Version → Save & Run All**. Do not launch an interactive run alongside it.

The batch restores the existing CUDA runtime/model/environment, validates software, performs fresh capability measurements for the restricted space, and runs the same four optimizers through matched checkpoints 10,15,20,25. Search has an eight-hour invocation window, confirmation a one-hour window; setup and capability validation take additional time. Faster evaluation is expected from the removed expensive workflows and lower parameter bounds, but exact completion time is not guaranteed. Incomplete budgets remain explicitly paused.

Download **campaign_v2_exports/campaign_v2_review.zip** and **campaign_v2_receipt.json** from the saved version's Output. Keep them in a folder labeled **restricted** and return both. Your full-space run produces the same export filenames: keep its downloads in a separate **full-space** folder.

## Exact restricted search space

| Dimension | Restricted values |
|---|---|
| Workflows | RAG; SubQuestion RAG |
| Embeddings | BAAI/bge-small-en-v1.5; BAAI/bge-base-en-v1.5 |
| Retrieval methods | Dense, sparse, hybrid |
| Retrieval top-k | 2–6 |
| Few-shot examples | Enabled/disabled; top-k 2–4 |
| Query decomposition | Enabled/disabled; 2–3 queries |
| Splitters | Recursive, sentence, token |
| Chunk exponent | 8–10 (256–1024) |
| Overlap fraction | 0–0.2, step 0.1 |
| Reranker | Enabled/disabled; top-k 2–4 |
| Additional context | Enabled/disabled; 2–4 nodes |
| HyDE | Enabled/disabled |
| Prompt templates | Default, concise, CoT |

Critique, ReAct and LATS workflows are excluded from this study. These are actual restrictions of workflow choices and expensive conditional parameter ranges, not merely a smaller count of possible embedding models.

The generator, pinned dataset, question-selection seed, objectives, 25 attempted candidates per method, ten search questions, ten starting configurations, population five, evolutionary operators and three generations match the full-space campaign. The starting configurations themselves differ because the spaces differ; initialization is matched across the four methods **within** each study. Failures/duplicates/interrupted attempts count.

## Validation and findings

This package explicitly changes only the common search-specification file and archives its original/new hashes. The original full release is not edited. It measures **11 representative restricted workflow/module cases on three questions each**, and checks both embeddings. Search is blocked unless those checks pass. Original full-space capability records are used only as model/data revision pins, not claimed as validation of the modified search space.

The same analysis exports F1/EM, logical generator tokens, observed fronts/configurations, fixed-reference HV, empirical pooled-front IGD, dominance coverage, convergence against attempts/tokens, failures, timing, parameter observations and evolutionary diagnostics. Comparisons require matching protocols and actual question IDs. Timings remain descriptive because index caches are shared within this study.

The 10-attempt checkpoint is initialization only. Checkpoints 15,20,25 contain algorithm-specific search. One candidate per method is selected by the predeclared maximum single-point HV rule and then evaluated on ten disjoint holdout questions after completed search. Holdout confirmation does not validate every search-front point.

## Reporting the two studies

Compare the baseline against the three optimizers **separately within each search space**. Do not pool full-space and restricted trials, substitute a full-space baseline for the restricted baseline, or report unequal budgets as equal. A completed restricted study is valid research about this declared subset even if the full study remains incomplete. It does not establish superiority across the full Syftr space. Single-seed/small-question limitations and any incomplete runs must be stated.

## Recovery

For this study, reuse its original restricted package and only its most recent downloaded campaign_v2_review.zip as private inputs in a fresh T4x2 session. Capsule checksums/source identities prevent accidentally resuming with the other study's results. Scratch weights, environment and indexes are recreated; recorded studies and sampler state resume. Completed restricted capability evidence is archived inside the capsule for verification/reuse. Do not edit manifests or bypass failed gates.

Software tests use synthetic workers; they are not model findings. Real restricted results still need to be measured on Kaggle.


Loader update: use the **_BATCH_FIXED.ipynb** notebook and see LOADER_FIX.md. Kaggle recursively unpacks nested archives; the fixed loader verifies their logical content and reconstructs them without another dataset upload.


Setup-test update: use the **_BATCH_READY.ipynb** notebook (supersedes FIXED). It also isolates the notebook-status fixture from live setup state. Research source bytes and the original ZIP remain unchanged, and every test still runs.
