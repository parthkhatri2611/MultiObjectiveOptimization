# Multi-objective Optimization of Local RAG Pipelines

Research Methodology project: comparing a local adaptation of Syftr’s multi-objective TPE baseline with **NSGA-II**, **Hybrid NSGA-II + MOTPE**, and **Adaptive NSGA-II**.

**Start here:** the final implementation actually executed on Kaggle is in [`final_project/02_implementation/native_exact`](final_project/02_implementation/native_exact). The original Syftr files at the repository root are retained as historical framework context; use the final snapshot for this study. Its files are preserved, including the original dependency lock and bundle manifest. Research modules are under `ourwork/`; the inherited Syftr framework is under `syftr/`.

## Completed experiment

We ran a declared **restricted subset** of Syftr’s conditional pipeline space, using Qwen2.5-7B-Instruct in GGUF Q4_K_M form, two Kaggle Tesla T4 GPUs, and HotPotQA. GPU 0 served generation; GPU 1 served BGE embeddings. All four methods used the same 10 fixed optimization questions, seed 11, 25 charged candidate attempts, and the same first 10 proposed configurations, independently evaluated for each method. The objectives were answer token F1 (maximize) and mean generator input + output tokens per question (minimize). Diagnostic judge tokens were recorded separately. Hypervolume used a fixed token reference of 50,000.

| Method | Attempts | Evaluated | Charged duplicates | Hypervolume | Best F1 |
|---|---:|---:|---:|---:|---:|
| Local Syftr MOTPE | 25 | 24 | 1 | 0.392754 | 0.398431 |
| NSGA-II | 25 | 25 | 0 | 0.371658 | 0.375238 |
| Hybrid NSGA-II + MOTPE | 25 | 24 | 1 | **0.444376** | **0.448571** |
| Adaptive NSGA-II | 25 | 25 | 0 | 0.442513 | **0.448571** |

Hybrid and adaptive hypervolume exceeded the local baseline by 13.14% and 12.67%, respectively, in this run. Plain NSGA-II was lower. These are **exploratory observations from one seed and ten optimization questions**, not evidence of general superiority. The native holdout confirmation did not produce usable results. The adaptive mutation multiplier stayed at 1 throughout the measured generations, so this run does not establish a benefit from adaptation. Hybrid’s highest-F1 candidate appeared before its first Bayesian injection. Repeated configurations also showed output variation. See the detailed report for these qualifications.

## Project contents

| Location under `final_project/` | Contents |
|---|---|
| `01_documentation/` | Detailed PDF, Word/Markdown explanations, figures and exported tables |
| `02_implementation/native_exact/` | Exact executed native restricted source, `pyproject.toml`, `uv.lock`, source manifest |
| `02_implementation/compact/` | Separate compact k/c/Bc experiment implementation |
| `03_results/native_restricted/` | Raw answers, request/call journals, candidate records, SQLite studies, Pareto fronts, audits and comparisons |
| `03_results/compact/` | Separate compact experiment results; do not pool with the native experiment |
| `04_reproducibility/` | CPU audit, plotting/report scripts, deployment instructions and recovery helpers |
| `06_presentation/` | Judge-facing PowerPoint, PDF and speaker guide |
| `PROJECT_CONTEXT.md` | Project history, decisions, recovery history and evidence limitations |

The native space includes RAG/sub-question RAG, BGE small/base embeddings, retrieval variants, retrieval count, chunking, prompts and conditional enhancement settings. The protocol and actual distribution files in each saved run are authoritative. This was not an exhaustive evaluation of the original Syftr space.

## Inspect and verify results without a GPU

Python 3.12 is recommended. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python final_project/04_reproducibility/offline_audit.py
cd final_project/02_implementation/native_exact
python -m unittest ourwork.optimizers.test_optimizers ourwork.optimizers.test_bridge -q
```

The audit recomputes primary metrics from archived answers and token journals, compares all four methods with the recorded audit, and works on a temporary copy. It does not contact an LLM, use GPU inference, modify the archived campaign, or deserialize sampler pickles. Expect 98 evaluated candidates and 980 question rows. Failed-count fields in the recorded summary include the two charged duplicate attempts; they are not two inference crashes.

## How the approaches work

- **Local Syftr MOTPE:** pinned Optuna 4.2.1 TPE, multi-objective quality/resource search, startup pool of ten configurations.
- **NSGA-II:** nondominated sorting, crowding distance, tournament selection, block crossover and active-gene mutation, elitist survival; population five.
- **Hybrid NSGA-II + MOTPE:** NSGA-II with two Bayesian proposals in each five-offspring generation. Those proposals consume the same evaluation budget.
- **Adaptive NSGA-II:** NSGA-II with mutation scaling triggered by insufficient hypervolume gain or diversity. Trigger logic is implemented, but the scaling did not activate in this run.

See `ourwork/optimizers/{nsga2,bridge,space}.py`, `ourwork/campaign_v2/`, and `ourwork/baseline/` in the final native snapshot. The pipeline builders reuse Syftr’s implementation; the local adapter, optimizer engine, evaluation journals and comparison workflow form our study implementation. Hybrid’s saved method identifier is `hybrid_motpe`.

## Run the native experiment again on Kaggle

The full shareable project ZIP includes `runtime_assets/syftr_restricted_v2.zip`, the original 175 MB restoration package. It is intentionally excluded from Git history. Model weights are not bundled and are downloaded by the verified setup. Saved sampler pickle checkpoints are excluded from this distribution; use a fresh campaign rather than treating these inspection artifacts as a resumable running session.

1. Upload that runtime ZIP as a **private Kaggle dataset** and attach it to a fresh notebook. Enable Internet and **GPU T4 x2**.
2. Import `final_project/04_reproducibility/native_deployment/SYFTR_RESTRICTED_V2_BATCH_SAFE.ipynb`. Read `SAFE_START.md` in the same folder. Run only one setup/controller at a time.
3. This historical deployment needed the included `RECOVER_OPTIMIZER_IMPORT_ORDER.py` launcher to preload Syftr’s hierarchical TPE before the temporary sampler factory substitution. Follow `REPRODUCTION_NOTES.md` before starting optimizer arms; do not rerun an already active controller.
4. Inspect status and logs, then export results and download them before ending the session. Setup and inference require real GPU time; completion time is not guaranteed.

The original source snapshot is deliberately not silently patched to hide the recovery history. This repository’s completed evidence can be audited independently of a new Kaggle run. For exact package dependencies use the snapshot’s `uv.lock` and its supported Python range (>=3.11.1, <3.13); the CPU-only requirements above are for auditing, not inference.

## Attribution and research scope

Based on **Syftr**, whose upstream source and MIT license are retained in the snapshot and [`LICENSE`](LICENSE). See the original framework documentation in [`docs/UPSTREAM_SYFTR_README.md`](docs/UPSTREAM_SYFTR_README.md) and references in the report/presentation. Qwen, BGE, HotPotQA, Optuna and other dependencies retain their respective licenses and attribution requirements. Downloaded model weights and runtime environments are not source authored by this project.

This repository reports the final completed restricted experiment. The compact k/c/Bc experiment is a separate study with its own protocol and evidence. Neither constitutes a full reproduction of every result in the Syftr paper.
