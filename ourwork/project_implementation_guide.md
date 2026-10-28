# Research implementation guide: Syftr baseline and optimizer comparison

**Prepared:** 2026-10-24  
**Working project:** `RM-project` (provided upstream Syftr checkout)  
**Output of this guide:** a staged plan to first run a local-model Syftr baseline, then compare the three optimizer methods in the presentation on a shared RAG setup.

## Executive plan

For the PC-specific hardware findings and exact Windows setup commands, use [pc_local_hardware_and_baseline_pipeline.md](pc_local_hardware_and_baseline_pipeline.md).

The project should proceed in this order:

1. Confirm the computer's RAM, GPU/VRAM, disk, and Windows/Python setup.
2. Fix one local generation model and one embedding model; use the same versions in every optimizer arm.
3. Use Syftr's prepared HotPotQA `train_hard` dataset. The repository already contains a loader and a study configuration for it; manual dataset copying is optional.
4. Make the smallest local RAG study work and save a baseline result from Syftr's TPE optimizer.
5. Freeze the search space, objective definitions, dataset splits, model, evaluation procedure, and evaluation budget.
6. Implement and compare the three methods from the slides: NSGA-II, hybrid NSGA-II + Bayesian optimization, and adaptive NSGA-II. Include the original Syftr TPE run as the required baseline.
7. Evaluate all methods on equal RAG-evaluation budgets and report repeated-run results.

Do not begin with long runs or the entire paper-scale configuration space. First prove the local pipeline works and is measurable. A reduced search space is acceptable for a student project if it is described honestly as Syftr-derived and held fixed across all methods.

## 1. Model choices

The optimizer comparison needs a fixed model environment. “Same model” should include the generator, answer judge, and embedding model; the optimizers should not be allowed to choose a different model in one arm.

### Suggested local generation models

- **First candidate to try: Qwen3 4B through Ollama.** It is small enough to be a practical initial local test on more machines than a large model. Ollama lists a Q4_K_M package around 2.5 GB; this is model-file size, not total RAM/VRAM required. Use the same exact model tag, quantization, context size, and generation settings for every arm. [Ollama model page](https://ollama.com/library/qwen3%3A4b)
- **If the computer has enough memory and acceleration: Qwen3 8B.** This may provide a stronger answer baseline but needs more resources. Decide only after checking the machine and a short throughput trial. [Ollama Qwen3 family](https://ollama.com/library/qwen3)
- **If the computer is constrained:** test Qwen3 1.7B for pipeline wiring, but treat it as a smoke-test model; its answers may be too weak for a convincing final RAG study.

Ollama supports a local OpenAI-compatible endpoint, and this Syftr checkout has an `openai_like` configuration path. That offers a plausible integration without a paid model endpoint. Validate a single prompt and function/tool behavior before a study; the code's normal generator and evaluator use supported Syftr model configuration, so an Ollama model should be registered there rather than only installed in Ollama. [Ollama OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility)

### Suggested embedding model

Start with `BAAI/bge-small-en-v1.5`, which is already Syftr's baseline RAG embedding in this checkout and can run locally. Keep it fixed. The project also supports local Hugging Face embeddings, but the chosen device and cache behavior must be verified. Do not add multiple embedding choices during the first algorithm comparison.

### Evaluation model / judge

The existing end-to-end HotPotQA study names `gpt-4o-mini` as its correctness judge; that would still use a paid endpoint. Replace that judge with the selected local model through the local-compatible provider, or implement a deterministic metric such as normalized exact match/F1. For the first Syftr baseline, a local judge preserves Syftr's `accuracy` objective with less custom evaluation work, but it is a noisy measure and must be identical across optimizer arms. Report the judge model and prompt. Use a small manually checked sample and exact-match/F1 as a secondary sanity check.

### Provisional recommendation

Given the measured 16 GB RAM and Intel shared-memory graphics, use **Qwen3 1.7B + BGE-small-en-v1.5** for the first local smoke run. After freeing memory and measuring inference, try Qwen3 4B if it is stable. Do not choose the final model until the hardware pilot; once selected, keep the exact model and settings fixed across all methods.

## 2. Dataset choice and how to add it

### Recommended dataset

Use **DataRobot-Research/HotPotQA, `train_hard`**. It is the prepared Syftr-compatible dataset, has a supported loader (`HotPotQAHF`) in `syftr/storage.py`, and has a study YAML in your checkout. HotPotQA focuses on questions that require combining evidence across Wikipedia passages. The DataRobot Hugging Face dataset card currently reports Apache-2.0 and a repository size around 276 MB; the loaded grounding corpus and indexes can need additional disk space. [Dataset page and download options](https://huggingface.co/datasets/DataRobot-Research/hotpotqa)

Syftr's docs describe the `train_hard` partitions as: sample 20, train 7,305, test 500, and holdout 7,836 QA pairs. Use the sample only for wiring checks, test for optimizer development, and holdout for final confirmation after the experiment protocol is frozen. Do not optimize on holdout.

### Do we need to manually put the dataset in the project folder?

**Usually, no.** The supplied `HotPotQAHF` loader downloads both the QA and grounding subsets from Hugging Face via `datasets.load_dataset()` when the study needs them. It stores them in Syftr's Hugging Face cache. The study YAML already identifies the dataset:

```yaml
dataset:
  xname: hotpotqa_hf
  subset: train_hard
```

The existing file `studies/gpt-oss1--cost--hotpotqa_hf-train_hard--train_hard.yaml` has the detailed dataset configuration. You can copy that study into a new study file and edit the new copy. Do not put downloaded files into `sources/`; that folder is read-only project reference material in this workspace.

### If you want the cached dataset inside `RM-project`

Create a cache directory such as `RM-project/data/huggingface`, then set the Syftr configuration path to it in the active `config.yaml` (keep `config.yaml` local and never commit secrets):

```yaml
paths:
  tmp_dir: "C:/Users/<your-user>/.../RM-project/tmp"
  huggingface_cache: "C:/Users/<your-user>/.../RM-project/data/huggingface"
```

Use the actual full path on your computer. Syftr's HotPotQA loader passes `cfg.paths.huggingface_cache` to Hugging Face Datasets. On first run, it will download the prepared data into that cache automatically. The cache will contain Hugging Face's storage layout, not a neat pair of CSV files; that is expected. Keeping cache outside the source checkout is also fine and avoids placing generated data in Git.

If you explicitly need to pre-download for offline use, use Hugging Face's supported download/cache flow and preserve the repo's expected dataset configs/subsets. Simply copying the original HotPotQA files into an arbitrary folder will not make this loader use them without a local dataset adapter.

## 3. Phase 0: record hardware and freeze a small protocol

Record:

- Windows version; Python version; RAM; GPU model and VRAM; free disk; whether CUDA is available.
- Syftr Git revision, dependency lock, Ollama version, downloaded model tag/digest, embedding model revision.
- Dataset subset and exact partitions; objective names and directions; study seed; trial count; sample count; pruning settings.

The repo supports Python `>=3.11.1,<3.13` and recommends Python 3.12.7 with `uv`. The dependency set includes Ray, Optuna, Transformers, Torch, LlamaIndex and vector/embedding integrations. Start with the repository's documented installation path; do not update dependencies midway through the comparison.

## 4. Phase 1: run a no-paid-API Syftr baseline

### 4.1 Configure the local model

1. Install Ollama and pull the selected model (begin with Qwen3 4B if its resource needs fit).
2. Start the local server and confirm the model answers one short question at `http://localhost:11434`.
3. In the local `config.yaml`, register the endpoint as a Syftr local/OpenAI-compatible model. Ollama's OpenAI-compatible endpoint is `http://localhost:11434/v1/`; its client examples use a placeholder API key that the local server ignores.
4. Register one local embedding choice, such as `BAAI/bge-small-en-v1.5`, and ensure the study uses local Hugging Face embeddings rather than an inference API.
5. Remove or exclude cloud-only LLMs from the active study's search lists, including agent subcomponents and judge configuration. A local generator alone does not eliminate API use if an evaluator, HyDE, agent, or reranker still points to a cloud provider.

### 4.2 Choose the first objective pair

For an end-to-end local run, the clearest initial pair supported in the supplied code is:

- maximize `accuracy` (with the local correctness judge),
- minimize `p80_time` (measured latency).

Syftr's `OptimizationConfig` supports `accuracy` or `retriever_recall` for objective 1, and `p80_time`, `llm_cost_mean`, or `retriever_context_length` for objective 2. Local inference does not incur the same vendor API charge, so `llm_cost_mean` is not an informative cost objective unless the team defines and documents a compute cost rate. If the research specifically wants **token usage**, the provided end-to-end objective set does not list a token-count objective: define exactly which tokens count and add/validate that metric before using it. For a retrieval-only pilot, `retriever_recall` + `retriever_context_length` already exists but is not an end-to-end generator result.

### 4.3 Start with a bounded study

1. Copy the existing HotPotQA `train_hard` study into a new local study file; keep the upstream file intact.
2. Set a short smoke run (20-question sample, a few trials, one concurrency worker).
3. Verify data loading, local generation, local judging, embeddings, SQLite/Optuna records, Ray workers, and that no external provider is called.
4. Increase to a development run on a fixed test subset and a small number of trials; save every candidate, prediction, quality score, latency, error, and resource log.
5. Run the Syftr TPE baseline with the final frozen study settings. Save the config, dataset/cache identifiers, model digest, seeds, full trial history, and Pareto points under `results/`.

This is **our local Syftr baseline**, not a direct reproduction of the paper's reported accuracy or costs; model, evaluator, dataset partition, compute, and study budget differ.

## 5. Phase 2: define the fair optimizer comparison

### Independent variable

Optimizer only:

1. Syftr TPE / MO-TPE reference.
2. NSGA-II.
3. Hybrid NSGA-II + Bayesian Optimization.
4. Adaptive / learning-guided NSGA-II.

The third method is **adaptive NSGA-II**, as confirmed by the user. It will use search feedback to adjust one or more evolutionary controls over time; it is not an RL method.

### Keep these identical

- Search-space definitions and conditional validity rules.
- RAG flow/evaluator, model, judge, prompt, embeddings, and any reranker.
- Dataset, partition, sample order, and holdout policy.
- Objective formulas, direction, latency measurement boundary, and failure treatment.
- Initial population/seeding policy, pruning policy, and total full-equivalent QA evaluations.
- Hardware, concurrency, and the number of random seeds.

Syftr's search space is conditional and hierarchical. Do not pass it to a flat genetic algorithm until there is a clear encoding/decoding layer: define a chromosome, legal value ranges, conditional activation rules, repair of invalid combinations, and duplicate handling. For a first comparison, constrain the variable search to a small Syftr-derived subset and explicitly freeze everything else. This makes the evaluation feasible and makes the scope reproducible.

### Define each proposed strategy before coding

- **NSGA-II:** generate a population, evaluate each valid flow with the common Syftr evaluator, rank by non-dominated sorting, preserve spread with crowding distance, then select/mutate/crossover within the same trial budget.
- **Hybrid NSGA-II + BO:** choose one mechanism, such as periodically asking a Bayesian surrogate to rank or propose offspring before expensive evaluation. Specify frequency, candidate count, injection and duplicate handling. Avoid loosely combining two independent algorithms without a measurable intervention.
- **Adaptive NSGA-II:** decide what changes based on search feedback (e.g., mutation rate from diversity/stagnation), how often it changes, and the adaptation rule. Keep ordinary NSGA-II as the ablation.

## 6. Phase 3: implement and compare

1. Create a shared objective/evaluation interface that receives a valid Syftr configuration and returns the same metric vector for every optimizer.
2. Keep the existing TPE implementation as the untouched reference path.
3. Add NSGA-II as a separate optimizer module; log proposed configurations before each expensive evaluation.
4. Add one hybrid mechanism on top of the same NSGA-II candidate/evaluation interface.
5. Add the adaptive strategy with logged control values (e.g., mutation rate by generation).
6. Add validation for invalid configs, duplicate configs, failed model calls, pruned partial evaluations, timeout, and resume behavior.
7. Use equal full-equivalent evaluation budgets (count QA/model/judge calls, not just “trials” or “generations”). Include optimizer overhead/training time separately from RAG inference time.
8. Run several independent seeds per method; store raw trials, configs, outputs, and environment manifest.

## 7. Evaluation and reporting

Primary comparison:

- Hypervolume of the quality-latency frontier using the same predeclared normalization and reference point for all methods.
- Quality at a fixed latency threshold and latency at a fixed quality threshold.
- Progress curves (hypervolume or best-so-far frontier by full-equivalent evaluations).

Additional reporting:

- Number of candidate proposals, valid unique flows, full and partial RAG evaluations, model/judge calls, and failures.
- Search overhead and total wall time, reported separately.
- Non-dominated configurations and their actual flow parameters.
- Mean, standard deviation, and confidence interval over seeds where feasible.

IGD requires a reference Pareto set, which is unknown in a black-box RAG search. If used, construct the reference set from the pooled evaluated results across all methods/seeds, state this, and avoid treating IGD as an absolute ground-truth metric. Do not select final settings using holdout results.

## 8. Project outputs to save

Under `ourwork/` keep the study protocol, dependency and hardware notes, local model setup, dataset/partition notes, design of each optimizer, and implementation log. Under the code project keep versioned configuration/study files (without secrets). Under `results/` save per-run configs, seeds, trial logs, predictions, metrics, plots, Pareto sets, and a summary table. Preserve the original checkout and track custom code/configs separately so results can be reproduced.

## 9. Immediate next actions

1. Send the computer's RAM, GPU/VRAM, and free disk so we can pick Qwen3 4B vs 8B as the fixed study model.
2. Confirm HotPotQA `train_hard` as the primary dataset.
3. Use the prepared dataset card: https://huggingface.co/datasets/DataRobot-Research/hotpotqa . You normally do not manually copy it; Syftr downloads it into its Hugging Face cache. If you want the cache in the repository, set `paths.huggingface_cache` in the local config to `RM-project/data/huggingface`.
4. Choose quality + latency as the first end-to-end objective pair, or tell us if context-token count is a required objective; the latter needs a defined metric in this code.
5. Run the short local Syftr smoke baseline before implementing any new optimizer.


