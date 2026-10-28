# Local Syftr pipeline for this PC

**Prepared:** 2026-10-24  
**Project checkout:** `RM-project` in this workspace (the user also has a Desktop project path; use the path that actually exists on the PC).  
**Experiment goal:** run the provided Syftr workflow locally first, save a reproducible TPE baseline, then compare it with NSGA-II, NSGA-II + Bayesian search, and adaptive NSGA-II using the same RAG space, model, data, and evaluation budget.

## 1. PC snapshot and practical implications

The machine check reported:

- CPU: Intel Core Ultra 7 155H.
- RAM: 15.7 GB total. Only about 1.2 GB was free during the snapshot; that is a point-in-time reading, but it is too little to start a local model at that moment.
- Graphics adapter reported by Windows: Intel Arc Pro Graphics. No NVIDIA GPU / `nvidia-smi` was available. Treat this as an Intel integrated/shared-memory setup; do not plan the experiment assuming dedicated VRAM.
- C: had about 97.8 GB free.
- Python available in the shell was 3.14.7; this Syftr checkout supports Python 3.11.1–3.12.x, so use 3.12.7 in a project virtual environment.
- `uv` 0.11.6 is available. Ollama client 0.24.0 is installed, but the current inspection session could not connect to its local server. Start/check Ollama on the PC itself before the model steps.

**Model choice for this hardware:** Qwen3 1.7B is the conservative first model for a smoke run. Its Ollama Q4_K_M download is about 1.4 GB; that is disk size, not total runtime RAM. Qwen3 4B Q4 is about 2.5 GB on disk and may be usable after closing memory-heavy apps, but measure it before committing to the full study. Do not start with 8B on a 16 GB shared-memory machine. Ollama lists the Qwen3 sizes on its model page: https://ollama.com/library/qwen3

At the time of the check, free RAM was the immediate bottleneck. Reboot/close heavy applications and recheck Task Manager before proceeding. Start with one concurrent trial and a small evaluation sample.

## 2. Keep source material and experiment outputs organized

Keep the provided Syftr checkout intact. Put project-created runbooks, local study YAMLs, logs, and exported result tables under `ourwork/` (or a dedicated results folder inside the actual project checkout). Keep downloaded Hugging Face data and generated indexes in cache folders so the dataset does not get mixed with source files. Do not commit local credentials, caches, or generated databases.

The repository has a prepared HotPotQA loader and a study for `DataRobot-Research/HotPotQA`, subset `train_hard`. You do **not** need to manually download or place the dataset in the repo: Syftr uses the Hugging Face `datasets` loader and its cache on first run. Dataset card: https://huggingface.co/datasets/DataRobot-Research/hotpotqa

The card and dataset version can change. Record the resolved dataset revision/date and the Syftr repo commit in the experiment notes. The repository study describes sample/train/test/holdout partitions; use `sample` for wiring and `test` for optimizer work, reserving `holdout` until the protocol is frozen.

## 3. Prepare the Windows Python environment

Open PowerShell in the Syftr repository root (the folder containing `pyproject.toml`). The exact checkout may be `C:\Users\parth\Desktop\Sem-5\Research_Methodology_Project\RM-project` or the workspace copy. Use the one that exists and keep all commands in that same folder.

```powershell
uv python install 3.12.7
uv venv --python 3.12.7
.\.venv\Scripts\Activate.ps1
uv sync --extra dev
uv pip install -e .
```

If PowerShell blocks activation, use the environment's interpreter directly instead of changing machine-wide execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

Ray is part of this pipeline. Ray documents Windows as beta, with slower performance and higher process-memory requirements than Linux. Keep concurrency at one initially. If the local Ray dashboard/worker path fails on native Windows, the fallback is WSL2 and the same project setup there; do not switch platforms until the native smoke run shows a concrete problem. Ray docs: https://docs.ray.io/en/latest/ray-overview/installation.html

## 4. Start and verify Ollama

Open Ollama normally from Windows so its local server is running, then use PowerShell:

```powershell
ollama --version
ollama pull qwen3:1.7b
ollama run qwen3:1.7b
```

Enter one short question and confirm a response, then exit the interactive prompt. Ollama exposes a local OpenAI-compatible API at `http://localhost:11434/v1/`; its compatibility guide says the API key value is ignored by Ollama, though Syftr's OpenAI-compatible provider requires a nonempty key field: https://docs.ollama.com/api/openai-compatibility

The Qwen model may initially use CPU or shared Intel graphics depending on Ollama's runtime support. Verify real memory use and generation speed in Task Manager before a longer run. For consistent comparisons, record exact Ollama version, model tag/quantization, context length, and whether GPU acceleration was active.

## 5. Configure Syftr to use the local model

Make a private local `config.yaml` in the Syftr root (or the configured Syftr config location). Start from `config.yaml.sample`; do not replace the sample. Add the local model entry under `local_models`, which this checkout automatically exposes to its OpenAI-like model configuration:

```yaml
local_models:
  default_api_key: "ollama"
  generative:
    - model_name: "qwen3:1.7b"
      api_base: "http://localhost:11434/v1/"
      api_key: "ollama"
      temperature: 0.0
      top_p: 0.95
      max_tokens: 256
      context_window: 4096
      is_chat_model: true
      is_function_calling_model: false
      timeout: 300
      max_retries: 0
```

`local_models.generative` is the schema in this checkout; it is different from adding an arbitrary `generative_models` provider entry. The model name is what the study's LLM fields must reference. Keep generation deterministic (`temperature: 0`) for the initial comparison. Do not use agent/tool-calling flows during the first local baseline because the selected small model is not being evaluated for reliable function calling.

Syftr's path settings support `results_dir`, `tmp_dir`, `huggingface_cache`, and `index_cache`. Set these to writable folders on the drive that has space, using forward slashes or quoted Windows paths. For example:

```yaml
paths:
  results_dir: "C:/Users/parth/Desktop/Sem-5/Research_Methodology_Project/ourwork/results"
  tmp_dir: "C:/Users/parth/Desktop/Sem-5/Research_Methodology_Project/ourwork/tmp"
  huggingface_cache: "C:/Users/parth/Desktop/Sem-5/Research_Methodology_Project/ourwork/cache/huggingface"
  index_cache: "C:/Users/parth/Desktop/Sem-5/Research_Methodology_Project/ourwork/cache/indexcache"
```

Change the root if the checkout is in a different location. These paths are examples; ensure each parent is writable. Keep `config.yaml` private/local and do not commit it.

## 6. Create a small, local-only study from the supplied HotPotQA study

Copy the existing study YAML to a new name, for example `studies/local_qwen3_hotpotqa_smoke.yaml`. Do not edit the supplied upstream study in place. In the copied study, make these changes:

1. Keep `dataset.xname: hotpotqa_hf` and `dataset.subset: train_hard`.
2. For the smoke run, set `dataset.partition_map.test: sample`. The HotPotQA sample partition is 20 examples in this repository's documentation. This maps Syftr's canonical evaluation/test requests to the sample partition.
3. Set `evaluation.llm_names` to the exact local model name registered in config, `qwen3:1.7b`.
4. Set `search_space.response_synthesizer_llm_config.llm_names` to only `qwen3:1.7b`.
5. Set `search_space.rag_modes` to only `rag` for this first study. Leave the core retrieval/chunking/RAG variables you intend to compare, but remove model lists that reference hosted models. Since the first run uses a single local model, it should not search over LLM choice.
6. Set `optimization.objective_1_name: accuracy` and `optimization.objective_2_name: p80_time`. Paid-API dollar cost is not a useful local objective unless a compute-cost rate is defined. `p80_time` measures local end-to-end time.
7. For smoke, set `optimization.num_trials: 2` (or 3), `optimization.num_eval_samples: 5`, `optimization.max_concurrent_trials: 1`, `optimization.cpus_per_trial: 1`, and `optimization.num_eval_batch: 1`. Use these only to validate the pipeline, not as publishable evidence.
8. Keep `optimization.sampler: tpe` for the baseline. Make sure recreate/reuse settings do not resume an old unrelated study; give this experiment a unique `name` and database study identity.
9. Remove/disable unrelated search variables that can invoke other LLMs (for example critique, sub-question, agent flows, and any hosted-model list). `rag_modes: [rag]` is the key guardrail, but inspect every remaining `llm_names` field in the new YAML.

For the embedding model, start with the supplied baseline embedding `BAAI/bge-small-en-v1.5`. Configure it to run via local Hugging Face embeddings and keep `optimization.use_hf_embedding_models: true` if the study's RAG retriever lists this model; set `optimization.embedding_device: cpu` initially to avoid unsupported/unstable Intel acceleration. Keep the embedding choice fixed to this one model across all algorithms. On first use, Hugging Face will download model files into the configured cache.

**Evaluation caution:** Syftr's `accuracy` in the supplied study is LLM-judged correctness. Using the same small Qwen model as both answer generator and judge is affordable and consistent but can be weak/noisy. Use the exact same judge prompt/model for all optimizer arms, and report it. Also compute exact match/F1 on saved answers as a secondary check if the project evaluation needs a less model-dependent metric. Do not compare the old GPT-judged results directly to local-judge scores as though only the optimizer changed.

## 7. Run the smoke study and preserve its evidence

In the active venv and repo root:

```powershell
syftr check
syftr run studies/local_qwen3_hotpotqa_smoke.yaml --follow
```

Syftr's README documents these commands. The first run may download dataset, grounding corpus, embedding weights, and build indexes. That can take time and disk; allow it to finish. Save the terminal output/log, effective YAML, Syftr commit hash, Ollama/model details, and resulting database/export under `ourwork/results/`.

A successful smoke run means the local model responds through the API, the data loader reaches the sample partition, retrieval/indexing works, at least one trial completes, and results include accuracy and `p80_time`. It does not establish a research result.

## 8. Grow from smoke run to a defensible baseline

After the smoke run succeeds:

- Run a modest pilot using the `test` partition (e.g., 50 examples if supported by study configuration; otherwise use the 500-example test partition and note it). Increase to 10–20 TPE trials first, concurrency one.
- Check model RAM, average throughput, Ray worker stability, evaluation failures, and total runtime. If 1.7B is reliable and there is adequate memory after a clean reboot, trial Qwen3 4B on the same sample before selecting the final model. Lock one model tag for all arms after this pilot.
- For a stronger baseline, increase the number of evaluated examples and trials only after measuring runtime. The research comparison should use identical example IDs and equal expensive objective evaluations across Syftr/TPE, NSGA-II, hybrid NSGA-II + Bayesian optimization, and adaptive NSGA-II.
- Repeat each optimizer with multiple random seeds if runtime permits. Report mean and spread (e.g., standard deviation or confidence interval), Pareto fronts, best/selected trade-offs, function-evaluation budget, and wall time.
- Preserve a final untouched holdout partition for confirmation. Never tune optimizers on holdout.

## 9. Comparison protocol to freeze before implementing the three methods

The four experiment arms are:

1. Syftr's existing TPE-based search: baseline.
2. NSGA-II.
3. NSGA-II + Bayesian optimization.
4. Adaptive NSGA-II (the corrected third method from the project plan; not RL).

Use the same Syftr-derived RAG search space, generator, judge, embedding model, dataset examples/split, answer prompt, objective definitions (`accuracy` and `p80_time`), number of evaluated examples, parallelism, and evaluation budget. Keep one result per trial with the full parameter vector, objectives, seed, runtime, and success/failure. Clearly describe any custom evaluation adapter or changes needed to connect Syftr to the new optimizers. Syftr does not provide these three proposed methods as ready-made samplers in the checked-out repository; they will be implemented in a later project phase.

A local model changes the absolute scores and latency relative to the paper's paid hosted models. The fair claim is a controlled comparison of optimizers in this local environment, not an exact reproduction of the paper's headline numbers.

## 10. Current blockers / things to verify on the PC

- Free RAM was about 1.2 GB at inspection. Free memory before model download/inference; capture a fresh Task Manager reading.
- Ollama is installed, but its server did not answer from the inspection session. Start Ollama on Windows and verify its API is reachable before installing anything else.
- Native Windows Ray can be slower and more memory-heavy. Start with one worker/trial and use WSL2 only if native execution fails.
- Python 3.14 is outside this repository's declared support; isolate Syftr in Python 3.12.7.
- Exact local model integration and the copied study must pass `syftr check` and a sample evaluation before expanding the budget.
