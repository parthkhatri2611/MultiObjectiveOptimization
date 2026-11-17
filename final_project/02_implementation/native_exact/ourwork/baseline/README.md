# Reproducible Syftr local-model baseline

This package prepares and executes the baseline before the proposed evolutionary algorithms are implemented. It uses the actual `syftr.studies.SearchSpace` and `syftr.tuner.qa_tuner.build_flow`; it does not replace Syftr with a three-variable RAG example.

## Selected model and research scope

- Generator and auxiliary LLM: **Qwen2.5-7B-Instruct, official Q4_K_M GGUF**.
- Serving: a pinned official llama.cpp build, CUDA offload, loopback OpenAI-compatible API.
- Context: 16,384; maximum output: 1,024 tokens per call; fixed temperature 0/top-p 1.
- Primary dataset: prepared `DataRobot-Research/hotpotqa`, `train_hard`, pinned dataset revision.
- Search families: ordinary RAG, SubQuestion, Critique, ReAct and LATS.
- Retrieval: dense, sparse, fusion; optional decomposition, HyDE, reranking, few-shot and additional neighboring context.
- Splitters: recursive, sentence and token, with paper-derived chunk/overlap ranges.
- Embeddings: all eleven paper models are declared. Each must pass a local-load capability check. Missing/failed models are not silently removed.
- Local correctness judge: the same model weights, fixed JSON prompt, passing threshold 4/5. Raw scores and responses are retained; this is not a calibrated reproduction of the paper's random hosted judges.

The exact generator catalog and decoding differ from the paper; they are held fixed across optimizer arms. The primary resource objective is actual mean generation-workflow input + output tokens, replacing historical API dollar prices. Primary quality is local judged accuracy, with benchmark-compatible answer F1 and EM also reported. Latency is measured separately; a latency-objective profile is provided. Do not compare results across differing objective profiles as though only the optimizer changed.

The model was chosen for Kaggle feasibility and software compatibility. It has not been declared empirically best across all available models. Its readiness and agent/tool behavior must pass the preflight and capability stages.

## Files and commands

| File | Purpose |
|---|---|
| `configs/*.json` | Immutable stage protocols and optional objective variants |
| `resolved_sources.json` | Pinned GGUF, HotPotQA and llama.cpp revisions |
| `spec.py` | Shared conditional space and capability cases |
| `adapter.py` | Actual Syftr flow construction, pinned data/models, generator tokenizer correction |
| `serve.py` | Build pinned CUDA server and verify downloaded model hashes |
| `proxy.py` | Journal every generation/judge HTTP request and actual usage |
| `worker.py` | Isolated full per-question evaluation and incremental saves |
| `run.py` | Owned, seeded TPE sampler, total budget, subprocess timeouts, resume |
| `records.py` | Provenance, atomic JSON, durable JSONL, protocol checks |
| `analyze.py` | Measured tables, Pareto/HV, observed component and question-group analysis |
| `confirm.py` | Holdout evaluation of search-selected trial IDs, without holdout search |
| `bootstrap.py` | Isolated locked Python 3.12 environment |
| `session.py` | Blocking batch execution; waits for setup/server/run |
| `package.py` | Allowlisted source ZIP and notebook export |
| `test_baseline.py` | Scoring, conditional sampling, usage proxy, recovery and metric tests |

Run lightweight local tests:

```bash
ourwork/baseline/.venv/bin/python -m unittest ourwork.baseline.test_baseline -v
python3 -m ourwork.baseline.package
```

The test environment intentionally omits model weights and the full LlamaIndex stack. Schema tests execute the real dependency-light classes extracted from `studies.py`; they validate conditional sampling but do not prove inference compatibility. Kaggle import and model smoke tests are mandatory.

## Stages

1. **Smoke:** 2 candidate attempts, 3 sample questions, ordinary RAG only. Infrastructure check; no research conclusion.
2. **Capabilities:** 14 deterministic representative cases on 3 sample questions, plus local loading/encoding checks for every embedding. Tests all five families and optional retrieval/prompt modules. Accuracy need not be high, but execution and complete accounting must succeed.
3. **Pilot:** 10 candidate attempts on 10 fixed test questions, all five families and full declared module space. Measure throughput, failures, judge parsing and truncation before committing the final budget.
4. **Baseline:** initial maximum of 30 attempts on 20 fixed test questions, 10 TPE startup observations; repeated seeds 11, 23 and 37. This budget is a feasibility target until pilot runtime is measured; larger studies can use a new frozen profile after pilot evidence. A session can stop before this target and resume; an incomplete budget is reported as incomplete.
5. **Confirmation:** explicitly selected trial IDs on a predeclared holdout manifest. No optimization on holdout.

Initial standard candidates are charged within the budget. Duplicate candidates, failures and interrupted attempts also consume attempts, while all known model calls/tokens are retained. Pruning is disabled across the comparison to make evaluation fidelity uniform. The fixed budget is candidate attempts on a fixed QA manifest, supplemented by actual model-call/token/time accounting. It is not a claim of equal deployment tokens across differently structured flows.

The main runner requires a matching successful preflight and complete capability report before pilot/baseline. A branch failure must be repaired or its exclusion explicitly versioned for **all** methods; it must not be hidden by shrinking one algorithm's space.

## Kaggle interactive setup

Use the generated notebook and attach the source ZIP privately. Kaggle may expand ZIP uploads into a source directory: the extraction cell must handle both forms. Keep Internet enabled during dependency/model/data setup. The active notebook may use Python 3.13; the project executes through its separate Python 3.12 environment.

Long dependency/model/index caches go under `/kaggle/temp/rm-syftr`; research artifacts go under `/kaggle/working/baseline_results`. Only the latter should be persisted as notebook outputs. Confirm actual scratch capacity before installing.

For a clean blocking batch run after source extraction:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-m', 'ourwork.baseline.session',
    '--stage', 'smoke', '--seed', '11',
    '--scratch', '/kaggle/temp/rm-syftr',
    '--output', '/kaggle/working/baseline_results'
], check=True)
```

Advance stages only after the prior output is inspected. The interactive notebook uses background processes to make progress visible; Save & Run All needs the blocking entry point above. Never launch the setup cell twice simultaneously. The runner holds an exclusive run-directory lock to prevent concurrent experiment writers.

## Structured results

```text
baseline_results/
  environment_setup.log
  server.log
  server_metadata.json          # revisions, GGUF/binary hashes, startup flags
  preflight.json                # allocated GPU/packages, usage and tool-call tests
  baseline-seed11/
    protocol.json               # source + model + data + objective + seed identity
    sources.json                # resolved HF revisions for every selected component
    environment-*.json
    packages.json
    questions_manifest.json     # fixed QA IDs and content hash
    grounding_manifest.json     # full corresponding corpus fingerprint
    effective_study.json
    distributions.json
    study.sqlite3
    sampler_checkpoint.json
    sampler-<hash>.pkl          # own verified sampler state, not a third-party file
    events.jsonl
    candidates/00000-<id>/
      job.json
      worker.log
      calls.jsonl               # raw prompts, completions, tool calls, endpoint usage
      requests.jsonl            # request starts, including calls interrupted before a response
      questions.jsonl           # answers, refs, judge/F1/EM, errors, durations
      progress.json
      result.json
    analysis/
      summary.json
      FINDINGS.md
      trials.csv
      pareto.csv / pareto.json
      convergence.csv
      parameter_observations.csv
      component_outcomes.csv
      human_audit.csv
      per_question.csv
      question_groups.csv
      failures.json
      pareto.png / convergence.png
```

Generated tables contain measured outcomes only. Unknown usage is not replaced with a character heuristic or zero. A candidate without complete verified generation usage cannot enter the reported frontier. API charge metadata is zero only to represent no paid API charge; it is not optimized or presented as zero compute cost.

Generation, nested tool/HyDE/reranking calls and judge calls are separated by evaluation phase. Prompt and completion tokens come from the serving endpoint. Raw prompts preserve the actual context supplied, even when response synthesis performs more than one generation. Judge overhead is search overhead, not workflow deployment cost.

Generation and judge errors are kept separate; missing outcomes are scored as failures rather than silently removed from the denominator. Candidates must reach the declared success threshold and complete the full manifest. Timeouts retain partial raw records, but partial objective estimates never become completed Pareto points.

## Resume and persistence

Save/download `baseline_results` before the Kaggle container ends. A file existing in a running draft is not proof of persistence. Restore outputs to the same writable location for resume. Restore the same source, package versions and serving binary/model settings; the runner rejects changed identities. Sampler state is hash-verified and belongs to this run. Do not restore untrusted pickle files.

An interrupted RUNNING trial becomes FAIL and consumes its attempt. Its partial QA/call history remains available. The runner resumes subsequent proposals with saved sampler state instead of restarting TPE or pretending the interrupted trial completed.

Rebuild/disposable index costs are reported as construction/total candidate wall time separately from per-query generation latency. Use the same cache policy for later algorithms. Hardware class and package versions must match for timing comparisons.

## Findings available from a completed baseline

- Quality/token and quality/latency trade-offs for observed configurations.
- Highest-quality, lowest-resource and balanced Pareto candidates, with concrete parameters.
- Hypervolume progression versus attempts, known model calls/tokens and wall time.
- Agentic versus non-agentic observations and failure patterns.
- Retrieval/splitter/embedding/prompt/HyDE/reranking/few-shot/neighbor appearance on the discovered frontier.
- Question-type and difficulty performance, raw error/truncation inspection and judge disagreements with EM/F1.
- Initialization versus later TPE behavior and repeated-seed variability.
- Holdout generalization of preselected candidates.

Component appearances are observational, not causal ablations. The pooled empirical reference for IGD is explicitly not the unknown global Pareto front. No conclusions about superiority over evolutionary algorithms are possible until their matched runs exist. The baseline does not claim to enumerate the enormous complete space or reproduce all paper figures with different models and judges.

The full paper dataset suite is represented in `spec.DATASETS`; HotPotQA is the first validated target. Other datasets require separate capability/data checks and their own frozen protocols. Do not represent unexecuted dataset profiles as measured findings.

## Changes relative to upstream execution

- Candidate generation is centralized and keeps an explicitly seeded Optuna 4.2.1 sampler; it avoids the upstream worker reload that loses sampler settings.
- A sequential subprocess evaluator invokes the same Syftr flow builder, avoiding expensive Ray worker model duplication. Ray remains a dependency; this is a documented scheduling adaptation, not the original distributed execution.
- Data revisions and model/tokenizer revisions are pinned; the generator tokenizer replaces the hard-coded GPT tokenizer for sentence/token chunking. Recursive splitting retains its embedding-tokenizer semantics, now revision-pinned.
- Native approximate F1 is replaced in exported answer metrics with multiset-based normalized F1 and HotPotQA yes/no safeguards. The optimization target is explicitly declared.
- Judge prompt JSON syntax, denominator treatment and full/partial accounting are corrected in the shared evaluator. Raw failures remain visible.
- A loopback usage proxy provides independent call-role/token evidence without relying on upstream aggregate pricing/tracing.
- The upstream global token-tracking handler is disabled in this adapter because its embedding-list serialization emits errors; endpoint journaling is the authoritative accounting path. The global LlamaIndex prompt tokenizer is set to the pinned generator tokenizer as well.

The upstream source files are preserved. All future optimizers must use this same evaluator and declared adaptations for a fair comparison.
