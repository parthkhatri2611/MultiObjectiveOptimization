# Research Methodology Project: Complete Syftr Analysis and Local Implementation Roadmap

**Review date:** 14 November 2026  
**Repository inspected:** `Research_Methodology_Project`  
**Recorded HEAD:** `7595f1d` (`Intitialized the syftr baseline implementation with its analysis`)  
**Purpose:** establish the technical and research foundation before selecting the final local model, running the baseline, and implementing the three proposed optimizers.

**Scope clarification from the user, 14 November 2026:** the research uses Syftr's hierarchical workflow search space. The three-variable `[k, c, Bc]` formulation is set aside. We compare and improve evolutionary search strategies on the same Syftr problem; we do not reduce the research to those three variables.

## 1. Project understanding and recommended direction

The project investigates **how the optimization algorithm affects the discovery of good RAG configurations under a limited evaluation budget**. Syftr supplies the base formulation, dataset preparation, RAG components, evaluation infrastructure, and TPE reference. Our three proposed approaches are:

1. NSGA-II.
2. Hybrid NSGA-II + Bayesian optimization.
3. Adaptive NSGA-II.

These are three **search algorithms**, not three new language models or three unrelated RAG architectures. Each must propose configurations to the same evaluator. The local generation model, embeddings, dataset examples, parameter bounds, answer scoring, and budget policy must remain the same across the four arms.

The intended sequence is sound:

```text
Understand paper + code + proposal
                  |
Choose and pilot a feasible local model
                  |
Validate common RAG evaluator and measurements
                  |
Run and preserve a local Syftr/TPE reference
                  |
Implement NSGA-II -> hybrid -> adaptive NSGA-II
                  |
Compare under the same protocol and repeated seeds
                  |
Confirm selected configurations on untouched holdout data
```

**Confirmed direction:** use Syftr's hierarchical workflow search space on a prepared Syftr benchmark, with locally available model substitutions shared by every optimizer. Preserve workflow families, module choices, dependent hyperparameters and validity rules as the research problem. First validate local components and measure resource feasibility, then freeze one common executable search-space specification and objective protocol. Any unsupported branch must be identified explicitly and handled identically across methods; local adaptation must not silently turn the study into a three-variable optimization problem.

The local experiment cannot reproduce the paper's absolute API-dollar results because the models, serving environment, and likely answer evaluator change. It can reproduce the **optimization methodology** and support a controlled comparison in our resource setting.

No model downloads, dependency installations, inference experiments, optimizer implementations, or research-result calculations were performed during this analysis. No existing source or proposal files were edited. Recommendations below are proposed designs, not measured outcomes.

## 2. Review coverage and evidence boundaries

### 2.1 What was examined

The review inventoried all 275 non-Git files present before this report was added. It covered the complete 44-page supplied Syftr PDF, all 26 slide XML parts in the proposal, existing project notes, the implementation and configuration structure, all study YAMLs, notebook contents/output state, tests and fixtures, and the active runtime's basic capabilities.

All 128 Python files were parsed for syntax and structural inventory. Core execution, search, dataset, model, retrieval, evaluation, token accounting, pruning, and storage paths were inspected in detail. Peripheral scripts and notebook families were mapped by their contents and roles. This is a source and methodology review, not a claim that every feature has been executed or every code branch has been verified.

Paper text was extracted page by page. Selected pages were also rendered and inspected, including Figure 4's pruning diagram and Figure A11's statistics table, which is an image and is missing from ordinary text extraction. Supplementary RAGGED and LLMLingua papers were inspected for their relevance to the proposal; the complete-paper analysis in this report concerns Syftr.

### 2.2 Source hierarchy

| Evidence | How it is used |
|---|---|
| Supplied Syftr PDF | What the base paper actually reports, including appendix qualifications |
| Current local implementation | What this checkout implements and how its execution path behaves |
| `PPTnall/RM.pptx` | The team's proposed research direction |
| Existing Markdown and progress DOCX | Prior planning, unresolved choices, and historical machine notes |
| Runtime inspection | What is available in the current shell, not every environment on the computer |
| Official model/runtime/Optuna documentation | External compatibility checks, with links in Section 20 |

When these disagree, this report identifies the disagreement instead of silently treating a planning statement as implementation evidence.

### 2.3 Git state

The configured remote is the team's fork, `https://github.com/parthkhatri2611/MultiObjectiveOptimization.git`; the README identifies the upstream project as DataRobot's Syftr.

Many files appear modified. A diff ignoring end-of-line whitespace removes the text-file differences and leaves the three PDF binaries, consistent with widespread line-ending differences and locally materialized Git LFS paper files. Therefore, the long modified-file list does **not** establish that the research algorithms have been implemented. Existing changes were preserved. No applicable `AGENTS.md` was found inside this checkout.

## 3. Folder-by-folder project map

| Location | Present contents and purpose | Relevance now |
|---|---|---|
| Root | README, license/contribution documents, Makefile, `pyproject.toml`, `uv.lock`, sample config/environment files | Installation, dependency constraints, configuration, project provenance |
| `PPTnall/` | 3 PDFs, `RM.pptx`, 2 extensive research/proposal Markdown files | Base paper, supporting literature, proposed algorithms and experimental framing |
| `ourwork/` | 5 Markdown notes and one progress/meeting DOCX before this report | Prior audits and plans; no experiment implementation here |
| `syftr/` | 85 files including the optimizer framework, flows, loaders, evaluation, plotting and scripts | Main reusable implementation |
| `studies/` | 57 YAML studies | Examples and historical research settings; not local-ready as a group |
| `examples/` | 5 notebooks | Getting started, long studies, prompt optimization, custom documents, IRT |
| `notebooks/` | 21 analysis notebooks | Pareto plots, pruning, seeding, cross-dataset analysis, errors and related studies |
| `docs/` | Provider/dataset/custom-flow/debugging guides plus visual assets | Important configuration references, some examples require schema checking |
| `tests/` | 75 files, including 26 YAML fixtures, Python tests, a SQLite database and pickle fixture | Upstream unit/functional/integration/end-to-end checks |

There are 83 YAML files overall: 57 study files plus 26 test YAMLs. All 26 notebooks are source material; the 21 analysis notebooks have no saved cell outputs. Notebook prose describing a successful study is not proof that the study ran in this checkout.

The only discovered SQLite database is `tests/data/syftr.db`. Read-only inspection found the fixture study `test-initialization`, with 4 COMPLETE and 6 RUNNING trials. This is test data, not our local-model baseline. The pickle fixture was identified but not deserialized to manufacture research results.

### 3.1 Existing planning documents

- `syftr_first_pass_audit.md` is explicitly superseded by the later code audit.
- `code_and_project_readiness_audit.md` correctly identifies the four-arm comparison, conditional search space, objective ambiguity, and need for local validation.
- `project_implementation_guide.md` gives a staged baseline-first plan and confirms adaptive NSGA-II rather than reinforcement learning.
- `pc_local_hardware_and_baseline_pipeline.md` contains historical Windows-specific observations and commands. Its local-embedding flag advice is incorrect for the current call path; see Section 10.
- `guide_progress_and_meeting_summary.md` and the progress DOCX distinguish completed review from pending implementation. They preserve earlier NQ-Open, token-objective, and compute discussions, but do not provide executed baseline results.
- The two long Markdown files in `PPTnall/` elaborate the optimizer comparison. Some abstract-style sentences describe implementations in the present tense; their existence must not be mistaken for completed code.

Historical Windows paths, Python versions, and memory readings must not replace a fresh Linux environment check. The active workspace is now a Linux path on a mounted volume.

## 4. Base paper identity and central contribution

**Paper:** *syftr: Pareto-Optimal Generative AI*  
**Authors:** Alexander Conway, Debadeepta Dey, Stefan Hackmann, Matthew Hausknecht, Michael Schmidt, Mark Steadman, Nick Volynets.  
**Supplied version:** arXiv:2505.20266v1, 26 May 2025.  
**Local source:** [Syftr base paper](../PPTnall/syftr-BasePaper.pdf).

Syftr treats a complete generative workflow as a candidate in a huge configuration space. A candidate includes the modules to use, which models they call, and their hyperparameters. It evaluates candidates on QA tasks and searches for a set of useful quality/resource trade-offs.

Its contribution is the combination of:

1. A broad hierarchical space containing agentic and non-agentic RAG flows.
2. Multi-objective Bayesian search using TPE.
3. Pareto-based early stopping of expensive evaluations.
4. Static, random, and transfer initialization.
5. Empirical analysis of how model and workflow choices affect accuracy, cost, and latency.

Syftr is an **optimizer/framework**, not a pretrained model requiring us to retrain its weights. In this project, “run Syftr on a local model” means using that model inside the workflows Syftr evaluates.

### 4.1 Why workflow optimization is necessary

A conventional RAG pipeline performs:

```text
Corpus -> chunking -> embeddings -> index
Query  -> retrieval -> optional transformations -> prompt -> LLM -> answer
```

Increasing retrieval depth may improve evidence coverage but increase context tokens. Smaller chunks may improve matching but split evidence. Reranking may improve precision but require more inference. Agents can decompose difficult questions but add calls and failure opportunities. A larger model may improve quality but consume more resources.

These decisions interact. Optimizing each component independently can miss good combinations. The paper positions Syftr against greedy component-wise optimization and systems focused mainly on prompt compilation or single-objective workflow construction.

### 4.2 Mathematical formulation

Let `x` be a valid workflow configuration, `D` the fixed evaluation questions, and `G` the grounding corpus:

```text
Q(x; D, G) = answer quality, to maximize
C(x; D, G) = resource objective, to minimize
```

Configuration `a` dominates `b` if:

```text
Q(a) >= Q(b) and C(a) <= C(b),
with at least one strict inequality.
```

The non-dominated observed configurations form an **empirical** Pareto set. Finite search over an enormous space does not prove global Pareto optimality.

For minimization-based algorithms, use `f(x) = [1 - Q(x), C(x)]`. Preserve the objective orientation when moving between Optuna's maximize/minimize interface and NSGA-II/HV calculations.

## 5. Syftr search space in detail

### 5.1 Top-level workflows

The paper describes five top-level families:

| Family | Main behavior | Extra resource burden |
|---|---|---|
| Non-agentic RAG | Retrieve context and synthesize an answer | Usually fewer LLM calls |
| SubQuestion RAG | Decompose a question, answer subquestions, combine | Multiple retrieval/generation steps |
| Critique RAG | Produce and critique/refine an answer | Reflection and correction calls |
| ReAct RAG | Interleave reasoning, tool use, and observations | Iterations and tool parsing |
| LATS RAG | Explore and evaluate reasoning/action trajectories | Tree expansion and rollouts |

An agent family can use the underlying RAG pipeline as a tool. Therefore its parameter space includes the RAG choices plus agent-specific controls. The checked-out code also includes no-RAG and Chain-of-Abstraction flows and additional datasets/models. The checkout is broader than the exact paper-v1 experiment.

### 5.2 Modules and dependent parameters

| Module | Paper options | Parameters/conditions |
|---|---|---|
| Generator | Small and larger hosted/open-weight models | Model, generation settings |
| Splitter | Sentence, token, recursive | Chunk size and overlap |
| Retriever | Dense, sparse, fusion | Retrieval depth; fusion mode and weights where relevant |
| Embeddings | 11 models in Table A3 | Active for dense/fusion and example retrieval |
| Reranker | Optional LLM reranking | Reranker model and retained top-k |
| HyDE | Optional hypothetical-document generation | Additional LLM and retrieval transformation |
| Prompt | Default, concise, CoT, dynamic few-shot | Example count and example embeddings when enabled |
| Additional context | Neighboring nodes around retrieved chunks | Number of neighbors |
| Agent controls | Depend on workflow family | Iterations, reflection, expansions, rollouts, etc. |

Table A1 specifies chunk sizes over a log-space range of 256-4096, overlap fractions 0.0-0.7 in 0.1 steps, and several top-k ranges of 2-20. The paper estimates roughly `10^23` possible flows. These are paper descriptions, not a guarantee that every current YAML uses identical ranges.

Current code represents chunk size as `2 ** splitter_chunk_exp`. Splitter implementations do not all use the same tokenizer. That distinction matters if we later describe `c` as local-model tokens.

### 5.3 Why the space is conditional

For sparse-only retrieval, a dense embedding choice is inactive. Reranker parameters matter only when reranking is enabled. LATS rollout count is irrelevant to ordinary RAG. Ignoring these conditions creates invalid configurations, meaningless mutations, duplicate effective pipelines, and misleading population diversity.

Any evolutionary encoding must distinguish an inactive gene from a valid numerical zero. Canonicalize configurations before duplicate detection and repair dependent values after mutation/crossover.

## 6. Syftr optimization mechanisms

### 6.1 Multi-objective TPE

Ordinary TPE separates prior observations into better and worse groups and estimates parameter densities `l(x)` and `g(x)`. Candidate selection favors settings with a high density ratio `l(x)/g(x)`. Its tree-structured formulation is useful for mixed numerical/categorical and conditional spaces.

In the multi-objective setting, ranking and weighting use Pareto/hypervolume information. The paper describes this through MO-TPE and expected hypervolume improvement. The actual integration in this checkout calls Optuna's `TPESampler`; there is no separate Syftr-coded GP-EHVI acquisition loop.

For the final report, explain the paper's EHVI motivation but describe the executed algorithm by the **actual Optuna version and sampler settings**. Do not equate a separate Gaussian-process EHVI algorithm, arbitrary weighted-sum search, and Optuna MOTPE.

### 6.2 Pareto Pruner

Running every candidate against every question is expensive. Syftr evaluates in batches and forms an optimistic quality/resource estimate:

```text
optimistic quality = estimated quality + uncertainty margin
optimistic resource = estimated resource - uncertainty margin
```

If even that optimistic point is dominated by a completed candidate, evaluation can stop early. Figure 4 uses the upper-left corner of a confidence rectangle in accuracy-versus-cost coordinates. The paper uses 90% confidence levels (`z = 1.645`), discusses approximately normal accuracy and log-normal cost distributions, and evaluates pruning in Appendix A3.

The supplied implementation's `acc_confidence` is `z * sqrt(a * (1-a) / n)`. Its function named `lognormal_confidence` returns `z * std(values, ddof=1) / sqrt(n)` on raw values; it does not visibly fit a log-normal distribution. Figure 4's caption also calls the resource value P80 cost, whereas `llm_cost_mean` in the current evaluator is a mean. These details must be reported from the executed code rather than assumed identical to paper prose.

Pruning is a heuristic with statistical assumptions, not a guarantee that a globally useful candidate cannot be removed. On tiny evaluation sets, normal intervals and percentile estimates are especially unstable.

### 6.3 Other pruning

`CostPruner` extrapolates partial cost against a per-trial limit. `RuntimePruner` extrapolates time but currently uses `time.process_time()`, which measures CPU process time rather than elapsed wall time. Local-server waits and other processes may therefore be poorly represented. Retain independent request/experiment wall-clock limits in any local implementation.

### 6.4 Initialization and transfer

- **Static seeding:** evaluate standard/default flows to provide useful initial observations.
- **Random startup:** sample configurations before density-guided search.
- **Transfer seeding:** select diverse promising configurations from earlier datasets and evaluate them on the target.

Appendix A4 lists static flows. Table A6 reports 100 random plus 46 static seeds for several main studies, and 60 random plus 46 static for InfiniteBench. Appendix A5 examines transferring diverse good configurations to HotPotQA. Its text also mentions a 10-trial random-start baseline; reproduce the specific experiment configuration rather than merging all these counts into one default.

Transfer here concerns configurations/optimization experience, not necessarily neural weight fine-tuning. Raw objective values from a different dataset must not be treated as measured target-dataset values.

## 7. Datasets and answer evaluation

### 7.1 Paper datasets

The paper uses a benchmark suite, not a single universal “Syftr dataset.” Table A7 reports:

| Dataset | Sample | Train | Test used in search | Holdout |
|---|---:|---:|---:|---:|
| HotPotQA train-hard | 20 | 7,305 | 500 | 7,836 |
| FinanceBench | 11 | 53 | 49 | 48 |
| CRAG3 music | 1 | 22 | 34 | 8 |
| CRAG3 sports | 7 | 39 | 46 | 11 |
| InfiniteBench | 5 | 116 | 115 | 115 |
| DRDocs | 5 | 10 | 80 | 10 |

These are the supplied paper's partition counts. Sample can overlap another partition; do not add sample sizes to infer unique dataset size. Confirm actual counts and IDs after loading a pinned revision.

Important task adaptations:

- HotPotQA uses the difficult `train_hard` subset and prepared Wikipedia fragments.
- FinanceBench has financial QA over SEC filings; the prepared version converts PDFs to Markdown and partitions grounding documents by company.
- CRAG uses Task 3 topic partitions, but retrieves across the prepared corpus rather than the official per-question 50 web results. The paper explicitly warns against comparing these scores directly with contest submissions.
- InfiniteBench uses English book QA as retrieval across multiple altered books, rather than simply supplying the whole relevant book.
- DRDocs is synthetic QA over DataRobot documentation.

### 7.2 Split semantics

Syftr calls the search-evaluation partition `test`. In conventional ML terminology this functions as **validation data for optimization**. The `train` split can provide few-shot examples; the `holdout` should be reserved for generalization checks.

Section 5 presents holdout as reserved for final evaluation, but Appendix A6 explicitly says the reported accuracy numbers use the test split and holdout was set aside for future use. Therefore, we should not describe the paper's headline results as independent holdout results.

For our study, retain the names in code and explain their roles:

```text
sample -> wiring/smoke testing
train  -> examples only if few-shot is enabled
test   -> optimizer development and search
holdout -> final confirmation after all decisions are frozen
```

### 7.3 Paper's LLM-as-a-judge protocol

Answers are compared with dataset references using correctness judges. Appendix A7 studies 447 responses across three datasets and three flow families, comparing human labels with ten judge configurations: individual judges, random judge selection, and majority consensus, each with default/modified prompts.

The main study selects the default-prompt random judge configuration for judge diversity without paying for consensus on every example. Table A8 reports random/default Pearson correlation 0.84, Cohen's kappa 0.29, and mean flow-accuracy difference 0.00 versus the human labels. Matching average accuracy does not imply agreement on individual examples.

There are presentation inconsistencies to preserve as caveats: A7 prose names o3-mini in the judge comparison, while Table A8 names gemini-pro; the human-label paragraph says greater than 4, while other passages describe passing as 4 or above. Our executed configuration must fix one exact judge list, prompt, parser, and threshold. The current factory uses a JSON score/reasoning prompt rather than assuming the printed appendix's plain-score format.

### 7.4 Dataset recommendation for this project

Use `DataRobot-Research/hotpotqa`, Syftr dataset name `hotpotqa_hf`, subset `train_hard` as the first primary dataset. It matches the base paper, has an implemented loader, and tests multi-hop QA without initial PDF conversion work. The public dataset page was reachable during this review. [Prepared dataset](https://huggingface.co/datasets/DataRobot-Research/hotpotqa).

`HotPotQAHF` loads:

```text
groundingdata_train_hard -> grounding text
qapairs_train_hard       -> questions, answers and metadata
```

Both are downloaded through `datasets.load_dataset()` into the configured cache. The full benchmark is not bundled in the project. Initial network access is still needed even if inference later runs offline.

The current loader does not pass an explicit `revision`. Add revision control or an immutable downloaded snapshot and record hashes before final experiments. Map QA and grounding splits together. Never shrink the corpus to only each question's gold supporting documents: that changes retrieval difficulty and the task.

For smoke testing, `partition_map.test: sample` maps both canonical test accesses to sample where implemented. For a pilot on a fixed subset of test questions, keep the full corresponding test grounding corpus and record the selected QA IDs. A simple first-N slice is consistent but may be biased; a seeded manifest is preferable for the final comparison.

Historical notes mention NQ-Open. It should not replace HotPotQA in this current same-Syftr-dataset plan without an explicit scope change. FinanceBench can be a later second domain if time permits; one dataset alone limits cross-domain conclusions.

## 8. What the paper's results establish

| Study | Reported finding | Correct interpretation for our project |
|---|---|---|
| Main Pareto searches | Non-agentic flows frequently occupy the frontier | Simple local RAG is a justified starting scope, not proof that agents never help |
| Appendix A7 component appearance figure | 88% of observed Pareto flows are non-agentic | Observational frequency among discovered flows, not a universal win probability |
| Default-flow comparison | Paper states about 6% more accuracy at baseline cost or 37% lower cost at baseline accuracy | Specific hosted-model experiments; Figure A9's roughly 54% to 60% average accuracy is about 6 percentage points |
| Abstract trade-off statement | Approximately 9x cheaper while preserving most accuracy | A different comparison from the fixed-baseline-cost/accuracy analysis |
| Larger-model substitution | Table A9: average +17.3 percentage points, average cost multiplier 67 | Model upgrades can help greatly, at large cost; do not confuse percentage points with relative percent |
| Agent-only FinanceBench | Over 70% accuracy, some flows near $10/call | High quality may require costly agentic computation |
| Latency study | A quality/latency frontier with a useful trade-off around 10 seconds in that setup | No prediction of latency on this laptop |
| Cross-dataset ranks | Modest Kendall correlations across datasets | Best workflows are dataset-dependent |
| Pruner ablation | Lower search cost and useful frontier discovery under cost limits | Must still be checked when metric/model/budget changes |
| Transfer initialization | Helpful startup/convergence in the reported study | Optional later extension; not required for the four-arm comparison |

The main paper also reports FinanceBench accuracy of 77.6%, versus 65.3% for Azure Assistant and 59.2% for Amazon Q under its setup. These are paper results, not current-service evaluations or expected local scores. The comparison includes dataset-specific tuning versus default assistants, which limits what it establishes.

### 8.1 Actual experiment scale

Figure A11, read visually from page 42, reports:

| Study | Successful | Failed | Pruned | LLM costs, USD |
|---|---:|---:|---:|---:|
| FinanceBench, agents only | 549 | 5 | 60 | 2,310.30 |
| InfiniteBench, agents only | 406 | 8 | 87 | 887.29 |
| CRAG music | 502 | 72 | 47 | 380.91 |
| CRAG sports | 390 | 61 | 136 | 629.67 |
| DRDocs | 525 | 30 | 69 | 339.66 |
| FinanceBench | 506 | 76 | 84 | 386.27 |
| HotPotQA | 503 | 17 | 204 | 125.13 |
| InfiniteBench | 475 | 4 | 52 | 764.99 |
| FinanceBench, latency | 478 | 13 | 44 | 1,359.08 |

These are reported LLM costs, not a complete accounting of cluster hardware and indexing. The paper describes a cluster with 696 CPUs, four A100 GPUs and four A6000 GPUs, plus autoscaling hosted embedding endpoints. Its scale cannot simply be copied onto a 16 GB shared-memory machine.

There are minor internal count discrepancies: A9 describes 476 evaluated FinanceBench agent trials, while A11 lists 549 successful trials for that study. Do not silently reconcile these as the same count or use one to infer an exact reproduction budget.

### 8.2 Limits of the paper as our baseline

- The experiment does not establish NSGA-II, our hybrid, or our adaptive method as inferior or superior; those comparisons remain to be performed.
- Large finite searches estimate a frontier, rather than certify the true frontier of `10^23` flows.
- Search-set tuning and judge noise can produce optimistic performance estimates.
- The judge study uses authors as human labelers and includes label review/modification; it is useful validation, with limitations.
- Pruning efficiency and model preference can change when API prices become local tokens or latency.
- Component appearance frequencies are affected by the search process and interactions; they are not independent causal ablations.
- The paper itself leaves generalized search cost-effectiveness analysis outside scope.

## 9. How the current code executes a study

### 9.1 Main call path

```text
CLI: syftr run <study.yaml>
  -> api.Study.from_file / Study.run
  -> Ray submission/runtime
  -> tuner.qa_tuner.run
  -> set_baselines
  -> optimization.StudyRunner.run
  -> get_study + seeding
  -> Ray Tune schedules tune_objective
  -> study.ask()
  -> qa_tuner.objective
  -> search_space.sample
  -> qa_tuner.evaluate / _evaluate / build_flow
  -> build retriever, index, models and prompt
  -> evaluation.eval_dataset
  -> per-question generation + scoring
  -> calculate_metrics
  -> set_trial stores params/metrics
  -> study.tell([quality, resource])
```

`qa_tuner.evaluate(params, study_config)` is a useful reuse boundary: it accepts concrete parameters and returns two objectives, metrics, and serialized flow parameters. It also calls worker preparation. It is not automatically a standalone pure function: tracing, configuration, datasets, pruning's study lookup, and model instrumentation remain relevant dependencies.

### 9.2 Implementation map

| Files/components | Responsibilities |
|---|---|
| `studies.py` | Strict Pydantic schemas; mixed/conditional spaces; sampling, defaults, cardinality, optimization/evaluation settings |
| `optimization.py` | Study creation/loading, maximize/minimize directions, Ray trial orchestration, initialization from studies |
| `optuna_helper.py` | Sampler factory, trial lookup/filtering, seeding, Pareto extraction, knee-point selection |
| `hierarchical_tpe.py` | Additional hierarchical TPE implementation using classifiers and private Optuna internals |
| `tuner/qa_tuner.py`, `tuner/core.py` | Candidate evaluation, flow/splitter construction, metadata persistence |
| `flows.py`, `agent_flows.py`, `tuner/agent_tuner.py` | Ordinary RAG, generator, retriever, judge and agent implementations/interfaces |
| `retrievers/build.py` | Dense/BM25/fusion retrieval and query decomposition |
| `retrievers/storage.py`, `utils/locks.py` | Dense-index caching, serialization, local/optional remote locking and cache access |
| `storage.py` | Prepared HF datasets, partition mapping, QA conversion, custom datasets |
| `llm.py`, `configuration.py` | Provider factories, local models, global paths and settings |
| `huggingface_helper.py`, `hf_endpoint_*`, `embeddings/timeouts.py` | Local/hosted embeddings and timeout handling |
| `evaluation/evaluation.py`, `evaluator_factory.py` | Generation/judging, retrieval scoring, batching, metrics, failures and pruning |
| `instrumentation/tokens.py`, `tracing.py`, `instrumentation/arize.py` | Tokens, call latency/cost, spans and optional tracing |
| `pruning.py`, `custom_metrics.py` | Resource and Pareto stopping; uncertainty and custom retrieval metrics |
| `baselines.py`, `transfer_learning.py` | Static/variation/agent seeds and selecting prior configurations |
| `prompts/`, `output_parsers/`, `templates.py`, template JSON | Answer/judge prompts, parsing, agent templates |
| `api.py`, `cli.py`, `ray/` | User-facing study controls and job lifecycle |
| `plotting/`, `analytics.py`, `sota.py`, stored flow/SOTA JSON | Analysis, reporting, prior/reference information |
| `prompt_optimization.py` | Optional second-stage textual optimization using Trace; separate from the proposed algorithms |
| `scripts/` | Flow builders, model replacement, study maintenance, system checks and historical experiments |
| `amazon.py`, readers, startup/helpers/event loop/patches/logger | Optional cloud storage, document loading and runtime support |

The JSON flow/SOTA assets are upstream reference material. They are not evidence of our new algorithms' results.

### 9.3 Study families

- `example-dr-docs.yaml`: small tutorial, useful for current schema structure.
- `gpt-oss1--cost--*` / `gpt-oss1--latency--*`: broad later model studies; often 3,000 trials and 50 examples.
- `cerebras*`: provider comparisons, including large local endpoint models and up to 10,000 trials.
- `llm2--thinking--*`: reasoning/model studies.
- `seeding1--training--*`: seed evaluation; some have `num_trials: 0` but substantial baseline lists. Zero optimization trials does not imply zero inference.
- `seeding1--testing-{random,silver,transfer}--*`: initialization comparisons.
- `judge-eval-*`: evaluator research, not primary RAG optimization.
- `retriever-only-*`: retrieval subproblems; some YAML fields are from older schemas and optional LLM enhancements remain enabled.
- `my-docs.yaml`: custom corpus example; not the requested same-Syftr-dataset experiment.

“Local-only” in a historical filename means locally served models in that experiment's infrastructure. It does not mean laptop-sized models or absence of every hosted dependency.

## 10. Concrete issues to resolve before trusting a local baseline

These findings are from source inspection unless otherwise stated. They have not been reproduced in an installed Syftr runtime during this review.

### 10.1 Sampler settings are lost on the worker load path

`get_study()` creates a configured sampler. However, `optimization.tune_objective()` subsequently calls `optuna.load_study()` with only storage/name, then calls `ask()` on that loaded object.

In pinned Optuna 4.2.1, loading without a sampler creates default TPE. Therefore the configured `multivariate=True`, `constant_liar=True`, startup count and hierarchical sampler choice are not guaranteed to reach candidate generation on this path. This remains TPE by default in that version; it is not evidence that NSGA-II is already integrated. [Optuna 4.2.1 load behavior](https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.load_study.html).

**Required action:** carry the intended sampler into the candidate-generating process or centralize candidate generation, verify the effective class/settings at `ask()`, and define seeded state/resume behavior. Merely recreating a sampler with the identical seed for every trial can reset its random sequence; preserve/advance state deliberately. Add a meaningful regression test for sampler propagation.

### 10.2 The embedding flag in the earlier runbook is reversed

The path is:

```text
optimization.use_hf_embedding_models
 -> build_rag_retriever(...)
 -> get_embedding_model(use_hf_endpoint_models=...)
```

If true and the model exists in the configured endpoint map, `get_hf_endpoint_embed_model()` selects a hosted endpoint. BGE-small is in that map.

For local Torch CPU embeddings, use:

```yaml
optimization:
  use_hf_embedding_models: false
  embedding_device: cpu
```

This corrects Section 6 of the earlier PC runbook. Do not copy its `true` setting into the local study.

### 10.3 New local model names can still break cost calculation

`LLMCallData.cost` raises `ValueError` for an unknown model name. `calculate_metrics()` calls cost extraction even when the selected objective is latency. Thus switching to `p80_time` does not automatically remove the need for a valid pricing/accounting entry.

The `local_models.generative` schema has no cost field. A supported alternative is registering the endpoint through `generative_models` with `provider: openai_like` and explicit cost metadata, or deliberately adapting accounting for unpriced local models.

For a smoke run, zero token prices may represent **no API charge** while preserving token/latency reporting. They must never be presented as zero compute cost or used as the meaningful minimized objective. Verify the endpoint's returned model name matches the pricing key. Keep local token and latency metrics even if API-charge accounting is zero.

### 10.4 Token accounting needs a local-model definition

`llm.get_tokenizer(name)` currently ignores `name` and returns the GPT-4o-mini tiktoken encoder. Sentence/token splitters and related token operations therefore do not establish true Qwen token counts. Recursive splitting instead uses an embedding-model tokenizer. Retrieval-only context length is approximated as `sum(len(text) // 4)`.

Use endpoint usage counts and/or the pinned generator tokenizer with the actual serialized chat template. Validate prompt and completion counts against a small known request. Distinguish chunking units from actual model-input tokens. Record any approximation explicitly; a character heuristic is unsuitable for the primary token objective.

### 10.5 Historical context-budget proposal is outside the confirmed scope

The slide formulation includes `Bc`, but the user has explicitly set `[k,c,Bc]` aside. Implementing a new context-budget decision variable is not a requirement for the current research. Retain Syftr's existing configuration semantics and apply common model context/output limits across arms. Validate tokenizer units and response-synthesizer behavior because long-context splitting/refinement can affect calls and resource measurements.

### 10.6 Small trial counts can become random-search-only

`OptimizationConfig.num_random_trials` defaults to 100. With a correctly propagated configured sampler, a 20-trial study with no qualifying prior completed seeds may never reach guided TPE. On the current worker reload path the default TPE startup behavior can differ, which reinforces the need to log effective settings.

Choose startup count relative to the real study budget and report when guided proposals begin. A two-trial smoke test is an integration check, not an optimization result.

### 10.7 Seeding and trial budgets are separate

`StudyRunner.run()` seeds the study before/alongside its main trial loop. `num_trials` bounds that loop, not necessarily the seeder evaluations. Timeout settings can allow seeding to continue in parallel. Baseline variations can add many evaluations.

For fair comparison, enumerate initialization candidates in advance and charge their evaluations to the same total budget. Disable unneeded automatic seed families or explicitly include their cost. Separate seed-selection randomness from optimizer randomness.

### 10.8 Partial evaluations may be marked COMPLETE

Pareto pruning can break out of evaluation, return metrics with `is_pruned`, and reach `study.tell()` as a completed objective pair. An Optuna COMPLETE state alone is therefore insufficient evidence of a full-question evaluation. `total_qa_pairs` can describe intended dataset size rather than actual successful calls.

Use explicit per-example attempt counts, evaluated IDs, error status, and full/partial markers. Never mix partial estimates and fully evaluated candidates as final frontier points without a declared policy and full reevaluation.

### 10.9 Answer scoring and error accounting need validation

- The class named `ExactMatchEvaluator` matches reference evidence against retrieved contexts; it is not normalized answer exact match.
- `core.f1_score()` uses a set intersection for common tokens instead of multiset overlap. It is not benchmark-standard repeated-token F1. It also lacks the usual explicit yes/no/noanswer disagreement handling used in HotPotQA scoring. Do not label it official HotPotQA F1 without aligning it to the benchmark.
- `calculate_metrics()` computes judged accuracy over results with boolean passing labels. Failed examples can be excluded, biasing results if failures differ across configurations.
- Its `num_total` is based on passing labels, and `num_success` subtracts error counts again; do not use these names blindly as experiment denominators.
- `_aeval_pair()` reads `response.text` when building a result even after generation failure can return a false-like placeholder. Validate this failure path before long runs.
- The default JSON judge prompt example contains invalid JSON punctuation. Small local judges may imitate it or produce unparseable outputs. Validate parsing, score range, and threshold; a prompt correction changes the evaluator and must apply to all arms.

Keep raw responses, failures and parsed scores. For the final project, use a tested scoring implementation and a fixed error policy. A legitimate model abstention can receive the task score; infrastructure failures should remain distinguishable and should consume recorded resources.

### 10.10 Configuration and reproducibility gaps

- `sampler` currently accepts only `tpe` and `hierarchical`; adding `sampler: nsga2` alone will fail schema validation.
- `objective_1_name` accepts accuracy/retriever recall; `objective_2_name` accepts P80 time, mean LLM cost, and retrieval context length. End-to-end token mean and answer F1 need deliberate integration.
- Older YAML fields such as `response_synthesizer_llms` may conflict with current strict schemas using `response_synthesizer_llm_config`. Parse the intended new study against the current classes.
- The configured TPE factory currently does not pass an explicit seed. Reproducibility requires control of sampler, Python/NumPy, data selection, model/runtime randomness, and scheduling.
- Repeated `suggest_*` calls with the same name inside a single Optuna trial generally reuse that trial's value. The current duplicate-retry loop should not be assumed to generate fresh candidates reliably. Central duplicate handling is preferable.
- Dense-index cache keys include selected splitter/embedding/dataset fields but not explicit dataset/model revisions or all tokenizer/context changes. Version the cache identity when these change.
- `syftr check` checks configured services/models; an indiscriminate check with historical models can attempt unwanted hosted connections. Prepare a local-only config first.
- A retrieval-only mode can still instantiate correctness evaluators and retain HyDE/query decomposition model calls. It is not automatically API-free.
- The lock pins Transformers 4.48.3. Qwen3's official card warns versions below 4.51.0 do not support its architecture for direct Transformers loading. A separate Ollama serving process avoids relying on Syftr's pinned Transformers to load Qwen3 generator weights, but tokenizer compatibility still needs testing. [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-4B).

## 11. Proposal analysis: what RM.pptx commits us to

The deck has 26 physical slide parts. Slide 3 contains no extracted text or picture objects. Some display numbers differ from physical slide positions. These are presentation organization issues, not experimental evidence.

| Physical slide(s) | Main content | Consequence |
|---|---|---|
| 1-2 | Quality/token trade-off and RAG motivation | Resource efficiency is central |
| 4 | `x = [k, c, Bc]`, maximize Q, minimize T | Historical formulation; explicitly set aside by the user |
| 5-6, 26 | Literature: RAG, LLMLingua, RAGGED, Syftr | Background supports efficiency/configuration optimization |
| 7-10 | Syftr full flow space, MO-TPE, pruning, seeding, datasets | Reference methodology |
| 11-12 | Change optimizer while holding problem/evaluation fixed | Controlled comparison |
| 13-14 | NSGA-II mechanism | First proposed implementation |
| 15-16 | Hybrid NSGA-II + BO, mechanism illustrative | Exact hybrid is still a design decision |
| 17 | Adaptive NSGA-II | Define signal, update rule and bounds |
| 18-20 | Four methods and shared evaluator/budget | Same candidate-to-objective interface |
| 21-23 | HV, IGD, diversity, runtime, convergence, ablations | Need optimizer-level metrics in addition to answer scores |
| 24-25 | Deliverables and final research framework | Implementations plus reproducible empirical analysis |

### 11.1 Resolve these wording/experimental mismatches

**Confirmed Syftr scope:** the user explicitly wants optimizer research on Syftr's hierarchical search space and has set `[k,c,Bc]` aside. Preserve Syftr's workflow/module structure. Substitute feasible local model catalogs and use the exact same resulting executable space for all optimizers. A single simple RAG flow is useful for infrastructure smoke testing only; it does not define the final research space.

**Fixed retriever versus searching retriever type:** hold the allowed space fixed across arms. If retriever type becomes a variable later, it varies identically within every arm. “Same RAG pipeline” should mean shared implementation and allowed choices, not necessarily identical realized candidates.

**Tokens versus cost versus latency:** these are distinct objectives. A latency smoke run is useful, but is not the token-focused final experiment. The baseline must be rerun under the final quality/token protocol before comparison.

**Adaptive versus RL:** the confirmed third method is adaptive NSGA-II. Do not introduce RL, reward-learning infrastructure, or claim a learned policy without actually implementing and evaluating one.

**Proposed versus completed:** no project-specific implementations or valid local baseline results were found. Present all three methods as planned until code and logs establish otherwise.

### 11.2 Supporting literature

LLMLingua performs coarse-to-fine prompt compression with budget control, token-level iterative compression and distribution alignment. It motivates token efficiency, but a context budget/truncation rule is not automatically an LLMLingua implementation.

The supplied RAGGED PDF is an anonymous review version titled *RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems*. It examines reader/retriever/context interactions and noise sensitivity, supporting controlled model-specific configuration experiments. The proposal's attributed title differs from the supplied PDF; verify the final published citation before submission.

Adding compression to only one optimizer arm would change the RAG problem. Any later compression option must be shared by all arms or evaluated as a separate intervention.

## 12. Local feasibility and model selection plan

**Execution update:** a fuller Ubuntu hardware audit and Kaggle assessment is now available in [Hardware and Kaggle execution plan](HARDWARE_AND_KAGGLE_EXECUTION_PLAN.md). Develop locally and validate Kaggle GPU execution for the expensive experiments; the final model catalog need not be limited by laptop memory. Root disk is currently near capacity.

### 12.1 Current observed environment

| Item | Observation during this review |
|---|---|
| CPU | Intel Core Ultra 7 155H |
| System memory | `/proc/meminfo` reports about 15.3 GiB total; available memory fluctuates |
| Active Python | 3.12.3, inside the project's supported 3.11.1-3.12.x range |
| Syftr packages in active interpreter | Torch, Optuna, Ray, Datasets and LlamaIndex not found |
| Executables on current PATH | `ollama`, `nvidia-smi`, `uv` not found |
| PDF tooling | Available and used for this review |

No NVIDIA capability was established. Absence of `nvidia-smi` is not a complete GPU inventory. Historical notes describe Intel shared graphics and a Windows Ollama installation; those do not prove availability in this Linux shell. No search of every system-wide virtual environment or model cache was performed.

### 12.2 Roles that must run locally

1. **Generator:** produces the RAG answer.
2. **Embeddings:** encode grounding chunks and queries if dense retrieval is selected.
3. **Evaluator:** deterministic answer scoring or a separately declared local judge.
4. **Optional auxiliary LLMs:** HyDE, decomposition, reranking, agents; initially disabled.

Changing only the generator is insufficient if the judge or embeddings still use hosted services.

### 12.3 Candidate models, without premature selection

| Candidate | Suggested role | Selection condition |
|---|---|---|
| Qwen3 1.7B, quantized | First integration pilot | Fits comfortably and produces usable final answers |
| Qwen3 4B, quantized | Stronger candidate for final comparison | Better answer quality with acceptable memory and runtime |
| Larger models | Later optional extension | Only after measured capacity/acceleration justifies them |

These are continuity candidates from the project's prior plan, not a claim that they are the best models available in November 2026. Avoid selecting a model solely because its weight file fits RAM: runtime also needs KV cache, embeddings, indexes, Python/Ray workers and the OS.

Qwen3 exposes thinking/non-thinking behavior; the official card documents explicit control and sampling guidance. Pin the mode and decoding settings and check actual endpoint behavior. Do not blindly copy the old temperature-zero recipe into a thinking-enabled run. [Qwen3 1.7B card](https://huggingface.co/Qwen/Qwen3-1.7B).

Ollama's local OpenAI-compatible API is a plausible serving boundary supported by the existing `openai_like` factory. Its documented local base URL is `http://localhost:11434/v1/`, with an ignored placeholder key. This is local server usage, distinct from Ollama's cloud service. [Official compatibility guide](https://docs.ollama.com/api/openai-compatibility).

### 12.4 Pilot selection criteria

For each candidate on the same small QA manifest, measure:

- Peak memory and whether swapping occurs.
- Answer completion rate, parseability and truncation.
- Correctness/EM/F1 on saved final answers, with manual spot checks.
- Generation time and input/output token counts.
- Context capacity actually configured at the server.
- Index construction/retrieval memory alongside inference.
- Repeatability and stability over more than one request.

Choose the smallest candidate that gives a meaningful, non-degenerate quality/token trade-off at a feasible throughput. If nearly every answer is wrong or every configuration has identical token use, optimizer comparison is uninformative. Pilot observations can determine design choices, but then freeze them and start fresh research runs.

Use one trial and one evaluation request at a time initially. Native model serving is a separate process; Ray resource declarations alone do not prevent that server from using all available threads/memory.

## 13. Recommended common experiment specification

Everything in this section is a proposed initial specification, to be finalized after the hardware pilot.

### 13.1 Shared Syftr hierarchical search space

The candidate is a complete valid Syftr workflow configuration, not a fixed three-gene vector. Use the existing search-space definitions and conditional rules as the common specification.

| Search component | Treatment in the research |
|---|---|
| Workflow family | Preserve Syftr's agentic/non-agentic choices where executable locally |
| Generator and auxiliary model choices | Substitute a declared feasible local model catalog, identical across arms |
| Retriever | Preserve dense, sparse and fusion choices and dependent settings |
| Embedding choices | Use a feasible local catalog with common versions and settings |
| Splitter | Preserve supported splitter choices and chunk/overlap parameters |
| Prompts and few-shot | Preserve supported choices and example-retrieval dependencies |
| HyDE, reranking, additional context | Preserve conditional enablement and associated parameters where supported |
| Agent controls | Preserve family-specific iterations, reflection, rollout and expansion parameters |
| Validity, defaults and constraints | One shared decoder/validator for all algorithms |

Local adaptation changes which model endpoints are available; it should preserve the underlying workflow-search problem as far as hardware and model capability allow. A single generator is acceptable if necessary, but auxiliary calls and agent tool behavior must still be validated. Do not claim the full paper's exact space if model catalogs or supported branches differ. Record an explicit capability matrix and a versioned final search-space manifest.

Use a simple fixed flow to verify inference, then test representative candidates from each intended branch before a full study. A failed agent/tool branch is a compatibility issue to investigate, not a reason to silently remove agentic choices from the research. If a branch cannot run, document the limitation and apply the same exclusion to every algorithm.

Evolutionary representation must support categorical, integer, continuous and inactive parameters. Canonicalize effective configurations, repair conditional dependencies, and ensure all optimizers have access to the same valid candidates. The research tests how algorithms navigate this structure under limited expensive evaluations.

### 13.2 Final objectives

For the token-focused proposal, the preferred main study is:

```text
maximize Q(x) = mean benchmark-compatible answer token F1 on fixed search questions
minimize T(x) = mean(input_tokens + output_tokens) per answer-generation query
```

Also report normalized answer exact match, latency, failure rate and actual retrieved/context tokens. F1 is a recommendation because it avoids repeatedly paying in inference time for a weak self-judge; it requires implementation/validation and is different from the paper's judged accuracy. Confirm it after pilot inspection. A fixed calibrated local judge is an alternative primary quality metric, with deterministic metrics as secondary checks.

If a local judge is used, separate its tokens and runtime from **deployment workflow** tokens. Judge overhead belongs to search cost. If auxiliary generator calls are later enabled, count every deployment call, including rewrites/reranking/agent turns. When the same model serves both generation and judging, distinguish them by call role/invocation, not model name alone.

For a minimal initial Syftr-compatible smoke run, `accuracy` plus `p80_time` is already supported. Label it an integration baseline. It does not replace the final TPE result using Q/T.

### 13.3 Budget definitions

Keep two concepts separate:

- **Candidate deployment cost T(x):** how many tokens the workflow uses per question.
- **Search budget:** how many expensive candidate/question evaluations the optimizer is allowed to purchase.

For a simple first comparison, disable Pareto/cost/runtime extrapolation pruning for all arms, retain identical hard safety timeouts, and give each arm `B` candidate attempts on the same `N` questions. Count seed evaluations within `B`. Record partial failures and actual calls rather than assuming the product `B*N` always completed.

If pruning is later enabled, candidate count alone is unfair. Report candidate attempts, fully evaluated candidates, actual QA attempts, generator/judge calls, tokens, elapsed time and partial work. Compare at a common expensive-work budget. Pruning policy must be identical across arms, and final frontier candidates must receive complete evaluation.

## 14. Baseline reproduction and implementation sequence

### Stage A: environment and dependency validation

Important declared constraints in `pyproject.toml` are:

| Component | Declared constraint/pin |
|---|---|
| Python | `>=3.11.1,<3.13`; README setup example uses 3.12.7 |
| Optuna | 4.2.1 |
| Ray | 2.46.0 |
| Torch | 2.7.0 |
| Transformers | 4.48.3 |
| LlamaIndex core | `>=0.12.52`, with concrete dependency resolution in `uv.lock` |
| Hugging Face Datasets | `>=3.3`, with concrete resolution in the lock |

The dependency graph includes many cloud-provider and agent integrations even if the experiment uses only local RAG. Installation, import compatibility and actual endpoint use are separate concerns. SQLite and single-machine Ray are supported defaults; PostgreSQL, an AWS bucket and a multi-GPU cluster are not required for the first local study.

1. Capture OS, CPU, actual GPU devices, RAM, disk and Python environment.
2. Prepare an isolated supported Python environment, preferring the lock before changing versions.
3. Check package resolution/imports and local Ray/SQLite behavior.
4. Establish local inference and embedding endpoints/models.
5. Warm/pre-cache required tokenizer and NLTK resources; worker preparation currently calls downloads.
6. Build a local-only config rather than carrying provider placeholders and large model lists forward.

Do not install every development extra unless needed. Keep local secrets, model files, caches and databases outside tracked source changes. The mounted filesystem and path spaces are not proven blockers, but environment/cache locations should be writable and performance should be measured.

### Stage B: one measured pipeline

1. Load the prepared sample QA and grounding splits and check IDs/schema.
2. Build a local index and inspect retrieved evidence for a few questions.
3. Run one plain RAG configuration with the selected candidate model.
4. Check actual request count, final answer, prompt packing, token counts and timings.
5. Validate scoring and the error path deliberately.
6. Save the effective config, model digest, data revision and raw per-question record.

### Stage C: bounded Syftr/TPE smoke reference

Use 2-3 candidates and about 5 sample questions to test orchestration. Disable automatic seed expansion and hosted branches. Fix sampler propagation and unknown-model accounting first. Use local embeddings with the corrected false flag. Keep a unique study name and one concurrent trial.

**Acceptance:** every intended candidate is traceable, generation/evaluation complete, effective sampler settings are logged, objectives are finite, requests are local, and output artifacts can be reopened. Passing this stage establishes wiring only.

### Stage D: freeze and run the real TPE baseline

After choosing the final model, Q/T definitions, search space, seeds and question manifest, run TPE from a fresh study. Set startup trials sensibly relative to budget and verify that guided search actually occurs. Retain the same implementation and common measurement corrections for every later optimizer.

Publishable baseline evidence consists of exact configurations, candidate history, raw answers, objective values, consumed work, runtime and failures. A printed “completed” status alone is insufficient.

### Stage E: implement and compare the three methods

Implement NSGA-II first, establish its validity on the common interface, then add one precisely specified hybrid and one precisely specified adaptation. Run the frozen baseline and all methods under the same protocol and repeat with matched seeds. Any later evaluator/model change invalidates direct comparison with the earlier baseline unless rerun.

## 15. Proposed algorithms: concrete starting designs

The following designs make the proposal implementable. They are recommendations for the next phase and have not been coded or shown to outperform TPE.

### 15.1 NSGA-II from scratch

Required pieces:

1. Candidate representation and validity/repair rules.
2. Initial population of size `P` drawn under the common initialization policy.
3. Shared objective evaluation.
4. Fast non-dominated sorting.
5. Crowding distance within each front.
6. Binary tournament parent selection by rank, then crowding distance.
7. Crossover and mutation respecting integer/categorical genes.
8. Elitist selection from parents plus offspring.
9. Duplicate handling, budget accounting and termination.

```text
Evaluate P initial candidates, charging them to B
while budget remains:
    rank parents; compute crowding distances
    select parents and create valid unseen offspring
    evaluate only the offspring allowed by remaining budget
    union = parents + evaluated offspring
    select P survivors by rank then crowding distance
    log population, objectives, diversity and cumulative work
return non-dominated archive of all fully evaluated candidates
```

Use per-gene categorical/ordinal mutation for the discrete initial space rather than arbitrary floating point mutations rounded after the fact. For a larger numerical space, choose bounded operators and document repair effects. Crowding distance must handle zero objective range without division by zero. End a partial final generation under a declared budget rule.

Choose `P` small enough for several generations; a population equal to the entire budget mostly tests initialization. Optuna includes an NSGA-II implementation that can serve as a later verification/reference, but configuring that library alone would not satisfy a from-scratch implementation claim.

### 15.2 Hybrid NSGA-II + Bayesian proposals

A manageable first design is **periodic MOTPE candidate injection into NSGA-II**:

1. Run ordinary NSGA-II as the outer population loop.
2. Maintain one shared history of all fully evaluated configurations/objectives.
3. Every `h` generations, propose `q` new candidates using MOTPE conditioned on that history.
4. Replace `q` ordinary offspring slots with these candidates; do not add free evaluations.
5. Evaluate through the same evaluator and include them in ordinary elitist survival.

Record `h`, `q`, BO startup behavior, history synchronization, proposal origin and BO computation time. Under this design call the method **NSGA-II + MOTPE injection**, a specific Bayesian hybrid. If the team instead selects a GP-based surrogate/acquisition, specify encoding, uncertainty/acquisition and fitting procedure separately; do not use the names interchangeably.

A useful ablation replaces BO injections with random injections at the same frequency/count. This tests whether any benefit comes from Bayesian guidance rather than simply extra diversity.

### 15.3 Adaptive NSGA-II

Start by adapting **mutation probability only**. Keeping other operators fixed makes the effect interpretable.

One proposed rule:

```text
At the end of each generation:
    compute HV with fixed normalization/reference
    measure valid unique configuration diversity
    if HV gain has stayed below epsilon for w generations
       or diversity falls below a declared threshold:
        p_mut = min(p_max, alpha * p_mut)
    else:
        p_mut = max(p_min, p_mut / beta)
```

Set the initial probability, bounds, `alpha`, `beta`, `w`, epsilon, diversity definition and update schedule before final runs. Measure diversity on the conditional Syftr representation: account for active genes, categorical choices and normalized numerical distances. Log the signal and every probability update.

This is feedback-driven adaptation. It should not be described as reinforcement learning or a learned policy merely because it uses search history. Compare directly with fixed-mutation NSGA-II and include adaptation overhead in runtime. Tune its hyperparameters on development/pilot work with the same tuning allowance as other methods.

## 16. Common software architecture for the next phase

Preserve the upstream framework and keep research-specific additions under `ourwork/` or a dedicated research package. A proposed structure is:

```text
ourwork/
  configs/          # versioned local study/search/experiment specifications
  data_manifests/   # revision, split, QA IDs, hashes
  experiment/      # configuration decoding and common evaluator adapters
  optimizers/      # TPE adapter, NSGA-II, hybrid, adaptive
  metrics/         # benchmark scoring, Pareto set, HV, reference IGD
  runners/         # smoke, baseline, repeated comparison, final evaluation
  results/         # untracked run outputs and manifests
```

Shared interface:

```text
optimizer.ask(history, budget_state) -> candidate
decode_and_validate(candidate) -> canonical concrete flow parameters
evaluate(parameters, fixed_protocol) -> objectives + per-example measurements
optimizer.tell(candidate, measured_result)
```

All arms must use the same candidate decoding, context packing, model adapter, dataset manifest, scoring code, failure handling, and logging. Common corrections to Syftr should be small, explicit, versioned and documented. Do not duplicate the entire RAG pipeline separately inside each optimizer.

Store optimizer state if a study is resumable, not just the objective table. Resume must recover population/generation/BO/adaptation and random state, or be explicitly documented as a new run using historical observations.

## 17. Metrics, statistical comparison and result artifacts

### 17.1 Two levels of evaluation

| Level | Required measurements |
|---|---|
| RAG configuration | Quality metric, EM, actual input/output tokens, context tokens, latency, failures |
| Optimizer | Pareto quality, HV progression, candidate/call budget, time to threshold, overhead, run-to-run variation |

Higher answer quality alone does not establish a better multi-objective optimizer if it was obtained at much greater token use or search cost.

### 17.2 Hypervolume

For minimization vectors `z = [1-Q, T]`, define fixed scaling, for example `z2 = T / S` where `S` is a predeclared token scale/cap. Choose a common worse reference point `r` and use it for all algorithms, seeds and checkpoints:

```text
HV(F; r) = volume of the union of rectangles [z1, r1] x [z2, r2]
           over non-dominated points z in F
```

Larger is better. Specify what happens if measured points exceed the reference bounds; avoid changing normalization/reference based on each arm's observed best/worst values. Plot HV versus both expensive evaluations and elapsed time. Define the budget origin to include common initial evaluations.

### 17.3 IGD and reference fronts

```text
IGD(F, R) = mean over reference points r in R of min distance(r, f), f in F
```

Smaller is better. Use normalized minimization objectives. The true full Syftr Pareto front is unknown. An exhaustively evaluated tiny synthetic space can validate the metric implementation, but is not a reference for the full research space. For the main comparison, use a declared pooled empirical reference from all methods/seeds and label it as such; it is not ground truth. Reference-set density and data leakage must be considered. Do not tune methods against final holdout-derived reference points.

### 17.4 Repetitions and uncertainty

Use matched independent seeds across algorithms, with a shared initialization list per seed where possible. Three runs can support a pilot; five or more are preferable if measured runtime permits. These counts are feasibility targets, not evidence of adequate power for every claim.

Report per-run values, mean and standard deviation, and paired differences/intervals where appropriate. Separate optimizer-seed variation from finite-question uncertainty. Bootstrap paired question IDs if estimating quality differences on the same evaluation set. Avoid treating each trial inside one search as an independent experimental replicate.

Timing needs controlled load and warm-up policy. Either prebuild common indexes and report this cost separately, or charge comparable construction costs consistently. A warm index in one arm and cold index in another can distort elapsed-time comparisons.

### 17.5 Minimum run record

```text
run identity, timestamp, code commit and working-tree diff
dataset revision, QA split/IDs, grounding corpus fingerprint
generator identifier/digest, quantization, server/tokenizer versions
embedding identifier/revision, index identity and build settings
prompt/chat template, thinking/decoding/output/context settings
optimizer name, seed, settings and initialization candidates
candidate ID, canonical params, proposal origin and generation
per-question raw answer, reference, score, failure/retry status
per-call role, input/output tokens and latency
candidate objective values, full/partial evaluation marker
cumulative attempted/completed candidates, calls, tokens and time
optimizer state, HV checkpoints, final Pareto configurations
```

`include_responses: true` preserves useful answers, but the current aggregate export is insufficient for all these per-query/per-call measurements. Extend logging deliberately; do not assume token means alone permit later forensic analysis.

### 17.6 Final deliverables

1. Reproducible environment/config/data manifests.
2. Working common evaluator with verified token and answer scoring.
3. Local TPE baseline and three implementations.
4. Full trial/per-example history for every run.
5. Pareto plots and HV versus budget/time.
6. Tables of final HV, quality at fixed token limits, tokens at fixed quality targets, runtime and failures.
7. Declared empirical-reference IGD if justified.
8. Hybrid/adaptive ablations.
9. Holdout confirmation of a predeclared selection policy, such as low-token, knee, and highest-quality search solutions.
10. An honest discussion of compute, single-model/single-dataset scope and unsuccessful outcomes.

Do not invent result tables now. Empty templates or formulas are acceptable; measured scores require stored inference evidence.

## 18. Verification and implementation priorities

### 18.1 Meaningful checks before long runs

- Verify the intended sampler actually produces worker-side proposals and retains seeded progression.
- Validate model-name accounting and local-only request destinations.
- Compare answer scoring against known benchmark examples, repeated tokens, yes/no disagreement and empty/failed output cases.
- Validate token usage on known prompts and chat templates; enforce context packing bounds.
- Test a deliberately failing generation request and a malformed judge response.
- Check non-dominated sorting/crowding/HV on tiny analytically known fronts, including ties and zero-range objectives.
- Check NSGA-II's population size, elitism, validity, duplicate and final-budget behavior.
- Check hybrid injection count replaces offspring rather than exceeding budget.
- Check adaptive updates occur at intended generations and stay within bounds.
- Confirm fresh versus resumed runs and complete versus partial evaluations cannot be confused.

These tests verify the research mechanism; they are more valuable than tests that merely restate configuration constants.

### 18.2 Prioritized work list

| Priority | Work | Why it comes first |
|---|---|---|
| P0 | Establish isolated runnable environment and measured model pilot | No inference feasibility has been proven in the active shell |
| P0 | Correct local embedding routing and unpriced-model handling | Avoid remote calls and metric crashes |
| P0 | Validate sampler propagation and state | Otherwise the named baseline can differ from the executed search |
| P0 | Freeze scoring/token/context definitions | These define the actual optimization problem |
| P1 | Control seeds, initialization and budgets | Required for a fair comparison |
| P1 | Validate failures and full/partial trial records | Avoid biased or falsely complete results |
| P1 | Run final-protocol TPE reference | Required baseline before interpreting new methods |
| P2 | Implement and verify ordinary NSGA-II | Foundation for both extensions |
| P2 | Implement one hybrid and one adaptive mechanism | Distinct, testable proposed approaches |
| P3 | Repeated experiments, reference-front metrics and ablations | Evidence for conclusions |
| P3 | Holdout confirmation and report/presentation revision | Generalization and defensible project claims |

## 19. Current status and next-session handoff

| Workstream | Evidence-based current state |
|---|---|
| Base paper | Reviewed, including appendix protocols, results and limitations |
| Proposal | Four-arm comparison understood; scope/objective ambiguities identified |
| Upstream code | Present, structurally parsed; critical runtime paths audited |
| Local model | Candidate family identified; final choice pending measured pilot |
| Full dataset | Loader present; no full dataset downloaded/validated in this review |
| Dependency environment | Active interpreter is supported, required Syftr packages absent |
| Local baseline results | No verified project baseline found |
| NSGA-II | Proposed, not project-implemented |
| Hybrid | Proposed, mechanism must be frozen before final implementation |
| Adaptive | Proposed, adaptation rule must be frozen before final implementation |
| Comparative results | Not available |

**Next concrete work:** verify local serving capacity and run a small Qwen3 1.7B pilot, then evaluate whether 4B is feasible. In parallel with that implementation phase, prepare a minimal local-only Syftr configuration, correct the shared routing/accounting/sampler issues, and verify one end-to-end RAG evaluation. Only then choose the final model and commit to experiment budgets.

The final research claim should take this form:

> We investigate whether NSGA-II and explicitly defined hybrid/adaptive evolutionary extensions can improve Pareto-front discovery and search efficiency relative to Syftr's TPE-based search, using the same hierarchical Syftr workflow space, feasible local models, prepared benchmark data and controlled evaluation budgets.

Whether the proposed methods win, tie, or lose is an experimental result. A careful negative result remains a valid research outcome.

## 20. Sources and traceability

### Local sources

- [Syftr paper, supplied v1](../PPTnall/syftr-BasePaper.pdf): Sections 1-7, Figures 1-5, Appendices A1-A12, Tables A1-A9, Figures A1-A13.
- [Research proposal](../PPTnall/RM.pptx): all 26 physical slide parts.
- [Proposal research notes](../PPTnall/README_syftr_optimization_research.md) and [presentation structure](../PPTnall/Final_PPT_Structure_syftr_Research.md).
- [Supporting LLMLingua paper](../PPTnall/LLMLingua.pdf) and [supporting RAGGED paper](../PPTnall/Ragged.pdf).
- [Prior readiness audit](code_and_project_readiness_audit.md), [implementation guide](project_implementation_guide.md), [historical PC guide](pc_local_hardware_and_baseline_pipeline.md), [progress summary](guide_progress_and_meeting_summary.md), and [progress/meeting document](Research_Project_Progress_Report_and_Meeting_Timeline.docx).
- [Project README](../README.md), [dependencies](../pyproject.toml), [lock](../uv.lock), [sample configuration](../config.yaml.sample).
- [Dataset documentation](../docs/datasets.md), [provider documentation](../docs/llm-providers.md), [flow extension guide](../docs/adding-new-flows.md).
- [Search/configuration](../syftr/studies.py), [runner](../syftr/optimization.py), [sampler factory](../syftr/optuna_helper.py), [QA tuner](../syftr/tuner/qa_tuner.py).
- [Evaluation](../syftr/evaluation/evaluation.py), [judge factory](../syftr/evaluation/evaluator_factory.py), [pruning](../syftr/pruning.py), [confidence functions](../syftr/custom_metrics.py), [answer F1](../syftr/core.py).
- [Model factories/tokenizer](../syftr/llm.py), [model schemas](../syftr/configuration.py), [embedding routing](../syftr/huggingface_helper.py), [token/cost tracking](../syftr/instrumentation/tokens.py).
- [Dataset loaders](../syftr/storage.py), [retriever builder](../syftr/retrievers/build.py), [cache identity](../syftr/retrievers/storage.py), [splitter builder](../syftr/tuner/core.py).

### External primary sources checked during this review

- [Prepared DataRobot HotPotQA dataset](https://huggingface.co/datasets/DataRobot-Research/hotpotqa): availability and prepared dataset identity; no runtime data loading performed.
- [Qwen3 1.7B official card](https://huggingface.co/Qwen/Qwen3-1.7B) and [Qwen3 4B official card](https://huggingface.co/Qwen/Qwen3-4B): candidate model behavior, thinking mode and compatibility.
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility): local endpoint interface.
- [Optuna 4.2.1 TPESampler](https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html): pinned-version TPE behavior and controls.
- [Optuna 4.2.1 load_study](https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.load_study.html) and [Study source](https://optuna.readthedocs.io/en/v4.2.1/_modules/optuna/study/study.html): default sampler on study reload.

### Validation performed for this document

File inventory, Python AST parsing, YAML inspection, notebook output-state inspection, PPTX/DOCX text extraction, PDF extraction and selected page rendering, read-only fixture-database inspection, Git diff interpretation, and active-interpreter/tool availability checks were performed. The full Syftr test suite and model experiments were not run because the active environment lacks the required packages and this phase is analysis rather than implementation.
