# Syftr project: code-grounded methodology and run-readiness audit

**Prepared:** 2026-10-24  
**Code reviewed:** local `RM-project` checkout (`main`, tracking `origin/main`, working tree reported clean).  
**Purpose:** explain the base approach, identify what this checkout needs to run, recommend a dataset, and flag choices to settle before coding new optimizers.

## 1. Scope and how to read the supplied material

The supplied paper, presentation, notes, and code are evidence about the project and its current proposal. Their proposed methods are not treated as instructions that override the user's current direction. The confirmed comparison is the original method plus NSGA-II, hybrid NSGA-II + Bayesian optimization, and adaptive NSGA-II.

The third proposed strategy is **adaptive NSGA-II**. It is distinct from reinforcement learning: it changes evolutionary behavior in response to search feedback, with a documented adaptation signal and rule.

## 2. What the Syftr paper does

Syftr treats a complete RAG workflow as one candidate in a large, **hierarchical and conditional** configuration space. First, the flow family is selected (the paper describes four agentic families plus a non-agentic RAG flow); then valid component choices and their dependent parameters are selected. Choices include the generation LLM, retrieval approach, embedding model, splitter and chunk settings, prompts/few-shot behavior, HyDE, reranking, and agentic modules. Because downstream parameters depend on earlier choices, the search space is not a simple flat vector of independent numeric variables.

The paper's optimization loop is:

1. Choose a candidate flow from the conditional search space.
2. Build and run the flow on question-answer examples grounded in a document corpus.
3. Record task quality and operational objectives such as cost or latency.
4. Update the multi-objective search and sample another flow.
5. Return the non-dominated configurations, which form a Pareto frontier.

The paper uses **multi-objective TPE**. TPE models the relative likelihood of parameter settings among good and poor observations. In the multi-objective setting, the search estimates Pareto trade-offs and uses an expected-hypervolume-improvement strategy. Syftr also adds a **Pareto Pruner**: during a flow's evaluation over examples, it estimates uncertainty around quality and cost; if a candidate's optimistic confidence bound is already dominated by the current frontier, it stops evaluating that candidate early. Seeding adds standard or random candidates, and transfer seeding can reuse useful configurations from prior studies.

This is a method for optimizing complete workflows, not only the three-variable vector shown on slide 4. The paper's main claimed results are tied to its benchmarks, model endpoints, judge setup, costs, and study budgets; those numbers should not be presented as expected results for our local experiment.

## 3. What the copied code actually implements

### Search and optimizer path

- `syftr/studies.py` defines the conditional search-space and optimization configuration, including objectives, trial counts, seeding, evaluation samples, concurrency, and pruning flags.
- `syftr/optimization.py` creates a two-objective Optuna study (`maximize` objective 1, `minimize` objective 2), then schedules evaluations with Ray. Each trial is asked from the Optuna study, evaluated through Syftr's objective/flow path, and told back to the study.
- `syftr/optuna_helper.py::get_sampler()` accepts `tpe` or `hierarchical`; it returns Optuna's `TPESampler` or Syftr's `HierarchicalTPESampler`. The package pins Optuna 4.2.1. **There is no NSGA-II or adaptive NSGA-II sampler integrated in this checkout.** The standard Optuna sampler plus multi-objective study is the likely reproduction baseline; hierarchical TPE is an additional Syftr option that should be separately identified in the baseline plan.
- `syftr/pruning.py` contains cost, runtime, and Pareto pruning. A strict comparison must decide whether to keep pruning active for all optimizers; otherwise one optimizer may receive fewer full QA evaluations per candidate.
- Syftr's setup also includes baselines and seeding. These consume evaluations or provide prior observations, so all methods need the same initialization policy and evaluation accounting.

### Existing dataset and study support

- `syftr/storage.py` includes loaders for HotPotQA, CRAG Task 3, FinanceBench, DRDocs, InfiniteBench, and other datasets. The prepared Hugging Face datasets include separate grounding and QA partitions.
- `studies/gpt-oss1--cost--hotpotqa_hf-train_hard--train_hard.yaml` provides a HotPotQA `train_hard` study configuration with `accuracy` and `llm_cost_mean`, `sampler: tpe`, and 50 evaluation samples in the copied file. It still names `gpt-4o-mini` as the LLM judge, and its space includes many model names; it is therefore not a no-credential, no-paid-API run as-is.
- `studies/retriever-only-hotpotqa.yaml` is a smaller and more affordable starting point: it optimizes `retriever_recall` and `retriever_context_length`, without full answer generation. It is a retrieval subproblem, however, and cannot by itself stand in for end-to-end RAG results.
- The study configuration maps `sample`, `train`, `test`, and `holdout` partitions. Keep tuning on `test`; preserve `holdout` for final evaluation and do not use it to choose configurations.

## 4. Paid services and local alternatives

The sample config contains credential examples for Azure OpenAI, Google Vertex, Cerebras, Together, Hugging Face endpoints, and other services. The provider documentation also supports `openai_like` endpoints (including self-hosted vLLM/TGI-compatible services), and the dependencies include LlamaIndex integrations for Hugging Face, Ollama, and vLLM. Hugging Face embeddings can run locally when the appropriate setting/device is enabled. **Therefore, the framework is not inherently restricted to paid OpenAI APIs**, but the selected study's models and judge must all be replaced or verified; merely changing the main generation model is not enough.

Potential no-vendor-API setup:

- Run a small generation model locally through a supported Ollama or self-hosted OpenAI-compatible vLLM endpoint, subject to available RAM/VRAM and throughput.
- Use local sentence-transformer / Hugging Face embeddings and local vector search.
- Replace LLM-as-a-judge with deterministic answer metrics where appropriate (e.g., normalized exact match or token F1), or run an open local judge and label that as a separate model component. Changing the judge changes the measured objective and must be held constant for all optimizer arms.
- Define the resource objective before experiments. API dollar cost is not meaningful for local inference unless we define an explicit accounting rate. A reproducible alternative is measured latency or compute time; context-token count is another distinct metric. Do not call these interchangeable.

Possible remaining external requirements: initial dependency/model/dataset downloads need internet; gated Hugging Face models may need account authorization; cloud endpoint studies need provider credentials; multi-machine Ray/PostgreSQL is optional for small local runs. The config says Optuna uses local SQLite by default, so PostgreSQL is not required for a single-machine smoke run.

## 5. Software and machine requirements to verify

The project's `pyproject.toml` supports Python `>=3.11.1,<3.13`; its dependency lock pins large packages including Ray 2.46.0, Optuna 4.2.1, Transformers 4.48.3, and Torch 2.7.0, as well as LlamaIndex and embedding/vector-store integrations. The README prescribes Python 3.12.7 with `uv sync --extra dev` followed by an editable install. It uses Ray for trials and SQLite by default for local Optuna storage.

Before selecting a local generation model, record the available GPU/VRAM, system RAM, disk space, OS, and whether CUDA is available. No source or dataset download was performed during this audit and no experiment/test was run. The exact environment has not yet been proven installable on the target machine.

## 6. Dataset recommendation

**Primary recommendation: Syftr's prepared `DataRobot-Research/hotpotqa`, `train_hard` subset.** It is already supported by the local loader and has a study YAML in this checkout. The documented partition sizes are sample 20, train 7,305, test 500, holdout 7,836. It tests multi-hop retrieval and reasoning, matches Syftr's intended QA-plus-grounding format, and avoids writing a new loader as the first task. Start with its 20-question sample for integration; then select a fixed modest subset of the test partition for optimizer development, and evaluate the chosen final configurations on the untouched holdout only once the protocol is frozen. Check the dataset card and licensing/terms before downloading.

This is the recommendation for the main optimizer comparison, not a claim that HotPotQA is universally best. CRAG Task 3 is closer to one of Syftr's paper benchmarks but has a more elaborate topic-based setup and the paper explicitly says its all-pages grounding setup differs from the official benchmark. FinanceBench is a useful compact domain test but adds PDF-grounding complexity. Do not mix datasets or use a dataset change to tune one optimizer differently.

## 7. Critical alignment decisions before implementation

### A. Define the actual search space

The slides say both “same full Syftr search space” and `x = [k, c, Bc]` (retrieval depth, chunk size, context budget). Those are different experiment scopes. A three-variable controlled subspace is much more realistic for a first implementation but is not full Syftr search. My recommendation is to begin with an explicitly reduced, reproducible Syftr-derived subspace and state which components are frozen; after the optimizer adapters work, expand only if compute permits.

### B. Resolve objective mismatch

The slide formulation uses task quality and token usage. Current end-to-end Syftr studies commonly use judged accuracy and model cost (`llm_cost_mean`), or latency; the retriever-only study uses recall and retrieved-context length. Choose one exact objective pair. If the goal is context-token budget, define which tokens count and at what point in the pipeline. If the goal is end-to-end quality versus latency, configure and measure latency consistently. Do not compare different objectives across optimizer arms.

### C. Define adaptive NSGA-II precisely

Choose which evolutionary control(s) adapt (for example, mutation rate), the feedback signal (such as population diversity or stalled hypervolume), update frequency, bounds, and adaptation rule. Log these values each generation. Compare against ordinary NSGA-II under the same expensive RAG-evaluation budget.

### D. Make a fair optimizer comparison

For each optimizer, hold constant the search-space bounds/conditional rules, RAG flow/evaluator, model and judge, data partition, objectives, seeding, treatment of pruning, and number of full-equivalent QA evaluations. Count partial pruned trials by their actual example calls and report both candidate count and model/evaluator calls. Use several random seeds if time allows. Report hypervolume with a predeclared reference point, quality at a fixed resource limit, and resource use; treat IGD cautiously because the true Pareto front is unknown, and define how the reference set is constructed. Optimization wall time should be reported separately from model evaluation time.

## 8. Suggested staged plan

1. **Inventory and smoke setup:** check machine capabilities, install the locked environment, test data loading and one supported local embedding / local LLM call. Do not begin a long study.
2. **Run the retrieval-only HotPotQA study** to verify local data, Ray/Optuna, metric logging, and results without a generator API. Treat this as infrastructure validation, not the final end-to-end outcome.
3. **Fix the initial RAG space and objectives** (recommend a small set of discrete choices plus chunk parameters, with a local generation model and a consistent answer metric).
4. **Run a bounded MO-TPE reference** with recorded per-trial configs and full-equivalent evaluation calls.
5. **Implement NSGA-II candidate generation** behind the same Syftr evaluator. Test validity/repair for conditional parameters and account for duplicate/invalid candidates.
6. **Choose one exact NSGA-II + BO design** (e.g., BO ranks candidate offspring or periodically injects candidates), then keep its extra model training out of the RAG-evaluation budget or account for it separately.
7. **Prototype adaptive NSGA-II** only after the standard NSGA-II path is stable, and record the adaptation signal, rule, and control values per generation.
8. Compare all methods on the frozen test protocol and use holdout once for final reporting.

## 9. Documents and code reviewed

- `PPTnall/syftr-BasePaper.pdf` — base paper, 44 pages.
- `PPTnall/RM.pptx` — 26-slide project presentation.
- `PPTnall/README_syftr_optimization_research.md` — research framing and proposed algorithms.
- `PPTnall/Final_PPT_Structure_syftr_Research.md` — intended slide narrative.
- `PPTnall/LLMLingua.pdf` and `PPTnall/Ragged.pdf` — supporting context-compression and RAG-design literature.
- `RM-project/README.md`, `pyproject.toml`, `config.yaml.sample`, `docs/llm-providers.md`, `docs/datasets.md`.
- `RM-project/syftr/optimization.py`, `optuna_helper.py`, `studies.py`, `storage.py`, and `pruning.py`.
- Relevant studies: `gpt-oss1--cost--hotpotqa_hf-train_hard--train_hard.yaml` and `retriever-only-hotpotqa.yaml`.

The next material decision from the group should be available compute and whether the preferred scope is the practical reduced search space first or attempting to reproduce the full conditional space immediately.
