# Research Project Progress and Meeting Summary

**Project:** Multi-Objective Optimization of RAG Search Strategies Using Syftr  
**Prepared for:** Project Guide / Professor  
**Prepared on:** 27 October 2026  
**Project directory:** `E:\Ongoing Projects\Research_Methodology_Project`  
**Current source revision:** `7174a26` (`main`)  

> **Record scope:** This summary is based on the project repository, supplied research and presentation materials, and project-planning notes available as of the date above. Actual meeting dates, the guide's detailed comments, and meeting-by-meeting action items were not included in those records. The meeting log at the end is intentionally left for the team to complete from its calendar or notes.

## 1. Executive summary

The team is developing a controlled comparison of multi-objective search strategies for RAG pipelines. Syftr is the selected base implementation and research foundation. Its central idea is to search a large, conditional space of RAG workflow configurations and return configurations along a Pareto frontier, balancing answer quality against an operational objective such as latency or cost.

The project plan is to establish the existing Syftr/TPE approach as the reference baseline, then evaluate three proposed methods on the same Syftr-derived search space:

1. NSGA-II.
2. NSGA-II combined with Bayesian optimization.
3. Adaptive NSGA-II.

The third proposed method is **adaptive NSGA-II**, as confirmed in project planning; it is not reinforcement learning. The team has completed a code and methodology review and prepared implementation and local-run guides. The final model, experiment protocol, and implementation are still to be completed. No Syftr baseline experiment or optimizer comparison has yet been run, so there are no experimental results to report at this stage.

## 2. Project objective and research question

**Research question:** Under a fixed RAG workflow search space, data split, evaluator, and evaluation budget, how do NSGA-II, NSGA-II combined with Bayesian optimization, and adaptive NSGA-II compare with Syftr's existing multi-objective TPE search in finding useful quality–efficiency trade-offs?

The comparison needs to isolate the optimizer as the changed factor. The generator, answer judge or metric, embedding model, RAG configurations, data examples, objective definitions, initialization, pruning policy, and evaluation budget should be held constant across the four methods.

## 3. Understanding of the base paper and Syftr

Syftr treats a complete RAG workflow as a candidate in a large, **hierarchical and conditional** search space. A configuration can include the RAG mode, retriever, embedding, document splitter and chunk settings, prompt/template, reranker, and optional agentic components. Some choices determine which later choices are valid, so the problem is not simply a flat vector of independent parameters.

At a high level, Syftr repeatedly proposes a workflow, evaluates it on QA examples grounded in documents, records its objectives, updates the search, and returns non-dominated workflows. The paper's approach uses multi-objective TPE with Pareto-oriented acquisition and a Pareto Pruner; seeding and transfer seeding are also part of the framework. The reported paper results depend on the paper's models, datasets, evaluators, and compute setup. They cannot be assumed to be the expected results for this project's local experiment.

The supplied repository implements an Optuna/Ray evaluation path. In the checked-out code, the available sampler choices include `tpe` and `hierarchical`; NSGA-II and adaptive NSGA-II are not already implemented as ready-to-use Syftr samplers. The proposed methods will need to be implemented and connected to the same workflow evaluator.

## 4. Work completed to date

- Reviewed the supplied Syftr base paper, project slide deck, presentation-structure notes, research README, and supporting RAG papers.
- Inspected the Syftr repository structure and key code areas for study configuration, search-space construction, sampling, Ray trial execution, evaluation, pruning, model configuration, and dataset loading.
- Prepared a code-grounded audit describing what the base method does, what the checked-out implementation supports, paid-service dependencies in the sample studies, experiment risks, and recommended next steps.
- Prepared a staged implementation guide and a PC-specific local execution guide.
- Confirmed the intended project comparison: Syftr/TPE baseline plus NSGA-II, NSGA-II + Bayesian optimization, and adaptive NSGA-II.
- Recorded the project assets in the selected directory: Syftr source code, the PPT and research reference materials under `PPTnall`, and project guides under `ourwork`.
- Checked the project Git revision as `7174a26` on `main`. The project materials are local, untracked documentation/reference folders; no Syftr source-code modifications for the new optimizers have been made yet.

## 5. Current decisions and open decisions

| Topic | Current direction | Status |
|---|---|---|
| Base implementation | Use the supplied Syftr repository and preserve its RAG workflow/evaluation path where practical. | Confirmed |
| Baseline optimizer | Run Syftr's existing multi-objective TPE approach as the reference. | Planned; not run |
| Proposed method 1 | NSGA-II. | Confirmed proposal; not implemented |
| Proposed method 2 | NSGA-II combined with Bayesian optimization. Exact hybrid mechanism still needs definition. | Confirmed proposal; design pending |
| Proposed method 3 | Adaptive NSGA-II, not RL. Define its feedback signal and adaptation rule before coding. | Confirmed proposal; design pending |
| Dataset | Syftr-compatible prepared HotPotQA, `train_hard` subset. Start with the small `sample` partition for integration; tune on `test`; reserve `holdout` for final confirmation. | Recommended; team confirmation and protocol pending |
| Local generator | A local open-weight model served through an OpenAI-compatible endpoint is planned. Several candidates have been surveyed, including Qwen and Gemma families. No final model/tag has been locked. | Pending |
| Embedding | `BAAI/bge-small-en-v1.5` is the proposed fixed local embedding candidate. | Provisional |
| Evaluation | A local judge or deterministic answer metric is needed to remove paid LLM calls. The evaluation method and prompt are not yet finalized. | Pending |
| Objectives | Accuracy plus latency (`p80_time`) is a practical end-to-end candidate. The presentation's token-usage objective does not directly match the current end-to-end Syftr objective list and needs an explicit definition if retained. | Pending guide/team decision |
| Search-space scope | Begin with a documented, reduced Syftr-derived RAG space if full conditional-space evaluation is not feasible; state clearly what is frozen or excluded. | Pending guide/team decision |
| Compute | Local PC setup and Colab were examined as options. A Colab T4 appeared selectable in the shared screenshot, but the full Ray-based Syftr run has not been validated there. | Pending pilot/decision |

## 6. Dataset and evaluation plan

The recommended dataset is the prepared `DataRobot-Research/HotPotQA` dataset with the `train_hard` subset, which is supported by the repository's loader and existing study configuration. The repository's documentation describes these partitions:

- `sample`: 20 examples, for setup and wiring checks.
- `train`: 7,305 examples.
- `test`: 500 examples, for development and optimizer comparison.
- `holdout`: 7,836 examples, to reserve for final confirmation after the protocol is fixed.

The loader downloads data from Hugging Face into its configured cache; manual dataset placement should not be needed for the supported loader. The dataset revision, question IDs, and evaluation settings should be recorded for reproducibility.

The existing end-to-end study uses an LLM correctness judge and a paid model, so it cannot be used unchanged for a no-paid-API experiment. The project must replace paid model/judge references and verify that no component such as a reranker, agent, or evaluator still calls a hosted endpoint. If an LLM judge is retained, use the exact same local judge and prompt for all optimizer arms. A deterministic answer metric such as normalized exact match or token F1 can be reported as an additional check.

## 7. Compute and software status

An earlier PC snapshot recorded an Intel Core Ultra 7 155H, approximately 15.7 GB RAM, Intel Arc shared-memory graphics, and Python 3.14 in the available shell. Syftr's repository declares support for Python 3.11.1 through 3.12.x and its README recommends Python 3.12.7. The local pipeline guide therefore recommends creating an isolated Python 3.12.7 environment and starting with one worker and a very small smoke run.

Ollama was found installed, but its server was not reachable from the inspection session. No model download, local inference, dependency installation, dataset download, or Syftr study execution was confirmed. A fresh check is needed before implementation begins.

Colab can be connected from VS Code through Google's Colab extension, but Colab's resources are temporary and vary by availability. Since Syftr uses Ray, the team's Colab configuration and applicable Colab usage restrictions need to be checked before using it for the full optimizer search. A short model or data-loading pilot is distinct from a full search experiment.

## 8. Work not yet completed

To avoid confusing planning with results, the following are explicitly still outstanding:

- Final generator model, exact model tag/quantization, and evaluator selection.
- Final objective pair and definition of token usage if it remains in the proposal.
- Frozen RAG search-space bounds and conditional parameter handling.
- Python environment and local/Colab execution setup.
- Data-loader and local-inference smoke run.
- End-to-end Syftr/TPE baseline and saved trial results.
- Implementation of NSGA-II, the selected NSGA-II + BO hybrid, and adaptive NSGA-II.
- Repeated-seed comparison, final holdout evaluation, statistical summaries, and final report.

## 9. Proposed next steps

1. Confirm with the guide whether the first study should use a reduced Syftr-derived space or attempt the full conditional space, and settle the primary objectives.
2. Select one local generator and one evaluation method after a short feasibility pilot. Freeze the exact settings for all optimizer arms.
3. Confirm HotPotQA `train_hard` and define fixed example IDs for smoke, development, and holdout use.
4. Configure the environment and run a small smoke study that verifies local model calls, data loading, RAG indexing, and metric recording.
5. Run and preserve a bounded Syftr/TPE baseline before modifying the optimizer path.
6. Implement NSGA-II and validate candidate encoding/repair for the conditional Syftr space.
7. Define the hybrid's specific Bayesian intervention, then define adaptive NSGA-II's feedback signal, rule, bounds, and update frequency.
8. Compare all four methods under equal model calls or full-equivalent QA evaluation budgets, record seeds and configurations, and report quality–efficiency Pareto results.
9. Use the reserved holdout only after the comparison protocol has been frozen.

## 10. Meeting record with the project guide

The materials available for preparing this report do not contain meeting dates, minutes, or a list of the guide's specific feedback. Please add only details from actual meetings; this section is a recording template, not a claim that these meetings occurred.

| Meeting | Date | Topics discussed | Guide's feedback / decisions | Actions agreed | Owner / target date | Status |
|---|---|---|---|---|---|---|
| 1 | [Add actual date] | [Add actual topics] | [Add actual feedback] | [Add agreed actions] | [Add owner/date] | [Not recorded yet] |
| 2 | [Add actual date] | [Add actual topics] | [Add actual feedback] | [Add agreed actions] | [Add owner/date] | [Not recorded yet] |
| 3 | [Add actual date] | [Add actual topics] | [Add actual feedback] | [Add agreed actions] | [Add owner/date] | [Not recorded yet] |

**Information needed to complete this section accurately:** meeting dates or approximate dates, what the team presented, the guide's recommendations or corrections, decisions made, and actions assigned. Calendar entries, messages, or brief notes are sufficient source material.

## 11. Supporting project records

- `ourwork/code_and_project_readiness_audit.md` — code-grounded review of the paper, implementation, dependencies, dataset, and risks.
- `ourwork/project_implementation_guide.md` — staged research implementation plan and optimizer-comparison protocol.
- `ourwork/pc_local_hardware_and_baseline_pipeline.md` — hardware snapshot and local setup/run plan.
- `PPTnall/RM.pptx` — project presentation.
- `PPTnall/syftr-BasePaper.pdf` — selected base paper.
- `PPTnall/README_syftr_optimization_research.md` and `PPTnall/Final_PPT_Structure_syftr_Research.md` — research framing and presentation structure.
- `PPTnall/LLMLingua.pdf` and `PPTnall/Ragged.pdf` — supporting literature.

---

**Short message to accompany this update:**  
*Dear Professor, please find attached our current project progress summary. It outlines our understanding of the Syftr base paper and implementation, the planned comparison methods, decisions still to be finalized, and the next steps. We have completed the initial paper/code review and prepared the implementation plan; the local baseline run and proposed optimizer implementations are still pending. We are also compiling the dated meeting record from our notes so that the meeting-by-meeting details and your feedback are recorded accurately.*
