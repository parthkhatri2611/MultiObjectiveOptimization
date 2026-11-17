# Speaker guide for the local Syftr optimizer study

Main presentation: slides1–28. Technical appendix: slides29–35. References: slides36–38. Use Presenter View to read the embedded notes.

Native evidence uses one seed and ten search questions. No usable native holdout result is available. The compact study uses a separate measured fitness replay protocol.

## 1 Pareto search for local RAG

Opening: Our project studies how the choice of optimizer changes the quality and token tradeoffs of a RAG pipeline. We adapt the Syftr framework to one accessible local Qwen model. We implement standard NSGA-II, a hybrid with MOTPE injections and an adaptive mutation variant. The main evidence is a restricted native experiment, not an exact reproduction of the paper catalogue. Introduce the team, then move to the practical configuration problem.

SOURCE AND EVIDENCE
Research Methodology Project. 17 November 2026.

## 2 The RAG configuration problem

Explain RAG in plain language. A retriever chooses passages from documents and a fixed generator answers using that context. Chunking, retrieval depth and extra modules change both answer quality and token usage. More context can help or distract, and nested workflows can make additional model calls. The research problem is to select useful tradeoffs under a limited evaluation budget. The diagram is conceptual, not an additional measured result.

SOURCE AND EVIDENCE
Background: Lewis et al. 2020. Syftr paper sections 1 to 3.

## 3 Syftr is our base framework

Syftr searches hierarchical RAG flows with multiobjective Bayesian optimization through Optuna MOTPE. The paper includes five workflow families, multiple hosted models, optional modules and Pareto pruning. Our research intervention is the proposal policy. We retain native flow construction but adapt model availability and evaluation. The baseline should therefore be described as a local Syftr MOTPE adaptation, not the exact original paper experiment.

SOURCE AND EVIDENCE
[R1] Conway et al. 2025. Supplied Syftr paper sections 3 to 5.

## 4 Our research questions

State the questions as an empirical comparison rather than a promise that the proposed methods will win. We ask which policy discovers the strongest measured quality and token frontier under the same budget, whether hybrid guidance helps, whether adaptive mutation actually activates, and how reliable a small-set result is. Later slides answer these questions using both positive and negative findings.

SOURCE AND EVIDENCE
Proposal: RM.pptx slides 11 to 25.

## 5 Major implementation decisions

Explain each decision and its cost. One pinned Qwen model avoids unavailable hosted APIs and controls capacity across arms. We use raw answer F1 because the local judge did not establish reliable primary correctness. Verified generator tokens replace API dollar costs. Restricting the space makes a real native study feasible. These decisions apply equally to every arm and limit generalization to the original full catalogue.

SOURCE AND EVIDENCE
Protocol and source: E1. Local judge calibration and context.md.

## 6 One evaluator for all four methods

The optimizer proposes active parameters. The adapter constructs the native Syftr flow. Qwen serves generation on physical GPU0. BGE embeddings use physical GPU1, isolated as logical cuda0 inside the worker. A proxy journals generator and judge calls separately. We reuse the evaluator and budget for every method. Independent re-evaluation still produced variation, so common code does not imply identical fitness.

SOURCE AND EVIDENCE
Implementation: baseline adapter.py worker.py proxy.py and campaign_v2/run.py.

## 7 The native restricted search space

The main experiment retains RAG and sub-question RAG, two embedding models, three retrieval types, three splitter choices and prompts. Optional modules include HyDE, few-shot retrieval, query decomposition, reranking and neighbor context. Module parameters only become active when their branch is enabled. This is broader than k, c and context budget, but smaller than the original paper catalogue.

SOURCE AND EVIDENCE
E1 executed restricted spec.py and distributions.json.

## 8 The experiment we actually ran

Distinguish counts carefully. We evaluate ten selected questions out of the 500 available search QA pairs, against 4972 grounding documents. Each arm has25 attempted configurations, including the same first10 parameter configurations independently re-evaluated and charged. The native experiment has one optimizer seed11.98 candidates passed eligibility and generated980 QA rows. The two remaining attempts were duplicate proposals. Repeated rows do not create980 independent questions.

SOURCE AND EVIDENCE
E1 protocols question and grounding manifests and final audit.

## 9 How we measure a candidate

Raw F1 is the mean of token overlap scores over all ten questions, with frozen normalization and special yes/no mismatch handling. Resource is the sum of prompt plus output tokens for every generator call in the workflow divided by ten. Judge calls remain separate diagnostics. Hypervolume summarizes the observed non-dominated quality-cost set with reference quality0 and tokens50000. Show direct cost thresholds later because this reference gives substantial weight to quality.

SOURCE AND EVIDENCE
Implementation metrics.py worker.py and report.py. Reference T = 50,000 tokens.

## 10 Baseline implementation with local MOTPE

We instantiate pinned Optuna4.2.1 TPESampler with seed11, startup10, multivariate true and constant_liar true, then optimize maximize F1 and minimize token cost through native SearchSpace.sample. We disable pruning for all methods so eligible configurations use the full ten questions. Shared initialization includes common seed settings and startup samples. The remaining15 attempts use the reference policy. We do not reimplement a separate EHVI solver.

SOURCE AND EVIDENCE
E1 baseline/run.py. [R1] Syftr. [R7] Optuna 4.2.1 documentation.

## 11 MOTPE improved beyond initialization

At25 attempts the reference completed24 configurations with one duplicate. Final HV is0.3927537959 and best F1 .3984313725. Initialization best F1 was.3752380952. Its total generator search tokens were322040. The cheapest eligible point is377.2 tokens per QA, but that low-cost point does not have the best F1. The chart shows cumulative frontier improvement, not independent validation. No native holdout result is available.
Evidence: E1 comparison/findings.json and baseline convergence.csv.

SOURCE AND EVIDENCE
Evidence E1: final matched25 raw audit. HotPotQA search set, seed 11.

## 12 Approach 1  NSGA-II

We choose NSGA-II because it explicitly maintains different tradeoff configurations in a small population. Our implementation performs non-dominated sorting, normalized crowding distance, binary tournament selection, variation and elitist environmental survival. It does not call Optuna NSGAIISampler. Ten shared measured settings initialize five survivors, then three generations each produce five charged offspring. Failed children do not become valid survivors.

SOURCE AND EVIDENCE
[R2] Deb et al. 2002. Exact implementation optimizers/nsga2.py.

## 13 Our genetic operators respect conditional modules

A genome stores every native distribution field, including inactive branches. Materialize calls the real SearchSpace with FixedTrial and produces only active concrete parameters. Block crossover keeps related fields together at probability.9. Mutation probability is min(.5,multiplier divided by active mutable count). A forced gene change prevents a direct parent copy. Global duplicates still consume attempts. Conditional distance normalizes numeric differences and counts branch presence differences.

SOURCE AND EVIDENCE
Implementation optimizers/space.py and bridge.py.

## 14 Standard NSGA-II finished below MOTPE

At25 attempts NSGA-II completed25 configurations without a duplicate. HV is.3716584094 versus reference.3927537959, a5.37 percent relative decrease. Best F1 .3752380952 stayed at a shared initialization point, while genetic offspring added lower-cost tradeoffs. This is an observed small-budget result. Five survivors and only three generations may restrict exploration, but that explanation was not tested with a population or generation ablation.

SOURCE AND EVIDENCE
Evidence E1: final matched25 raw audit. HotPotQA search set, seed 11.

## 15 Approach 2  Hybrid NSGA-II with MOTPE

The hybrid hypothesis combines population recombination with proposals guided by successful history. Every five-child generation has three genetic slots followed by two MOTPE injection slots. A fresh pinned TPE sampler imports the actual successful active parameters, distributions and measured objective pairs. A separate deterministic BO random stream seeds it. Across three generations, six of the15 post-startup slots are injections. Every injection consumes the same budget.

SOURCE AND EVIDENCE
Exact implementation EvolutionSearch.ask and _bayesian_proposal in nsga2.py.

## 16 Hybrid led the first three methods

Hybrid final HV.4443762473 exceeds MOTPE by13.14 percent and standard NSGA-II by about19.56 percent. It completed24 configurations with one duplicate, and best observed F1.4485714286. The comparison uses the same model, ten questions and25 attempts. This is descriptive evidence on one seed, not proof of stable general superiority. The next slide separates the hybrid overall result from injection-specific attribution.

SOURCE AND EVIDENCE
Evidence E1: final matched25 raw audit. HotPotQA search set, seed 11.

## 17 Genetic proposals supplied the strongest hybrid points

Trial11 was genetic and reached the hybrid maximum F1.4485714286 at1762.8 tokens per QA. Genetic trial16 reached .4252380952 at402.1 tokens. The first injection occurs after trial11, and injection maximum F1 was.3752380952. This establishes the origin of the observed best quality, not that the injection component has no value. Bayesian injections could still affect history or later proposals. A zero-injection ablation with repeated seeds is needed for causality.

SOURCE AND EVIDENCE
E1 hybrid optimizer_state.json trial origins and raw measurements.

## 18 Approach 3  Adaptive mutation NSGA-II

This variant uses generation-level feedback, not a separately trained learning model. Begin with multiplier1. After each generation compute HV gain and conditional population diversity. Gain at most1e-6 increments stagnation. One stagnant generation or diversity below.15 multiplies mutation strength by1.5, capped5. Otherwise divide by1.2 but floor at1. Per active gene probability is capped.5. Crossover, population and survival stay the same as standard NSGA-II.

SOURCE AND EVIDENCE
EvolutionSearch._finish_generation and ConditionalSpace.offspring.

## 19 The adaptation trigger never fired

The implementation is present, but in all three generations the used and next multipliers stayed1. HV gains were positive above the stagnation threshold and diversities stayed above.15. We must not credit improved scores to a stronger mutation rate. Independent fitness variation can alter selection and later trajectories even at nominally equal strength. The table is exact recorded generation diagnostics, not a hypothetical adaptation example.

SOURCE AND EVIDENCE
E1 adaptive optimizer_generation_log.json.

## 20 Adaptive came close to hybrid

At25 attempts adaptive completed25 eligible configurations with no duplicates. Its best F1 .4485714286 ties hybrid. HV .4425132498 is12.67 percent above MOTPE and about.42 percent below hybrid. Its frozen finalist trial19 reaches the same best quality at1435.7 tokens per QA, versus hybrid trial11 at1762.8. That does not mean its whole front dominates hybrid. The adaptation trigger did not activate.

SOURCE AND EVIDENCE
Evidence E1: final matched25 raw audit. HotPotQA search set, seed 11.

## 21 All four observed Pareto fronts

The scatter chart contains the final observed front points only. Lower token cost and higher F1 are desirable. The chart shows discrete non-dominated measurements without interpolation. Hybrid has a strong low-token point near402, while adaptive reaches the tied highest F1 at lower cost than the hybrid highest quality point. Therefore aggregate HV ranking does not identify one method as cheapest at every quality level.

SOURCE AND EVIDENCE
E1 audited pareto_fronts.csv. Front points only. Token reference 50,000.

## 22 Most improvement arrived early after initialization

The curves use end-of-attempt hypervolume and include duplicate charges. The first10 attempts share configurations but independently re-evaluate them. Small differences at that point are measured variation, not a policy advantage. Hybrid and adaptive improve early, followed by smaller later gains. A flat curve across a few attempts is not proof of global convergence. No error bars appear because there is only one native optimizer seed.

SOURCE AND EVIDENCE
E1 per-arm convergence.csv. First 10 attempts are shared initialization.

## 23 Hybrid offered the strongest low-token quality

Read the table as best search F1 among candidates whose mean generator tokens per QA satisfy the ceiling. At500 tokens, hybrid .425238 is substantially above the other observed arms. At2000 tokens hybrid and adaptive reach their joint maximum. A threshold on full generator tokens is different from a context-token cap. This is search evidence and does not establish unseen QA performance.

SOURCE AND EVIDENCE
E1 recomputed best_f1_at_tokens_le fields.

## 24 One answer reversal changed mean F1 by 0.10

Three arms evaluated identical active parameter configuration6f466924 at trial11. Standard NSGA-II answered no on question5ae2cd3b5542996483e64a6d while hybrid and adaptive answered yes, matching the reference. The difference on that one question is enough to move the ten-question mean by.1. Exact cause is not isolated: generation, retrieval ordering or nested prompt behavior may contribute. Temperature0 alone does not establish end-to-end determinism.

SOURCE AND EVIDENCE
E1 trial11 raw questions.jsonl and repeated_configuration_variation.csv.

## 25 A separate compact study found no final winner

The compact fallback measured27 settings of k in2,4,6; c256,512,1024; Bc512,1024,2048 on five search questions. It replayed four optimizers over the fixed measured fitness landscape for ten seeds with25 logical queries. All methods reached the same measured frontier and final HV .5229264 using reference5000. Seven of ten initial pools already contained the complete front. This is a useful negative result, not evidence that all optimizers are equivalent in the larger native space.

SOURCE AND EVIDENCE
Evidence E2: compact27 audited measured landscape and ten-seed fitness replay.

## 26 Answers to our research questions

Summarize without overstating. On RQ1 the observed native winner is hybrid. On RQ2 standard NSGA-II does not beat MOTPE. On RQ3 the injection causal contribution is unresolved and adaptive mutation strength never changed. On RQ4 we observed repeat variation and no usable native holdout. The compact study reinforces that an easy small landscape can hide optimizer differences.

SOURCE AND EVIDENCE
E1 and E2 audited findings and generation diagnostics.

## 27 Questions this study cannot yet answer

We explicitly did not complete multiple native optimizer seeds, a usable native holdout, a hybrid intervention ablation, an adaptive-trigger activation study or the full paper catalogue. Shared cache reuse and fixed method order also prevent an unbiased wall-clock winner. These are limitations and next experiments, not missing values to fill. We can report the completed search now without another long Kaggle run.

SOURCE AND EVIDENCE
Study scope and limitations from final audited dossier.

## 28 Conclusion

Close with what the evidence supports. We built a working local adaptation of native Syftr with three implemented alternative search policies and auditable usage accounting. At25 attempts per arm hybrid attained the strongest observed frontier, standard NSGA-II did not outperform the reference, and adaptive mutation benefit remained untested because it never activated. One seed and ten search questions make this an exploratory comparison. The appendix provides exact implementation and evidence for questions.

SOURCE AND EVIDENCE
Completed native restricted study and separate compact27 study.

## 29 Appendix  Exact implementation map

If asked to show code, open the exact source returned from the final campaign. spec.py contains the restricted override, adapter prepares native StudyConfig, worker and proxy collect answers and usage, space.py materializes conditional phenotypes, nsga2.py implements operators and bridge.py connects them to Optuna storage. report.py re-audits every eligible objective. No training or new inference occurs in the offline report pipeline.

SOURCE AND EVIDENCE
FINAL_RESEARCH_PROJECT/02_implementation/native_exact.

## 30 Appendix  Full final comparison

These values are final search metrics at25 attempts per arm. Duplicates explain eligible counts24 or25 and still consume budget. IGD is to the pooled empirical front with common cost scaling, not an unknown global optimum. The native source and raw record audit reproduced these numbers offline. Frontier configuration count can include equal objective ties.

SOURCE AND EVIDENCE
E1 AUDIT_at_25.json and comparison/findings.json.

## 31 Appendix  Total search resource usage

Generator tokens are actual logged input plus output usage across all generator workflow calls. Diagnostic judge tokens are separate. Totals differ from deployment cost of one chosen finalist. Standard NSGA-II spent fewer generator tokens but attained a smaller frontier. Because setup/index cache state and method order differ, these totals do not establish a GPU runtime or energy speedup.

SOURCE AND EVIDENCE
E1 known_generator_tokens and known_judge_tokens per-arm ledgers.

## 32 Appendix  Compact holdout exposed a quality drop

The compact experiment has its own5 held-out questions and three deduplicated frozen configurations. All selected search means were.56, but holdout F1 was.2571 for c256/c512 and.0571 for c1024. Those40 method/seed selections collapse to only3 distinct settings, so the repeated selections are not independent experiments. This holdout cannot substitute for native finalist confirmation and must not be pooled with native metrics.

SOURCE AND EVIDENCE
E2 holdout_measurements.json and compact audit.

## 33 Appendix  What counts as an ablation

The proposal included future ablations. We did not complete systematic controlled comparisons of population size, crossover probability, injection frequency or mutation trigger settings. The compact study measures parameter settings but is a different fixed-pipeline experiment. A valid injection ablation holds the shared protocol and seed design fixed while changing only injection slots, then repeats runs. This prevents attributing an overall hybrid winner automatically to the new component.

SOURCE AND EVIDENCE
RM proposal planned ablations. Completed evidence boundary from E1 and E2.

## 34 Appendix  Evidence ready for inspection

The final export reconstructed from four verified parts contains177 bundle members and1338 capsule files. The curated project ZIP includes documentation, exact executed source, raw native and compact outputs, deployment assets and CPU-only analysis tools. The offline audit recomputes98 eligible objectives from980 question rows and request/completion token journals without starting GPU inference. Model caches and credentials are excluded.

SOURCE AND EVIDENCE
Project MANIFEST.json final export receipt and offline audit result.

## 35 Appendix  Model and execution settings

All active LLM modules use the same pinned Qwen generator with temperature0 and top_p1. The native output cap is1024 and context16384, unlike the compact output cap256. The common worker uses request300, question600 and candidate3600 second timeouts. These are safety bounds rather than observed runtimes. Source and model revisions, actual package output and full parameter dictionaries are preserved in E1.

SOURCE AND EVIDENCE
E1 protocol.json server_metadata.json and packages.json.

## 36 References  Research foundations

Use these sources for literature context. The Syftr paper is the supplied base paper and native framework. NSGA-II provides non-dominated sorting and elitist survival. The MOTPE references explain the Bayesian component. Lewis et al. provides foundational retrieval-augmented generation background. Bibliographic references are not additional project experiments.

SOURCE AND EVIDENCE
References. Full URLs and source roles appear in these speaker notes.
R1 https://arxiv.org/abs/2505.20266
R2 https://doi.org/10.1109/4235.996017
R3 Referenced in Syftr [47] and [48]. Optuna links: https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html
R4 https://arxiv.org/abs/2005.11401

## 37 References  Data models and runtime

The project used a pinned DataRobot Research adaptation of HotPotQA train_hard rather than all113000 original examples. We use Qwen2.5-7B-Instruct GGUF Q4_K_M, BGE small/base, Optuna4.2.1, llama.cpp and native Syftr/LlamaIndex integration. Full revisions are stored in the protocol. These sources document the components, not independent evidence that a particular optimizer is superior.

SOURCE AND EVIDENCE
Official papers model cards and pinned software records.
R5 https://aclanthology.org/D18-1259/ and https://huggingface.co/datasets/DataRobot-Research/hotpotqa
R6 https://arxiv.org/abs/2412.15115 and https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
R7 https://arxiv.org/abs/1907.10902 and https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html
R8 https://huggingface.co/BAAI/bge-small-en-v1.5 and https://huggingface.co/BAAI/bge-base-en-v1.5 and https://github.com/ggml-org/llama.cpp and https://github.com/datarobot/syftr and https://docs.llamaindex.ai/

## 38 References  Supporting literature and project evidence

The supplied LLMLingua and RAGGED papers motivated discussion of prompt efficiency and RAG configuration. We did not evaluate LLMLingua compression as a new intervention. The supplied RAGGED PDF is anonymized, so we do not invent its authors or venue. The proposal and all final quantitative findings are traceable to preserved project documents and manifests. Native holdout remains unavailable.

SOURCE AND EVIDENCE
Supplied papers proposal and immutable result archives.
Local references PPTnall/LLMLingua.pdf and PPTnall/Ragged.pdf. Background only.
E1 FINAL_RESEARCH_PROJECT/03_results/native_restricted and its original receipts.
E2 FINAL_RESEARCH_PROJECT/03_results/compact. Proposal PPTnall/RM.pptx. Full dossier under FINAL_RESEARCH_PROJECT/01_documentation.
