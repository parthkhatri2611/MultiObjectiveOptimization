# Research Project README
## Multi-Objective Optimization of RAG Search Strategies Based on syftr

## 1. Project Overview

This project studies how different **multi-objective optimization strategies** perform when searching the same large RAG configuration space.

Our selected base paper is:

> **syftr: Pareto-Optimal Generative AI**  
> Alexander Conway et al., 2025

The key idea of syftr is to formulate the design of Generative AI / RAG flows as a **multi-objective optimization problem**, where configurations are evaluated according to competing objectives such as task quality/accuracy and cost.

syftr uses **Bayesian optimization**, specifically a **Multi-Objective Tree-of-Parzen Estimators (MO-TPE)** approach, together with mechanisms such as **EHVI-based acquisition, Pareto pruning, and seeding**.

The paper describes a search space containing more than \(10^{23}\) possible RAG flows and reports that syftr can identify substantially cheaper Pareto-optimal flows while retaining high task accuracy.

---

# 2. Core Research Problem

RAG systems have many configuration choices.

A RAG flow can vary in components such as:

- LLM
- Retriever
- Embedding model
- Document splitter
- Chunking parameters
- Prompt
- Reranker
- Retrieval settings
- Agentic components
- Reasoning strategies
- Other pipeline components

When these choices are combined, the configuration space becomes extremely large.

Therefore, evaluating every possible configuration is impractical.

The central problem is:

> **How can we efficiently search a very large RAG configuration space to discover configurations that provide good task quality while controlling cost/efficiency?**

This is naturally a **multi-objective optimization problem**.

---

# 3. Base Paper: syftr

The selected base paper is:

**syftr: Pareto-Optimal Generative AI**

The paper proposes a framework for searching large GenAI/RAG configuration spaces using multi-objective Bayesian optimization.

The important concept is the **Pareto frontier**.

Instead of looking for a single configuration that is best according to one metric, syftr searches for configurations representing different trade-offs between competing objectives.

For example:

```text
Higher Quality
      ↑
      │             ●
      │          ●
      │       ●
      │    ●
      │ ●
      └──────────────────→ Cost
```

A configuration can therefore be:

- cheaper but slightly lower quality,
- more expensive but higher quality,
- or somewhere between these extremes.

The user can then select a configuration according to the desired quality-cost trade-off.

---

# 4. What Optimization Strategy Does syftr Use?

A very important point:

> **syftr does NOT use NSGA-II.**

The original syftr optimization strategy is based on **Bayesian Optimization**, specifically **Multi-Objective Tree-of-Parzen Estimators (MO-TPE)**.

The optimization process uses information from previous trials to guide future evaluations.

Conceptually:

```text
Search Space
     ↓
Select Configuration
     ↓
Run RAG
     ↓
Measure Objectives
     ↓
Update Optimization Model
     ↓
Select Next Promising Configuration
     ↓
Repeat
```

Important syftr components include:

### MO-TPE

Tree-of-Parzen Estimators model promising and less-promising regions of the search space.

### EHVI

Expected Hypervolume Improvement can be used to guide multi-objective search toward potentially useful regions of objective space.

### Pareto-Pruner

syftr can stop poor trials early when intermediate results indicate that they are unlikely to become competitive with the current Pareto frontier.

### Seeding

The search can be initialized with informative configurations instead of relying entirely on uninformed exploration.

---

# 5. Our Research Idea

Our research does **not** change the syftr search space.

This is a critical part of the project.

We want to keep:

```text
SAME SEARCH SPACE
SAME RAG CONFIGURATION PROBLEM
SAME OBJECTIVES
SAME DATA
SAME RAG EVALUATION
SAME COMPUTATIONAL/EVALUATION BUDGET
```

and change only:

```text
OPTIMIZATION STRATEGY
```

Therefore, the project can be summarized as:

```text
                 SYFTR SEARCH SPACE
                        │
          ┌─────────────┼─────────────┐
          │             │             │
        MO-TPE       NSGA-II      Hybrid/Adaptive
       Baseline      Proposed        Proposed
          │             │             │
          └─────────────┼─────────────┘
                        ↓
                 Same RAG Evaluation
                        ↓
                Same Objectives
                        ↓
                 Pareto Fronts
                        ↓
              Quantitative Comparison
```

The main research question is therefore:

> **How does changing the optimization strategy affect the efficiency and quality of Pareto-optimal RAG configuration discovery within the same syftr search space?**

---

# 6. What Exactly Are We Changing?

We are investigating three alternative optimization strategies.

## 6.1 NSGA-II

NSGA-II is a population-based evolutionary multi-objective optimization algorithm.

Its main concepts include:

- Population
- Fitness/objective evaluation
- Non-dominated sorting
- Crowding distance
- Selection
- Crossover
- Mutation
- Generation-wise evolution

Conceptually:

```text
Initial Population
        ↓
Evaluate RAG Configurations
        ↓
Non-Dominated Sorting
        ↓
Crowding Distance
        ↓
Selection
        ↓
Crossover + Mutation
        ↓
New Population
        ↓
Repeat
```

The purpose of investigating NSGA-II is to study how an explicit population-based Pareto search behaves on the syftr configuration space.

---

# 7. Hybrid NSGA-II + Bayesian Optimization

The second proposed strategy combines evolutionary search with Bayesian/model-guided search.

The general idea is:

```text
              NSGA-II
                 ↓
       Population Exploration
                 ↓
       Identify Promising Regions
                 ↓
       Bayesian Guidance
                 ↓
      Generate/Refine Candidates
                 ↓
            Next Population
```

The exact hybrid mechanism is part of the research implementation and should be finalized experimentally.

Possible design choices include:

- Bayesian guidance for candidate selection
- Bayesian generation of promising candidates
- Bayesian local refinement
- Periodic Bayesian intervention
- Using an acquisition function to inject candidates into the evolutionary population

The final implementation should clearly define the mechanism and keep the comparison fair.

---

# 8. Adaptive / Learning-Guided NSGA-II

The third proposed strategy introduces adaptation into NSGA-II.

Instead of keeping all evolutionary parameters fixed throughout optimization, the algorithm can use information from previous generations to modify its search behavior.

Conceptually:

```text
Evaluate Population
        ↓
Observe Search Progress
        ↓
Learn From Search History
        ↓
Adjust Exploration / Exploitation
        ↓
Modify Search Behavior
        ↓
Generate Next Population
        ↓
Repeat
```

Possible adaptive components include:

- Mutation rate
- Crossover rate
- Exploration intensity
- Exploitation intensity
- Candidate generation strategy
- Population diversity control

The exact adaptive mechanism will be defined as part of the research implementation.

It should not be presented as finalized until implemented and experimentally evaluated.

---

# 9. What We Are NOT Doing

The following points are important.

### We are NOT:

- Creating a completely new RAG architecture.
- Designing a new retrieval model.
- Changing the fundamental syftr search space.
- Claiming beforehand that NSGA-II is better than Bayesian optimization.
- Claiming beforehand that the hybrid method is better.
- Claiming beforehand that adaptive NSGA-II is better.
- Comparing algorithms using different evaluation budgets.
- Changing the RAG pipeline for different algorithms.

### We ARE:

- Keeping the base problem fixed.
- Keeping the search space fixed.
- Keeping the evaluation environment fixed.
- Reproducing the original optimization baseline.
- Implementing alternative optimization strategies.
- Comparing their behavior experimentally.
- Measuring Pareto-front quality and search efficiency.

---

# 10. Experimental Variables

The most important experimental principle is:

> **The optimization strategy is the primary independent variable.**

### Controlled variables

The following should remain constant:

| Component | Experimental Setting |
|---|---|
| Search Space | Same syftr search space |
| Dataset | Same |
| LLM | Same |
| Retriever | Same |
| Embedding Model | Same |
| RAG Pipeline | Same |
| Objectives | Same |
| Evaluation Procedure | Same |
| Evaluation Budget | Same |
| Number of Runs | Same |
| Random Seed Protocol | Controlled |
| Optimization Strategy | Changed |

This allows us to attribute observed differences primarily to the optimization strategy.

---

# 11. Algorithms Being Compared

The main comparison is:

```text
                 Optimization Strategies
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
      MO-TPE           NSGA-II        Hybrid NSGA-II
     syftr              Proposed        + Bayesian
        │                                   │
        └────────────────┬──────────────────┘
                         │
                  Adaptive NSGA-II
                      Proposed
```

### Primary baseline

**MO-TPE**

This is mandatory because it is the optimization strategy of the base paper.

### Proposed approaches

1. NSGA-II
2. Hybrid NSGA-II + Bayesian Optimization
3. Adaptive / Learning-Guided NSGA-II

### Optional simple baselines

Depending on feasibility:

- Random Search
- Structured/Grid Search

These provide additional reference points but are not the core research contribution.

---

# 12. Fair Experimental Comparison

The algorithms must be compared under equivalent conditions.

The most important constraint is the **evaluation budget**.

RAG evaluation is expensive.

Therefore, if:

```text
MO-TPE → 1,000 evaluations
NSGA-II → 10,000 evaluations
```

then a direct comparison would not be fair.

Instead, we should define a common budget such as:

```text
Maximum RAG evaluations = B
```

and give every optimization method the same budget.

We can also compare performance as a function of evaluation count.

---

# 13. Objectives

The exact objectives should remain consistent across all optimization strategies.

The central idea is the trade-off between:

### Objective 1 — Task Quality

Examples may include:

- Answer accuracy
- Exact Match
- F1
- Another task-specific RAG quality metric

### Objective 2 — Cost / Efficiency

Depending on the implementation, this may include:

- Token usage
- LLM cost
- Context cost
- Latency
- Or the cost metric used consistently by the base implementation

The exact objective definitions must remain identical across all algorithms.

---

# 14. Pareto Optimization

Because the objectives conflict, there is generally no single universally optimal configuration.

For example:

```text
Configuration A
High quality
High cost

Configuration B
Medium quality
Medium cost

Configuration C
Lower quality
Low cost
```

A configuration is **Pareto-dominated** if another configuration is at least as good in every objective and strictly better in at least one.

The remaining configurations form the **non-dominated set**.

This is the Pareto frontier.

Our algorithms will attempt to discover this frontier.

---

# 15. Evaluation Metrics

We will evaluate both:

## A. Pareto-front quality

### Hypervolume (HV)

Measures the objective-space volume dominated by the obtained Pareto front relative to a chosen reference point.

### IGD

Measures the distance between the obtained Pareto front and a reference Pareto front.

### Number of Non-Dominated Solutions

Measures how many non-dominated configurations are discovered.

---

## B. Search efficiency

### Number of RAG Evaluations

How many expensive evaluations are needed?

### Runtime

How much optimization time is required?

### Convergence

How quickly does the optimizer approach a strong Pareto frontier?

### Cost per Improvement

How many additional evaluations are required to obtain meaningful improvement in the objective space?

---

# 16. Experimental Workflow

The complete research workflow is:

```text
                 Base syftr Search Space
                          ↓
                 Define Common Objective
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
            MO-TPE     NSGA-II    Hybrid/Adaptive
              ↓           ↓           ↓
              └───────────┼───────────┘
                          ↓
                 Generate RAG Config
                          ↓
                    Run RAG Flow
                          ↓
                  Evaluate Quality
                          +
                  Measure Cost
                          ↓
                 Store Trial Result
                          ↓
                 Update Optimizer
                          ↓
                     Continue
                          ↓
                 Obtain Pareto Set
                          ↓
             HV / IGD / Convergence
                          ↓
                Compare Algorithms
```

---

# 17. Research Questions

## RQ1

**How does NSGA-II perform compared with syftr's original MO-TPE strategy when searching the same RAG configuration space?**

## RQ2

**Can a hybrid NSGA-II + Bayesian Optimization strategy improve Pareto-front discovery or search efficiency compared with the individual strategies?**

## RQ3

**Can an adaptive/learning-guided NSGA-II strategy improve exploration and exploitation during optimization?**

## RQ4

**How do different optimization strategies affect the quality-cost trade-off of the discovered RAG configurations?**

## RQ5

**How does optimization efficiency change when all strategies are given the same evaluation budget?**

---

# 18. Hypotheses

The project should treat these as hypotheses to test, not established facts.

### H1

Different optimization strategies will produce different Pareto-front search behavior on the same RAG configuration space.

### H2

Population-based evolutionary search may discover diverse non-dominated configurations through explicit population evolution.

### H3

Combining Bayesian guidance with evolutionary search may provide a useful balance between model-guided exploitation and population-based exploration.

### H4

Adaptive search behavior may affect convergence and exploration/exploitation balance compared with fixed NSGA-II parameters.

These hypotheses must be evaluated experimentally rather than assumed to be true.

---

# 19. Ablation Studies

Once the main algorithms are working, we should analyze why they behave differently.

## NSGA-II ablations

Test:

- Population size
- Mutation rate
- Crossover rate
- Number of generations

## Hybrid ablations

Test:

- Bayesian intervention frequency
- Number of Bayesian candidates
- Candidate injection strategy
- Acquisition strategy

## Adaptive ablations

Test:

- Adaptation frequency
- Adaptation signal
- Mutation adaptation
- Crossover adaptation
- Exploration/exploitation policy

The goal is to understand which components contribute to observed performance.

---

# 20. Expected Outputs

The research should produce:

### 1. Pareto Fronts

For every optimization strategy.

### 2. Hypervolume Curves

Hypervolume vs number of evaluations.

### 3. Convergence Curves

Optimization progress over time/evaluations.

### 4. Evaluation Efficiency

Performance achieved for a fixed number of RAG evaluations.

### 5. Runtime Comparison

Optimization overhead.

### 6. Non-Dominated Configuration Sets

Actual RAG configurations discovered by each algorithm.

### 7. Statistical Comparison

Multiple independent runs should be used where computationally feasible.

Report:

- Mean
- Standard deviation
- Confidence intervals where appropriate

---

# 21. What Would Constitute a Successful Result?

The research does not require NSGA-II to outperform MO-TPE.

A scientifically useful result can be any clearly demonstrated finding, for example:

```text
NSGA-II
→ different Pareto-front behavior

Hybrid
→ different convergence characteristics

Adaptive
→ different exploration behavior

MO-TPE
→ strong model-guided search under certain conditions
```

The important contribution is to establish **empirical evidence about the behavior of different optimization strategies on the same problem**.

---

# 22. Implementation Philosophy

The implementation should be modular.

Recommended architecture:

```text
project/
│
├── data/
│
├── config/
│
├── rag/
│   ├── pipeline
│   ├── retrieval
│   └── evaluation
│
├── search_space/
│   └── syftr_search_space
│
├── optimizers/
│   ├── mo_tpe
│   ├── nsga2
│   ├── hybrid_nsga2_bo
│   └── adaptive_nsga2
│
├── evaluation/
│   ├── pareto
│   ├── hypervolume
│   ├── igd
│   └── convergence
│
├── experiments/
│   ├── baseline
│   ├── nsga2
│   ├── hybrid
│   └── adaptive
│
├── results/
│
├── plots/
│
└── README.md
```

The key design principle is that all optimizers should call the same underlying RAG evaluation interface.

For example:

```text
Optimizer
    ↓
candidate configuration
    ↓
RAG evaluator
    ↓
objective values
    ↓
Optimizer
```

This prevents algorithm-specific changes from leaking into the RAG evaluation process.

---

# 23. Reproducibility

Every experiment should record:

- Dataset version
- Search-space version
- Model version
- Retriever
- Embedding model
- Random seed
- Optimizer parameters
- Evaluation budget
- Objective definitions
- Runtime
- Hardware/environment
- Final Pareto solutions
- Intermediate trial history

This is essential because stochastic optimization algorithms can produce different results across runs.

---

# 24. Recommended Experiment Protocol

For each optimization method:

```text
1. Initialize experiment
2. Fix random seed
3. Load same dataset
4. Load same RAG pipeline
5. Load same syftr search space
6. Set same objectives
7. Set same evaluation budget
8. Run optimizer
9. Store every evaluated configuration
10. Compute final Pareto set
11. Compute HV
12. Compute IGD
13. Measure runtime
14. Store convergence history
15. Repeat for multiple seeds
16. Aggregate results
```

Then compare the distributions of results rather than relying on one run.

---

# 25. Important Terminology

## Search Space

The set of all possible configurations that the optimizer is allowed to evaluate.

## Candidate / Configuration

One specific point in the search space.

## Objective

A quantity that the optimizer attempts to optimize.

## Pareto Dominance

One solution is no worse in every objective and better in at least one.

## Pareto Frontier

The set of non-dominated solutions.

## Multi-Objective Optimization

Optimization involving multiple objectives simultaneously.

## Bayesian Optimization

A model-based optimization strategy that uses observations from previous evaluations to guide future evaluations.

## MO-TPE

Multi-Objective Tree-of-Parzen Estimators, the Bayesian optimization strategy used by syftr.

## NSGA-II

A population-based evolutionary multi-objective optimization algorithm based on non-dominated sorting and diversity preservation.

## Hypervolume

A metric for evaluating the objective-space coverage of a Pareto front.

## IGD

A metric for measuring the distance between an obtained Pareto front and a reference front.

## Exploration

Searching previously under-explored regions.

## Exploitation

Focusing search around regions already identified as promising.

---

# 26. Current Research Position

At this stage, the project can be summarized as:

```text
BASE PAPER
syftr
   ↓
Original optimizer
MO-TPE / Bayesian Optimization
   ↓
RESEARCH GAP
Limited investigation of alternative
optimization strategies within the same framework
   ↓
OUR RESEARCH
Keep syftr search space fixed
   ↓
Change optimization strategy
   ↓
NSGA-II
Hybrid NSGA-II + Bayesian
Adaptive NSGA-II
   ↓
CONTROLLED EXPERIMENT
Same RAG pipeline
Same objectives
Same evaluation budget
   ↓
EVALUATION
Pareto quality
Convergence
Efficiency
Runtime
   ↓
RESEARCH CONTRIBUTION
Empirical comparison and analysis
of optimization strategies for RAG search
```

---

# 27. One-Line Project Definition

> **We investigate how alternative multi-objective optimization strategies—NSGA-II, hybrid NSGA-II + Bayesian Optimization, and adaptive NSGA-II—perform relative to syftr's original MO-TPE strategy when searching the same RAG configuration space under controlled evaluation conditions.**

---

# 28. Short Abstract

This project investigates optimization strategies for large-scale Retrieval-Augmented Generation configuration spaces using the syftr framework as the base. syftr formulates GenAI/RAG flow design as a multi-objective optimization problem and uses Multi-Objective Tree-of-Parzen Estimators for Bayesian search. Rather than modifying the underlying syftr search space, this work keeps the search space, RAG pipeline, objectives, and evaluation conditions fixed and investigates alternative optimization strategies. NSGA-II, a hybrid NSGA-II + Bayesian Optimization strategy, and an adaptive/learning-guided NSGA-II strategy are implemented and compared against the original MO-TPE baseline. The methods are evaluated using Pareto-front quality, hypervolume, IGD, convergence, number of RAG evaluations, and runtime. The objective is to experimentally understand how the choice of optimization strategy affects the discovery of Pareto-optimal RAG configurations and the efficiency of searching a large configuration space.

---

# 29. Important Research Principle

The project should always be explained using this distinction:

> **We are not proposing a new RAG search space.**

> **We are investigating new ways to search the existing syftr space.**

This distinction should remain consistent across:

- README
- PPT
- Research paper
- Code
- Experiments
- Project presentation
- Viva/judge answers

