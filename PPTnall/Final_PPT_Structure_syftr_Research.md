# Final PPT Structure — syftr Optimization Research Project

## Project Title

**Multi-Objective Optimization of RAG Search Strategies**

### Base Paper
**syftr: Pareto-Optimal Generative AI**

### Core Research Idea

> Keep the **same syftr search space and RAG evaluation framework**, but investigate how different optimization strategies perform when searching that space.

---

# SECTION 1 — INTRODUCTION

## Slide 1 — Title Slide

### Title
**Multi-Objective Optimization of RAG Search Strategies**

### Subtitle
**Investigating Alternative Optimization Strategies within the syftr Framework**

### Include
- Team members
- Faculty / professor
- Course: Research Methodology
- Institution
- Date

### Visual
A simple:

```text
RAG Search Space
       ↓
Multi-Objective Optimization
       ↓
Pareto-Optimal Configurations
```

Do not overload the title slide.

---

## Slide 2 — Why This Problem Matters

### Title
**Why Optimize RAG Configurations?**

### Content

RAG performance depends on many configuration choices:

- LLM
- Retriever
- Embedding model
- Chunking / splitter
- Prompt
- Reranking
- Retrieval settings
- Agentic components
- Reasoning strategies

As these choices are combined, the configuration space becomes extremely large.

### Key challenge

Improving quality can increase:

- Token usage
- Computational cost
- Latency
- Monetary cost

Therefore:

> **RAG configuration is naturally a multi-objective optimization problem.**

### Visual

```text
             RAG Configuration
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      Task Quality        Cost/Efficiency
          ↑                   ↓
          └────── Trade-off ──┘
```

---

# SECTION 2 — LITERATURE REVIEW

## Slide 3 — Evolution of the Problem

### Title
**From RAG to RAG Optimization**

Show the literature progression:

```text
RAG
│
├── Retrieval + Generation
│
↓
RAG Efficiency
│
├── Context Compression
│
↓
RAG Configuration
│
├── Retriever / Chunking / Prompt / LLM choices
│
↓
Multi-Objective Optimization
│
└── Pareto-optimal RAG configurations
```

### Message

Research progressively moved from:

> **building RAG systems**

to:

> **optimizing how RAG systems are configured.**

---

## Slide 4 — Key Literature

### Title
**Key Literature Supporting the Research**

### Papers

| Work | Main Contribution | Relevance |
|---|---|---|
| Lewis et al. (2020) — RAG | Introduced the RAG framework | Foundation |
| Jiang et al. (2023) — LLMLingua | Context compression | Token efficiency |
| Hsia et al. (2025) — RAGGED | Studies RAG design choices | Configuration optimization |
| Conway et al. (2025) — syftr | Multi-objective optimization of GenAI/RAG flows | **Base paper** |

### Message

The literature establishes:

1. RAG quality depends on system design.
2. Context/token efficiency matters.
3. RAG contains many configurable components.
4. syftr directly formulates the configuration problem as multi-objective optimization.

---

## Slide 5 — Literature Gap

### Title
**What Is Still Open?**

### Existing work

Existing research studies:

- RAG architectures
- Context compression
- RAG configuration choices
- Multi-objective optimization
- Bayesian optimization for large configuration spaces

### Research opportunity

The base paper uses a particular optimization strategy:

> **Multi-Objective Bayesian Optimization using MO-TPE**

Our research asks:

> **How do alternative optimization strategies behave when searching the same syftr configuration space?**

### Important

Do not claim that syftr is inadequate.

The research gap is:

> **A comparative investigation of alternative optimization strategies within the same search framework.**

---

# SECTION 3 — BASE PAPER: SYFTR

## Slide 6 — Selected Base Paper

### Title
**Selected Base Paper — syftr**

### Paper

**syftr: Pareto-Optimal Generative AI**

Authors:
Alexander Conway et al.

Published as arXiv paper in 2025.

### Why this paper is central

syftr:

- formulates GenAI/RAG flow design as multi-objective optimization,
- searches a very large configuration space,
- seeks Pareto-optimal solutions,
- jointly considers task quality and cost,
- uses Bayesian optimization.

The paper describes a search space exceeding \(10^{23}\) possible RAG flows.

### Visual

Use the syftr architecture/search-space figure from the paper if permitted by your presentation requirements.

---

## Slide 7 — What Does syftr Actually Do?

### Title
**How syftr Solves the RAG Configuration Problem**

### Flow

```text
Large RAG Configuration Space
            ↓
Generate Candidate Flow
            ↓
Execute RAG Flow
            ↓
Measure Objectives
            ↓
Update Optimization Model
            ↓
Select Next Candidate
            ↓
Repeat
            ↓
Pareto Frontier
```

### Key idea

syftr does not search for one single "best" configuration.

It searches for a set of configurations representing different quality-cost trade-offs.

### Output

> **Pareto-optimal GenAI/RAG flows**

---

# SECTION 4 — SYFTR OPTIMIZATION MECHANISM

## Slide 8 — Original Optimization Strategy

### Title
**Original syftr Optimization Strategy — MO-TPE**

### Important clarification

> **syftr does not use NSGA-II.**

Its optimization strategy is based on:

**Multi-Objective Tree-of-Parzen Estimators (MO-TPE)**.

### Concept

TPE models promising and less-promising regions of the search space using density estimation.

```text
Previous Evaluations
        ↓
Good / Promising Trials
        +
Less-Promising Trials
        ↓
TPE Model
        ↓
Select Promising Configuration
        ↓
Evaluate
        ↓
Update Model
```

### Message

This provides the baseline that our proposed strategies will be compared against.

---

## Slide 9 — Important Components of syftr

### Title
**Key Components of the syftr Optimization Framework**

### 1. MO-TPE
Bayesian optimization strategy used for searching the configuration space.

### 2. EHVI
Expected Hypervolume Improvement helps guide multi-objective search.

### 3. Pareto-Pruner
Can terminate poor trials early when intermediate results indicate that they are unlikely to become competitive.

### 4. Seeding / Search Initialization
Informative initial configurations can help bootstrap optimization in a very large search space.

### Overall

```text
MO-TPE
  +
EHVI
  +
Pareto Pruning
  +
Informative Search Initialization
       ↓
Efficient Multi-Objective Search
```

---

# SECTION 5 — RESEARCH GAP

## Slide 10 — Research Gap

### Title
**Research Gap: Optimization Strategy**

### Current base-paper approach

```text
syftr
  ↓
Large Search Space
  ↓
MO-TPE / Bayesian Optimization
  ↓
Pareto Frontier
```

### Our research

We retain:

- Same search space
- Same RAG problem
- Same objectives
- Same evaluation framework

and investigate:

```text
MO-TPE
   vs
NSGA-II
   vs
Hybrid NSGA-II + Bayesian
   vs
Adaptive NSGA-II
```

### Research gap statement

> **The research investigates whether alternative or combined optimization strategies exhibit different Pareto-search behavior and evaluation efficiency on the same syftr problem.**

---

# SECTION 6 — RESEARCH OBJECTIVE

## Slide 11 — Research Objective

### Title
**Research Objective**

### Main objective

> **To investigate and compare alternative multi-objective optimization strategies for searching the syftr RAG configuration space under controlled experimental conditions.**

### Specific objectives

1. Reproduce the original syftr optimization baseline.
2. Implement NSGA-II on the same search space.
3. Develop a hybrid NSGA-II + Bayesian strategy.
4. Develop an adaptive / learning-guided NSGA-II strategy.
5. Compare Pareto-front quality.
6. Compare search efficiency.
7. Analyze convergence and exploration/exploitation behavior.

---

# SECTION 7 — WHY NSGA-II?

## Slide 12 — Why Investigate NSGA-II?

### Title
**Why Investigate NSGA-II?**

### NSGA-II provides a different search paradigm.

Instead of building a probabilistic surrogate model, it uses:

- Population-based search
- Non-dominated sorting
- Crowding distance
- Selection
- Crossover
- Mutation

### Concept

```text
Population
   ↓
Evaluate
   ↓
Non-Dominated Sorting
   ↓
Diversity Preservation
   ↓
Selection
   ↓
Crossover + Mutation
   ↓
New Population
```

### Research motivation

This gives us an opportunity to investigate how **population-based Pareto search** behaves on the same configuration problem handled by syftr's Bayesian approach.

---

## Slide 13 — MO-TPE vs NSGA-II

### Title
**Two Different Search Paradigms**

| Aspect | syftr MO-TPE | NSGA-II |
|---|---|---|
| Main paradigm | Bayesian / model-based | Evolutionary / population-based |
| Search mechanism | Density/model-guided | Population evolution |
| Candidate generation | Model-guided | Crossover + mutation |
| Pareto handling | Multi-objective acquisition | Non-dominated sorting |
| Diversity mechanism | Acquisition/search model | Crowding distance |
| Search history | Surrogate model | Population/archive |
| Main experimental question | How does model-based search behave? | How does evolutionary search behave? |

### Important

This is a **methodological comparison**, not a claim that one method is universally superior.

---

# SECTION 8 — OUR THREE PROPOSED STRATEGIES

## Slide 14 — Strategy 1: NSGA-II

### Title
**Proposed Strategy 1 — NSGA-II**

### Algorithm

```text
Initialize Population
        ↓
Evaluate Configurations
        ↓
Non-Dominated Sorting
        ↓
Crowding Distance
        ↓
Selection
        ↓
Crossover
        ↓
Mutation
        ↓
Evaluate Offspring
        ↓
Select Next Generation
        ↓
Repeat
```

### Goal

Investigate the behavior of an evolutionary multi-objective optimizer on the syftr search space.

---

## Slide 15 — Strategy 2: Hybrid NSGA-II + Bayesian Optimization

### Title
**Proposed Strategy 2 — Hybrid NSGA-II + Bayesian Optimization**

### Concept

```text
                 NSGA-II
                    ↓
          Population Exploration
                    ↓
          Identify Promising Areas
                    ↓
          Bayesian Guidance
                    ↓
        Generate / Refine Candidates
                    ↓
             Next Population
```

### Motivation

Combine two forms of search:

**Evolutionary exploration**
+
**Model-guided exploitation**

### Important

The exact hybrid mechanism is an implementation/research decision and should be experimentally defined rather than presented as already established.

---

## Slide 16 — Strategy 3: Adaptive / Learning-Guided NSGA-II

### Title
**Proposed Strategy 3 — Adaptive NSGA-II**

### Basic idea

Standard NSGA-II uses predefined evolutionary parameters.

Our adaptive version can use search history to modify the search behavior.

```text
Evaluate Population
        ↓
Observe Progress
        ↓
Learn From Search History
        ↓
Adjust Search Parameters
        ↓
Change Exploration / Exploitation
        ↓
Generate Next Population
```

### Possible adaptive components

- Mutation rate
- Crossover rate
- Exploration intensity
- Exploitation intensity
- Candidate-generation behavior
- Diversity control

---

# SECTION 9 — RESEARCH DESIGN

## Slide 17 — Four Optimization Strategies

### Title
**Controlled Algorithmic Comparison**

```text
                    SAME SYFTR SEARCH SPACE
                             │
       ┌─────────────┬───────┼────────┬─────────────┐
       ↓             ↓       ↓        ↓
     MO-TPE       NSGA-II  Hybrid   Adaptive
     Baseline     Proposed  NSGA-II  NSGA-II
                              + BO
       │             │       │        │
       └─────────────┴───────┴────────┘
                             ↓
                    Same RAG Evaluation
                             ↓
                     Same Objectives
                             ↓
                      Compare Results
```

### Core principle

> **Only the optimization strategy is changed.**

---

# SECTION 10 — EXPERIMENTAL RESEARCH DESIGN

## Slide 18 — Same Search Space, Different Optimizers

### Title
**Controlled Experimental Setup**

### Keep constant

| Component | Setting |
|---|---|
| Search Space | Same syftr search space |
| RAG Pipeline | Same |
| Dataset | Same |
| LLM | Same |
| Retriever | Same |
| Embedding Model | Same |
| Objectives | Same |
| Evaluation Procedure | Same |
| Evaluation Budget | Same |
| Number of Runs | Controlled |
| Optimization Strategy | **Changed** |

### Central statement

> **The optimization strategy is the primary independent variable.**

---

# SECTION 11 — OPTIMIZATION STRATEGIES UNDER STUDY

## Slide 19 — Baseline and Proposed Methods

### Title
**Optimization Methods Being Compared**

### Reference baseline

**Original syftr MO-TPE**

### Proposed methods

1. **NSGA-II**
2. **Hybrid NSGA-II + Bayesian Optimization**
3. **Adaptive / Learning-Guided NSGA-II**

### Optional simple baselines

- Random Search
- Structured/Grid Search, if computationally feasible

### Important

MO-TPE must remain a primary baseline because it is the optimization strategy used by the base paper.

---

# SECTION 12 — FAIR COMPARISON

## Slide 20 — Fair Experimental Comparison

### Title
**How Will We Ensure a Fair Comparison?**

### All methods use:

```text
Same Search Space
        ↓
Same RAG Pipeline
        ↓
Same Dataset
        ↓
Same Objectives
        ↓
Same Evaluation Budget
        ↓
Same Evaluation Procedure
        ↓
Multiple Independent Runs
        ↓
Statistical Analysis
```

### Most important constraint

**Equal RAG evaluation budget**

Example:

```text
MO-TPE     → B evaluations
NSGA-II    → B evaluations
Hybrid     → B evaluations
Adaptive   → B evaluations
```

This avoids giving one optimizer an unfair computational advantage.

---

# SECTION 13 — EVALUATION METRICS

## Slide 21 — How Will We Evaluate the Optimizers?

### Title
**Evaluation Metrics**

### Pareto-front quality

#### Hypervolume (HV)

Measures objective-space coverage of the obtained Pareto front relative to a reference point.

#### IGD

Measures the distance between the obtained Pareto front and a reference Pareto front.

#### Number of Non-Dominated Solutions

Measures the size of the discovered non-dominated set.

### Search efficiency

- Number of RAG evaluations
- Runtime
- Convergence speed
- Performance vs evaluation budget

### Core principle

We evaluate both:

> **What solutions were found?**

and

> **How efficiently were they found?**

---

# SECTION 14 — EXPERIMENTAL WORKFLOW

## Slide 22 — Complete Experimental Workflow

### Title
**Experimental Workflow**

```text
             SYFTR SEARCH SPACE
                     ↓
            Generate Configuration
                     ↓
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
     MO-TPE       NSGA-II       Hybrid/Adaptive
       ↓             ↓              ↓
       └─────────────┼──────────────┘
                     ↓
                RAG Pipeline
                     ↓
              Evaluate Quality
                     +
               Measure Cost
                     ↓
              Store Trial Result
                     ↓
              Update Optimizer
                     ↓
                  Repeat
                     ↓
              Pareto Frontier
                     ↓
           HV / IGD / Convergence
                     ↓
            Algorithm Comparison
```

### Important

Every optimizer ultimately interacts with the same RAG evaluation function.

---

# SECTION 15 — RESEARCH QUESTIONS

## Slide 23 — Research Questions

### RQ1

**How does NSGA-II perform compared with the original MO-TPE strategy used by syftr when searching the same RAG configuration space?**

### RQ2

**Can a hybrid NSGA-II + Bayesian Optimization strategy improve Pareto-front discovery or search efficiency compared with the individual strategies?**

### RQ3

**Can an adaptive/learning-guided NSGA-II strategy improve exploration and exploitation during optimization?**

### RQ4

**How do different optimization strategies affect the quality-cost trade-off of the discovered RAG configurations?**

### RQ5

**How does optimization efficiency change when all strategies are given the same evaluation budget?**

---

# SECTION 16 — BASELINES

## Slide 24 — Baseline Design

### Title
**Baseline and Comparison Methods**

| Category | Method | Purpose |
|---|---|---|
| Base-paper baseline | MO-TPE | Reproduce syftr |
| Simple baseline | Random Search | Reference for uninformed exploration |
| Optional baseline | Grid/Structured Search | Reference if feasible |
| Proposed | NSGA-II | Evolutionary search |
| Proposed | Hybrid NSGA-II + BO | Combined search |
| Proposed | Adaptive NSGA-II | Learning-guided search |

### Message

The most important comparison is:

**syftr MO-TPE vs our proposed optimization strategies.**

---

# SECTION 17 — RESULT ANALYSIS

## Slide 25 — What Will We Analyze?

### Title
**Result Analysis**

### A. Pareto Front

Compare:

- Coverage
- Convergence
- Diversity
- Quality-cost trade-off

### B. Hypervolume

Plot:

**HV vs Number of Evaluations**

### C. Convergence

Plot optimization progress over evaluations.

### D. Efficiency

Compare:

- RAG evaluations
- Runtime
- Objective improvement

### Example

```text
Hypervolume
    ↑
    │              ─────
    │           ───
    │        ───
    │     ───
    │  ───
    └────────────────────→ Evaluations
```

---

# SECTION 18 — ABLATION & ALGORITHM ANALYSIS

## Slide 26 — Understanding Algorithm Behavior

### Title
**Ablation and Algorithm Analysis**

The objective is not only to compare final results.

We also want to understand:

> **Why do the optimization strategies behave differently?**

### NSGA-II

Analyze:

- Population size
- Mutation rate
- Crossover rate
- Number of generations

### Hybrid

Analyze:

- Bayesian intervention frequency
- Number of Bayesian candidates
- Candidate injection strategy
- Acquisition strategy

### Adaptive

Analyze:

- Adaptation frequency
- Adaptation signal
- Mutation adaptation
- Crossover adaptation
- Exploration/exploitation policy

---

# SECTION 19 — RESEARCH CONTRIBUTION

## Slide 27 — Expected Research Contribution

### Title
**Research Contribution**

### Contribution 1
**Controlled Optimization-Strategy Comparison**

Compare syftr's MO-TPE with alternative optimization strategies.

### Contribution 2
**Evolutionary Search within the syftr Framework**

Investigate NSGA-II without changing the underlying search space.

### Contribution 3
**Hybrid Optimization**

Investigate combining evolutionary search with Bayesian guidance.

### Contribution 4
**Adaptive Optimization**

Investigate learning-guided changes in evolutionary search.

### Contribution 5
**Empirical Analysis**

Compare:

- Pareto-front quality
- Convergence
- Evaluation efficiency
- Runtime
- Quality-cost trade-offs

---

# SECTION 20 — DELIVERABLES

## Slide 28 — Final Deliverables

### Title
**Project Deliverables**

```text
1. Reproduce syftr MO-TPE baseline
                ↓
2. Implement NSGA-II
                ↓
3. Implement Hybrid NSGA-II + Bayesian
                ↓
4. Implement Adaptive NSGA-II
                ↓
5. Common evaluation framework
                ↓
6. Pareto-front comparison
                ↓
7. Quantitative evaluation
                ↓
8. Ablation studies
                ↓
9. Optimization behavior analysis
```

### Final output

A systematic empirical comparison of optimization strategies for RAG configuration search.

---

# SECTION 21 — FUTURE SCOPE

## Slide 29 — Future Scope

### Optimization

- Other evolutionary algorithms
- Other Bayesian optimization methods
- Surrogate-assisted evolutionary algorithms
- Other learning-guided optimization strategies

### RAG

- Additional datasets
- Additional LLMs
- Additional retrievers
- Additional RAG components

### Objectives

Potential future objectives:

- Accuracy
- Token cost
- Latency
- Monetary cost
- Retrieval quality
- Context length

### Important

These are future extensions and are not part of the controlled core experiment unless implemented.

---

# SECTION 22 — CONCLUSION

## Slide 30 — Conclusion

### Title
**Conclusion**

### Final story

```text
RAG has a large configuration space
             ↓
syftr formulates it as multi-objective optimization
             ↓
syftr uses MO-TPE / Bayesian Optimization
             ↓
We keep the same syftr search space
             ↓
We change the optimization strategy
             ↓
NSGA-II
Hybrid NSGA-II + Bayesian
Adaptive NSGA-II
             ↓
Same RAG evaluation
             ↓
Same objectives
             ↓
Same evaluation budget
             ↓
Compare Pareto quality + search efficiency
```

### Final statement

> **Our research investigates how different optimization strategies search the same syftr RAG configuration space and how this affects Pareto-front quality, convergence, and evaluation efficiency.**

---

# FINAL PRESENTATION STORY

The entire PPT should communicate one continuous story:

```text
                    PROBLEM
                       ↓
        RAG has many configuration choices
                       ↓
             Huge Search Space
                       ↓
        Multi-Objective Optimization
                       ↓
                 LITERATURE
                       ↓
                    syftr
                       ↓
       syftr uses MO-TPE / Bayesian Search
                       ↓
               RESEARCH GAP
                       ↓
   What happens if we change the optimizer?
                       ↓
             KEEP EVERYTHING SAME
                       ↓
             Same syftr Search Space
             Same RAG Pipeline
             Same Objectives
             Same Evaluation Budget
                       ↓
              CHANGE ONLY OPTIMIZER
                       ↓
       ┌────────┬──────────┬──────────┐
       ↓        ↓          ↓          ↓
     MO-TPE  NSGA-II   Hybrid BO   Adaptive
    Baseline           + NSGA-II   NSGA-II
       └────────┴──────────┴──────────┘
                       ↓
              CONTROLLED EXPERIMENT
                       ↓
             Pareto Front Quality
             Convergence
             Hypervolume
             IGD
             Runtime
             Evaluation Efficiency
                       ↓
               RESEARCH FINDINGS
```

# One-Sentence Project Definition

> **We keep the syftr RAG search space fixed and investigate whether different multi-objective optimization strategies—NSGA-II, hybrid NSGA-II + Bayesian Optimization, and adaptive NSGA-II—exhibit different Pareto-search quality and efficiency compared with syftr's original MO-TPE strategy.**

# Presentation Design Principle

The PPT should consistently distinguish:

**What the base paper does**
→ syftr + MO-TPE

**What our research changes**
→ optimization strategy

**What remains fixed**
→ syftr search space + RAG evaluation

**What we measure**
→ Pareto quality + search efficiency

**What we do not assume**
→ that any proposed optimizer is better before experimentation
