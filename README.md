# LLM_JSSP_masking

Step-by-step LLM scheduling for Job Shop Scheduling Problems (JSSP) with **decode-time feasibility control**, **self-labeled supervision**, and **RL fine-tuning**.

This repository studies a specific question:

> Can an LLM act as an end-to-end scheduler for JSSP while enforcing feasibility **during decoding**, not only after generation?

The core answer explored here is **yes, at the decision level**:

- represent scheduling as a sequence of feasible step decisions
- decode only from the currently feasible action set
- validate every transition in an explicit scheduling environment
- optimize makespan with supervised learning and RL

## 1. Research Scope

This work focuses on **JSSP step-by-step decision making**, not on optimization modeling or heuristic code generation.

The repository supports two formulations:

- **Serial schedule-construction mode**
  - benchmark-oriented formulation
  - the model chooses the next operation order
- **Dispatch mode**
  - event-driven formulation
  - the model chooses among jobs dispatchable at the current decision epoch

The training/evaluation pipeline includes:

- self-labeled step dataset generation
- supervised fine-tuning for policy / reason / mixed supervision
- constrained inference with feasible-action masking
- RL fine-tuning (`reinforce`, `grpo`, `bopo`)

## 2. Why This Research Matters

Recent LLM-for-CO papers show that LLMs can produce promising solutions, but the main gap is still:

- **feasibility is often improved, not guaranteed**
- many methods rely on **sampling, post-hoc filtering, or repair**
- scheduling-specific objectives such as **makespan** are often handled indirectly
- many LLM papers target **formulation**, **hyper-heuristics**, or **solver generation**, rather than direct online scheduling decisions

This work targets a stricter setting:

- **end-to-end step policy**
- **explicit feasible action set**
- **invalid action probability forced to zero at decode time**
- **makespan-oriented scheduling state and transition features**

## 3. Literature Review: End-to-End LLMs for CO and Scheduling

Discovery source:
- [awesome-fm4co: LLMs for Combinatorial Optimization](https://github.com/ai4co/awesome-fm4co/blob/main/README.md)

Selection rule used here:
- prioritize papers that are
  - end-to-end or close to end-to-end
  - relevant to **feasibility control**
  - relevant to **makespan / scheduling quality**
  - especially relevant to **JSSP / FSSP / scheduling**

### 3.1 Main Comparison Table

| Work | Problem | End-to-end direct solver | Feasibility mechanism | Objective optimization | Hard feasibility at inference | Main limitation relative to this work |
|---|---|---:|---|---|---:|---|
| [LLMs can Schedule (2024)](https://arxiv.org/abs/2408.06993) | JSSP | O | supervised JSSP dataset + sampling | makespan-oriented scheduling | X | JSSP-specific and end-to-end, but no explicit decode-time invalid-action elimination is emphasized in the abstract |
| [Starjob: Dataset for LLM-Driven Job Shop Scheduling (2025)](https://arxiv.org/abs/2503.01877) | JSSP | O | supervised dataset + LoRA fine-tuning | makespan minimization | X | strong scheduling baseline, but feasibility is improved through training rather than enforced by a hard feasible-action decoder |
| [Self-Guiding Exploration for Combinatorial Problems (2024)](https://arxiv.org/abs/2405.17950) | TSP, VRP, BPP, AP, KP, JSSP | O | prompting-based self-guided search | improves optimization performance | X | broad and end-to-end, but no explicit hard constraint guarantee at token/action decoding time |
| [ACCORD (2025)](https://arxiv.org/abs/2506.11052) | TSP, VRP, Knapsack, FlowShop, JSSP, BinPacking | O | autoregressive constraint-satisfying generation + routing/dynamic attention | feasibility-aware output structure, broad CO performance | X | closest prior art in spirit; however the abstract emphasizes improved feasibility rather than an explicit 100% decoder-level guarantee |
| [Large Language Models as End-to-end Combinatorial Optimization Solvers (2025)](https://arxiv.org/abs/2509.16865) | multiple CO tasks | O | SFT + FOARL + Best-of-N | explicitly optimizes feasibility and solution quality | X | feasibility is mitigated via RL and sampling; still depends on multi-sample search rather than hard masking by construction |
| [Hard Constraints Meet Soft Generation / FALCON (2026)](https://arxiv.org/abs/2602.01090) | multiple CO tasks | O | grammar-constrained decoding + repair layer + adaptive BoN | BOPO preference optimization with objective-gap weighting | O | achieves perfect feasibility, but uses repair and adaptive sampling; feasibility is not enforced purely by a stepwise feasible-action mask in a scheduling environment |
| **This work** | **JSSP** | **O (step-level)** | **feasible-action masking + environment transition checks** | **makespan-oriented step policy with SFT/RL** | **O (with respect to the modeled action space)** | **JSSP-focused and requires problem-specific env/mask engineering** |

### 3.2 What the Prior Literature Says

#### Feasibility-oriented works

- **LLM-as-end-to-end CO solvers** shows that feasibility and optimality can be improved jointly with SFT + RL + Best-of-N, but feasibility is still something to be *learned and sampled for*, not strictly blocked at generation time. Source: [arXiv:2509.16865](https://arxiv.org/abs/2509.16865)
- **ACCORD** is one of the most relevant prior works because it explicitly focuses on autoregressive constraint-satisfying generation. It improves feasibility through representation and dynamic routing, but the abstract does not claim a universal 100% theorem-style guarantee. Source: [arXiv:2506.11052](https://arxiv.org/abs/2506.11052)
- **FALCON** is the strongest recent feasibility-oriented baseline among LLM solvers. It explicitly claims 100% feasibility through grammar-constrained decoding, repair, and adaptive Best-of-N sampling. Source: [arXiv:2602.01090](https://arxiv.org/abs/2602.01090)

#### Makespan / scheduling-oriented works

- **LLMs can Schedule** demonstrates that JSSP can be approached directly with LLMs and dataset-based supervision. Source: [arXiv:2408.06993](https://arxiv.org/abs/2408.06993)
- **Starjob** strengthens this line by introducing a dedicated JSSP dataset and showing strong makespan improvements over dispatching rules and neural baselines. Source: [arXiv:2503.01877](https://arxiv.org/abs/2503.01877)
- **ACCORD** is also relevant for makespan-oriented scheduling because it covers both FlowShop and JSSP in an end-to-end framework. Source: [arXiv:2506.11052](https://arxiv.org/abs/2506.11052)

### 3.3 Adjacent Works Not Directly Comparable

Some papers in the `awesome-fm4co` list are important, but they are not the closest baselines for this repository because they are **not direct end-to-end schedulers**.

- formulation/modeling assistants:
  - OptiMUS
  - ORLM
  - LLMOPT
  - Towards Foundation Models for MILP
- hyper-heuristic / algorithm-generation approaches:
  - OPRO
  - ReEvo
  - EoH
  - SeEvo for dynamic JSSP

These are valuable, but they answer a different question from this repository.  
This repository studies **direct decision generation under feasibility constraints**, not optimization modeling or heuristic synthesis.

## 4. Limitations of Existing Research

Based on the papers above, the main limitations of prior work can be summarized as follows.

### 4.1 Feasibility is often improved, not guaranteed

- Many end-to-end LLM solvers improve feasibility through
  - better prompting
  - supervised fine-tuning
  - RL
  - multi-sample selection
- But without hard decode-time blocking, invalid actions can still be generated.

### 4.2 Sampling and repair increase inference cost

- Best-of-N, adaptive BoN, and post-hoc repair can work well,
- but they increase runtime and complicate online decision making.

This matters especially in scheduling and dispatching, where:

- decisions may need to be produced repeatedly
- infeasible actions should ideally be impossible, not merely unlikely

### 4.3 Scheduling state is often treated too globally

- Sequence-level generation can hide the fact that scheduling is a sequence of constrained local decisions.
- For JSSP, the important question is often not only
  - “what is the final sequence?”
- but also
  - “what actions are feasible **now**?”
  - “what action best improves future makespan **from this state**?”

### 4.4 Many works are not aligned with real dispatching settings

- Serial schedule generation is useful for benchmarks,
- but real manufacturing decisions are often event-driven.

That is why this repository explicitly supports both:

- **serial** decision making
- **dispatch** decision epochs

## 5. What Is New in This Work

### 5.1 Feasibility is enforced at decision time

The core differentiator is:

- **invalid actions are removed before sampling**
- not only penalized after generation

Concretely, this work combines:

- feasible-action masking
- environment-backed transition checks
- structured step-by-step scheduling decisions

This means feasibility is controlled **inside decoding**, not only by:

- training losses
- post-hoc filtering
- repair

### 5.2 End-to-end scheduling is reformulated as constrained step policy learning

Instead of generating a full schedule as unconstrained free-form text, this work learns:

- a **step policy**
- conditioned on the current scheduling state
- over the **current feasible action set**

This provides a cleaner bridge between:

- LLM sequence modeling
- scheduling environment dynamics
- RL fine-tuning

### 5.3 Makespan optimization is tied to state transitions

This repository does not only ask the model to produce a valid next action.
It also encodes features related to scheduling quality, such as:

- current completion state
- job/machine readiness
- remaining work
- candidate transition effects
- current or projected makespan-related signals

So the model is trained to choose **feasible** actions that are also **makespan-reducing**.

### 5.4 Serial and dispatch are both supported

This is a practical contribution that is rare in LLM scheduling work:

- **serial mode**
  - good for benchmark-aligned schedule construction
- **dispatch mode**
  - closer to event-driven real-time scheduling

This lets the same framework speak both to:

- academic benchmark evaluation
- practical deployment-oriented decision making

### 5.5 Self-labeling makes large-scale supervised data practical

The repository converts teacher schedules into step-level training data:

- policy-only rows
- reason-only rows
- mixed action+reason rows

This enables:

- scalable supervision
- policy/reason separation
- direct RL fine-tuning after SFT

## 6. Paper-Ready Differentiation Statement

The following wording is appropriate for a paper-positioning section.

### 6.1 One-sentence positioning

> Unlike prior LLM-based CO solvers that primarily improve feasibility through prompting, supervised learning, RL, sampling, or repair, our method enforces feasibility directly at the decision level by masking infeasible actions during decoding and validating every transition in an explicit scheduling environment.

### 6.2 Short paragraph positioning

> Existing end-to-end LLM approaches to combinatorial optimization have demonstrated encouraging gains in feasibility and objective quality, but most treat feasibility as something to be learned, sampled for, or repaired after generation. In contrast, our work reformulates JSSP as a constrained step-by-step decision problem and integrates feasible-action masking directly into decoding. As a result, the model is trained and evaluated not as a free-form sequence generator, but as an end-to-end scheduling policy operating only over valid decisions. This makes the method particularly suitable for makespan-oriented scheduling settings where invalid actions should be impossible rather than merely low-probability.

### 6.3 Difference from the strongest related works

#### vs. LLMs as End-to-end CO Solvers

- They improve feasibility and quality with SFT + RL + Best-of-N.
- We instead make feasibility a **decoder-side structural property** of the policy.

#### vs. ACCORD

- ACCORD is very close in spirit and is one of the most relevant baselines.
- The main difference is that our method uses an explicit scheduling environment and feasible-action masking at each decision step, rather than relying primarily on output structuring and dynamic autoregressive generation to improve feasibility.

#### vs. FALCON

- FALCON is the strongest feasibility-guaranteed LLM baseline.
- The key difference is that FALCON combines constrained decoding with repair and adaptive Best-of-N, whereas our method targets **direct feasible step selection** inside a scheduling environment.
- In short:
  - FALCON: grammar + repair + adaptive sampling
  - ours: feasible action mask + transition-valid environment policy

#### vs. Starjob / LLMs can Schedule

- These works show that JSSP can be solved end-to-end with LLMs.
- Our difference is not merely “use an LLM for JSSP,” but:
  - **decode-time feasibility control**
  - **explicit step environment**
  - **serial/dispatch dual formulation**
  - **policy/RL integration under hard feasible-action restriction**

## 7. Strengths and Weaknesses of This Research

### Strengths

- **Feasibility by construction at the action level**
  - invalid actions are blocked before decoding completes
- **Scheduling-specific**
  - explicitly designed for makespan-oriented JSSP decisions
- **End-to-end**
  - the model still acts directly as a scheduler, not only as a modeling or heuristic assistant
- **RL-compatible**
  - constrained action space integrates naturally with policy improvement
- **Two deployment views**
  - serial benchmark mode
  - dispatch/event-driven mode

### Weaknesses / Honest limitations

- **Problem-specific engineering is required**
  - environment, masking, and prompt/state design are domain-aware
- **Guarantee scope depends on the modeled constraints**
  - if a real-world constraint is not encoded in the environment/mask, it is not guaranteed
- **JSSP-focused**
  - broader CO generalization still requires additional problem-specific work
- **Prompt/state length can become large**
  - especially for larger instances or richer transition features

## 8. Repository Components

### Data generation

- [generate_jssp_step_dataset.py](generate_jssp_step_dataset.py)

### Core environments and prompting

- [llm_jssp/utils/jssp_step_env.py](llm_jssp/utils/jssp_step_env.py)
- [llm_jssp/utils/jssp_dispatch_env.py](llm_jssp/utils/jssp_dispatch_env.py)
- [llm_jssp/utils/step_prompting.py](llm_jssp/utils/step_prompting.py)
- [llm_jssp/utils/step_prompting_dispatch.py](llm_jssp/utils/step_prompting_dispatch.py)
- [llm_jssp/utils/step_reasoning.py](llm_jssp/utils/step_reasoning.py)

### Local source entrypoints

- [unified_trainer.py](unified_trainer.py)
- [inference_jssp_fssp.py](inference_jssp_fssp.py)
- [RL_jssp_fssp.py](RL_jssp_fssp.py)

### Self-contained notebooks

- [notebooks/colab_00_generate_step_dataset.ipynb](notebooks/colab_00_generate_step_dataset.ipynb)
- [notebooks/colab_01_upload_datasets_to_hf.ipynb](notebooks/colab_01_upload_datasets_to_hf.ipynb)
- [notebooks/colab_02_train_step_lora_full.ipynb](notebooks/colab_02_train_step_lora_full.ipynb)
- [notebooks/colab_03_inference_step_full.ipynb](notebooks/colab_03_inference_step_full.ipynb)
- [notebooks/colab_04_upload_model_to_hf.ipynb](notebooks/colab_04_upload_model_to_hf.ipynb)
- [notebooks/colab_05_rl_full.ipynb](notebooks/colab_05_rl_full.ipynb)
- [notebooks/colab_06_rl_compare.ipynb](notebooks/colab_06_rl_compare.ipynb)

## 9. Practical Claim of This Repository

The practical claim of this repository is not:

- “LLMs can solve CO if we sample enough candidates.”

It is instead:

> For JSSP, an LLM can be trained as a step-level scheduling policy that remains end-to-end while enforcing feasibility directly at decode time through a feasible-action mask and explicit environment transitions.

That is the main research distinction of this repository.
