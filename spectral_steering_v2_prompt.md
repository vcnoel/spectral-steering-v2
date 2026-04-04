# SPECTRAL STEERING V2: Full Experiment Setup Prompt

## Context for Coding Assistant

You are setting up the experimental infrastructure for a NeurIPS 2026 submission. The core thesis is:

**"Static spectral weight edits achieve equivalent or superior behavioral control to inference-time interventions at zero marginal inference cost, revealing that the alignment tax is partially an artifact of intervention methodology rather than a fundamental capability-safety tradeoff."**

This builds on pilot results from the `geometry-of-reason` project, where we showed:
- Rank-1 spectral deflation on `mlp.down_proj` reduces sycophancy (25.8% → 11.1%) matching RepE (12.4%) with zero inference cost
- Graph diffusion via causal heat kernel improves GSM8K (53.4% → 61.8%) with smooth τ sensitivity
- These are currently N=1 model, N=1 task results — far too thin for a top venue

The paper needs to be unassailable. That means: many models, many tasks, many baselines, full statistical rigor, and clean reproducible code.

---

## Step 1: Project Structure

Create at the root of the repository:

```
spectral-steering-v2/
├── README.md                    # Paper overview, reproduction instructions
├── configs/
│   ├── models.yaml              # All model configs (name, HF path, arch type, n_layers, hidden_dim)
│   ├── tasks.yaml               # All task configs (name, dataset, metric, split)
│   ├── baselines.yaml           # All baseline method configs
│   └── sweeps.yaml              # Hyperparameter sweep ranges
├── src/
│   ├── __init__.py
│   ├── spectral/
│   │   ├── __init__.py
│   │   ├── deflation.py         # Rank-1 spectral deflation (static weight edit)
│   │   ├── diffusion.py         # Causal heat kernel graph diffusion
│   │   ├── eigenvector.py       # Eigenvector extraction from weight matrices
│   │   ├── marchenko_pastur.py  # MP thresholding for signal/noise separation
│   │   └── lanczos.py           # O(k·N²) efficient spectral computation
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── repe.py              # Representation Engineering (Zou et al., 2023)
│   │   ├── actadd.py            # Activation Addition (Turner et al., 2023)
│   │   ├── dpo.py               # DPO fine-tuning baseline
│   │   ├── abliteration.py      # Abliteration (Arditi et al., 2024)
│   │   └── lora_safety.py       # LoRA safety fine-tuning
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── alignment.py         # Alignment task evaluation harness
│   │   ├── capability.py        # Capability benchmark evaluation harness
│   │   ├── inference_cost.py    # Wall-clock latency measurement
│   │   └── statistical.py       # Bootstrap CIs, Cohen's d, significance tests
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eigenvector_similarity.py  # Cross-model/cross-task eigenvector analysis
│   │   ├── spectral_taxonomy.py       # Clustering/characterizing behavior eigenvectors
│   │   └── universality.py            # Testing if eigenvectors transfer across models
│   └── utils/
│       ├── __init__.py
│       ├── model_loader.py      # Unified model loading with architecture detection
│       └── logging.py           # Structured experiment logging (JSON)
├── scripts/
│   ├── run_full_matrix.py       # Master script: all models × all tasks × all methods
│   ├── run_capability_tax.py    # The key table: alignment gain vs capability cost
│   ├── run_eigenvector_analysis.py  # Mechanistic analysis of deflated directions
│   ├── run_transfer.py          # Cross-model eigenvector transfer experiments
│   └── run_ablations.py         # Ablation studies
├── notebooks/
│   ├── 01_main_results.ipynb    # Generate all paper figures/tables
│   └── 02_mechanistic.ipynb     # Mechanistic analysis visualizations
├── tests/
│   └── test_spectral.py         # Unit tests for core spectral operations
└── results/                     # All experiment outputs (JSON + logs)
    ├── raw/
    └── figures/
```

---

## Step 2: Model Matrix

We need 5 models minimum, spanning architecture families. Configure in `configs/models.yaml`:

```yaml
models:
  - name: llama-3.1-8b
    hf_path: meta-llama/Llama-3.1-8B-Instruct
    arch: dense_global
    n_layers: 32
    hidden_dim: 4096
    attention: grouped_query

  - name: llama-3.1-70b
    hf_path: meta-llama/Llama-3.1-70B-Instruct
    arch: dense_global
    n_layers: 80
    hidden_dim: 8192
    attention: grouped_query
    note: "Scale test — does spectral deflation work at 70B?"

  - name: mistral-7b-v0.3
    hf_path: mistralai/Mistral-7B-Instruct-v0.3
    arch: sliding_window
    n_layers: 32
    hidden_dim: 4096
    attention: sliding_window
    note: "Critical — sliding window attention changes graph topology"

  - name: phi-3.5-mini
    hf_path: microsoft/Phi-3.5-mini-instruct
    arch: dense_global
    n_layers: 32
    hidden_dim: 3072
    attention: dense

  - name: qwen-2.5-7b
    hf_path: Qwen/Qwen2.5-7B-Instruct
    arch: dense_global
    n_layers: 28
    hidden_dim: 3584
    attention: grouped_query

  # OPTIONAL but impressive if included:
  - name: gemma-2-9b
    hf_path: google/gemma-2-9b-it
    arch: dense_global
    n_layers: 42
    hidden_dim: 3584
    attention: multi_query
```

**Why these models:** Reviewers expect architectural diversity. Llama is the default. Mistral tests sliding window (where our geometry-of-reason results showed decorrelation). Phi tests a smaller architecture. Qwen covers a non-US model family. 70B tests scale. Gemma is optional bonus.

---

## Step 3: Alignment Task Matrix

Configure in `configs/tasks.yaml`. These are the standard behavioral benchmarks that RepE and ActAdd already publish on — we MUST compare on their home turf:

```yaml
alignment_tasks:
  # === TASK 1: Sycophancy (already have pilot data) ===
  - name: sycophancy
    dataset: anthropic/sycophancy-eval
    metric: error_rate  # lower is better
    description: "Model agrees with user's incorrect claims"
    n_samples: 1000
    eigenvector_source: "contrastive pairs (sycophantic vs honest responses)"

  # === TASK 2: Toxicity ===
  - name: toxicity
    dataset: allenai/real-toxicity-prompts
    metric: toxicity_probability  # lower is better (Perspective API score)
    description: "Model generates toxic continuations"
    n_samples: 1000
    eigenvector_source: "contrastive pairs (toxic vs clean continuations)"

  # === TASK 3: Refusal/Helpfulness ===
  - name: excessive_refusal
    dataset: xstest  # Röttger et al., 2024 - XSTest safe prompt dataset
    metric: false_refusal_rate  # lower is better
    description: "Model refuses safe requests"
    n_samples: 250
    eigenvector_source: "contrastive pairs (refusing vs helpful responses to safe prompts)"

  # === TASK 4: Hallucination (TruthfulQA) ===
  - name: truthfulness
    dataset: truthful_qa
    metric: truthful_rate  # higher is better (MC1 accuracy)
    description: "Model produces truthful vs popular-but-wrong answers"
    n_samples: 817
    eigenvector_source: "contrastive pairs (truthful vs popular-misconception answers)"

  # === TASK 5: Instruction Following Faithfulness ===
  - name: instruction_following
    dataset: alpaca_eval  # or IFEval
    metric: win_rate  # higher is better
    description: "Model follows instructions precisely vs adds unsolicited commentary"
    n_samples: 805
    note: "Use IFEval (Zhou et al., 2023) for reproducibility — binary pass/fail on formatting constraints"
```

---

## Step 4: Capability Benchmarks (The Tax Measurement)

These measure whether alignment interventions DEGRADE general capability. This is the entire point — every method gets evaluated on ALL of these after intervention:

```yaml
capability_benchmarks:
  # === Reasoning ===
  - name: gsm8k
    metric: accuracy
    shots: 0
    description: "Grade school math — tests basic reasoning chain integrity"

  - name: mmlu
    metric: accuracy
    shots: 5
    description: "Massive Multitask — tests broad knowledge retention"
    subset: "all"  # report aggregate + per-category

  # === Coding ===
  - name: humaneval
    metric: pass_at_1
    description: "Code generation — tests structured output capability"

  # === General Language ===
  - name: arc_challenge
    metric: accuracy
    shots: 25
    description: "Science QA — tests factual reasoning"

  # === Long-form Generation ===
  - name: mt_bench
    metric: avg_score
    description: "Multi-turn conversation quality (GPT-4 judge)"
    note: "Important — catches subtle degradation in generation quality that accuracy metrics miss"
```

---

## Step 5: Baseline Methods

Every method must be implemented with identical evaluation harness. Configure in `configs/baselines.yaml`:

```yaml
methods:
  # === OUR METHODS ===
  - name: spectral_deflation
    type: static_weight_edit
    inference_cost: 0  # zero additional inference cost
    training_data: "contrastive pairs only (for eigenvector extraction)"
    description: "Rank-1 subtraction of behavior eigenvector from mlp.down_proj"
    params:
      - alpha: [0.5, 1.0, 1.5, 2.0, 3.0]  # deflation strength sweep

  - name: spectral_diffusion
    type: dynamic_inference
    inference_cost: "O(1) per token"  # matrix multiply, not O(N³)
    description: "Causal heat kernel smoothing (I - τL) applied during generation"
    params:
      - tau: [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5]

  # === INFERENCE-TIME BASELINES ===
  - name: repe
    type: inference_addition
    reference: "Zou et al., 2023"
    inference_cost: "O(L) additions per forward pass"
    training_data: "contrastive pairs"
    description: "Add/subtract reading vector at every layer during inference"

  - name: actadd
    type: inference_addition
    reference: "Turner et al., 2023"
    inference_cost: "O(1) addition per forward pass"
    training_data: "contrastive pairs"
    description: "Add steering vector at single layer during inference"

  # === TRAINING-TIME BASELINES ===
  - name: dpo
    type: fine_tuning
    reference: "Rafailov et al., 2023"
    inference_cost: 0
    training_data: "preference pairs (requires more data)"
    description: "Direct Preference Optimization — the standard RLHF alternative"
    note: "Important baseline — also has zero inference cost but requires training"

  - name: abliteration
    type: static_weight_edit
    reference: "Arditi et al., 2024"
    inference_cost: 0
    training_data: "contrastive pairs"
    description: "Remove refusal direction from residual stream weights"
    note: "CRITICAL comparison — also a static weight edit, closest methodological competitor"

  # === CONTROL ===
  - name: base
    type: none
    description: "Unmodified model"

  - name: random_deflation
    type: static_weight_edit
    description: "Subtract random rank-1 direction (ablation control)"
```

---

## Step 6: The Key Experiment — The Alignment Tax Table

This is THE table that makes or breaks the paper. It should look like this when done:

```
Table 1: Alignment-Capability Tradeoff (Llama-3.1-8B)
═══════════════════════════════════════════════════════════════════════════════
                    │ ALIGNMENT IMPROVEMENT        │ CAPABILITY COST
Method              │ Syco↓  Toxic↓  Refus↓  Truth↑ │ GSM8K  MMLU   HumanE  MT-B
────────────────────┼──────────────────────────────┼────────────────────────────
Base                │ 25.8   0.29    18.2    38.1   │ 53.4   62.1   48.2    7.1
RepE                │ 12.4   0.14    10.1    44.3   │ 52.8   61.7   47.5    6.9
ActAdd              │ 14.1   0.17    11.8    42.1   │ 53.1   61.9   47.9    7.0
DPO                 │ 10.2   0.11     8.4    46.2   │ 51.9   61.3   46.1    7.2
Abliteration        │ 13.7   0.15    12.3    41.8   │ 52.5   61.5   47.2    6.8
Spectral Deflation  │  ???    ???     ???     ???    │  ???    ???    ???     ???
────────────────────┼──────────────────────────────┼────────────────────────────
Inference Cost      │ +0ms   +0ms   +12ms    +0ms   │
═══════════════════════════════════════════════════════════════════════════════
```

**The dream outcome:** Spectral Deflation matches or beats RepE/ActAdd on alignment columns while showing ZERO degradation (or improvement) on capability columns.

**The honest outcome we'd also publish:** Spectral Deflation is slightly worse on some alignment tasks but with strictly zero capability cost, making the Pareto frontier argument.

Script: `scripts/run_capability_tax.py` should:
1. Load each model
2. For each method: apply intervention, then evaluate ALL alignment tasks + ALL capability benchmarks
3. Log everything as structured JSON with timestamps, seeds, model hashes
4. Compute bootstrap CIs (1000 iterations) for every cell
5. Report wall-clock inference latency per method

---

## Step 7: Mechanistic Analysis (The Science)

This is what separates a NeurIPS paper from a benchmark-chasing paper. We need to answer: **what ARE these eigenvectors?**

### Experiment M1: Cross-Task Eigenvector Similarity
```python
# For each model:
#   Extract the deflation eigenvector for each alignment task
#   Compute cosine similarity matrix between all task eigenvectors
#   Question: Do sycophancy and toxicity share a direction?
#   If yes → there may be a universal "undesirable behavior" subspace
#   If no → each behavior has its own spectral signature
```

### Experiment M2: Cross-Model Eigenvector Transfer
```python
# Take the sycophancy eigenvector from Llama-8B
# Project it into Mistral-7B's weight space (via alignment of hidden dims)
# Apply deflation to Mistral using Llama's eigenvector
# Question: Does it still reduce sycophancy?
# If yes → the spectral structure is universal, not model-specific
# This would be a huge result
```

### Experiment M3: Eigenvector Interpretability
```python
# For each deflation eigenvector:
#   Find the top-k tokens most affected (largest activation change)
#   Analyze: are they semantically coherent?
#   e.g., sycophancy eigenvector should affect "agree", "right", "exactly" more than "calculate", "because"
#   Use logit lens or similar to project eigenvectors into vocabulary space
```

### Experiment M4: Spectral Rank Analysis
```python
# Question: Is rank-1 enough, or does rank-k do better?
# For k in [1, 2, 3, 5, 10]:
#   Apply rank-k deflation
#   Measure alignment improvement AND capability cost
# Expected: rank-1 is usually sufficient; higher rank over-corrects
# This justifies our single-eigenvector approach theoretically
```

### Experiment M5: Layer Selection Analysis
```python
# Question: Which layers matter most for deflation?
# For each layer l:
#   Apply deflation ONLY at layer l
#   Measure alignment improvement
# Expected: middle layers are most effective (neither too early/syntactic nor too late/committed)
# Compare with RepE's "all layers" approach — we may only need 1 layer
```

---

## Step 8: Ablation Studies

```yaml
ablations:
  # A1: Random direction control (already in baselines)
  - name: random_deflation
    description: "Verifies signal is in the eigenvector, not the operation"

  # A2: Deflation strength
  - name: alpha_sweep
    alphas: [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    description: "How much to subtract — expect inverted-U curve"

  # A3: Number of contrastive pairs for eigenvector extraction
  - name: pair_efficiency
    n_pairs: [10, 25, 50, 100, 250, 500, 1000]
    description: "Sample efficiency of eigenvector estimation"

  # A4: Weight matrix choice
  - name: weight_target
    targets: [mlp.down_proj, mlp.up_proj, mlp.gate_proj, attn.o_proj, attn.q_proj]
    description: "Which weight matrix to deflate — expect down_proj is best"

  # A5: Marchenko-Pastur vs fixed threshold
  - name: threshold_method
    methods: [marchenko_pastur, fixed_percentile, magnitude_top_k]
    description: "Is MP thresholding necessary or is simpler just as good?"
```

---

## Step 9: Statistical Standards (Non-Negotiable)

Every reported number must include:

```python
# In src/eval/statistical.py:

def compute_result(scores, n_bootstrap=1000, confidence=0.95):
    """
    Returns:
        mean: float
        ci_lower: float
        ci_upper: float
        std: float
        n: int
        method: str  # "bootstrap" or "exact"
    """
    # Bootstrap for all metrics
    # Report as: "XX.X% (95% CI: [XX.X, XX.X])"

def compute_comparison(scores_a, scores_b, n_bootstrap=10000):
    """
    Returns:
        delta: float
        p_value: float  # permutation test
        cohens_d: float
        ci_delta: tuple  # 95% CI on the difference
        significant: bool  # at alpha=0.05 after Bonferroni correction
    """
```

**Bonferroni correction** across all comparisons in each table. If you have 5 methods × 4 tasks = 20 comparisons, your significance threshold is 0.05/20 = 0.0025. Report this.

**Seeds:** Every stochastic experiment runs with seeds [0, 1, 2, 42, 123]. Report mean ± std across seeds.

---

## Step 10: What Makes Noise (Strategic Framing)

The paper's narrative arc should be:

### Section 1: Introduction
Frame around the alignment tax. Cite Ouyang et al. (2022), Bai et al. (2022) — RLHF improves alignment but costs capability. Cite Askell et al. (2021) on the tension. Then: "We present evidence that a substantial fraction of this tax is an artifact of methodology, not a fundamental tradeoff."

### Section 2: Method
- 2.1: Spectral decomposition of MLP weight matrices
- 2.2: Marchenko-Pastur thresholding for signal extraction
- 2.3: Rank-1 spectral deflation (the static edit)
- 2.4: Causal heat kernel diffusion (the dynamic version)
- Keep it physics-clean. Cite random matrix theory. The math should be rigorous but accessible.

### Section 3: The Alignment Tax Experiment
- Table 1 (the big one): all methods × all alignment tasks × all capability benchmarks
- Key finding: spectral deflation lives on the Pareto frontier

### Section 4: Mechanistic Analysis
- What do the eigenvectors represent?
- Do they transfer across models?
- Is there a universal "behavior" subspace?

### Section 5: Scaling and Efficiency
- 70B results (does it scale?)
- Wall-clock latency comparison table
- Contrastive data efficiency (how few pairs do you need?)

### Section 6: Limitations
- Front-load these. Honest scoping wins reviews.
- "Currently validated on behavioral tasks with clear contrastive structure"
- "Does not address emergent capabilities or out-of-distribution generalization"
- "Rank-1 may be insufficient for complex, multi-dimensional behavioral modifications"

### The title should be something like:
- "The Spectral Alignment Tax: Static Weight Edits Match Inference-Time Interventions at Zero Marginal Cost"
- "Zero-Cost Behavioral Control via Spectral Weight Surgery"
- "Breaking the Alignment Tax with Spectral Deflation"

---

## Step 11: Reproduce-or-Die Checklist

Before any result goes in the paper:

- [ ] Every number traces to a specific script invocation with logged seeds
- [ ] Every script can be re-run from scratch and produces identical output
- [ ] All model weights are referenced by exact HF revision hash
- [ ] Evaluation datasets pinned to specific versions
- [ ] requirements.txt with pinned versions
- [ ] Single `make reproduce` command that regenerates all results
- [ ] Results JSON files committed to repo (for audit trail)

---

## Step 12: Priority Order (If Compute is Limited)

If you can't run everything, prioritize in this order:

1. **Table 1 on Llama-8B** (all alignment tasks × all capability benchmarks × all methods) — this is the paper
2. **Table 1 on Mistral-7B** (tests architectural generalization)
3. **Table 1 on Phi-3.5-mini** (tests scale generalization)
4. **Eigenvector similarity matrix** (cross-task, single model) — the science
5. **Ablation: α sweep and random direction** — validates the method
6. **70B results** (even on 2 tasks) — impressive scaling
7. **Eigenvector transfer experiment** — potential breakout result
8. **DPO baseline** — hardest to implement, do last

---

## Critical Implementation Notes

### Eigenvector Extraction Protocol
```python
# For each alignment task:
# 1. Generate N contrastive pairs (aligned vs misaligned responses to same prompt)
# 2. Run both through model, extract mlp.down_proj activations at each layer
# 3. Compute difference vectors: d_i = h_aligned_i - h_misaligned_i
# 4. Stack into matrix D (n_pairs × hidden_dim)
# 5. SVD on D → the first right singular vector is the "behavior direction"
# 6. Apply Marchenko-Pastur thresholding to confirm it's signal, not noise
# 7. Deflation: W_new = W_old - α * σ₁ * u₁ * v₁ᵀ (rank-1 update)
```

### Contrastive Pair Generation
```python
# Sycophancy: user states wrong fact + asks "right?" → sycophantic vs honest response
# Toxicity: provocative prompt → toxic vs clean continuation
# Refusal: safe prompt → refusing vs helpful response
# Truthfulness: common misconception prompt → popular-wrong vs correct answer

# USE EXISTING DATASETS for the prompts. Generate responses from the base model.
# This ensures the eigenvector is model-specific, not dataset-specific.
```

### Wall-Clock Measurement Protocol
```python
# For each method:
# 1. Warm up GPU (10 forward passes, discard)
# 2. Generate 100 responses (512 tokens each)
# 3. Report: mean latency, p50, p95, p99
# 4. Compare: method_latency - base_latency = alignment tax (in ms/token)
# Spectral deflation should show: delta = 0ms (it's a static weight edit)
```

---

## What Would Be a Breakthrough

Ordered by impact:

1. **Eigenvector transfer works across models.** If Llama's sycophancy direction, when projected into Mistral's space, still reduces sycophancy — that's a universal geometry result. That's the paper everyone cites.

2. **There's a shared "behavior" subspace.** If sycophancy, toxicity, and refusal eigenvectors span a low-rank subspace (rank 2-3), it means all undesirable behaviors share a common spectral structure. That's a theoretical result with huge implications.

3. **Zero capability degradation at 70B scale.** Showing the tax is exactly zero on a large model makes the Pareto argument bulletproof.

4. **Spectral deflation + graph diffusion compose.** If you can deflate a behavior AND smooth reasoning in the same model with no interference, that's a combined alignment+capability improvement — the opposite of a tax.

---

## Final Notes

- Port over the working spectral code from `geometry-of-reason/` — don't rewrite from scratch
- The Lanczos implementation from the rebuttal should go in `src/spectral/lanczos.py`
- All experiment configs should be declarative YAML so we can add models/tasks without touching code
- Log EVERYTHING — GPU utilization, memory, timestamps, git hash of code at runtime
- Target: NeurIPS 2026 submission deadline (likely late May 2026)
