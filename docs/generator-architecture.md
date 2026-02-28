# Generator Module — Architecture & How It Works

## Overview

The generator module (`g3nat/models/generator.py`) contains **`SequenceOptimizer`** — learns DNA sequences that maximize the change in electrical transmission upon hybridization (complementary strand binding). Uses a differentiable GNN predictor with Softmax Straight-Through gradient flow and adaptive entropy. Based on Fast SeqProp (Linder & Seelig, 2021).

For ground-truth physics evaluation, use `create_hamiltonian()` and `calculate_NEGF()` from `g3nat.utils.physics` directly.

```
                        GOAL
    ┌─────────────────────────────────────────────┐
    │  Find sequence S such that:                 │
    │                                             │
    │  T(single-stranded S) ≠ T(double-stranded S)│
    │                                             │
    │  Maximize: ||T_single - T_double||₁         │
    └─────────────────────────────────────────────┘
```

### References

- Linder, J., & Seelig, G. (2021). Fast activation maximization for molecular sequence design. *BMC Bioinformatics*, 22, 510. https://doi.org/10.1186/s12859-021-04437-5

---

## SequenceOptimizer

### How It Works

The optimizer maintains three sets of **learnable parameters** optimized jointly via Adam against a frozen GNN predictor:

- **Logits** `[N, 4]` — per-position nucleotide preferences
- **Gamma** `[4]` — per-channel scale (controls sampling entropy adaptively)
- **Beta** `[4]` — per-channel offset

```
┌──────────────────────────────────────────────────────────────────┐
│                      SequenceOptimizer                           │
│                                                                  │
│  ┌──────────────────┐     ┌────────────────────────────────┐    │
│  │  Learnable Params │     │  Frozen Predictor               │    │
│  │                   │     │  (DNATransportGNN or            │    │
│  │  logits [N, 4]   │     │   DNATransportHamiltonianGNN)   │    │
│  │  gamma  [4]      │     │                                 │    │
│  │  beta   [4]      │     │  Graph ──► GNN ──► T(E)        │    │
│  │                   │     │                                 │    │
│  │  Instance Norm    │     │                                 │    │
│  │  ──► Softmax ST   │     │                                 │    │
│  │  ──► one-hot      │     │                                 │    │
│  └──────────────────┘     └────────────────────────────────┘    │
│           │                          ▲                           │
│           ▼                          │                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Graph Construction                          │    │
│  │  one-hot ──► sequence_to_graph() topology                │    │
│  │           ──► replace node features with ST one-hot      │    │
│  │           ──► build single-stranded & double-stranded    │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Loss = -||T_single(E) - T_double(E)||₁                 │    │
│  │  (minimize negative = maximize difference)               │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Instance Normalization + Softmax ST

The key innovation from Fast SeqProp: per-channel instance normalization with learnable scale (γ) prevents logit drift, while γ adaptively controls sampling entropy — replacing manual temperature schedules.

```
    Learnable Params                          Output

    logits [N, 4]                         ┌──────────────┐
         │                                │              │
         ▼                                │  Instance    │
    Instance Norm                         │  Norm + γ,β  │
    per-channel across positions          │      │       │
    μ_j = mean(logits[:, j])              │      ▼       │
    σ_j = std(logits[:, j])               │  Softmax     │
         │                                │      │       │
         ▼                                │      ▼       │
    (logits - μ) / σ                      │  Categorical │
         │                                │  sample      │
         ▼                                │      │       │
    * gamma + beta                        │      ▼       │
    (γ controls entropy)                  │  one-hot     │
                                          │  [N, 4]      │
    gamma [4] ─── learnable               └──────────────┘
    beta  [4] ─── learnable

    N = sequence length
    4 channels = [A, T, G, C]
```

### Softmax Straight-Through Estimator

The forward pass samples **discrete one-hot** vectors from the categorical distribution (keeping the predictor in-distribution). The backward pass substitutes the **softmax gradient** for the non-differentiable sampling step.

```
    FORWARD PASS                       BACKWARD PASS
    ┌──────────────────────┐          ┌──────────────────────┐
    │ scaled_logits        │          │ Gradients flow through│
    │        │             │          │ continuous softmax     │
    │        ▼             │          │                        │
    │   softmax(scaled)    │          │ ∂L/∂logits computed   │
    │        │             │          │ via softmax Jacobian   │
    │        ▼             │          │                        │
    │   Categorical sample │          │ ∂L/∂γ adapts entropy  │
    │        │             │          │ ∂L/∂β shifts channels │
    │        ▼             │          │                        │
    │   [1, 0, 0, 0]      │          │ All updated by Adam   │
    │   (discrete one-hot) │          └──────────────────────┘
    │                      │
    │ Predictor sees hard  │
    │ one-hot (in-distrib.)│
    └──────────────────────┘
```

### Adaptive Entropy (γ)

Instead of a fixed temperature schedule, the learnable scale parameter γ controls sampling entropy per channel. Adam optimizes γ alongside logits:

```
    γ_j large  →  scaled logits spread out  →  sharper softmax  →  lower entropy
    γ_j small  →  scaled logits compressed  →  flatter softmax  →  higher entropy

    Adaptive behavior:
    - When gradient signals are consistent: γ grows (exploitation)
    - When gradient signals are inconsistent: γ shrinks (exploration)
```

### Training vs Eval Mode

```
    TRAINING MODE                          EVAL MODE
    ┌──────────────────────┐              ┌──────────────────────┐
    │ Instance Norm logits │              │ Instance Norm logits │
    │        │             │              │        │             │
    │        ▼             │              │        ▼             │
    │   softmax ──►        │              │   argmax ──► one_hot │
    │   categorical sample │              │   [1, 0, 0, 0]      │
    │        │             │              │                      │
    │        ▼             │              │ Deterministic:       │
    │   [1, 0, 0, 0]      │              │ always same output   │
    │                      │              │ for given logits     │
    │ Stochastic sample    │              │                      │
    │ with ST gradient     │              │ No sampling noise    │
    └──────────────────────┘              └──────────────────────┘
```

### Graph Construction with Soft Features

`sequence_to_graph()` converts a DNA string into a graph with hard one-hot node features. Hard one-hot values have **no gradient**, so the optimizer can't learn through them.

**Solution:** Use `sequence_to_graph()` for the graph **topology** only (edges, contacts), then swap in the straight-through one-hot features for the primary strand:

```
    Step 1: Get topology from dummy sequence
    ┌─────────────────────────────────────────┐
    │  sequence_to_graph("AAAA")              │
    │                                         │
    │  Returns:                               │
    │    Nodes: [contact_L, contact_R,        │
    │            A, A, A, A]                  │
    │    Edges: backbone + contact edges      │
    │    Features: HARD one-hot (no gradient) │
    └─────────────────────────────────────────┘
                        │
                        ▼
    Step 2: Replace primary strand features
    ┌─────────────────────────────────────────┐
    │  Node 0 (left contact):   [0,0,0,0]    │  ← keep (not optimized)
    │  Node 1 (right contact):  [0,0,0,0]    │  ← keep (not optimized)
    │  Node 2 (position 0):    one_hot[0]    │  ← REPLACE (has grad)
    │  Node 3 (position 1):    one_hot[1]    │  ← REPLACE (has grad)
    │  Node 4 (position 2):    one_hot[2]    │  ← REPLACE (has grad)
    │  Node 5 (position 3):    one_hot[3]    │  ← REPLACE (has grad)
    └─────────────────────────────────────────┘
```

### Single-Stranded vs Double-Stranded Graphs

Both graphs are built per optimization step. The primary strand uses soft features (differentiable); the complement uses hard features (not optimized).

```
    SINGLE-STRANDED (no complement)         DOUBLE-STRANDED (with complement)
    ┌───────────────────────────┐           ┌───────────────────────────────────┐
    │                           │           │                                   │
    │  [L]──A──T──G──C──[R]    │           │  [L]──A──T──G──C──[R]            │
    │                           │           │       |  |  |  |                  │
    │  6 nodes, ~10 edges       │           │       T──A──C──G                 │
    │  (2 contacts + 4 bases)   │           │                                   │
    │                           │           │  10 nodes, ~24 edges              │
    │  Primary features:        │           │  (2 contacts + 4 primary + 4 comp)│
    │  straight-through one-hot │           │                                   │
    │  (differentiable via ST)  │           │  Primary features: ST one-hot     │
    │                           │           │  Complement features: HARD        │
    │                           │           │  (only primary is optimized)      │
    └───────────────────────────┘           └───────────────────────────────────┘

    [L] = left electrode contact       ── = backbone bond
    [R] = right electrode contact       | = hydrogen bond
```

### Optimization Loop (Single Step)

```
               ┌─────────────────┐
               │  Instance Norm  │
               │  per-channel    │
               │  * γ + β        │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Softmax ST     │
               │  categorical    │
               │  sample         │
               │  ──► one_hot    │
               └────────┬────────┘
                        │
               ┌────────┴────────┐
               │                 │
               ▼                 ▼
      ┌──────────────┐  ┌──────────────────┐
      │  decode to   │  │  Keep one_hot     │
      │  "ATGC"      │  │  (ST gradient)    │
      │  (discrete)  │  │                    │
      └──────┬───────┘  └────────┬───────────┘
             │                   │
             ▼                   │
      ┌───────────────┐         │
      │  reverse      │         │
      │  complement   │         │
      └──────┬────────┘         │
             │                   │
    ┌────────┴───────────────────┤
    │                            │
    ▼                            ▼
┌──────────────────┐   ┌──────────────────┐
│  Build Graph     │   │  Build Graph     │
│  SINGLE-stranded │   │  DOUBLE-stranded │
│  Soft primary    │   │  Soft primary    │
│  No complement   │   │  Hard complement │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│  Frozen Predictor│   │  Frozen Predictor│
│  ──► T_single(E) │   │  ──► T_double(E) │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Loss = -||diff||₁   │
         │  loss.backward()     │
         │  optimizer.step()    │
         │  (updates logits,    │
         │   gamma, and beta)   │
         └──────────────────────┘
```

### Gradient Flow Path

```
    ┌──────────────────────────────────────────────────────────┐
    │                   GRADIENT FLOW                          │
    │                                                          │
    │  Logits      Inst.Norm   Softmax   Graph    Frozen  Loss│
    │  γ, β        + γ, β     ST        Features  Pred.      │
    │                                                          │
    │  θ           scaled     one_hot   data.x   GNN     L   │
    │  │            │          │         │        layers   │   │
    │  │            │          │         │        │        │   │
    │  ◄────────────◄──────────◄─────────◄────────◄────────┘   │
    │       ▲              ▲        ▲        ▲        ▲        │
    │  ∂L/∂θ        ∂L/∂γ,β    ST est.  ∂L/∂x   ∂L/∂T(E)    │
    │  ∂L/∂γ                                                   │
    │  ∂L/∂β        γ adapts   straight- passes   passes      │
    │               entropy    through   through  through      │
    │  ALL updated             gradient          (frozen =     │
    │  by Adam                                    no update,   │
    │                                             but grads    │
    │                                             still flow)  │
    └──────────────────────────────────────────────────────────┘
```

### Loss Function

The loss uses **L1 norm** (sum of absolute differences) rather than L2, because L1 avoids over-weighting isolated resonance peaks in the transmission spectrum — a single sharp peak shouldn't dominate the optimization.

```
    T_single(E) ─────┐
                     │    element-wise
                     ├──► subtraction ──► diff(E)
                     │
    T_double(E) ─────┘

                          diff(E) ──► ||·||₁ ──► negate ──► loss


    With energy mask (optional):

    diff(E) ──► × mask(E) ──► ||·||₁ ──► negate ──► loss

    mask = [1, 1, 0, 0, ..., 0, 1, 1]
            ▲                     ▲
            only these energy     and these
            points contribute     energy points
            to the loss
```

---

## Typical Workflow

```
    ┌─────────────────────────────────┐
    │ 1. Optimize with GNN            │
    │    (fast, differentiable)       │
    │                                 │
    │    SequenceOptimizer            │
    │    + frozen DNATransportGNN     │
    │    ──► optimized sequence       │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ 2. Validate with exact physics  │
    │    (slow, exact)                │
    │                                 │
    │    create_hamiltonian()         │
    │    + calculate_NEGF()           │
    │    ──► ground-truth T(E)        │
    └─────────────────────────────────┘
```

### Example: Optimize and Validate

```python
import numpy as np
import torch
from g3nat.models.generator import SequenceOptimizer
from g3nat.utils.physics import create_hamiltonian, calculate_NEGF
from g3nat.evaluation import load_trained_model

# Step 1: Optimize with GNN
opt = SequenceOptimizer(seq_length=4)
predictor, energy_grid, device = load_trained_model("trained_models/model.pth")

losses = opt.optimize(predictor, num_steps=500, lr=0.1, log_every=100)

# Get optimized sequence
opt.eval()
with torch.no_grad():
    one_hot, _ = opt()
sequence = opt.decode_sequence(one_hot)
complement = opt.get_complement(sequence)

print(f"Optimized: 5'-{sequence}-3'")
print(f"Complement: 5'-{complement}-3'")

# Step 2: Validate against exact physics
energy_grid = np.linspace(-3, 3, 100)

H_s, GL_s, GR_s = create_hamiltonian(sequence, '_' * len(sequence))
trans_single, dos_single = calculate_NEGF(H_s, GL_s, GR_s, energy_grid)

H_d, GL_d, GR_d = create_hamiltonian(sequence, complement)
trans_double, dos_double = calculate_NEGF(H_d, GL_d, GR_d, energy_grid)

score = np.sum(np.abs(np.log10(trans_single + 1e-30) - np.log10(trans_double + 1e-30)))
print(f"Ground-truth score: {score:.4f}")
```

### Using an Energy Mask

```python
# Only optimize transmission difference in the [-1, 0] eV window
energy_grid = torch.linspace(-3, 3, 100)
mask = ((energy_grid >= -1) & (energy_grid <= 0)).float()

losses = opt.optimize(predictor, num_steps=500, energy_mask=mask, log_every=100)
```

---

## API Reference

### SequenceOptimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_length` | int | required | Number of DNA bases to optimize |

| Learnable Parameter | Shape | Init | Description |
|---------------------|-------|------|-------------|
| `logits` | `[N, 4]` | `randn` | Per-position nucleotide logits |
| `gamma` | `[4]` | `ones` | Per-channel scale (adaptive entropy) |
| `beta` | `[4]` | `zeros` | Per-channel offset |

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `()` | `(one_hot [N,4], scaled_logits [N,4])` |
| `decode_sequence` | `(one_hot)` | DNA sequence string |
| `get_complement` | `(sequence: str)` | Watson-Crick reverse complement string |
| `build_graph_with_soft_features` | `(soft_bases, complementary_sequence=None)` | `Data` graph |
| `compute_loss` | `(transmission_single, transmission_double, energy_mask=None)` | scalar loss tensor (negative L1 norm) |
| `optimize` | `(predictor, num_steps, lr=0.1, energy_mask=None, log_every=100)` | `List[float]` of losses |

---

## File Structure

```
g3nat/
  models/
    generator.py          ← SequenceOptimizer
    standard.py           ← DNATransportGNN (predictor option 1)
    hamiltonian.py        ← DNATransportHamiltonianGNN (predictor option 2)
    __init__.py           ← exports all model classes
  graph/
    construction.py       ← sequence_to_graph() used for topology
  utils/
    physics.py            ← create_hamiltonian(), calculate_NEGF() (ground-truth physics)
  __init__.py             ← top-level exports

tests/
  test_models/
    test_generator.py     ← tests for SequenceOptimizer
```
