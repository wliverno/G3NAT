# Sequence Optimizer - Architecture & How It Works

## Overview

The Sequence Optimizer learns DNA sequences optimal for **sensing applications**. It finds sequences where hybridization (binding of the complementary strand) produces the largest possible change in electrical transmission — making the sensor maximally sensitive.

This approach is based on **Fast SeqProp** (Linder & Seelig, 2020), which uses direct logit optimization with straight-through Gumbel-Softmax instead of training a generative neural network.

```
                        GOAL
    ┌─────────────────────────────────────────────┐
    │  Find sequence S such that:                 │
    │                                             │
    │  T(single-stranded S) ≠ T(double-stranded S)│
    │                                             │
    │  Maximize: ||T_single - T_double||₂         │
    └─────────────────────────────────────────────┘
```

### References

- Linder, J., & Seelig, G. (2020). Fast activation maximization for molecular sequence design. *BMC Bioinformatics*, 21, 510. https://doi.org/10.1186/s12859-020-03846-2
- Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR 2017*. https://arxiv.org/abs/1611.01144

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      SequenceOptimizer                           │
│                                                                  │
│  ┌──────────────────┐     ┌────────────────────────────────┐    │
│  │  Learnable Logits │     │  Frozen Predictor               │    │
│  │  [N, 4]           │     │  (DNATransportGNN or            │    │
│  │                   │     │   DNATransportHamiltonianGNN)   │    │
│  │  logits ──►       │     │                                 │    │
│  │  Gumbel-Softmax   │     │  Graph ──► GNN ──► T(E)        │    │
│  │  (hard=True)      │     │                                 │    │
│  │  ──► one-hot      │     │                                 │    │
│  └──────────────────┘     └────────────────────────────────┘    │
│           │                          ▲                           │
│           │                          │                           │
│           ▼                          │                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Graph Construction                          │    │
│  │  one-hot ──► sequence_to_graph() topology                │    │
│  │           ──► replace node features                      │    │
│  │           ──► single-stranded & double-stranded          │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Loss = -||T_single(E) - T_double(E)||₂                 │    │
│  │  (minimize negative = maximize difference)               │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Direct Logit Optimization

Unlike a generative model that maps random latent vectors through an MLP, the `SequenceOptimizer` maintains **learnable logits directly** — one 4-dimensional vector per sequence position:

```
    Learnable Logits                    Output
    ┌─────────────────┐            ┌──────────────┐
    │                 │            │              │
    │  logits         │            │  Normalize   │
    │  [N, 4]         │──────────► │  (subtract   │
    │                 │            │   mean)      │
    │  nn.Parameter   │            │      │       │
    │  (directly      │            │      ▼       │
    │   optimized)    │            │  Gumbel-     │
    │                 │            │  Softmax     │
    └─────────────────┘            │  (hard=True) │
                                   │      │       │
                                   │      ▼       │
                                   │  one-hot     │
                                   │  [N, 4]      │
                                   └──────────────┘

    N = sequence length
    4 channels = [A, T, G, C]
```

### Why Direct Logits?

The previous MLP-based approach (`DNASequenceGenerator`) suffered from:
1. **Random z sampling** — new random input each step created extreme variance
2. **Gumbel noise compounding** with z noise at batch_size=1
3. **Soft features out-of-distribution** for the predictor (trained on hard one-hot)

Direct logit optimization eliminates all three issues.

### Straight-Through Gumbel-Softmax

The straight-through estimator (Jang et al., 2017) uses `hard=True`: the forward pass produces discrete one-hot vectors (keeping the predictor in-distribution), while the backward pass computes gradients through the continuous softmax approximation.

```
    FORWARD PASS                       BACKWARD PASS
    ┌──────────────────────┐          ┌──────────────────────┐
    │ logits + Gumbel noise│          │ Gradients flow through│
    │        │             │          │ continuous softmax     │
    │        ▼             │          │ (as if hard=False)     │
    │   gumbel_softmax     │          │                        │
    │   (hard=True)        │          │ ∂L/∂logits computed   │
    │        │             │          │ via soft approximation │
    │        ▼             │          │                        │
    │   [1, 0, 0, 0]      │          │ Logits updated by Adam │
    │   (discrete one-hot) │          └──────────────────────┘
    │                      │
    │ Predictor sees hard  │
    │ one-hot (in-distrib.)│
    └──────────────────────┘
```

### Temperature Annealing

Temperature (tau) controls the sharpness of the Gumbel-Softmax distribution and is linearly annealed during optimization:

```
    tau = 2.0 (start, exploratory):  gradients spread across bases
    tau = 0.1 (end, near-discrete):  gradients concentrated on chosen base

    Linear schedule: tau(t) = tau_start + (tau_end - tau_start) * t / (num_steps - 1)
```

### Training vs Eval Mode

```
    TRAINING MODE                          EVAL MODE
    ┌──────────────────────┐              ┌──────────────────────┐
    │ logits + Gumbel noise│              │ logits               │
    │        │             │              │        │             │
    │        ▼             │              │        ▼             │
    │   gumbel_softmax     │              │   argmax ──► one_hot │
    │   (hard=True)        │              │   [1, 0, 0, 0]      │
    │        │             │              │                      │
    │        ▼             │              │ Deterministic:       │
    │   [1, 0, 0, 0]      │              │ always same output   │
    │                      │              │ for given logits     │
    │ Discrete but with    │              │                      │
    │ Gumbel noise for     │              │ No Gumbel noise      │
    │ exploration          │              │                      │
    └──────────────────────┘              └──────────────────────┘
```

---

## Graph Construction with Soft Features

### The Gradient Flow Problem

`sequence_to_graph()` converts a DNA string into a graph with hard one-hot node features. But hard one-hot values have **no gradient** — the optimizer can't learn.

**Solution:** Use `sequence_to_graph()` only for the graph **topology** (edges, contacts), then swap in the straight-through one-hot features:

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

    Straight-through: forward is hard one-hot,
    backward computes gradients via soft approximation
```

### Single-Stranded vs Double-Stranded Graphs

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

---

## Optimization Loop

### Single Step

```
               ┌─────────────────┐
               │  Anneal tau     │
               │  tau(t) = ...   │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Normalize      │
               │  logits         │
               │  (subtract mean)│
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Gumbel-Softmax │
               │  (hard=True)    │
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
         │  Loss = -||diff||₂   │
         │  loss.backward()     │
         │  optimizer.step()    │
         └──────────────────────┘
```

### Gradient Flow Path

```
    ┌──────────────────────────────────────────────────────────┐
    │                   GRADIENT FLOW                          │
    │                                                          │
    │  Logits      ST Gumbel    Graph        Frozen     Loss  │
    │  [N, 4]      Softmax      Features     Predictor        │
    │                                                          │
    │  θ           one_hot      data.x       GNN        L    │
    │  │            │           │            layers      │    │
    │  │            │           │            │           │    │
    │  ◄────────────◄───────────◄────────────◄───────────┘    │
    │       ▲              ▲          ▲            ▲           │
    │  ∂L/∂θ         ST estimate  ∂L/∂x      ∂L/∂T(E)        │
    │                                                          │
    │  UPDATED        straight-   passes     passes           │
    │  by Adam        through     through    through           │
    │                 gradient              (frozen =          │
    │                                       no update,        │
    │                                       but gradients     │
    │                                       still flow)       │
    └──────────────────────────────────────────────────────────┘
```

### Loss Function

```
    T_single(E) ─────┐
                     │    element-wise
                     ├──► subtraction ──► diff(E)
                     │
    T_double(E) ─────┘

                          diff(E) ──► ||·||₂ ──► negate ──► loss


    With energy mask (optional):

    diff(E) ──► × mask(E) ──► ||·||₂ ──► negate ──► loss

    mask = [1, 1, 0, 0, ..., 0, 1, 1]
            ▲                     ▲
            only these energy     and these
            points contribute     energy points
            to the loss
```

---

## Example: Optimizing a 4-Base Sensor Sequence

### Setup

```python
from g3nat.models.generator import SequenceOptimizer
from g3nat.evaluation import load_trained_model

# Create optimizer for 4-base sequences
opt = SequenceOptimizer(seq_length=4)

# Load a pre-trained predictor (will be frozen during optimization)
predictor, energy_grid, device = load_trained_model("trained_models/hamiltonian_2000x_4to10BP_5000epoch.pth")
```

### Optimization

```python
# Optimize for 500 steps with tau annealing
losses = opt.optimize(
    predictor,
    num_steps=500,
    tau_start=2.0,    # start exploratory
    tau_end=0.1,      # end near-discrete
    lr=0.1,           # higher lr for direct logit optimization
    log_every=100
)

# Output:
# Step 100/500 | Loss: -0.3421 | Seq: ATCG | tau: 1.622
# Step 200/500 | Loss: -0.5187 | Seq: GTCA | tau: 1.243
# Step 300/500 | Loss: -0.6893 | Seq: GCTA | tau: 0.864
# Step 400/500 | Loss: -0.7234 | Seq: GCTA | tau: 0.486
# Step 500/500 | Loss: -0.7301 | Seq: GCTA | tau: 0.100
#
# Loss becomes more negative = difference is growing = better sensor
```

### Inference

```python
# Get optimized sequence
opt.eval()
with torch.no_grad():
    one_hot, _ = opt(tau=0.1)

sequence = opt.decode_sequence(one_hot)
complement = opt.get_complement(sequence)

print(f"Optimized sequence:     5'-{sequence}-3'")
print(f"Reverse complement:     5'-{complement}-3'")
```

### Using an Energy Mask

```python
import torch

# Only optimize transmission difference in the [-1, 0] eV window
energy_grid = torch.linspace(-3, 3, 100)
mask = ((energy_grid >= -1) & (energy_grid <= 0)).float()

losses = opt.optimize(
    predictor,
    num_steps=500,
    energy_mask=mask,    # focus optimization on this window
    log_every=100
)
```

---

## API Reference

### SequenceOptimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_length` | int | required | Number of DNA bases to optimize |

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(tau: float)` | `(one_hot [N,4], normalized_logits [N,4])` |
| `decode_sequence` | `(one_hot)` | DNA sequence string |
| `get_complement` | `(sequence: str)` | Watson-Crick reverse complement string |
| `build_graph_with_soft_features` | `(soft_bases, complementary_sequence=None)` | `Data` graph |
| `compute_loss` | `(transmission_single, transmission_double, energy_mask=None)` | scalar loss tensor |
| `optimize` | `(predictor, num_steps, tau_start=2.0, tau_end=0.1, lr=0.1, energy_mask=None, log_every=100)` | `List[float]` of losses |

### optimize() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictor` | nn.Module | required | Pre-trained predictor (will be frozen) |
| `num_steps` | int | required | Number of optimization steps |
| `tau_start` | float | 2.0 | Initial Gumbel-Softmax temperature |
| `tau_end` | float | 0.1 | Final Gumbel-Softmax temperature |
| `lr` | float | 0.1 | Learning rate for Adam optimizer |
| `energy_mask` | Tensor or None | None | Mask for energy sub-window optimization |
| `log_every` | int | 100 | Print progress every N steps |

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
  __init__.py             ← top-level exports

tests/
  test_models/
    test_generator.py     ← tests covering all SequenceOptimizer functionality
```
