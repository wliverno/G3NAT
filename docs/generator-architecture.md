# DNA Sequence Generator - Architecture & How It Works

## Overview

The DNA Sequence Generator learns to produce DNA sequences that are optimal for **sensing applications**. It finds sequences where hybridization (binding of the complementary strand) produces the largest possible change in electrical transmission — making the sensor maximally sensitive.

```
                        GOAL
    ┌─────────────────────────────────────────────┐
    │  Find sequence S such that:                 │
    │                                             │
    │  T(single-stranded S) ≠ T(double-stranded S)│
    │                                             │
    │  Maximize: ||T_single - T_double||          │
    └─────────────────────────────────────────────┘
```

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      GeneratorTrainer                            │
│                                                                  │
│  ┌──────────────────┐     ┌────────────────────────────────┐    │
│  │  DNASequence      │     │  Frozen Predictor               │    │
│  │  Generator        │     │  (DNATransportGNN or            │    │
│  │                   │     │   DNATransportHamiltonianGNN)   │    │
│  │  z ──► MLP ──►    │     │                                 │    │
│  │  Gumbel-Softmax   │     │  Graph ──► GNN ──► T(E)        │    │
│  │  ──► soft bases   │     │                                 │    │
│  └──────────────────┘     └────────────────────────────────┘    │
│           │                          ▲                           │
│           │                          │                           │
│           ▼                          │                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Graph Construction                          │    │
│  │  soft bases ──► sequence_to_graph() topology             │    │
│  │              ──► replace node features                   │    │
│  │              ──► single-stranded & double-stranded       │    │
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

## Component 1: DNASequenceGenerator

### Architecture

```
         Latent Space              MLP Network              Output
    ┌─────────────────┐    ┌─────────────────────┐    ┌──────────────┐
    │                 │    │                     │    │              │
    │  z ~ N(0, 1)    │──► │ Linear(32, 128)     │    │  logits      │
    │  shape: [B, 32] │    │ ReLU                │    │  [B, N, 4]   │
    │                 │    │ Linear(128, 128)    │──► │              │
    │  (sampled or    │    │ ReLU                │    │      │       │
    │   provided)     │    │ Linear(128, N×4)    │    │      ▼       │
    │                 │    │                     │    │  Gumbel-     │
    └─────────────────┘    └─────────────────────┘    │  Softmax     │
                                                      │      │       │
                                                      │      ▼       │
                                                      │ soft_bases   │
                                                      │ [B, N, 4]   │
                                                      └──────────────┘

    B = batch size, N = sequence length
    4 channels = [A, T, G, C] probabilities
```

### Gumbel-Softmax: The Key Trick

The core challenge: DNA sequences are **discrete** (A, T, G, or C), but neural networks need **continuous, differentiable** values to learn via gradient descent.

**Gumbel-Softmax** bridges this gap by producing "soft" one-hot vectors that are differentiable but approximate discrete choices:

```
    Hard one-hot (discrete, NOT differentiable):
    Position 1:  [1, 0, 0, 0]  = A         ← no gradient can flow

    Soft one-hot (continuous, differentiable):
    Position 1:  [0.85, 0.05, 0.08, 0.02]  ← gradient flows!
                  ↑ A    ↑ T   ↑ G   ↑ C

    Still "means A" (argmax = 0), but gradients can adjust the weights
```

**Temperature (tau) controls sharpness:**

```
    tau = 2.0 (exploratory):    [0.40, 0.25, 0.20, 0.15]  ← spread out
    tau = 1.0 (balanced):       [0.70, 0.12, 0.10, 0.08]  ← mostly one
    tau = 0.1 (near-discrete):  [0.97, 0.01, 0.01, 0.01]  ← almost one-hot
```

### Training vs Eval Mode

```
    TRAINING MODE                          EVAL MODE
    ┌──────────────────────┐              ┌──────────────────────┐
    │ logits + Gumbel noise│              │ logits               │
    │        │             │              │        │             │
    │        ▼             │              │        ▼             │
    │   gumbel_softmax     │              │   softmax / tau      │
    │   (hard=False)       │              │        │             │
    │        │             │              │        ▼             │
    │        ▼             │              │   argmax ──► one_hot │
    │   [0.7, 0.1, ...]   │              │   [1, 0, 0, 0]      │
    │                      │              │                      │
    │ Differentiable, but  │              │ Deterministic:       │
    │ stochastic (random   │              │ same z always gives  │
    │ Gumbel noise added)  │              │ same sequence        │
    └──────────────────────┘              └──────────────────────┘
```

---

## Component 2: Graph Construction with Soft Features

### The Gradient Flow Problem

`sequence_to_graph()` converts a DNA string into a graph with hard one-hot node features. But hard one-hot values have **no gradient** — the generator can't learn.

**Solution:** Use `sequence_to_graph()` only for the graph **topology** (edges, contacts), then swap in the soft features:

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
    │  Node 2 (position 0):    soft_bases[0] │  ← REPLACE with soft
    │  Node 3 (position 1):    soft_bases[1] │  ← REPLACE with soft
    │  Node 4 (position 2):    soft_bases[2] │  ← REPLACE with soft
    │  Node 5 (position 3):    soft_bases[3] │  ← REPLACE with soft
    └─────────────────────────────────────────┘

    Now: topology is correct AND gradients flow through node features
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
    │  Primary features: SOFT   │           │  (2 contacts + 4 primary + 4 comp)│
    │  (differentiable)         │           │                                   │
    │                           │           │  Primary features: SOFT           │
    │                           │           │  Complement features: HARD        │
    │                           │           │  (only primary is optimized)      │
    └───────────────────────────┘           └───────────────────────────────────┘

    [L] = left electrode contact       ── = backbone bond
    [R] = right electrode contact       | = hydrogen bond
```

---

## Component 3: Training Loop

### Single Training Step

```
                    ┌─────────────────┐
                    │  Sample z       │
                    │  z ~ N(0, 1)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Generator MLP  │
                    │  z ──► logits   │
                    │  ──► soft_bases │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
           ┌──────────────┐  ┌──────────────────┐
           │  decode to   │  │  Keep soft_bases  │
           │  "ATGC"      │  │  (differentiable) │
           │  (discrete)  │  │                    │
           └──────┬───────┘  └────────┬───────────┘
                  │                   │
                  ▼                   │
           ┌───────────────┐         │
           │  reverse      │         │
           │  complement   │         │
           │  "ATGC" ──►   │         │
           │  "GCAT"       │         │
           └──────┬────────┘         │
                  │                   │
         ┌────────┴───────────────────┤
         │                            │
         ▼                            ▼
┌──────────────────┐        ┌──────────────────┐
│  Build Graph     │        │  Build Graph     │
│  SINGLE-stranded │        │  DOUBLE-stranded │
│                  │        │                  │
│  Soft primary    │        │  Soft primary    │
│  No complement   │        │  Hard complement │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  Frozen Predictor│        │  Frozen Predictor│
│  ──► T_single(E) │        │  ──► T_double(E) │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Loss Computation    │
         │                      │
         │  diff = T_single     │
         │       - T_double     │
         │                      │
         │  loss = -||diff||₂   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  loss.backward()     │
         │  optimizer.step()    │
         │                      │
         │  Gradients flow:     │
         │  loss ──► predictor  │
         │  (frozen, passes     │
         │   gradients through) │
         │  ──► soft node feats │
         │  ──► generator MLP   │
         └──────────────────────┘
```

### Gradient Flow Path (Detail)

```
    ┌──────────────────────────────────────────────────────────┐
    │                   GRADIENT FLOW                          │
    │                                                          │
    │  Generator    Gumbel-      Graph        Frozen     Loss  │
    │  MLP params   Softmax      Features     Predictor        │
    │                                                          │
    │  W₁, W₂, W₃   soft        data.x       GNN        L    │
    │    │            │           │            layers      │   │
    │    │            │           │            │           │   │
    │    ◄────────────◄───────────◄────────────◄───────────┘   │
    │         ▲              ▲          ▲            ▲          │
    │    ∂L/∂W         ∂L/∂soft    ∂L/∂x      ∂L/∂T(E)        │
    │                                                          │
    │  UPDATED         passes      passes     passes           │
    │  by optimizer    through     through    through           │
    │                                         (frozen =        │
    │                                          no update,      │
    │                                          but gradients   │
    │                                          still flow)     │
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

## Example: Training a 4-Base Sensor Sequence

### Setup

```python
from g3nat.models import DNASequenceGenerator, DNATransportGNN
from g3nat.models.generator import GeneratorTrainer
import torch

# Create generator for 4-base sequences
generator = DNASequenceGenerator(
    seq_length=4,       # 4 DNA bases
    latent_dim=32,      # 32-dim latent space
    hidden_dim=128,     # 128 hidden units
    num_hidden_layers=2,# 2 hidden layers
    tau=1.0             # Gumbel-Softmax temperature
)

# Load a pre-trained predictor (frozen during generator training)
predictor = DNATransportGNN(
    hidden_dim=128,
    num_layers=4,
    output_dim=100      # 100 energy points
)
# predictor.load_state_dict(torch.load("trained_predictor.pth"))

# Create trainer
trainer = GeneratorTrainer(
    generator=generator,
    predictor=predictor,
    lr=1e-3              # learning rate for generator only
)
```

### Training

```python
# Train for 500 steps
losses = trainer.train(num_steps=500, log_every=100)

# Output:
# Step 100/500 | Loss: -0.3421 | Seq: ATCG
# Step 200/500 | Loss: -0.5187 | Seq: GTCA
# Step 300/500 | Loss: -0.6893 | Seq: GCTA
# Step 400/500 | Loss: -0.7234 | Seq: GCTA
# Step 500/500 | Loss: -0.7301 | Seq: GCTA
#
# Loss becomes more negative = difference is growing = better sensor
```

### Inference

```python
# Generate optimized sequence
generator.eval()
with torch.no_grad():
    soft_bases, _ = generator(batch_size=1)

sequence = generator.decode_sequences(soft_bases)[0]
complement = generator.get_complement(sequence)

print(f"Optimized sequence:     5'-{sequence}-3'")
print(f"Reverse complement:     5'-{complement}-3'")
# Optimized sequence:     5'-GCTA-3'
# Reverse complement:     5'-TAGC-3'
#
# In the double-stranded structure:
#   5'-G-C-T-A-3'   (primary)
#   3'-C-G-A-T-5'   (complement, antiparallel)
```

### What the Optimized Sequence Means

```
    Before hybridization              After hybridization
    (single-stranded DNA)             (double-stranded DNA)

    5'─G─C─T─A─3'                    5'─G─C─T─A─3'
    [L]         [R]                   [L] |  |  |  | [R]
                                      3'─C─G─A─T─5'

    T_single(E)                       T_double(E)
    ┌────────────────┐                ┌────────────────┐
    │    ╱╲          │                │         ╱╲     │
    │   ╱  ╲         │                │   ╱╲   ╱  ╲   │
    │  ╱    ╲        │                │  ╱  ╲ ╱    ╲  │
    │ ╱      ╲───    │                │ ╱    ╳      ╲ │
    │╱           ╲   │                │╱    ╱ ╲      ╲│
    └────────────────┘                └────────────────┘
           E (eV)                            E (eV)

    The generator found a sequence where these two
    transmission curves differ maximally.

    Large ||T_single - T_double|| means:
    ──► Hybridization produces a big conductance change
    ──► This sequence is an excellent DNA sensor candidate
```

### Using an Energy Mask

```python
import torch

# Only optimize transmission difference in the [-1, 0] eV window
energy_grid = torch.linspace(-3, 3, 100)
mask = ((energy_grid >= -1) & (energy_grid <= 0)).float()

trainer = GeneratorTrainer(
    generator=generator,
    predictor=predictor,
    lr=1e-3,
    energy_mask=mask    # focus optimization on this window
)

losses = trainer.train(num_steps=500, log_every=100)

# Now the generator optimizes for maximum transmission
# difference specifically in the [-1, 0] eV energy range
```

---

## API Reference

### DNASequenceGenerator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_length` | int | required | Number of DNA bases to generate |
| `latent_dim` | int | 32 | Dimension of latent vector z |
| `hidden_dim` | int | 128 | Hidden layer width |
| `num_hidden_layers` | int | 2 | Number of hidden layers |
| `tau` | float | 1.0 | Gumbel-Softmax temperature |

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(z=None, batch_size=1)` | `(soft_bases [B,N,4], logits [B,N,4])` |
| `decode_sequences` | `(soft_bases)` | `List[str]` of DNA sequences |
| `get_complement` | `(sequence: str)` | Watson-Crick reverse complement string |

### GeneratorTrainer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generator` | DNASequenceGenerator | required | The generator to train |
| `predictor` | nn.Module | required | Pre-trained predictor (will be frozen) |
| `mode` | str | `'maximize_difference'` | Training objective |
| `lr` | float | 1e-3 | Learning rate for generator |
| `energy_mask` | Tensor or None | None | Mask for energy sub-window optimization |

| Method | Signature | Returns |
|--------|-----------|---------|
| `train_step` | `()` | `(loss: float, sequences: List[str])` |
| `train` | `(num_steps, log_every=100)` | `List[float]` of losses |
| `build_graph_with_soft_features` | `(soft_bases, complementary_sequence=None)` | `Data` graph |
| `compute_loss` | `(transmission_single, transmission_double, energy_mask=None)` | scalar loss tensor |

---

## File Structure

```
g3nat/
  models/
    generator.py          ← DNASequenceGenerator + GeneratorTrainer
    standard.py           ← DNATransportGNN (predictor option 1)
    hamiltonian.py        ← DNATransportHamiltonianGNN (predictor option 2)
    __init__.py           ← exports all model classes
  graph/
    construction.py       ← sequence_to_graph() used for topology
  __init__.py             ← top-level exports

tests/
  test_models/
    test_generator.py     ← 19 tests covering all generator functionality
```
