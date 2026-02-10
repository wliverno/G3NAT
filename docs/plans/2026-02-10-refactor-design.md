# G3NAT Codebase Refactor Design

**Date:** 2026-02-10
**Status:** Design Complete, Ready for Implementation
**Goal:** Refactor codebase into clean modules with proper separation of concerns, using TDD to ensure behavior doesn't change

## Priorities

1. **Separation of concerns** - Models, training, data loading, evaluation in separate modules
2. **Code reusability** - Shared utilities between TB and pickle training properly factored
3. **Discoverability** - Clear module structure for new users
4. **Testability** - TDD ensures refactor doesn't break existing behavior

## Target Module Structure

```
G3NAT/
├── g3nat/                      # Main package
│   ├── __init__.py            # Clean API: import g3nat
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Shared base classes
│   │   ├── standard.py        # DNATransportGNN
│   │   └── hamiltonian.py     # DNATransportHamiltonianGNN
│   ├── graph/                  # Core innovation - graph construction
│   │   ├── __init__.py
│   │   ├── construction.py    # sequence_to_graph
│   │   ├── features.py        # Feature engineering (extensible)
│   │   └── rules.py           # Pairing/interaction rules (extensible)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py        # DNATransportDataset
│   │   ├── loaders.py         # DataLoader utilities
│   │   ├── synthetic.py       # Tight-binding data generation
│   │   └── pickle.py          # Pickle file loading
│   ├── training/              # Unified training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py         # Trainer class
│   │   ├── config.py          # TrainingConfig
│   │   ├── callbacks.py       # Checkpointing, logging
│   │   ├── metrics.py         # Loss functions
│   │   └── utils.py           # setup_device, LengthBucketBatchSampler
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── graphs.py          # DNA graph visualization
│   │   └── results.py         # Training curves, predictions
│   └── utils/
│       ├── __init__.py
│       ├── device.py          # Device management
│       └── physics.py         # NEGF solvers, Hamiltonian utilities
├── scripts/                    # Clean entry points
│   ├── train.py               # Unified training (TB + pickle)
│   ├── evaluate.py
│   ├── visualize.py
│   └── ablate.py
├── tests/                      # TDD test suite
│   ├── baseline/              # Capture current behavior
│   ├── test_graph/
│   ├── test_models/
│   ├── test_data/
│   ├── test_training/
│   └── test_integration/
└── docs/
    └── plans/
```

## File Migration Map

### Models
- `models.py` (lines 1-116: DNATransportGNN) → `g3nat/models/standard.py`
- `models.py` (lines 116-720: DNATransportHamiltonianGNN) → `g3nat/models/hamiltonian.py`
- `models.py` (lines 720+: train_model) → `g3nat/training/trainer.py`

### Graph Construction
- `dataset.py` (sequence_to_graph) → `g3nat/graph/construction.py`
- Future extensibility: `g3nat/graph/features.py`, `rules.py` (empty for now)

### Data
- `dataset.py` (DNATransportDataset) → `g3nat/data/datasets.py`
- `data_generator.py` → `g3nat/data/synthetic.py`
- `load_pickle_data.py` → `g3nat/data/pickle.py`
- `data_generator.py` (NEGF/Hamiltonian helpers) → `g3nat/utils/physics.py`

### Training Utilities
- `train_from_TB.py` (utilities) → `g3nat/training/utils.py`, `callbacks.py`
- `train_from_TB.py` + `train_from_pickles.py` (main) → `scripts/train.py` (unified)

### Visualization
- `visualize_dna_graph.py` → `g3nat/visualization/graphs.py`

### Ablation
- `ablate.py` → `scripts/ablate.py` (updated imports only)

## Key Design Decisions

### Unified Training Pipeline
**Current:** Separate `train_from_TB.py` and `train_from_pickles.py` with duplicated code
**New:** Single `scripts/train.py` using unified `Trainer` class

```python
from g3nat.training import Trainer, TrainingConfig
from g3nat.data import generate_tight_binding_data, load_pickle_directory

# Works for both data sources
trainer = Trainer(model, train_loader, val_loader, config)
results = trainer.fit()
```

### Graph Module (Core Innovation)
**Rationale:** `sequence_to_graph()` is the key methodological contribution
- Elevated to `g3nat/graph/` module (not buried in data)
- Structure allows future extensions (RNA, metal modifications)
- Clean API: `from g3nat.graph import sequence_to_graph`

### Extensibility Without Feature Creep
**Structure supports future extensions:**
- `g3nat/graph/features.py` - New base encodings (RNA, modified bases)
- `g3nat/graph/rules.py` - New interaction types (metal coordination)

**But refactor ONLY moves existing code:**
- Current DNA bases (A, T, G, C) unchanged
- Current edge types (backbone, hydrogen_bond, contact) unchanged
- TDD verifies: same inputs → same outputs

## TDD Strategy

### Phase 1: Capture Baselines (Before Refactor)
```python
tests/baseline/test_baseline_*.py

# Save current outputs as ground truth
- Model predictions on test inputs
- Dataset construction outputs
- Graph construction (node features, edges)
- Training loop outputs (losses, checkpoints)
```

### Phase 2: Refactor with Verification
```python
tests/test_*/test_*.py

# Each test verifies new code produces identical output
def test_sequence_to_graph_unchanged():
    graph_new = sequence_to_graph("ACGT", ...)  # New location
    baseline = load_baseline("sequence_to_graph_ACGT.pkl")
    assert_graphs_identical(graph_new, baseline)
```

### Phase 3: Integration Tests
```python
tests/test_integration/test_end_to_end.py

# Verify complete pipeline produces same results
- Same losses with fixed random seed
- Same model weights after training
- Can load old checkpoints in new code
```

## Code Quality Standards

### Comments (No AI Fluff)
- ✅ Document WHY when non-obvious
- ✅ Add context for scientific decisions
- ✅ Concise docstrings for public APIs
- ❌ NO comments restating obvious code

### API Design
**Package-level imports:**
```python
# Clean for papers/documentation
import g3nat

model = g3nat.DNATransportHamiltonianGNN(...)
graph = g3nat.sequence_to_graph("ACGT")
results = g3nat.train_model(model, ...)
```

## Implementation Phases

### Phase 1: Setup & Baselines
1. Create new directory structure
2. Write baseline capture tests
3. Run baselines, save outputs

### Phase 2: Core Modules (Order Matters)
1. `g3nat/utils/` - No dependencies
2. `g3nat/graph/` - Depends on utils
3. `g3nat/models/` - Depends on utils
4. `g3nat/data/` - Depends on graph, utils
5. `g3nat/training/` - Depends on models, data

### Phase 3: Scripts & Cleanup
1. `scripts/train.py` - Unified training
2. Update `ablate.py` imports
3. Remove old files
4. Update documentation

### Phase 4: Verification
1. Run all tests
2. Train a model end-to-end
3. Compare with old training run
4. Document migration for future users

## Success Criteria

- ✅ All baseline tests pass (identical outputs)
- ✅ Integration tests pass (end-to-end training works)
- ✅ Can run full training pipeline (TB and pickle)
- ✅ Code organized by responsibility (models ≠ training ≠ data)
- ✅ Clean imports for scientific publication
- ✅ No circular dependencies
- ✅ Extensible structure for future work

## Non-Goals (Future Work)

These are explicitly OUT OF SCOPE for this refactor:
- ❌ Adding RNA support
- ❌ Adding metal modifications
- ❌ Changing APIs beyond reorganization
- ❌ Performance optimizations
- ❌ New features

The structure allows these later, but the refactor only moves existing code.

## Notes

- Scientific publication target: Clean module structure makes methods section easier
- Solo user (no backward compatibility concerns): Can make breaking changes
- Test-driven: Behavior preservation is critical
- Extensibility: Structure supports future research directions without implementing them now
