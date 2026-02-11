# G3NAT Module Structure (Post-Refactor)

## Package Organization

```
g3nat/
├── models/          # Model architectures (GNN, Hamiltonian)
├── graph/           # Graph construction (sequence_to_graph)
├── data/            # Data loading (synthetic, pickle, datasets)
├── training/        # Training loop, config, callbacks
├── utils/           # Device management, physics utilities
├── visualization/   # Plotting and visualization
└── evaluation/      # Evaluation metrics
```

## Quick Start

```python
import g3nat

# Create model
model = g3nat.DNATransportHamiltonianGNN(...)

# Create graph
graph = g3nat.sequence_to_graph("ACGT", ...)

# Train
results = g3nat.train_model(model, train_loader, val_loader)
```

## Training Scripts

**Unified training:** `scripts/train.py`
- Supports both TB and pickle data
- Single interface for all training

**Ablation studies:** `scripts/ablate.py`

## Migration Guide

Old import → New import:
- `from models import DNATransportGNN` → `from g3nat.models import DNATransportGNN`
- `from dataset import sequence_to_graph` → `from g3nat.graph import sequence_to_graph`
- `from models import train_model` → `from g3nat.training import train_model`

## Testing

Run test suite:
```bash
python -m pytest tests/ -v
```

All tests verify refactored code produces identical behavior to original.
