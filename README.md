# Graph Neural Network Nucleic Acid Transport (G3NAT)

A compact Graph Neural Network project for predicting DNA transport properties (Density of States and Transmission) using PyTorch Geometric.

### Core modules
- `models.py`: GNN models (standard and Hamiltonian), NEGF projection, training utilities, loading and inference helpers
- `dataset.py`: DNA sequence to graph conversion and dataset creation utilities
- `data_generator.py`: Synthetic data generation via a simple tight-binding DNA model + NEGF
- `main.py`: End-to-end training pipeline and sample prediction/plotting
- `visualize_dna_graph.py`: NetworkX-based visualization of DNA graphs

### Install
```bash
pip install -r requirements.txt
```

### Train (quick start)
```bash
python main.py \
  --num_samples 2000 \
  --seq_length 8 \
  --num_energy_points 100 \
  --model_type standard \
  --batch_size 32 \
  --num_epochs 100 \
  --learning_rate 1e-3
```
Outputs (model checkpoint, curves, sample prediction plots) are saved under `./outputs` and `./checkpoints`.

To resume training, pass `--resume_from path/to/checkpoint_latest.pth`.

### Generate synthetic data only
Use helpers in `data_generator.py`:
```python
from data_generator import create_sample_data
sequences, sequences_complement, dos_data, trans_data, energy_grid = create_sample_data(
    num_samples=1000, seq_length=8, num_energy_points=100
)
```

### Inference with a trained model
```python
from models import load_trained_model, predict_sequence

model, energy_grid, device = load_trained_model('outputs/dna_transport_standard_model.pth')
dos_pred, trans_pred = predict_sequence(
    model,
    sequence="ACGTACGT",
    complementary_sequence="TGCATGCA",  # or omit to treat as single strand with blanks
    left_contact_positions=0,
    right_contact_positions=7,
    left_contact_coupling=0.1,
    right_contact_coupling=0.2,
)
```

### Visualize graphs
```python
from dataset import sequence_to_graph
from visualize_dna_graph import visualize_dna_graph

G = sequence_to_graph("ACGTACGT", "TGCATGCA", left_contact_positions=0, right_contact_positions=7)
fig, ax = visualize_dna_graph(G, "ACGTACGT", "TGCATGCA")
```

### Tests
All compatible tests are consolidated in `tests/test_all.py` (pytest-style smoke tests):
```bash
# optional: install dev deps
pip install -e .[dev]
# run tests
pytest -q
```

### Notes
- Node features: 4 one-hot features (A, T, G, C)
- Edge features: 5 values per edge: [backbone_onehot, hbond_onehot, contact_onehot, directionality, coupling]
- Hamiltonian NEGF implementation is vectorized for stability; transmission/DOS are returned as log10-safe values for training stability in `DNATransportHamiltonianGNN`.

### Contact configuration defaults
- **Default policy (graph + generator)**: left contact attaches to position `0` of the primary strand; right contact attaches to position `len(primary_sequence) - 1` of the primary strand.
- **Specifying contacts**: pass `int`, `List[int]`, or `Tuple[str, Union[int, List[int]]]` where the first element of the tuple is `'primary'` or `'complementary'` to target a specific strand.
  - Example:
    ```python
    from dataset import sequence_to_graph

    G = sequence_to_graph(
        primary_sequence="ACGTACGT",
        complementary_sequence="TGCATGCA",
        left_contact_positions=("primary", 0),
        right_contact_positions=("primary", 7),
        left_contact_coupling=0.1,
        right_contact_coupling=0.2,
    )
    ```
- **Complementary indexing**: positions for the complementary strand are 0-indexed into the provided `complementary_sequence` string.
- **Consistency**: the simple physics generator in `data_generator.create_hamiltonian` follows the same default (primary-end) policy. Dataset helpers will not override explicitly provided contact positions.

### Hamiltonian construction semantics
- In `DNATransportHamiltonianGNN`, the Hamiltonian is constructed directly from the graph:
  - **Nodes** contribute onsite blocks (diagonal terms).
  - **Edges** contribute coupling blocks (off-diagonal terms).
