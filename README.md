# Graph Neural Network Nucleic Acid Transport (G3NAT)

A compact Graph Neural Network project for predicting DNA transport properties (Density of States and Transmission) using PyTorch Geometric.

![DNA graph example](DNAGraphExample.png)

**Demo:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13gInyEBZVMuL1ma-jB5U1pHU917bT9U8?usp=sharing)

Full DNA DFT dataset status: _In Progress_

### Core modules
- `g3nat/models/`: GNN models (standard and Hamiltonian), NEGF projection
- `g3nat/graph/`: DNA sequence to graph conversion (core innovation)
- `g3nat/data/`: Dataset creation, synthetic data generation, pickle loading
- `g3nat/training/`: Training loop, configuration, callbacks
- `g3nat/utils/`: Device setup, physics utilities (Hamiltonian, NEGF)
- `g3nat/visualization/`: NetworkX-based visualization of DNA graphs
- `g3nat/evaluation/`: Model loading and inference helpers
- `scripts/train.py`: Unified training script for both synthetic and pickle data

### Install
```bash
pip install -r requirements.txt
```

### Quick start
To use synthetic data from an approximate tight-binding hamiltonian:
```bash
python scripts/train.py \
  --data_source tb \
  --num_samples 2000 \
  --seq_length 8 \
  --num_energy_points 100 \
  --model_type hamiltonian \
  --batch_size 32 \
  --num_epochs 100 \
  --learning_rate 1e-3
```
Outputs (model checkpoint, curves, sample prediction plots) are saved under `./outputs` and `./checkpoints`.

To resume training, pass `--resume_from path/to/checkpoint_latest.pth`.


### Inference with a trained model
You can create and load trained models to use for transport prediction, and the access the tight binding hamiltonian direcly from the model:
```python
from g3nat.models import load_trained_model, predict_sequence

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
H_TB = model.H[0].detach().cpu().numpy()  # Converts PyTorch tensor to NumPy array
```

### Visualize graphs
```python
from g3nat.graph import sequence_to_graph
from g3nat.visualization import visualize_dna_graph

G = sequence_to_graph("ACGTACGT", "TGCATGCA", left_contact_positions=0, right_contact_positions=7)
fig, ax = visualize_dna_graph(G, "ACGTACGT", "TGCATGCA")
```


### Training from dataset (pickle format)

For this work, we have generated a dataset of 500 structures with 4 contact positions for a total of 2000 data points **(NOTE: in progress, link to be uploaded when complete)**

To use this data set, ensure that all pickle files are in the correct directory, and use the unified training script:

```bash
python scripts/train.py \
    --data_source pickle \
    --data_dir /path/to/pickle/files \
    --model_type hamiltonian \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_heads 4 \
    --n_orb 1 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --output_dir ./my_results \
    --checkpoint_dir ./my_checkpoints
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
    from g3nat.graph import sequence_to_graph

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
- **Consistency**: the simple physics generator in `g3nat.utils.create_hamiltonian` follows the same default (primary-end) policy. Dataset helpers will not override explicitly provided contact positions.

### Hamiltonian construction semantics
- In `DNATransportHamiltonianGNN`, the Hamiltonian is constructed directly from the graph:
  - **Nodes** contribute onsite blocks (diagonal terms).
  - **Edges** contribute coupling blocks (off-diagonal terms).
