# Graph Neural Network Nucleic Acid Transport

A Graph Neural Network (GNN) implementation for predicting DNA transport properties, including Density of States (DOS) and transmission coefficients.

## Overview

This project implements a Graph Neural Network using PyTorch Geometric to predict electronic transport properties of DNA sequences. The model converts DNA sequences into graph representations and uses Graph Attention Networks (GAT) to predict:

- Density of States (DOS) as a function of energy
- Transmission coefficients for electronic transport

## Features

- **DNA Sequence to Graph Conversion**: Converts DNA sequences into graph representations with nodes representing bases and edges representing physical interactions
- **Graph Attention Network**: Uses GAT layers with multi-head attention for feature learning
- **Multi-task Learning**: Simultaneously predicts DOS and transmission properties
- **Synthetic Data Generation**: Includes utilities for generating sample data for testing and development

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd G3NAT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main training script:
```bash
python main.py
```

### Custom Training

```python
from models import DNATransportGNN, train_model
from dataset import DNATransportDataset
from data_generator import create_sample_data

# Generate data
sequences, dos_data, transmission_data, energy_grid = create_sample_data(
    num_samples=1000, seq_length=8, num_energy_points=100
)

# Create dataset
dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)

# Initialize model
model = DNATransportGNN(
    node_features=8,
    edge_features=4,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    output_dim=100
)

# Train model
train_losses, val_losses = train_model(model, train_loader, val_loader)
```

## Project Structure

```
G3NAT/
├── main.py                 # Main training script
├── dna_transport_gnn.py    # Core GNN model and dataset classes
├── data_generator.py       # Synthetic data generation utilities
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── examples/             # Example scripts and notebooks
    └── basic_usage.py
```

## Model Architecture

The model consists of:

1. **DNA Sequence to Graph Converter**: Converts DNA sequences into graph representations with:
   - Node features: One-hot encoding + physical properties (purine/pyrimidine, molecular weight, etc.)
   - Edge features: Coupling strengths, backbone connections, hydrogen bonding

2. **Graph Attention Network**: 
   - Multiple GAT layers with residual connections
   - Multi-head attention mechanism
   - Layer normalization

3. **Output Heads**:
   - DOS prediction head
   - Transmission coefficient prediction head

## Data Format

### Input
- DNA sequences as strings (e.g., "ATGCATGC")
- Energy grid for DOS/transmission calculations

### Output
- Density of States (DOS) as function of energy
- Transmission coefficients (0-1 range)

## Training

The model is trained using:
- **Loss Function**: MSE loss for both DOS and transmission predictions
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau for learning rate scheduling
- **Regularization**: Dropout and gradient clipping

## Results

After training, the model saves:
- Trained model weights (`dna_transport_model.pth`)
- Training curves plot (`training_results.png`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dna_transport_gnn,
  title={DNA Transport GNN: Graph Neural Networks for DNA Transport Property Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/G3NAT}
}
``` 