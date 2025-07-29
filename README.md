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

### Example using pre-trained model

```python
from models import load_trained_model, predict_sequence
from data_generator import create_sample_data, getTransmissionDOS

# Generate data
sequences, sequences_complement, dos_data, transmission_data, energy_grid = create_sample_data(
    num_samples=1000, seq_length=8, num_energy_points=100
)

# Create dataset
dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)

# Load trained model
model, energy_grid, device = load_trained_model('trained_model.pth')

# Predict DOS and transmission for each sequence
dos_preds = []
transmission_preds = []
for seq, seq_complementary in zip(sequences, sequences_complement)
    dos_pred, transmission_pred = predict_sequence(model, seq, seq_complementary, energy_grid=energy_grid)
    dos_preds.append(dos_pred)
    transmission_preds.append(transmission_pred)
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

``` 