"""Inference utilities for loading and using trained models."""
import torch
import numpy as np
from typing import Tuple, Union, List
from torch_geometric.data import Batch

from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph


def load_trained_model(model_path: str, device: str = 'auto') -> Tuple[Union[DNATransportGNN, DNATransportHamiltonianGNN], np.ndarray, torch.device]:
    """
    Load a trained DNA Transport GNN model.

    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        Tuple of (model, energy_grid, device)
    """
    if device == 'auto':
        device_tensor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_tensor = torch.device(device)

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device_tensor}")

    # Load the saved model (allow numpy arrays for energy grid)
    checkpoint = torch.load(model_path, map_location=device_tensor, weights_only=False)

    # Extract model arguments
    args = checkpoint.get('args', {})
    energy_grid = checkpoint.get('energy_grid', np.linspace(-3, 3, 100))

    # Detect model type from state dict keys
    state_dict = checkpoint['model_state_dict']
    model_type = None

    # Check for Hamiltonian model (new DNATransportHamiltonianGNN has onsite_proj/coupling_proj)
    if any('onsite_proj' in key for key in state_dict.keys()) and any('coupling_proj' in key for key in state_dict.keys()):
        model_type = 'hamiltonian'
        print("Detected DNATransportHamiltonianGNN model")
    # Check for standard model (has dos_proj and transmission_proj layers)
    elif any('dos_proj' in key for key in state_dict.keys()) and any('transmission_proj' in key for key in state_dict.keys()):
        model_type = 'standard'
        print("Detected DNATransportGNN model")
    # Check for simple Hamiltonian model (has H_proj but no NEGF components)
    elif any('H_proj' in key for key in state_dict.keys()) and not any('NEGF' in key for key in state_dict.keys()):
        model_type = 'simple_hamiltonian'
        print("Detected DNAHamiltonianGNN model (legacy)")
    else:
        # Default to standard model
        model_type = 'standard'
        print("Could not detect model type, defaulting to DNATransportGNN")

    # Initialize model with same architecture
    if model_type == 'hamiltonian':
        model = DNATransportHamiltonianGNN(
            hidden_dim=args.get('hidden_dim', 128),
            num_layers=args.get('num_layers', 4),
            num_heads=args.get('num_heads', 4),
            energy_grid=energy_grid,
            dropout=args.get('dropout', 0.1),
            n_orb=args.get('n_orb', 1),
            enforce_hermiticity=args.get('enforce_hermiticity', True),
            solver_type=args.get('solver_type', 'frobenius'),
            use_log_outputs=args.get('use_log_outputs', True),
            log_floor=args.get('log_floor', 1e-16),
            complex_eta=args.get('complex_eta', 1e-12),
            conv_type=args.get('conv_type', 'gat')
        )
        print("DNATransportHamiltonianGNN initialized successfully")
    elif model_type == 'simple_hamiltonian':
        # Legacy support - for now just use hamiltonian model
        print("Warning: Legacy DNAHamiltonianGNN detected. Using DNATransportHamiltonianGNN instead.")
        model = DNATransportHamiltonianGNN(
            hidden_dim=args.get('hidden_dim', 128),
            num_layers=args.get('num_layers', 4),
            num_heads=args.get('num_heads', 4),
            energy_grid=energy_grid,
            dropout=args.get('dropout', 0.1),
            n_orb=args.get('n_orb', 1),
            enforce_hermiticity=args.get('enforce_hermiticity', True),
            solver_type=args.get('solver_type', 'frobenius'),
            use_log_outputs=args.get('use_log_outputs', True),
            log_floor=args.get('log_floor', 1e-16),
            complex_eta=args.get('complex_eta', 1e-12),
            conv_type=args.get('conv_type', 'transformer')
        )
    else:  # standard
        model = DNATransportGNN(
            hidden_dim=args.get('hidden_dim', 128),
            num_layers=args.get('num_layers', 4),
            num_heads=args.get('num_heads', 4),
            output_dim=args.get('num_energy_points', 100),
            dropout=args.get('dropout', 0.1),
            conv_type=args.get('conv_type', 'transformer')
        )
        print("DNATransportGNN initialized successfully")

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_tensor)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")

    return model, energy_grid, device_tensor


def predict_sequence(
    model: Union[DNATransportGNN, DNATransportHamiltonianGNN],
    sequence: str,
    complementary_sequence: str,
    left_contact_positions: Union[int, List[int]] = None,
    right_contact_positions: Union[int, List[int]] = None,
    left_contact_coupling: float = 0.1,
    right_contact_coupling: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict DOS and transmission for a DNA sequence.

    Args:
        model: Trained DNATransportGNN or DNATransportHamiltonianGNN model
        sequence: DNA sequence string (e.g., "ACGTACGT")
        complementary_sequence: Complementary DNA sequence string (e.g., "__GCATGCAT__")
        left_contact_positions: Position(s) for left contact
        right_contact_positions: Position(s) for right contact (default: last position)
        left_contact_coupling: Coupling strength for left contact
        right_contact_coupling: Coupling strength for right contact

    Returns:
        Tuple of (transmission_pred, dos_pred) arrays
    """
    if right_contact_positions is None:
        right_contact_positions = len(sequence) - 1

    print(f"Predicting for sequence: {sequence}")
    print(f"                         {complementary_sequence[::-1]}")
    print(f"Left contact at position {left_contact_positions}, coupling: {left_contact_coupling}")
    print(f"Right contact at position {right_contact_positions}, coupling: {right_contact_coupling}")

    # Convert sequence to graph
    graph = sequence_to_graph(
        primary_sequence=sequence,
        complementary_sequence=complementary_sequence,
        left_contact_positions=left_contact_positions,
        right_contact_positions=right_contact_positions,
        left_contact_coupling=left_contact_coupling,
        right_contact_coupling=right_contact_coupling
    )

    if graph is None:
        raise ValueError(f"Failed to create graph for sequence: {sequence}")

    # Create batch (single graph)
    batch_data = Batch.from_data_list([graph])
    batch_data = batch_data.to(next(model.parameters()).device)

    # Make prediction
    with torch.no_grad():
        dos_pred, transmission_pred = model(batch_data)

        # Convert to numpy arrays
        dos_pred = dos_pred.cpu().numpy()[0]  # Remove batch dimension
        transmission_pred = transmission_pred.cpu().numpy()[0]

    print(f"Prediction completed!")
    print(f"DOS range: [{dos_pred.min():.4f}, {dos_pred.max():.4f}]")
    print(f"Transmission range: [{transmission_pred.min():.4f}, {transmission_pred.max():.4f}]")

    return transmission_pred, dos_pred
