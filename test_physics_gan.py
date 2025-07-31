#!/usr/bin/env python3
"""
Test script for physics-informed GAN training using data_generator.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split

# Import our modules
from data_generator import create_sample_data
from dataset import sequence_to_graph, DNATransportDataset, create_dna_dataset
from models import train_physics_informed, DNAHamiltonianGNN, PhysicsDiscriminator
from torch_geometric.data import Data


def create_data_loaders(num_samples=100, seq_length=8, num_energy_points=100, batch_size=16):
    """
    Create train and validation data loaders using data_generator.py
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Maximum sequence length
        num_energy_points: Number of energy points
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, energy_grid)
    """
    print(f"Generating {num_samples} DNA sequences...")
    
    # Generate data using data_generator
    primary_sequences, complementary_sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=num_samples,
        seq_length=seq_length,
        num_energy_points=num_energy_points
    )
    
    print(f"Generated {len(primary_sequences)} sequences")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    # Filter out sequences that might cause issues during graph conversion
    print("Filtering sequences and preparing data...")
    valid_indices = []
    
    for i, (seq, comp_seq) in enumerate(zip(primary_sequences, complementary_sequences)):
        # Basic validation - check if sequences are valid
        if len(seq) > 0 and len(comp_seq) > 0:
            valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} valid sequences out of {len(primary_sequences)}")
    
    # Convert lists to numpy arrays first, then filter
    dos_data_array = np.array(dos_data)
    transmission_data_array = np.array(transmission_data)
    
    # Filter data to only include valid sequences
    filtered_sequences = [primary_sequences[i] for i in valid_indices]
    filtered_complementary_sequences = [complementary_sequences[i] for i in valid_indices]
    filtered_dos_data = dos_data_array[valid_indices]  # Numpy array indexing
    filtered_transmission_data = transmission_data_array[valid_indices]  # Numpy array indexing
    
    print(f"Filtered data shapes: DOS {filtered_dos_data.shape}, Transmission {filtered_transmission_data.shape}")
    
    # Create dataset using the proper function from dataset.py
    print("Creating dataset with proper batching...")
    
    dataset = create_dna_dataset(
        sequences=filtered_sequences,
        dos_data=filtered_dos_data,
        transmission_data=filtered_transmission_data,
        energy_grid=energy_grid,
        complementary_sequences=filtered_complementary_sequences,
        graph_converter_func=sequence_to_graph,
        left_contact_positions=0,
        right_contact_positions=0,  # Will be automatically updated to len(sequence) - 1 for each sequence
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    # Split dataset indices
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices, val_indices = train_test_split(
        range(dataset_size), 
        test_size=0.2, 
        random_state=42
    )
    
    # Create train and validation datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, energy_grid


def test_physics_informed_training():
    """Test the physics-informed training function"""
    print("=" * 50)
    print("Testing Physics-Informed GAN Training")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, energy_grid = create_data_loaders(
        num_samples=50,  # Small dataset for testing
        seq_length=8,
        num_energy_points=100,
        batch_size=8
    )
    
    # Test parameters
    max_len_dna = 8
    gammaL = np.array([0.1] + [0.0] * 15)  # Left contact on first site
    gammaR = np.array([0.0] * 7 + [0.1]+ [0.0]*8)  # Right contact on last site
    
    print(f"Max DNA length: {max_len_dna}")
    print(f"GammaL: {gammaL}")
    print(f"GammaR: {gammaR}")
    
    # Run training
    print("\nStarting physics-informed training...")
    try:
        generator, train_losses, val_losses = train_physics_informed(
            train_loader=train_loader,
            val_loader=val_loader,
            energy_grid=energy_grid,
            max_len_dna=max_len_dna,
            gammaL=gammaL,
            gammaR=gammaR,
            epochs=5,  # Short training for testing
            lr=1e-3,
            device=device
        )
        
        print("\nTraining completed successfully!")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        # Test prediction on a single sample
        print("\nTesting prediction on a single sample...")
        generator.eval()
        with torch.no_grad():
            # Get a single batch
            batch = next(iter(val_loader))
            batch = batch.to(device)
            
            # Generate Hamiltonian
            H_pred = generator(batch)
            print(f"Generated Hamiltonian shape: {H_pred.shape}")
            
            # Test physics discriminator
            discriminator = PhysicsDiscriminator(energy_grid, max_len_dna)
            discriminator = discriminator.to(device)
            
            T_pred, DOS_pred = discriminator(H_pred, gammaL, gammaR)
            print(f"Predicted transmission shape: {T_pred.shape}")
            print(f"Predicted DOS shape: {DOS_pred.shape}")
            
            # Compare with targets
            batch_size = T_pred.size(0)
            num_energy_points = T_pred.size(1)
            
            target_T = batch.transmission.view(batch_size, num_energy_points)
            target_DOS = batch.dos.view(batch_size, num_energy_points)
            
            t_loss = F.mse_loss(T_pred, target_T)
            dos_loss = F.mse_loss(DOS_pred, target_DOS)
            
            print(f"Transmission MSE: {t_loss.item():.4f}")
            print(f"DOS MSE: {dos_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components separately"""
    print("\n" + "=" * 50)
    print("Testing Individual Components")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test data generation
    print("1. Testing data generation...")
    try:
        train_loader, val_loader, energy_grid = create_data_loaders(
            num_samples=10, seq_length=6, batch_size=4
        )
        print("✓ Data generation successful")
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False
    
    # Test generator model
    print("2. Testing generator model...")
    try:
        generator = DNAHamiltonianGNN(
            energy_grid=energy_grid,
            max_len_dna=6
        ).to(device)
        
        # Test forward pass
        batch = next(iter(train_loader))
        batch = batch.to(device)
        
        # Debug: Check input shapes
        print(f"Debug: batch.x shape = {batch.x.shape}")
        print(f"Debug: batch.edge_attr shape = {batch.edge_attr.shape}")
        print(f"Debug: Expected node_features = 6, edge_features = 5")
        
        H_pred = generator(batch)
        print(f"✓ Generator forward pass successful, output shape: {H_pred.shape}")
    except Exception as e:
        print(f"✗ Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test physics discriminator
    print("3. Testing physics discriminator...")
    try:
        discriminator = PhysicsDiscriminator(energy_grid, max_len_dna=6)
        discriminator = discriminator.to(device)
        
        gammaL = np.array([0.1] + [0.0] * 11)
        gammaR = np.array([0.0] * 11 + [0.1])
        
        T_pred, DOS_pred = discriminator(H_pred, gammaL, gammaR)
        print(f"✓ Physics discriminator successful")
        print(f"  Transmission shape: {T_pred.shape}")
        print(f"  DOS shape: {DOS_pred.shape}")
    except Exception as e:
        print(f"✗ Physics discriminator test failed: {e}")
        return False
    
    print("✓ All individual components working!")
    return True


if __name__ == "__main__":
    print("Starting Physics-Informed GAN Tests")
    print("=" * 60)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test full training
        training_ok = test_physics_informed_training()
        
        if training_ok:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! Physics-informed GAN is working correctly.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Training test failed. Check the error messages above.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Component tests failed. Check the error messages above.")
        print("=" * 60) 