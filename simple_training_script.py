#!/usr/bin/env python3
"""
Simple training script for DNA Transport GNN using existing modules.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import existing modules
from data_generator import create_sample_data
from dataset import create_dna_dataset, sequence_to_graph
from models import train_physics_informed, DNAHamiltonianGNN, PhysicsDiscriminator
from utils import setup_logging, plot_training_curves, plot_dos_comparison, plot_transmission_comparison, calculate_metrics, print_metrics, create_output_directory, set_random_seed
from visualize_dna_graph import visualize_dna_graph
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def main():
    """Main training pipeline."""
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create output directory
    output_dir = create_output_directory("outputs", "simple_training_experiment")
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting simple DNA transport training experiment")
    
    # Parameters
    num_samples = 2000
    seq_length = 8
    num_energy_points = 100
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3
    max_len_dna = 8
    
    logger.info(f"Parameters: samples={num_samples}, seq_length={seq_length}, epochs={epochs}")
    
    # Generate data using existing function
    logger.info("Generating sample data...")
    primary_sequences, complementary_sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=num_samples,
        seq_length=seq_length,
        num_energy_points=num_energy_points
    )
    
    logger.info(f"Generated {len(primary_sequences)} sequences")
    logger.info(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    # Convert lists to numpy arrays
    dos_data_array = np.array(dos_data)
    transmission_data_array = np.array(transmission_data)
    
    # Create dataset using existing function
    logger.info("Creating dataset...")
    dataset = create_dna_dataset(
        sequences=primary_sequences,
        dos_data=dos_data_array,
        transmission_data=transmission_data_array,
        energy_grid=energy_grid,
        complementary_sequences=complementary_sequences,
        graph_converter_func=sequence_to_graph,
        left_contact_positions=0,
        right_contact_positions=0,  # Will be automatically updated to len(sequence) - 1 for each sequence
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    # Split dataset
    dataset_size = len(dataset)
    train_indices, val_indices = train_test_split(
        range(dataset_size), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Define contact parameters
    gammaL = np.array([0.1] + [0.0] * 15)  # Left contact on first site
    gammaR = np.array([0.0] * 7 + [0.1] + [0.0] * 8)  # Right contact on last site
    
    # Train model using existing function
    logger.info("Starting physics-informed training...")
    generator, train_losses, val_losses = train_physics_informed(
        train_loader=train_loader,
        val_loader=val_loader,
        energy_grid=energy_grid,
        max_len_dna=max_len_dna,
        gammaL=gammaL,
        gammaR=gammaR,
        epochs=epochs,
        lr=learning_rate,
        device=device
    )
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Plot training curves using existing function
    logger.info("Plotting training curves...")
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        filepath=os.path.join(output_dir, 'training_curves.png')
    )
    
    # Test prediction on a single sample
    logger.info("Testing prediction on validation sample...")
    generator.eval()
    with torch.no_grad():
        # Get a single batch
        batch = next(iter(val_loader))
        batch = batch.to(device)
        
        # Generate Hamiltonian
        H_pred = generator(batch)
        logger.info(f"Generated Hamiltonian shape: {H_pred.shape}")
        
        # Test physics discriminator
        discriminator = PhysicsDiscriminator(energy_grid, max_len_dna)
        discriminator = discriminator.to(device)
        
        T_pred, DOS_pred = discriminator(H_pred, gammaL, gammaR)
        logger.info(f"Predicted transmission shape: {T_pred.shape}")
        logger.info(f"Predicted DOS shape: {DOS_pred.shape}")
        
        # Compare with targets
        batch_size = T_pred.size(0)
        num_energy_points = T_pred.size(1)
        
        target_T = batch.transmission.view(batch_size, num_energy_points)
        target_DOS = batch.dos.view(batch_size, num_energy_points)
        
        # Calculate metrics using existing function
        t_metrics = calculate_metrics(target_T.cpu().numpy(), T_pred.cpu().numpy())
        dos_metrics = calculate_metrics(target_DOS.cpu().numpy(), DOS_pred.cpu().numpy())
        
        logger.info("Transmission metrics:")
        print_metrics(t_metrics, "  ")
        logger.info("DOS metrics:")
        print_metrics(dos_metrics, "  ")
        
        # Plot comparison for first sample using existing functions
        logger.info("Creating comparison plots...")
        plot_transmission_comparison(
            true_trans=target_T[0].cpu().numpy(),
            pred_trans=T_pred[0].cpu().numpy(),
            energy_grid=energy_grid,
            filepath=os.path.join(output_dir, 'transmission_comparison.png'),
            title="Transmission Comparison (First Sample)"
        )
        
        plot_dos_comparison(
            true_dos=target_DOS[0].cpu().numpy(),
            pred_dos=DOS_pred[0].cpu().numpy(),
            energy_grid=energy_grid,
            filepath=os.path.join(output_dir, 'dos_comparison.png'),
            title="DOS Comparison (First Sample)"
        )
    
    # Create a sample graph visualization using existing function
    logger.info("Creating sample graph visualization...")
    sample_seq = primary_sequences[0]
    sample_comp_seq = complementary_sequences[0]
    
    sample_graph = sequence_to_graph(
        primary_sequence=sample_seq,
        complementary_sequence=sample_comp_seq,
        left_contact_positions=0,
        right_contact_positions=len(sample_seq) - 1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    fig, ax = visualize_dna_graph(sample_graph, sample_seq, sample_comp_seq)
    plt.savefig(os.path.join(output_dir, 'sample_dna_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"All results saved to: {output_dir}")
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main() 