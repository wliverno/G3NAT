"""
Example demonstrating how to use gamma debugging functionality in the dataset.

This example shows how to:
1. Create a dataset with specific gamma vectors for each sequence
2. Access gamma vectors during training for debugging
3. Verify that the gamma values used in graph creation match those stored in the dataset
"""

import numpy as np
import torch
import sys
import os

# Add the parent directory to the path so we can import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import create_dna_dataset, sequence_to_graph, create_default_gamma_vectors

def main():
    # Example DNA sequences
    sequences = ['ATCG', 'GCTA', 'TAGC']
    
    # Create synthetic data
    num_samples = len(sequences)
    num_energy_points = 50
    energy_grid = np.linspace(-3, 3, num_energy_points)
    
    # Synthetic DOS and transmission data
    dos_data = np.random.rand(num_samples, num_energy_points)
    transmission_data = np.random.rand(num_samples, num_energy_points)
    
    # Create default gamma vectors (seq_length * 2 for each sequence)
    max_seq_length = max(len(seq) for seq in sequences)
    gamma_l, gamma_r = create_default_gamma_vectors(sequences)
    
    print("Creating dataset with gamma vectors:")
    print(f"Gamma vectors shape: {gamma_l.shape}")
    print(f"Default gamma L: {gamma_l}")
    print(f"Default gamma R: {gamma_r}")
    print()
    
    # Create dataset with gamma vectors
    dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=dos_data,
        transmission_data=transmission_data,
        energy_grid=energy_grid,
        gamma_l=gamma_l,
        gamma_r=gamma_r
    )
    
    print("Dataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    print()
    
    # Access individual samples and verify gamma vectors
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Sequence: {sequences[i]}")
        print(f"  Gamma L vector: {sample.gamma_l.tolist()}")
        print(f"  Gamma R vector: {sample.gamma_r.tolist()}")
        
        # Find non-zero gamma values
        non_zero_l = torch.nonzero(sample.gamma_l).flatten().tolist()
        non_zero_r = torch.nonzero(sample.gamma_r).flatten().tolist()
        print(f"  Non-zero Gamma L positions: {non_zero_l}")
        print(f"  Non-zero Gamma R positions: {non_zero_r}")
        
        # Verify that the gamma values in the graph edges match the stored values
        # Find contact edges in the graph
        edge_attr = sample.edge_attr
        edge_index = sample.edge_index
        
        # Contact edges have edge_attr[:, 2] == 1 (contact type)
        contact_mask = edge_attr[:, 2] == 1
        contact_edges = edge_index[:, contact_mask]
        contact_couplings = edge_attr[contact_mask, 4]  # Coupling strength is in column 4
        
        print(f"  Contact couplings in graph: {contact_couplings.unique().tolist()}")
        print()
    
    # Example of custom gamma vectors
    print("Creating custom gamma vectors:")
    custom_gamma_l = np.zeros((num_samples, max_seq_length * 2))
    custom_gamma_r = np.zeros((num_samples, max_seq_length * 2))
    
    # Set custom contact positions
    custom_gamma_l[0, 1] = 0.15  # Sequence 0: left contact at position 1
    custom_gamma_r[0, 2] = 0.18  # Sequence 0: right contact at position 2
    custom_gamma_l[1, 0] = 0.12  # Sequence 1: left contact at position 0
    custom_gamma_r[1, 3] = 0.25  # Sequence 1: right contact at position 3
    custom_gamma_l[2, 2] = 0.20  # Sequence 2: left contact at position 2
    custom_gamma_r[2, 1] = 0.16  # Sequence 2: right contact at position 1
    
    print(f"Custom gamma L: {custom_gamma_l}")
    print(f"Custom gamma R: {custom_gamma_r}")
    print()
    
    # Create dataset with custom gamma vectors
    custom_dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=dos_data,
        transmission_data=transmission_data,
        energy_grid=energy_grid,
        gamma_l=custom_gamma_l,
        gamma_r=custom_gamma_r
    )
    
    # Verify custom gamma values
    for i in range(len(custom_dataset)):
        sample = custom_dataset[i]
        non_zero_l = torch.nonzero(sample.gamma_l).flatten().tolist()
        non_zero_r = torch.nonzero(sample.gamma_r).flatten().tolist()
        print(f"Custom sample {i}:")
        print(f"  Non-zero Gamma L positions: {non_zero_l}")
        print(f"  Non-zero Gamma R positions: {non_zero_r}")
        print(f"  Gamma L values: {[sample.gamma_l[pos].item() for pos in non_zero_l]}")
        print(f"  Gamma R values: {[sample.gamma_r[pos].item() for pos in non_zero_r]}")
        print()
    
    # Example of how to use gamma vectors during training
    print("Example training loop with gamma vector debugging:")
    for i, sample in enumerate(dataset):
        # During training, you can access gamma vectors for debugging
        gamma_l_vector = sample.gamma_l
        gamma_r_vector = sample.gamma_r
        
        # Find active contact positions
        active_l_positions = torch.nonzero(gamma_l_vector).flatten()
        active_r_positions = torch.nonzero(gamma_r_vector).flatten()
        
        print(f"Training sample {i}:")
        print(f"  Active left contacts: {active_l_positions.tolist()}")
        print(f"  Active right contacts: {active_r_positions.tolist()}")
        print(f"  Left coupling strengths: {[gamma_l_vector[pos].item() for pos in active_l_positions]}")
        print(f"  Right coupling strengths: {[gamma_r_vector[pos].item() for pos in active_r_positions]}")
        
        # Your model forward pass would go here
        # model_output = model(sample)
        
        # You can log gamma vectors for debugging
        # logger.info(f"Sample {i} gamma vectors: L={gamma_l_vector.tolist()}, R={gamma_r_vector.tolist()}")
    
    print("\nGamma debugging example completed!")

if __name__ == "__main__":
    main() 