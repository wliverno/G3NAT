#!/usr/bin/env python3
"""
Debug script to test checkpoint functionality directly.
"""

import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Add current directory to path to import modules
sys.path.append('.')

from models import DNATransportGNN, train_model
from data_generator import create_sample_data
from dataset import create_dna_dataset
from main import save_checkpoint, load_checkpoint, save_progress_file, create_checkpoint_callback, create_progress_callback

def test_checkpoint_functionality():
    """Test checkpoint functionality directly."""
    
    print("=== Direct Checkpoint Test ===")
    
    # Create test data
    print("1. Creating test data...")
    primary_sequences, complementary_sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=20,
        seq_length=4,
        num_energy_points=50
    )
    
    # Create dataset
    dos_data_array = np.array(dos_data)
    transmission_data_array = np.array(transmission_data)
    
    dataset = create_dna_dataset(
        sequences=primary_sequences,
        dos_data=dos_data_array,
        transmission_data=transmission_data_array,
        energy_grid=energy_grid,
        complementary_sequences=complementary_sequences,
        left_contact_positions=0,
        right_contact_positions=-1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    # Split dataset
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    
    dataset_size = len(dataset)
    train_indices, val_indices = train_test_split(
        range(dataset_size), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model
    print("2. Initializing model...")
    model = DNATransportGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        output_dim=50,
        dropout=0.1
    )
    
    # Setup checkpoint directory
    checkpoint_dir = "./debug_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks
    args = {
        'learning_rate': 1e-3,
        'num_epochs': 5,
        'checkpoint_frequency': 2
    }
    
    checkpoint_callback = create_checkpoint_callback(checkpoint_dir, args, energy_grid)
    progress_callback = create_progress_callback(checkpoint_dir, args)
    
    print("3. Starting training with checkpointing...")
    
    # Train for a few epochs
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-3,
        device='cpu',
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=2,
        checkpoint_callback=checkpoint_callback,
        progress_callback=progress_callback
    )
    
    print("4. Checking checkpoint files...")
    
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"Checkpoint directory contents: {files}")
        
        if "checkpoint_latest.pth" in files:
            print("✓ Checkpoint file found!")
            
            # Test loading checkpoint
            print("5. Testing checkpoint loading...")
            try:
                start_epoch, loaded_train_losses, loaded_val_losses, optimizer = load_checkpoint(
                    os.path.join(checkpoint_dir, 'checkpoint_latest.pth'),
                    model, torch.device('cpu')
                )
                print(f"✓ Checkpoint loaded successfully! Epoch: {start_epoch}")
                print(f"  Train losses: {len(loaded_train_losses)}")
                print(f"  Val losses: {len(loaded_val_losses)}")
                return True
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {e}")
                return False
        else:
            print("✗ No checkpoint file found")
            return False
    else:
        print("✗ Checkpoint directory not found")
        return False

if __name__ == "__main__":
    success = test_checkpoint_functionality()
    
    if success:
        print("\n=== Debug Test PASSED ===")
        print("Checkpoint functionality is working correctly!")
    else:
        print("\n=== Debug Test FAILED ===")
        print("Checkpoint functionality has issues.")
        sys.exit(1) 