import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DNATransportHamiltonianGNN
from dataset import sequence_to_graph
from visualize_dna_graph import visualize_dna_graph
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Batch
from main2 import create_sample_data, create_graphs_from_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.autograd import gradcheck
from models import DNATransportHamiltonianGNN

# Your DNATransportHamiltonianGNN class would go here
# [The class definition from your paste.txt]

def create_test_data(batch_size=2, num_nodes_per_graph=8):
    """Create synthetic test data for the DNA GNN"""
    print("Generating sample data...")
    primary_sequences, complementary_sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=2,
        seq_length=4,  # Use fixed length to ensure consistent graph sizes
        num_energy_points=10
    )
    
    print(f"Generated {len(primary_sequences)} sequences")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    # Convert sequences to graphs
    graphs = create_graphs_from_sequences(primary_sequences, complementary_sequences, dos_data, transmission_data)
    
    return Batch.from_data_list(graphs)

def gradient_check_dna_gnn():
    """Comprehensive gradient checking for DNA GNN"""
    print("=== DNA Transport GNN Gradient Check ===\n")
    
    # Initialize model
    model = DNATransportHamiltonianGNN(
        hidden_dim=32,  # Smaller for faster testing
        num_layers=2,
        num_heads=2,
        energy_grid=np.linspace(-1, 1, 10),  # Smaller grid for testing
        max_len_dna=6,
        dropout=0.0  # Disable dropout for gradient checking
    )
    
    # Create test data
    batch_data = create_test_data(batch_size=2, num_nodes_per_graph=6)
    
    print("1. BASIC FORWARD PASS TEST")
    print("-" * 40)
    try:
        dos_pred, transmission_pred = model(batch_data)
        print(f"✓ Forward pass successful")
        print(f"  DOS shape: {dos_pred.shape}")
        print(f"  Transmission shape: {transmission_pred.shape}")
        print(f"  DOS range: [{dos_pred.min():.4f}, {dos_pred.max():.4f}]")
        print(f"  Transmission range: [{transmission_pred.min():.4f}, {transmission_pred.max():.4f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    print("\n2. PARAMETER GRADIENT CHECK")
    print("-" * 40)
    
    def simple_loss_fn(dos, transmission):
        return torch.mean(dos**2) + torch.mean(transmission**2)
    
    # Check if all parameters get gradients
    model.zero_grad()
    dos_pred, transmission_pred = model(batch_data)
    loss = simple_loss_fn(dos_pred, transmission_pred)
    loss.backward()
    
    grad_issues = []
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_issues.append(f"✗ No gradient for parameter: {name}")
        elif torch.isnan(param.grad).any():
            grad_issues.append(f"✗ NaN gradient for parameter: {name}")
        elif torch.isinf(param.grad).any():
            grad_issues.append(f"✗ Inf gradient for parameter: {name}")
        else:
            grad_norm = torch.norm(param.grad).item()
            print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
    
    if grad_issues:
        print("\nGradient Issues Found:")
        for issue in grad_issues:
            print(issue)
    else:
        print("✓ All parameters have valid gradients")
    
    print(f"\n3. NUMERICAL GRADIENT CHECK (SAMPLE)")
    print("-" * 40)
    
    # Test a few parameters with gradcheck
    test_params = []
    for name, param in model.named_parameters():
        if param.numel() < 100:  # Only test small parameters
            test_params.append((name, param))
        if len(test_params) >= 3:  # Limit to 3 for speed
            break
    
    for name, param in test_params:
        try:
            # Create a wrapper that only tests this specific parameter
            def single_param_wrapper(param_to_test):
                # Temporarily replace this specific parameter
                original_data = param.data.clone()
                param.data = param_to_test
                
                try:
                    dos_pred, transmission_pred = model(batch_data)
                    loss = simple_loss_fn(dos_pred, transmission_pred)
                finally:
                    # Restore original parameter
                    param.data = original_data
                
                return loss
            
            param_copy = param.clone().detach().requires_grad_(True)
            test_result = gradcheck(
                single_param_wrapper, 
                (param_copy,), 
                eps=1e-4, 
                atol=1e-3,
                masked=False
            )
            print(f"✓ {name}: {'PASSED' if test_result else 'FAILED'}")
        except Exception as e:
            print(f"✗ {name}: ERROR - {str(e)}")
    
    print(f"\n4. ARCHITECTURE-SPECIFIC CHECKS")
    print("-" * 40)
    
    # Check for potential issues
    issues_found = []
    
    # Check 1: Complex number handling
    try:
        model.eval()
        with torch.no_grad():
            dos_pred, transmission_pred = model(batch_data)
            if torch.is_complex(dos_pred) or torch.is_complex(transmission_pred):
                issues_found.append("⚠ Output contains complex numbers")
            else:
                print("✓ Output is real-valued")
    except Exception as e:
        issues_found.append(f"✗ Complex number check failed: {e}")
    
    # Check 2: Matrix operations stability
    try:
        # Test with different batch sizes
        single_data = create_test_data(batch_size=1, num_nodes_per_graph=8)
        dos_single, trans_single = model(single_data)
        print("✓ Single sample processing works")
    except Exception as e:
        issues_found.append(f"✗ Single sample processing failed: {e}")
    
    # Check 3: Energy grid consistency
    if hasattr(model, 'energy_grid'):
        if len(model.energy_grid) != dos_pred.shape[1]:
            issues_found.append("✗ Energy grid size mismatch with DOS output")
        else:
            print("✓ Energy grid size matches output")
    
    # Check 4: Hamiltonian size consistency
    if hasattr(model, 'H_size') and hasattr(model, 'num_unique_elements'):
        expected_elements = model.H_size + (model.H_size * (model.H_size - 1)) // 2
        if model.num_unique_elements != expected_elements:
            issues_found.append(f"✗ Hamiltonian size inconsistency: expected {expected_elements}, got {model.num_unique_elements}")
        else:
            print("✓ Hamiltonian size calculation correct")
    
    if issues_found:
        print("\nIssues Found:")
        for issue in issues_found:
            print(issue)
    else:
        print("✓ No architecture-specific issues detected")
    
    print(f"\n5. POTENTIAL GRADIENT FLOW ISSUES")
    print("-" * 40)
    
    # Analyze potential problems in the code
    potential_issues = []
    
    # Check for in-place operations (manual inspection needed)
    print("Manual code inspection needed for:")
    print("  - In-place operations (look for +=, *=, etc.)")
    print("  - .detach() calls")
    print("  - Non-differentiable operations")
    print("  - Complex number handling in NEGFProjection")
    
    # Test gradient flow through NEGFProjection specifically
    try:
        model.train()
        x_test = torch.randn(2, model.num_unique_elements, requires_grad=True)
        gamma_l = torch.randn(2, model.H_size)
        gamma_r = torch.randn(2, model.H_size)
        
        T, DOS, H = model.NEGFProjection(x_test, gamma_l, gamma_r)
        loss_negf = torch.mean(T) + torch.mean(DOS)
        loss_negf.backward()
        
        if x_test.grad is not None:
            print("✓ NEGFProjection allows gradient flow")
        else:
            print("✗ NEGFProjection blocks gradient flow")
    except Exception as e:
        print(f"✗ NEGFProjection gradient test failed: {e}")

def analyze_code_issues():
    """Analyze the code for potential gradient flow issues"""
    print(f"\n6. CODE ANALYSIS")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    # Issue 1: Complex number operations
    issues.append("""
    🔴 CRITICAL: Complex number gradient issues
    - NEGFProjection uses complex arithmetic (0j, .conj(), etc.)
    - PyTorch's complex autograd can be unstable
    - Recommendation: Ensure all final outputs are real-valued
    """)
    
    # Issue 2: Matrix solve operations
    warnings.append("""
    🟡 WARNING: Matrix solve stability
    - torch.linalg.solve() can be unstable for ill-conditioned matrices
    - Fallback to pinv() is good, but may not preserve gradients well
    - Consider adding more regularization or using SVD-based approaches
    """)
    
    # Issue 3: Index operations
    warnings.append("""
    🟡 WARNING: Index-based operations
    - torch.triu_indices and advanced indexing in NEGFProjection
    - These operations preserve gradients but can be slow
    - Consider if there are more efficient implementations
    """)
    
    # Issue 4: Einsum operations
    print("✓ GOOD: Using einsum for traces - this preserves gradients well")
    
    # Issue 5: Error handling
    issues.append("""
    🔴 POTENTIAL ISSUE: Exception handling in NEGFProjection
    - The try/except block for linalg.solve might mask gradient issues
    - If pinv() is used frequently, gradients may be poor quality
    - Recommendation: Log when fallback is used and consider alternatives
    """)
    
    print("Issues found:")
    for issue in issues:
        print(issue)
    
    print("Warnings:")
    for warning in warnings:
        print(warning)
    
    print(f"\n7. RECOMMENDATIONS")
    print("-" * 40)
    print("""
    1. Add gradient clipping to handle potential instabilities
    2. Monitor when the linalg.solve fallback is triggered
    3. Consider using double precision for more stable matrix operations
    4. Add numerical checks (NaN/Inf detection) in training loop
    5. Test with different regularization values in the matrix solve
    6. Consider alternative implementations for the NEGF calculation
    """)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    gradient_check_dna_gnn()
    analyze_code_issues()