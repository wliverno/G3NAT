import numpy as np
from typing import List, Tuple
import random


def create_sample_data(num_samples: int = 1000, seq_length: int = 8, 
                      num_energy_points: int = 100) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate random DNA sequences
    bases = ['A', 'T', 'G', 'C']
    sequences = []
    for _ in range(num_samples):
        seq = ''.join(np.random.choice(bases, seq_length))
        sequences.append(seq)
    
    # Energy grid (in eV)
    energy_grid = np.linspace(-3, 3, num_energy_points)
    
    # Generate synthetic DOS and transmission data
    dos_data = np.zeros((num_samples, num_energy_points))
    transmission_data = np.zeros((num_samples, num_energy_points))
    
    for i, seq in enumerate(sequences):
        # Simple model: DOS depends on GC content and sequence
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        
        # Create peaks around HOMO/LUMO
        homo_energy = -1.5 + 0.5 * gc_content + 0.1 * np.random.randn()
        lumo_energy = 1.0 + 0.3 * gc_content + 0.1 * np.random.randn()
        
        # DOS with Gaussian peaks
        dos_data[i] = (np.exp(-(energy_grid - homo_energy)**2 / 0.2) + 
                      np.exp(-(energy_grid - lumo_energy)**2 / 0.3))
        
        # Transmission with transport gap
        transmission_data[i] = 1 / (1 + np.exp(10 * (np.abs(energy_grid) - 0.5)))
        
        # Add some noise
        dos_data[i] += 0.1 * np.random.randn(num_energy_points)
        transmission_data[i] += 0.05 * np.random.randn(num_energy_points)
        transmission_data[i] = np.clip(transmission_data[i], 0, 1)
    
    return sequences, dos_data, transmission_data, energy_grid


def generate_realistic_dna_sequences(num_samples: int = 1000, 
                                   min_length: int = 6, 
                                   max_length: int = 12) -> List[str]:
    """
    Generate more realistic DNA sequences with biological constraints.
    
    Args:
        num_samples: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        List of DNA sequences
    """
    bases = ['A', 'T', 'G', 'C']
    sequences = []
    
    for _ in range(num_samples):
        # Random length within bounds
        seq_length = random.randint(min_length, max_length)
        
        # Generate sequence with some biological constraints
        seq = []
        for i in range(seq_length):
            if i > 0:
                # Avoid long runs of the same base (biological constraint)
                if seq[-1] in ['A', 'T'] and random.random() < 0.3:
                    # Prefer G or C after A/T
                    seq.append(random.choice(['G', 'C']))
                elif seq[-1] in ['G', 'C'] and random.random() < 0.3:
                    # Prefer A or T after G/C
                    seq.append(random.choice(['A', 'T']))
                else:
                    seq.append(random.choice(bases))
            else:
                seq.append(random.choice(bases))
        
        sequences.append(''.join(seq))
    
    return sequences


def create_sequence_variants(base_sequence: str, num_variants: int = 10) -> List[str]:
    """
    Create variants of a base sequence by introducing mutations.
    
    Args:
        base_sequence: Original DNA sequence
        num_variants: Number of variants to create
        
    Returns:
        List of variant sequences
    """
    bases = ['A', 'T', 'G', 'C']
    variants = [base_sequence]
    
    for _ in range(num_variants - 1):
        variant = list(base_sequence)
        
        # Random number of mutations (1-3)
        num_mutations = random.randint(1, min(3, len(base_sequence)))
        
        for _ in range(num_mutations):
            # Random position
            pos = random.randint(0, len(variant) - 1)
            # Random new base (different from current)
            current_base = variant[pos]
            new_base = random.choice([b for b in bases if b != current_base])
            variant[pos] = new_base
        
        variants.append(''.join(variant))
    
    return variants


def calculate_sequence_features(sequences: List[str]) -> dict:
    """
    Calculate various features for DNA sequences.
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        Dictionary with sequence features
    """
    features = {
        'gc_content': [],
        'length': [],
        'base_counts': {'A': [], 'T': [], 'G': [], 'C': []},
        'purine_content': [],
        'pyrimidine_content': []
    }
    
    for seq in sequences:
        # GC content
        gc_count = seq.count('G') + seq.count('C')
        features['gc_content'].append(gc_count / len(seq))
        
        # Length
        features['length'].append(len(seq))
        
        # Base counts
        for base in ['A', 'T', 'G', 'C']:
            features['base_counts'][base].append(seq.count(base))
        
        # Purine content (A, G)
        purine_count = seq.count('A') + seq.count('G')
        features['purine_content'].append(purine_count / len(seq))
        
        # Pyrimidine content (T, C)
        pyrimidine_count = seq.count('T') + seq.count('C')
        features['pyrimidine_content'].append(pyrimidine_count / len(seq))
    
    return features


def save_sequences_to_file(sequences: List[str], filename: str):
    """
    Save DNA sequences to a text file.
    
    Args:
        sequences: List of DNA sequences
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence_{i+1}\n")
            f.write(f"{seq}\n")


def load_sequences_from_file(filename: str) -> List[str]:
    """
    Load DNA sequences from a text file.
    
    Args:
        filename: Input filename
        
    Returns:
        List of DNA sequences
    """
    sequences = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line.startswith('>') and line:  # Skip header lines
            sequences.append(line.upper())
    
    return sequences 