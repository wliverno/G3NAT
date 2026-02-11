"""
Synthetic data generation for DNA transport simulations.
"""
import numpy as np
from typing import List, Tuple
import random

from g3nat.utils import create_hamiltonian, calculate_NEGF


def generate_tight_binding_data(num_samples: int = 1000, seq_length: int = 10,
                                num_energy_points: int = 100, min_length: int = -1) -> Tuple[List[str], List[str], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Generate sample data for demonstration."""
    np.random.seed(42)

    # Generate random DNA sequences
    bases = ['A', 'T', 'G', 'C']
    complementary_bases = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    primary_sequences = []
    complementary_sequences = []
    for _ in range(num_samples):
        if min_length == -1:
            min_length = seq_length
        length = np.random.randint(min_length, seq_length+1)
        seq = ''.join(np.random.choice(bases, length))
        seq_complementary = ''.join(complementary_bases[base] for base in seq)[::-1]
        # Remove bases from complementary sequence for 1 in 10 sequences
        num_remove = np.random.randint(0, length) if np.random.random() > 0.9 else 0
        pos = np.random.choice(range(length), size=num_remove, replace=False)
        seq_complementary = ''.join(seq_complementary[i] if i not in pos else '_' for i in range(length))
        # Add to list
        primary_sequences.append(seq)
        complementary_sequences.append(seq_complementary)

    # Energy grid (in eV)
    energy_grid = np.linspace(-3, 3, num_energy_points)

    # Generate synthetic DOS and transmission data
    dos_data, transmission_data = [], []
    for seq, seq_complementary in zip(primary_sequences, complementary_sequences):
        trans, dos = getTransmissionDOS(
            seq=seq,
            seq_complementary=seq_complementary,
            energy_grid=energy_grid
        )
        dos_data.append(np.log10(dos))
        transmission_data.append(np.log10(trans))

    return primary_sequences, complementary_sequences, dos_data, transmission_data, energy_grid


def getTransmissionDOS(seq: str,
                       seq_complementary: str = None,
                       energy_grid: np.ndarray = None,
                       GammaL: np.ndarray = None,
                       GammaR: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transmission and DOS for a given sequence using NEGF method.
    This is a convenience function that combines Hamiltonian generation and NEGF calculation.

    Args:
        seq: Primary DNA sequence 5' to 3'
        seq_complementary: Complementary DNA sequence 5' to 3' (optional)
        energy_grid: Energy grid
        GammaL: Left coupling vector for DNA nodes only (optional)
        GammaR: Right coupling vector for DNA nodes only (optional)

    Returns:
        Tuple of transmission and DOS data (raw values, not log10)
    """
    if energy_grid is None:
        energy_grid = np.linspace(-3, 3, 100)

    # Create Hamiltonian and default gamma vectors
    H, GammaL_default, GammaR_default = create_hamiltonian(seq, seq_complementary)

    # Use provided gamma vectors or defaults
    if GammaL is None:
        GammaL = GammaL_default
    if GammaR is None:
        GammaR = GammaR_default

    # Calculate using NEGF
    return calculate_NEGF(H, GammaL, GammaR, energy_grid)


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
