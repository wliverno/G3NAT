import numpy as np
from typing import List, Tuple
import random


def create_sample_data(num_samples: int = 1000, seq_length: int = 10, 
                      num_energy_points: int = 100) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate random DNA sequences
    bases = ['A', 'T', 'G', 'C']
    complementary_bases = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    sequences = []
    for _ in range(num_samples):
        length = np.random.randint(1,seq_length)
        seq = ''.join(np.random.choice(bases, length))
        sequences.append(seq)
        
        
    
    # Energy grid (in eV)
    energy_grid = np.linspace(-3, 3, num_energy_points)
    
    # Generate synthetic DOS and transmission data
    dos_data, transmission_data = [], []
    for seq in sequences:
        seq_complementary = ''.join(complementary_bases[base] for base in seq)[::-1]
        dos, trans = getTransmissionDOS(
            seq=seq, 
            seq_complementary=seq_complementary, 
            energy_grid=energy_grid
        )
        dos_data.append(np.log10(dos))
        transmission_data.append(np.log10(trans))
    
    return sequences, dos_data, transmission_data, energy_grid

def getTransmissionDOS(seq: str,
                       seq_complementary: str = None,
                       energy_grid: np.ndarray = None,
                       GammaL: np.ndarray = None,
                       GammaR: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transmission and DOS for a given sequence.

    Args:
        seq: Primary DNA sequence 5' to 3'
        seq_complementary: Complementary DNA sequence 5' to 3' (optional)
        energy_grid: Energy grid
        GammaL: Left coupling value list
        GammaR: Right coupling value list
        complementary_bases: Dictionary of complementary bases

    Returns:
        Tuple of transmission and DOS data
    """         
    if seq_complementary is None:
        seq_complementary = '_' * len(seq)
    if GammaL is None:
        GammaL = np.zeros(len(seq)*2)
        GammaL[0] = 0.1 # Default coupling strength on first site of primary strand
    if GammaR is None:
        GammaR = np.zeros(len(seq)*2)
        GammaR[len(seq)-1] = 0.1 # Default coupling strength on last site of primary strand
    if energy_grid is None:
        energy_grid = np.linspace(-3, 3, 100)

    assert len(seq) == len(seq_complementary), "Primary and complementary sequences must have the same length"
    assert len(GammaL) == len(seq)*2, "GammaL must have the same length as the primary sequence"
    assert len(GammaR) == len(seq)*2, "GammaR must have the same length as the primary sequence"

    # Initialize Hamiltonian and coupling matrices
    H = np.zeros((len(seq)*2, len(seq)*2))

    # From Roche et al, 2003 10.1103/PhysRevLett.91.228101
    onsite_energies = {
        'A': -0.49,
        'T': -1.39,
        'G': 0.00,
        'C': -1.12,
        '_': 0.00,
    }

    # From Voityuk et al, 2001 (10.1063/1.1352035)
    HBond_energies = {
        'AA': 0.0,
        'CC': 0.0,
        'GG': 0.0,
        'TT': 0.0,
        'AT': 0.034,
        'AG': 0.0,
        'AC': 0.0,
        'CT': 0.0,
        'CG': 0.050,
        'CA': 0.0, 
        'GT': 0.0,
        'GC': 0.050,
        'GA': 0.0,
        'TA': 0.034,
        'TC': 0.0,
        'TG': 0.0,
    }
    nn_energies = {
        'AA': 0.030,
        'CC': 0.041,
        'GG': 0.084,
        'TT': 0.158,
        'AT': 0.105,
        'AG': 0.049,
        'AC': 0.061,
        'CT': 0.100,
        'CG': 0.042,
        'CA': 0.029, 
        'GT': 0.137,
        'GC': 0.110,
        'GA': 0.089,
        'TA': 0.086,
        'TC': 0.076,
        'TG': 0.085,
    }

    # Fill in the Hamiltonian
    full_seq = seq + seq_complementary[::-1]
    for i in range(len(full_seq)):
        for j in range(len(full_seq)):
            base_1 = full_seq[i]
            base_2 = full_seq[j]
            if i == j:
                H[i, j] = onsite_energies[base_1]
            elif base_1 == '_' or base_2 == '_':
                continue
            elif i%len(seq) == j%len(seq):
                H[i, j] = HBond_energies[base_1 + base_2]
            elif i == j+1 and i//len(seq) == j//len(seq):
                BP = base_1 + base_2
                H[i, j] = nn_energies[BP]
            elif i == j-1 and i//len(seq) == j//len(seq):
                BP = base_2 + base_1
                H[i, j] = nn_energies[BP]
    
    # Start energy grid loop
    transmission_data = np.zeros(len(energy_grid))
    dos_data = np.zeros(len(energy_grid))
    for n, energy in enumerate(energy_grid):
        # Set up Green's function
        sumSig = -0.5j * (np.diag(GammaL) + np.diag(GammaR))
        Gr = np.linalg.inv(np.eye(len(seq)*2)*energy - H - sumSig)

        # Calculate transmission
        gamma1Gr = np.array([GammaL * row for row in Gr])
        gamma2Ga = np.array([GammaR * row for row in Gr.conj().T])
        T = 0
        for i in range(len(seq)*2):
            T += np.dot(gamma1Gr[i, :], gamma2Ga[:, i])
        transmission_data[n] = np.real(T)

        # Calculate DOS
        DOS = -1 *np.trace(np.imag(Gr))
        dos_data[n] = DOS
        
    return transmission_data, dos_data

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