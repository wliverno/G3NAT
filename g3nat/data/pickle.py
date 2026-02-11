"""
Utilities for loading DNA transport data from pickle files.

This module provides functions to load and parse pickle files containing
DNA transport calculations, including sequences, DOS, transmission, and contact information.
"""

import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import glob

complementary_bases = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def load_single_pickle(pickle_path: str) -> Optional[Dict]:
    """
    Load a single pickle file and extract DNA transport data.

    Args:
        pickle_path: Path to the pickle file

    Returns:
        Dictionary with keys: sequence, complementary_sequence, dos, transmission,
        energy_grid, contact_type, coupling, left_contact_pos, right_contact_pos
        Returns None if file cannot be loaded or is missing required fields.
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Extract required fields
        if 'sequence' not in data:
            print(f"Warning: {pickle_path} missing 'sequence' field, skipping")
            return None

        sequence = data['sequence'].upper()

        # Extract complementary sequence if available, otherwise generate it using complementary_bases
        complementary_sequence = data.get('complementary_sequence',
                                          ''.join(complementary_bases[base] for base in sequence)[::-1])

        # Extract DOS and Transmission log10 transformed
        dos = np.log10(np.array(data['DOS']))
        transmission = np.log10(np.array(data['T']))
        energy_grid = np.array(data['Egrid'])
        energy_grid = energy_grid - np.mean(energy_grid)

        # Extract contact information
        contacts = data.get('contacts', {})
        contact_type = contacts.get('contact_type', 'same')  # default to 'same'
        coupling = contacts.get('coupling_eV', 0.1)  # default coupling

        # Determine contact positions based on contact_type
        # 'same': both contacts on primary strand (5' to 3')
        # 'cross': left on primary 5', right on complementary 5'
        if contact_type == 'same':
            left_contact_pos = 0  # First position of primary strand
            right_contact_pos = len(sequence) - 1  # Last position of primary strand
            left_strand = 'primary'
            right_strand = 'primary'
        elif contact_type == 'cross':
            left_contact_pos = 0  # First position of primary strand
            right_contact_pos = 0  # First position of complementary strand
            left_strand = 'primary'
            right_strand = 'complementary'
        else:
            print(f"Warning: Unknown contact_type '{contact_type}' in {pickle_path}, defaulting to 'same'")
            left_contact_pos = 0
            right_contact_pos = len(sequence) - 1
            left_strand = 'primary'
            right_strand = 'primary'

        return {
            'sequence': sequence,
            'complementary_sequence': complementary_sequence,
            'dos': dos,
            'transmission': transmission,
            'energy_grid': energy_grid,
            'contact_type': contact_type,
            'coupling': coupling,
            'left_contact_pos': (left_strand, left_contact_pos),
            'right_contact_pos': (right_strand, right_contact_pos),
            'filename': os.path.basename(pickle_path)
        }

    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None


def load_pickle_directory(directory: str, pattern: str = "*.pkl") -> Tuple[List[str], List[str],
                                                                            np.ndarray, np.ndarray,
                                                                            np.ndarray, List[Dict]]:
    """
    Load all pickle files from a directory.

    Args:
        directory: Directory containing pickle files
        pattern: Glob pattern for pickle files (default: "*.pkl")

    Returns:
        Tuple of:
        - sequences: List of primary DNA sequences
        - complementary_sequences: List of complementary DNA sequences
        - dos_data: Array of DOS data [num_samples, num_energy_points]
        - transmission_data: Array of transmission data [num_samples, num_energy_points]
        - energy_grid: Energy grid array [num_energy_points] (from first file)
        - contact_configs: List of contact configuration dictionaries
    """
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(directory, pattern))

    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {directory} matching pattern '{pattern}'")

    print(f"Found {len(pickle_files)} pickle files in {directory}")

    sequences = []
    complementary_sequences = []
    dos_list = []
    transmission_list = []
    contact_configs = []
    energy_grid = None

    # Load each file
    for pickle_file in sorted(pickle_files):
        data = load_single_pickle(pickle_file)

        if data is None:
            continue

        # Store energy grid from first valid file
        if energy_grid is None:
            energy_grid = data['energy_grid']
        else:
            # Verify energy grids match
            if not np.allclose(energy_grid, data['energy_grid']):
                print(f"Warning: Energy grid mismatch in {data['filename']}, skipping")
                continue

        # Append data
        sequences.append(data['sequence'])
        complementary_sequences.append(data['complementary_sequence'])
        dos_list.append(data['dos'])
        transmission_list.append(data['transmission'])

        # Store contact configuration
        contact_configs.append({
            'contact_type': data['contact_type'],
            'coupling': data['coupling'],
            'left_contact_pos': data['left_contact_pos'],
            'right_contact_pos': data['right_contact_pos'],
            'filename': data['filename']
        })

    if len(sequences) == 0:
        raise ValueError(f"No valid data loaded from {directory}")

    # Convert to arrays
    dos_data = np.array(dos_list)
    transmission_data = np.array(transmission_list)

    print(f"Successfully loaded {len(sequences)} samples")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    print(f"DOS shape: {dos_data.shape}")
    print(f"Transmission shape: {transmission_data.shape}")

    return sequences, complementary_sequences, dos_data, transmission_data, energy_grid, contact_configs


if __name__ == "__main__":
    # Test loading a single file
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]

        if os.path.isfile(path):
            print(f"Loading single file: {path}")
            data = load_single_pickle(path)
            if data:
                print(f"\nSequence: {data['sequence']}")
                print(f"Complementary: {data['complementary_sequence']}")
                print(f"Contact type: {data['contact_type']}")
                print(f"Coupling: {data['coupling']} eV")
                print(f"Left contact: {data['left_contact_pos']}")
                print(f"Right contact: {data['right_contact_pos']}")
                print(f"Energy grid points: {len(data['energy_grid'])}")
                print(f"DOS shape: {data['dos'].shape}")
                print(f"Transmission shape: {data['transmission'].shape}")
        elif os.path.isdir(path):
            print(f"Loading directory: {path}")
            sequences, comp_seqs, dos, trans, egrid, configs = load_pickle_directory(path)
            print(sequences)
            print(comp_seqs)
            print(dos)
            print(trans)
            print(egrid)
            print(configs)
            print(f"\nLoaded {len(sequences)} sequences")
            print(f"First sequence: {sequences[0]}")
            print(f"Contact types: {[c['contact_type'] for c in configs]}")
        else:
            print(f"Path not found: {path}")
    else:
        print("Usage: python load_pickle_data.py <pickle_file_or_directory>")
