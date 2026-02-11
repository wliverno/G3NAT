import numpy as np
from typing import Tuple


def create_hamiltonian(seq: str,
                      seq_complementary: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Hamiltonian matrix and default gamma vectors for a DNA sequence.

    Args:
        seq: Primary DNA sequence 5' to 3'
        seq_complementary: Complementary DNA sequence 5' to 3' (optional)

    Returns:
        Tuple of (H_matrix, GammaL, GammaR)
    """
    if seq_complementary is None:
        seq_complementary = '_' * len(seq)

    assert len(seq) == len(seq_complementary), "Sequence and complementary sequence must have the same length"
    assert '_' not in seq, "Primary sequence cannot contain \'_\'"

    # Build tight-binding Hamiltonian parameters
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

    H = np.zeros((len(full_seq), len(full_seq)))
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

    ind = 0
    for i in range(len(full_seq)):
        if full_seq[i] == '_':
            H = np.delete(H, ind, axis=0)
            H = np.delete(H, ind, axis=1)
        else:
            ind += 1

    # - Left contact couples to the first PRIMARY base
    # - Right contact couples to the LAST PRIMARY base
    GammaL = np.zeros(ind)
    GammaL[0] = 0.1

    GammaR = np.zeros(ind)
    GammaR[len(seq)-1] = 0.1

    return H, GammaL, GammaR


def calculate_NEGF(H: np.ndarray,
                  GammaL: np.ndarray,
                  GammaR: np.ndarray,
                  energy_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transmission and DOS using NEGF method.
    This implementation should match NEGFProjection() in models.py exactly.

    Args:
        H: Hamiltonian matrix [H_size, H_size]
        GammaL: Left coupling vector [H_size]
        GammaR: Right coupling vector [H_size]
        energy_grid: Energy grid [num_energy_points]

    Returns:
        Tuple of transmission and DOS data (raw values, not log10)
    """
    H_size = H.shape[0]
    assert GammaL.shape == (H_size,), f"GammaL must have shape ({H_size},)"
    assert GammaR.shape == (H_size,), f"GammaR must have shape ({H_size},)"

    transmission_data = np.zeros(len(energy_grid))
    dos_data = np.zeros(len(energy_grid))

    # Add small imaginary part for numerical stability (match models.py)
    delta = 1e-12j

    for n, energy in enumerate(energy_grid):
        # Self-energy matrices (match models.py exactly)
        SigmaL = -0.5j * np.diag(GammaL)
        SigmaR = -0.5j * np.diag(GammaR)
        sigTot = SigmaL + SigmaR

        # Green's function (match models.py)
        A = energy * np.eye(H_size) - H - sigTot + delta
        try:
            Gr = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices (match models.py)
            Gr = np.linalg.pinv(A + 1e-8j * np.eye(H_size))

        # Calculate DOS: -Im(Tr(Gr))/pi (match models.py)
        dos_data[n] = -np.trace(np.imag(Gr))/np.pi

        # Calculate transmission: Tr(ΓL * Gr * ΓR * Ga) (match models.py)
        Ga = Gr.conj().T
        GammaL_diag = np.diag(GammaL)
        GammaR_diag = np.diag(GammaR)

        # Matrix multiplication: ΓL * Gr * ΓR * Ga (match models.py)
        temp1 = np.dot(GammaL_diag, Gr)
        temp2 = np.dot(temp1, GammaR_diag)
        Tcoh = np.dot(temp2, Ga)

        transmission_data[n] = np.real(np.trace(Tcoh))

    # Ensure positive values (should be by construction)
    transmission_data = np.clip(transmission_data, 1e-16, None)
    dos_data = np.clip(dos_data, 1e-16, None)

    return transmission_data, dos_data
