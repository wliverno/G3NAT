# DNA Dataset Generation and Transport Calculation Tools

Tools for generating DNA structures, running DFT calculations, and computing electronic transmission properties.

## Prerequisites

- **NAB** (Nucleic Acid Builder) from Amber Classic: https://github.com/dacase/nabc
- Python 3 with NumPy
- Gaussian 16
- MATLAB
- SLURM

## Tools

### `dnabuilder` - Generate DNA structures

Generates PDB and Gaussian input files from DNA sequences using NAB.

**Usage:**
```bash
./dnabuilder -s "SEQUENCE" -t TYPE
```

**Options:**
- `-s`: DNA sequence (e.g., "ATCGATCG")
- `-t`: DNA type (`A` or `B`, default: B)

**Setup:** Edit `BUILDDIR` in the script to point to your NAB installation.

**Example:**
```bash
./dnabuilder -s "GGCCGG" -t B
# Creates: ggccgg.pdb and ggccgg.gjf
```

### `TransportSetup.py` - Generate parameter files

Parses PDB files and generates `Parameters.txt` for transmission calculations. Automatically finds HOMO-LUMO from Gaussian log and eigen files to set energy range.

**Usage:**
```bash
python TransportSetup.py PDB_FILE --mode {same,cross} --gamma VALUE
```

**Options:**
- `--mode`: `same` (5'→3' same strand) or `cross` (5'→5' cross-strand), default: `same`
- `--gamma`: Coupling strength (eV), default: 0.1

**Example:**
```bash
python TransportSetup.py ggccgg.pdb --mode cross --gamma 0.6
# Creates: Parameters_ggccgg.txt
# Requires: ggccgg.log and ggccgg_eigen.mat in same directory
```

### MATLAB Functions

MATLAB functions for processing DFT outputs and computing transmission properties. Developed by Hashem Mohammad, Jianqing Qi, and Yiren Wang in the [Quantum Devices Lab](https://sites.uw.edu/anantmp/) at the University of Washington.

**`readMAT.m`**: Extracts Fock and Overlap matrices from Gaussian MAT files and computes Hamiltonian
- Converts Fock matrix to orthogonalized Hamiltonian
- Generates `{strand}_eigen.mat` with orbital energies
- Generates `{strand}.mat` with Hamiltonian matrix

**`DNATransmission_Ballistic.m`**: Computes ballistic transmission through DNA
- Uses non-equilibrium Green's function method
- Calculates transmission between left and right contacts
- Outputs `Tran_{strand}_gammaL_{gammaL}_gammaR_{gammaR}.mat`

**`DOS_calc.m`**: Computes density of states (DOS) for the molecule
- Calculates total DOS and per-atom DOS contributions
- Uses Green's function method with broadening parameter
- Outputs `DOS_{strand}_gammaL_{gammaL}_gammaR_{gammaR}.mat` with `Energy`, `DOS`, and `DOSAtom` arrays

### SLURM Scripts

**`combined_script.slurm`**:  Master pipeline script for automated DNA transmission dataset generation
- Completes the following steps:
   1) Generates a DNA sequence between 4-8 units.
   2) Builds the molecular structure and Gaussian input files using NAB tools.
   3) Runs a first Gaussian calculation to produce checkpoint and log files.
   4) Modifies the Gaussian input to enable matrix output and inserts the required trailer for matrix dumping.
   5) Runs a second Gaussian calculation to produce the Hamiltonian and overlap matrices in .mat format.
   6) Converts Gaussian matrix output using readmat and MATLAB processing.
   7) Runs transmission simulations (ballistic or decoherence) using MATLAB transport scripts.
   8) Organizes outputs into structured run folders.
   9) Converts results into pickle files for machine learning training within the G3NAT framework.


**`TransportScript.slurm`**: Sets up multiple transmission runs
- Edit `PDB_FILE` (line 17) and `CASES` array (lines 44-49)
- Requires `.mat` Hamiltonian file in current directory

**`run_transmission.slurm`**: Runs MATLAB transmission/DOS calculations
- Called automatically by `TransportScript.slurm`
- Manual: `sbatch run_transmission.slurm RUN_NUMBER DESCRIPTION`

## Workflow (Fully automated pipeline) 

The dataset generation process is fully automated through `combined_script.slurm`.

### 1. Edit the SLURM script (if needed)

Inside `combined_script.slurm`, adjust:
- DNA sequence length range (4–8 by default)
- Number of sequences to generate
- Contact mode (same / cross)
- Coupling strengths (gamma values)

### 2. Submit the pipeline

```bash
sbatch combined_script.slurm
```
## Directory Structure 

After execution, results are organized as: 
```
DNA_SEQUENCE/
├── run1/
│   ├── Parameters.txt
│   ├── Tran_sequence_gammaL_X_gammaR_X.mat
│   ├── DOS_sequence_gammaL_X_gammaR_X.mat
│   ├── metadata.txt
│   └── run_config.txt
├── run2/
├── run3/
├── run4/
├── Parameters.txt
├── sequence.mat
├── sequence_eigen.mat
├── transmission outputs
├── DOS outputs
└── pickle files
```

Original `.pdb` and `.gjf` files remain in the main directory.

## Common Issues

- Small (~9 KB) `.mat` file → Gaussian matrix output not enabled properly.
- `g16: command not found` → Load module `chem/g16`.
- `readmat: Permission denied` → Ensure executable permissions.
- Missing HOMO-LUMO range → Ensure `.log` and `_eigen.mat` exist.

## Credits

- **MATLAB transmission functions**: Developed by Hashem Mohammad and Yiren Wang in the [Quantum Devices Lab](https://sites.uw.edu/anantmp/) at the University of Washington (Prof. M. P. Anantram's group)
- **NAB**: Nucleic Acid Builder from Amber Classic (https://github.com/dacase/nabc)

## Notes

- NAB installation: Install from https://github.com/dacase/nabc and set `BUILDDIR` in `dnabuilder`
- PDB files must have TER records separating strands
- `TransportSetup.py` automatically finds HOMO-LUMO from `.log` and `_eigen.mat` files to set energy range (HOMO±1eV, 200 points)
- MATLAB functions must be in MATLAB path or same directory as scripts
