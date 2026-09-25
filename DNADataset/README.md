# DNA Dataset Generation and Transport Calculation Tools

Tools for generating DNA structures, running DFT calculations, and computing electronic transmission properties.

## Published dataset

This section is a datasheet for the dataset released on Zenodo (https://doi.org/10.5281/zenodo.22964053), structured after
the question set in Gebru et al., "Datasheets for Datasets" (see `docs/references.md`).
It documents `transport.h5` and `matrices.h5`, the two files that make up the Zenodo
record; the pipeline used to build them is documented in the rest of this README and in
`export_hdf5.py` / `export_matrices_hdf5.py`, whose root attrs this section quotes
verbatim in places so it cannot silently drift from the shipped files.

### Motivation

Coherent-transport observables for DNA duplexes, computed from all-electron DFT, released
so that coarse-grained Hamiltonian models (of the kind trained in this repository) can be
trained and evaluated by others without re-running the DFT/NEGF pipeline. The dataset also
supports any other use of DFT-derived electronic structure and NEGF transport for short DNA
duplexes.

### Composition

`transport.h5` has 2109 run groups at path `/<sequence>/<run>`: 2077 training records
(520 duplexes, 4-8 bp, up to 4 contact/coupling variants each) and 32 held-out records
(8 duplexes: 4 at 12 bp, 4 at 16 bp, x 4 variants each). Every
run group carries an attr `split` set to `"train"` or `"heldout"`.

Per run group:
- `Egrid`: 201 points, eV, the window HOMO +/- 1 eV at 0.01 eV spacing, centred on that
  record's own `energy_reference_eV` (an absolute energy, not shared across records).
- `T`, `DOS`, `DOSAtom`: transmission, total density of states, and per-atom-resolved DOS.
- an `atoms` table (element, name, resname, resseq, xyz) and the `contacts` fields
  (`left_atoms`, `right_atoms`, `coupling_eV`, `contact_type`).

`matrices.h5` has one group per sequence, `/<sequence>/{fock, overlap, basis}`, for the
528 sequences that have at least one transport record: 520 training-length sequences and
the 8 held-out sequences. `fock` and `overlap` are the Gaussian Fock and AO overlap
matrices in Hartree, in the non-orthogonal atomic-orbital basis (they are NOT the
orthogonalized Hamiltonian used to compute `T`/`DOS`; see Preprocessing below). `basis` is
the per-row orbital map: `atom_index` (0-based, into the `atoms` table of the matching
`transport.h5` sequence), `shell_index`, `shell_type`, `component`, and `type_code`
(Gaussian's own per-function code, `1000*l + component_index`). Each sequence group also
carries an attr `label_source`, `"derived"` for the 520 training sequences or `"derived;
verified against this sequence's Gaussian matrix-element file"` for the 8 held-out
sequences, whose orbital map was additionally checked against that sequence's own Gaussian
matrix-element dump.

The held-out sequences are:
- 12 bp: `acttgacagtca`, `atatatatatat`, `gggggggggggg`, `gtcaggatctga`
- 16 bp: `ggataggcttagaatt`, `gggggggggggggggg`, `gtcaggatctgacagt`, `tcagtgctaagtcatg`

Three sequences (`atattg`, `cactc`, `tggaa`) have Fock/overlap matrices in the upstream
pipeline output but no complete transport records, because their upstream transport output
was incomplete; they are excluded from both files. Two training sequences, `cgtat` and
`gcctgg`, have fewer than 4 run variants each because three configurations for those two
sequences were corrupted upstream and dropped rather than shipped.

### Collection process

Geometry: idealized NAB fiber B-DNA (`dnabuilder`, documented below), one geometry per
duplex, no MD, no per-sequence relaxation. DFT: Gaussian 16, B3LYP/6-31G(d,p), implicit
water (SCRF continuum), single-point on the idealized geometry, net charge
`-2*(N_bp - 1)` for an `N_bp`-bp duplex, closed-shell singlet. Transport: NEGF with
wide-band-limit contacts, per the `contact_model` attr (see Contacts below).

### Preprocessing

- The transport results (`T`, `DOS`, `DOSAtom`) are computed from the Lowdin-orthogonalized
  Hamiltonian `H0 = S^-1/2 F S^-1/2`, not from the raw Fock matrix. `H0` itself is not
  stored in either file; `matrices.h5` ships the un-orthogonalized `F` and `S` so a reader
  can reconstruct `H0` if wanted.
- The upstream `_eigen.mat` orbital-energy files are in Hartree and unsorted; do not assume
  ascending order or eV units when working from them directly.
- `DOSAtom` rows follow PDB atom order, and residue identity is the PDB `resseq` column
  (strand membership follows from `resseq`; see "Strand identity" below).
- The 6-31G(d,p) basis is Cartesian (6 `d` functions per shell, not 5 spherical). Per-atom
  function counts: H 5, C/N/O 15, P 19.
- Every duplex is built from one idealized Watson-Crick geometry; the geometry does not
  vary with sequence except at the base atoms. A/T and G/C onsite contributions are
  therefore confounded with geometry by construction -- this dataset cannot separate a
  sequence-electronic effect from a sequence-geometric one, because geometry is
  (by construction) not a free variable here.

### Energy reference

Root attr `energy_convention` from `transport.h5`, quoted verbatim:

> Egrid is RAW/absolute. Each record's grid is centred on THAT sequence's HOMO, so
> energy_reference_eV differs per record. WARNING: the reference is a composition proxy
> (AT-only vs GC-only sequences differ by 0.813 eV, 13.6 sigma, zero overlap), so
> comparing a fixed RELATIVE energy across sequences reintroduces a base-composition
> confound.

In words: every record's grid runs from HOMO - 1 eV to HOMO + 1 eV of its own duplex, so
0 eV relative energy is a different absolute energy in different records. The absolute
HOMO of a GC-only duplex is 0.81 eV higher than that of an AT-only duplex, yet both sit at
the 0 eV reference.

### Strand identity

Root attr `strand_identity` from `transport.h5`, quoted verbatim:

> PDB chainID is blank in these structures (the builder does not set it) and is
> therefore NOT exported. Strand identity comes from resseq: for a duplex of L base
> pairs, residues 1..L are the primary strand 5'->3' and residues L+1..2L are the
> complementary strand, also written 5'->3' and therefore antiparallel to the primary.
> This is why complementary_sequence is the REVERSE complement of sequence.

### Contacts

Root attr `run_map` from `transport.h5`, quoted verbatim:

> run1=(0.1 eV, same) run2=(0.1 eV, cross) run3=(0.6 eV, same) run4=(0.6 eV, cross). same:
> left contact = residue 1 (primary strand 5' end), right contact = residue L (primary
> strand 3' end). cross: left = residue 1, right = residue L+1 (complementary strand 5'
> end). Each run group also stores left_residue/right_residue (resseq) and
> left_atoms/right_atoms (1-based).

Root attr `contact_model`, quoted verbatim:

> Sigma_L,R = -i*Gamma/2 * I; wide-band limit, energy-independent, purely imaginary, no
> real part and no work function. Applied to EVERY atomic orbital of EVERY atom in the
> terminal RESIDUE (the full nucleotide: base, sugar and phosphate). coupling_eV is used
> for both leads (gammaL == gammaR). There is therefore no physical Fermi level in this
> model.

### Detailed schema (transport.h5 record fields)

Group path in `transport.h5` is `/<sequence>/<run>`, `<sequence>` the lowercase base
sequence (e.g. `aaac`) and `<run>` one of `run1`..`run4` (see Contacts above for what each
run means physically). The intermediate `/<sequence>` group carries no attrs or datasets
of its own; it exists only to hold the `run1..run4` subgroups.

**Layout note:** the pickle nests `left_atoms`, `right_atoms`, `coupling_eV` and
`contact_type` inside one `contacts` dict; in the HDF5 they are flattened onto the run
group. `left_atoms`/`right_atoms` are int32 datasets, `coupling_eV`/`contact_type` are
attrs, and `energy_reference_eV` and `split` are also run-group attrs. There is no HDF5
group or dataset literally named `contacts`.

Per-run-group fields:

| name | HDF5 kind | dtype | shape | units | meaning |
|---|---|---|---|---|---|
| `Egrid` | dataset | float64 | `(n_energy,)` | eV | Absolute (not relative) energy grid; see Energy reference above before comparing across records. |
| `DOS` | dataset | float64 | `(n_energy,)` | 1/eV | Total density of states, `DOS = -(1/pi) Im Tr(G^r)`. One spin channel, bare, no `2e^2/h`. |
| `T` | dataset | float64 | `(n_energy,)` | dimensionless | Transmission, `T = Tr(Gamma_L G^r Gamma_R G^a)`. One spin channel, bare Landauer trace, not a conductance. |
| `DOSAtom` | dataset | float64 | `(n_atoms, n_energy)` | 1/eV | Per-atom-resolved DOS; row order matches `atoms/*`; summing over atoms reproduces `DOS`. |
| `gjf_text` | dataset | UTF-8 string, scalar | -- | -- | Full text of the Gaussian `.gjf` input for this record's DFT calculation. |
| `complementary_sequence` | dataset | UTF-8 string, scalar | -- | -- | Reverse complement of `sequence` (the group name), not a position-wise complement: `aaac` has complementary `gttt`, not `tttg`, because `sequence[i]` pairs with `complementary_sequence[L-1-i]`. |
| `left_atoms` / `right_atoms` | dataset | int32 | `(n_left,)` / `(n_right,)` | -- | 1-based indices into `atoms/*` for the atoms carrying that contact's self-energy. |
| `coupling_eV` | attr | float64 | scalar | eV | Wide-band-limit coupling used for both leads in this run. |
| `contact_type` | attr | string | scalar | -- | `"same"` or `"cross"`. |
| `energy_reference_eV` | attr | float64 | scalar | eV | This record's own HOMO reference (`Egrid.mean()`); differs per record. |
| `split` | attr | string | scalar | -- | `"train"` or `"heldout"`. |

`atoms` subgroup (`/<sequence>/<run>/atoms`):

| name | dtype | shape | units | meaning |
|---|---|---|---|---|
| `element` | UTF-8 string array | `(n_atoms,)` | -- | Element symbol, PDB file order. |
| `name` | UTF-8 string array | `(n_atoms,)` | -- | PDB atom name. |
| `resname` | UTF-8 string array | `(n_atoms,)` | -- | PDB residue name (base identity). |
| `resseq` | int32 | `(n_atoms,)` | -- | PDB residue sequence number, 1-based, non-decreasing; determines strand identity (Strand identity above). |
| `xyz` | float64 | `(n_atoms, 3)` | Angstrom | Atomic Cartesian coordinates, PDB file order (also the `DOSAtom` row order). |

`atoms/chain` is deliberately not present (PDB chainID is blank in every source
structure; strand membership is read from `resseq` instead), and there is no
`n_orbitals` field in `transport.h5` -- the per-atom orbital counts live in
`matrices.h5`'s `basis` group.

Root attrs not already quoted above, verbatim: `units_energy`: `"eV"`;
`units_xyz`: `"Angstrom"`; `dos_definition`: `"DOS = -(1/pi) Im Tr(G^r); bare, no
2e^2/h"`; `transmission_definition`: `"T = Tr(Gamma_L G^r Gamma_R G^a); bare Landauer
trace, no 2e^2/h"`; `spin`: `"Spin-restricted closed-shell Fock (alpha only). DOS and T
are ONE spin-degenerate channel; double for total-electron DOS or conductance."`;
`atom_index_base`: `"contacts left_atoms/right_atoms are 1-BASED into atoms/*"`;
`level_of_theory`: `"B3LYP/6-31G(d,p) in implicit water (SCRF continuum,
scrf=(solvent=water)); single-point on the idealized geometry, no optimization."`;
`charge_and_multiplicity`: `"Net charge -2(L-1) for an L-bp duplex (every internal
phosphodiester phosphate deprotonated, free 5'/3'-OH termini, no counterions);
closed-shell singlet."`; `orthogonalization`: `"H0 = S^-1/2 F S^-1/2 (Lowdin
symmetric)"`; `geometry`: `"Idealized NAB fiber B-DNA template (dnabuilder); no MD, no
per-sequence relaxation. Geometry varies only through base identity (e.g. twist SD 1.01
deg, rise SD 0.005 A over the training set), so conformational and electronic effects
are not separable in this dataset."`; `regime`: `"Coherent, ballistic, zero-bias
only."`;
`limitations`: `"Fock and overlap matrices, with a per-row orbital map, are in the
companion file matrices.h5 (same sequence keys); H0 = S^-1/2 F S^-1/2 is not stored. A
contact's matrix rows are all rows whose basis atom lies in its residue."`;
`split`: `"Each run group has attr split='train' (lengths 4-8, the training set) or
'heldout' (lengths 12 and 16, never used in training)."`; `license`: `"CC-BY-4.0"`.

Sequence-length distribution of the 2077 training records (520 duplexes, 4-8 bp):

| sequence length (bases) | distinct sequences | records |
|---|---|---|
| 4 | 135 | 540 |
| 5 | 113 | 451 |
| 6 | 83 | 330 |
| 7 | 93 | 372 |
| 8 | 96 | 384 |

Records by run in the training split: run1 = 520, run2 = 519, run3 = 519, run4 = 519
(the CGTAT/GCCTGG gaps noted in Composition above). Atom counts (length of
`atoms/element`) range from 250 to 510 across the training split.

### Uses

Intended use: training and evaluating coarse-grained Hamiltonian or direct (physics-blind)
predictors of DNA coherent transport, and any other reuse of DFT-derived electronic
structure and NEGF transport for short DNA duplexes. This is not a model of real,
fluctuating DNA: the geometry is a single idealized fiber-B-DNA conformation with no
thermal motion, and the transport regime is coherent, ballistic and zero-bias only (no
inelastic scattering, no finite bias, no explicit temperature broadening beyond whatever
numerical broadening the Green's-function calculation itself carries).

### Distribution

Distributed on Zenodo under CC-BY-4.0 (https://doi.org/10.5281/zenodo.22964053). The record contains:
- `transport.h5` (about 1.41 GB)
- `matrices.h5` (about 13.79 GB)
- `geom_cache.tar`, holding `geom_cache/geometry_v2.pkl` and
  `geom_cache/geometry_heldout_L12_L16.pkl`, the X3DNA-DSSR edge-geometry caches read by
  `--geom_cache` (see "Edge geometry (X3DNA / DSSR)" in the top-level `README.md`).
  Extract it at the repository root (`tar -xf geom_cache.tar`) to get `geom_cache/`.
- `SHA256SUMS`, checksums for the three files above

### Maintenance

New data, if any, will be released as new versions of the same Zenodo record rather than
as a separate one.

### Reading the archive

```python
import h5py

with h5py.File("transport.h5", "r") as h:
    g = h["aaac/run1"]                  # any (sequence, run) group actually present
    split = g.attrs["split"]            # "train" or "heldout"
    Egrid, T, DOS = g["Egrid"][:], g["T"][:], g["DOS"][:]

with h5py.File("matrices.h5", "r") as h:
    m = h["aaac"]
    F, S = m["fock"][:], m["overlap"][:]           # Hartree, non-orthogonal AO basis
    atom_index = m["basis/atom_index"][:]          # 0-based, into transport.h5's atoms table
```

### Converting back to the pickle format the training code reads

```bash
# training split
python DNADataset/import_hdf5.py transport.h5 pickle_files_v2 --split train

# held-out split: import_hdf5.py writes every record from one call into a single
# directory without separating by duplex length, so import into a scratch directory
# first, then move the 12- and 16-character-sequence records into place
python DNADataset/import_hdf5.py transport.h5 <scratch_dir> --split heldout
# then move the 12 bp records into DNADataset/validation_L12/pickles/
# and the 16 bp records into DNADataset/validation_L16/pickles/
```

---

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
- `--mode`: `same` (5'->3' same strand) or `cross` (5'->5' cross-strand), default: `same`
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
   2) Builds the molecular structure and Gaussian input files using NAB tools (the
      .gjf already carries the matrix-output flags and .mat trailer, as of 2026-08-09).
   3) Derives a first-pass SCF input by STRIPPING the matrix flags and trailer, and
      runs it to produce checkpoint and log files.
   4) Runs the unmodified input as a second Gaussian pass, restarting from the
      checkpoint, to dump the Fock and overlap matrices in .mat format.
   5) Converts Gaussian matrix output using readmat and MATLAB processing.
   6) Runs transmission simulations (ballistic or decoherence) using MATLAB transport scripts.
   7) Organizes outputs into structured run folders.
   8) Converts results into pickle files for machine learning training within the G3NAT framework.


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
- DNA sequence length range (4-8 by default)
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
    run1/
        Parameters.txt
        Tran_sequence_gammaL_X_gammaR_X.mat
        DOS_sequence_gammaL_X_gammaR_X.mat
        metadata.txt
        run_config.txt
    run2/
    run3/
    run4/
    Parameters.txt
    sequence.mat
    sequence_eigen.mat
    transmission outputs
    DOS outputs
    pickle files
```

Original `.pdb` and `.gjf` files remain in the main directory.

## Common Issues

- Small (~9 KB) `.mat` file -> Gaussian matrix output not enabled properly.
- `g16: command not found` -> Load module `chem/g16`.
- `readmat: Permission denied` -> Ensure executable permissions.
- Missing HOMO-LUMO range -> Ensure `.log` and `_eigen.mat` exist.

## Credits

- **MATLAB transmission functions**: Developed by Hashem Mohammad and Yiren Wang in the [Quantum Devices Lab](https://sites.uw.edu/anantmp/) at the University of Washington (Prof. M. P. Anantram's group)
- **NAB**: Nucleic Acid Builder from Amber Classic (https://github.com/dacase/nabc)

## Notes

- NAB installation: Install from https://github.com/dacase/nabc and set `BUILDDIR` in `dnabuilder`
- PDB files must have TER records separating strands
- `TransportSetup.py` automatically finds HOMO-LUMO from `.log` and `_eigen.mat` files to set energy range (HOMO +/- 1 eV inclusive at 0.01 eV, i.e. 201 points)
- MATLAB functions must be in MATLAB path or same directory as scripts

