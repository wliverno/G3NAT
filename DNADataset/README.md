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
- `TransportSetup.py` automatically finds HOMO-LUMO from `.log` and `_eigen.mat` files to set energy range (HOMO±1eV inclusive at 0.01 eV, i.e. 201 points)
- MATLAB functions must be in MATLAB path or same directory as scripts

---

## Dataset archive format (HDF5)

> **The DFT data itself is not yet distributed.** The archive described here is released
> with the preprint; until then this section is the format specification only, and no
> download exists. The generation tooling above is public and complete, so the records can
> be regenerated from scratch.

This section documents the single HDF5 file produced by `export_hdf5.py` -- the archive that
will be distributed. It is written entirely from the pickled records
(`convert_to_pickle.py`, functions `build_record`/`validate_record`), not from the raw
`.mat`/`.pdb` files directly, so this text is a mechanical transcription of
`DNADataset/export_hdf5.py` and should never drift from the shipped file. If anything
below appears to disagree with the archive file itself, trust the file's own root attrs
(section 2 quotes them verbatim) over this prose.

### Regenerating the archive

Two steps. Step 1 is a SLURM array job (pure parsing of already-computed DFT/transport
output into pickle files); step 2 is a single process (packs those pickle files into the
one HDF5 archive):

```bash
sbatch DNADataset/run_regeneration.sh                  # -> pickle_files_v2/
python DNADataset/export_hdf5.py pickle_files_v2 g3nat_dna_transport.h5
```

### 1. Layout and schema

Group path: `/<sequence>/<run>`, where `<sequence>` is the lowercase base sequence (for
example `aaac`) and `<run>` is one of `run1`, `run2`, `run3`, `run4` (see the run table
below for what each one means physically). Each `(sequence, run)` pair present in the
file is one independent DFT + NEGF transport calculation. The intermediate `/<sequence>`
group itself carries no attrs or datasets -- it exists only to hold the `run1..run4`
subgroups for that sequence.

**Important layout note:** the HDF5 flattens `contacts` relative to the pickle. The
pickle nests `left_atoms`, `right_atoms`, `coupling_eV` and `contact_type` inside one
`contacts` dict. In the HDF5, `left_atoms` and `right_atoms` are int32 **datasets**
directly on the run group, and `coupling_eV` / `contact_type` are **attrs** on the run
group, not datasets. `energy_reference_eV` is also a run-group attr. There is no HDF5
group or dataset literally named `contacts`.

#### Root attrs (14 total; global to the file, identical for every group)

`units_energy`, `units_xyz`, `energy_convention`, `dos_definition`,
`transmission_definition`, `strand_identity`, `spin`, `contact_model`,
`orthogonalization`, `atom_index_base`, `geometry`, `regime`, `run_map`, `limitations`.

Every one of these 14 is quoted verbatim in section 2 below (or in section 3/4/6, where
noted). There is no `n_orbitals` root attr and no `n_orbitals` field anywhere in the
archive.

#### Per-run group: `/<sequence>/<run>`

| name | HDF5 kind | dtype | shape | units | meaning |
|---|---|---|---|---|---|
| `Egrid` | dataset | float64 | `(n_energy,)` | eV | Absolute (NOT relative) energy grid for this record. Read section 3 before comparing across records. |
| `DOS` | dataset | float64 | `(n_energy,)` | 1/eV | Total density of states, `DOS = -(1/pi) Im Tr(G^r)`. One spin channel; bare, no `2e^2/h` factor. |
| `T` | dataset | float64 | `(n_energy,)` | dimensionless | Transmission, `T = Tr(Gamma_L G^r Gamma_R G^a)`. One spin channel; bare Landauer trace, NOT a conductance -- no `2e^2/h` factor is applied. |
| `DOSAtom` | dataset | float64 | `(n_atoms, n_energy)` | 1/eV | Per-atom-resolved DOS. Row order matches `atoms/*` order. Summing over atoms (axis 0) reproduces `DOS` (checked at generation time). |
| `gjf_text` | dataset | UTF-8 string, scalar | `()` | -- | Full text of the Gaussian `.gjf` input file for the DFT calculation this record came from. |
| `complementary_sequence` | dataset | UTF-8 string, scalar | `()` | -- | The REVERSE complement of `sequence` (the group name one level up), NOT a position-wise complement. See section 4. |
| `left_atoms` | dataset | int32 | `(n_left,)` | -- | 1-BASED indices into this run's `atoms/*` arrays: atoms carrying the left (injection) contact self-energy. `n_left` depends on which base sits at that terminus. |
| `right_atoms` | dataset | int32 | `(n_right,)` | -- | 1-BASED indices into `atoms/*`: atoms carrying the right (extraction) contact self-energy. |
| `coupling_eV` | attr (run group) | float64 | scalar | eV | Wide-band-limit coupling `Gamma` used for BOTH leads in this run (`gammaL == gammaR`): 0.1 for run1/run2, 0.6 for run3/run4. |
| `contact_type` | attr (run group) | string | scalar | -- | `"same"` (both contacts on the same strand, 5'->3') or `"cross"` (contacts on opposite strands, 5' end to 5' end) -- the `--mode` option of `TransportSetup.py`, documented earlier in this README. |
| `energy_reference_eV` | attr (run group) | float64 | scalar | eV | The absolute energy this record's `Egrid` is centred on (`Egrid.mean()`), i.e. this record's own HOMO reference. DIFFERS PER RECORD -- see section 3. |

#### `atoms` subgroup: `/<sequence>/<run>/atoms`

| name | dtype | shape | units | meaning |
|---|---|---|---|---|
| `element` | UTF-8 string array | `(n_atoms,)` | -- | Element symbol per atom, PDB file order. |
| `name` | UTF-8 string array | `(n_atoms,)` | -- | PDB atom name per atom. |
| `resname` | UTF-8 string array | `(n_atoms,)` | -- | PDB residue name (base identity, e.g. `DA`/`DT`/`DG`/`DC`). |
| `resseq` | int32 | `(n_atoms,)` | -- | PDB residue sequence number, 1-based, non-decreasing in file order. Determines strand identity -- see section 4. |
| `xyz` | float64 | `(n_atoms, 3)` | Angstrom | Atomic Cartesian coordinates, PDB file order (also the `DOSAtom` row order). |

**Deliberately not present:** `atoms/chain` (PDB chainID is blank in every source
structure -- see `strand_identity` in section 4 for how strand membership is actually
recorded) and `n_orbitals` (per-atom AO counts; the archive ships no Fock/Overlap/H0 for
AO-indexed data to be checked against it -- see section 6).

#### Run table (also the `run_map` root attr)

| run | `coupling_eV` (eV) | `contact_type` |
|---|---|---|
| run1 | 0.1 | same |
| run2 | 0.1 | cross |
| run3 | 0.6 | same |
| run4 | 0.6 | cross |

### 2. Physics conventions (verbatim from the root HDF5 attrs)

Quoted directly from `export_hdf5.py` so this text cannot drift from the shipped file.
Read all 14 before drawing conclusions from this dataset. The six below are the ones
most likely to be silently misread, and are flagged first.

**`contact_model`** -- wide-band-limit leads; NO physical Fermi level. Verbatim:

> Sigma_L,R = -i*Gamma/2 * I; wide-band limit, energy-independent, purely imaginary, no
> real part and no work function. Applied to EVERY atomic orbital of EVERY atom in the
> terminal base. coupling_eV is used for both leads (gammaL == gammaR). There is
> therefore no physical Fermi level in this model.

In plain terms: the leads are NOT a real metal electrode with a work function. `Sigma` is
a constant, purely imaginary self-energy applied uniformly to every orbital of every atom
listed in `left_atoms`/`right_atoms`, at every energy in `Egrid`. Because `Sigma` has no
real part and no energy dependence, there is no level-alignment calculation and no
physical Fermi level anywhere in this dataset -- "the Fermi level" is not a quantity you
can look up or derive from these files.

**`dos_definition`** and **`transmission_definition`** -- no `2e^2/h` anywhere. Verbatim:

> DOS = -(1/pi) Im Tr(G^r); bare, no 2e^2/h
> T = Tr(Gamma_L G^r Gamma_R G^a); bare Landauer trace, no 2e^2/h

`DOS` and `T` are the bare mathematical quantities, not SI physical observables. In
particular `T` is NOT a conductance: to obtain a linear-response two-terminal conductance
in Siemens you must multiply by `2e^2/h` yourself (roughly 7.748e-5 S), on top of the
spin doubling described next if you want a total-electron rather than single-channel
value. The archive does not perform this multiplication and does not store `2e^2/h`.

**`spin`** -- ONE spin-degenerate channel. Verbatim:

> Spin-restricted closed-shell Fock (alpha only). DOS and T are ONE spin-degenerate
> channel; double for total-electron DOS or conductance.

The underlying DFT is spin-restricted (closed-shell): only the alpha (spin-up) channel
is computed, and by construction the beta channel is identical to it. `DOS` and `T` as
stored are for that ONE spin channel. If you want the total-electron DOS or a
two-terminal conductance that counts both spins, double these values yourself -- they are
not pre-doubled.

**`orthogonalization`** -- Lowdin symmetric orthogonalization. Verbatim:

> H0 = S^-1/2 F S^-1/2 (Lowdin symmetric)

The tight-binding-like Hamiltonian used internally to produce `Egrid`/`DOS`/`T` (not
itself stored -- see section 6) is the Lowdin-symmetrically-orthogonalized Fock matrix,
`H0 = S^-1/2 F S^-1/2`, where `F` is the Gaussian Fock matrix and `S` is the AO overlap
matrix. This is a specific, non-unique choice of orthogonalization. A reader who
reconstructs any part of the electronic structure from `gjf_text` / re-run Gaussian
output using a different orthogonalization convention (canonical, symmetric-but-not-
Lowdin, or a non-orthogonalized AO basis) will not reproduce the `DOS`/`T` recorded here.

**`atom_index_base`** -- contact atom indices are 1-BASED. Verbatim:

> contacts left_atoms/right_atoms are 1-BASED into atoms/*

`left_atoms` and `right_atoms` index into the `atoms/*` arrays (`element`, `name`,
`resname`, `resseq`, `xyz`) and into the rows of `DOSAtom`, counting from 1, NOT from 0.
In a 0-based language (Python/NumPy, C), subtract 1 from every value before using it as
an array index. MATLAB arrays are already 1-based, so `left_atoms`/`right_atoms` can be
used directly there with no adjustment.

**`geometry`** -- fixed idealized NAB fiber B-DNA; no MD; no per-sequence relaxation.
Verbatim:

> Idealized NAB fiber B-DNA template (dnabuilder); no MD, no per-sequence relaxation.
> Geometry is IDENTICAL across sequences except for base identity, so all transport
> variation is electronic, not conformational.

Every structure in this archive comes from the same idealized fiber B-DNA template
generated by NAB's `dnabuilder` (documented earlier in this README), with only base
identities substituted in. There is no molecular-dynamics equilibration, no per-sequence
geometry relaxation or optimization, and no experimentally determined structure. This is
deliberate: any differences observed in `DOS`/`T` across sequences in this archive are
attributable to electronic structure (sequence and base composition), not to
conformational differences, because geometry essentially does not vary across sequences
here. It also means this archive cannot be used to study sequence-dependent DNA
structure/flexibility effects on transport -- by construction, structure does not vary
that way in this dataset.

The remaining root attrs, quoted verbatim (each is discussed in more detail in the
section indicated):

- **`units_energy`**: `"eV"` -- unit for every energy-valued field (`Egrid`,
  `energy_reference_eV`, `coupling_eV`).
- **`units_xyz`**: `"Angstrom"` -- unit for `atoms/xyz`.
- **`energy_convention`**: quoted in full in section 3.
- **`strand_identity`**: quoted in full in section 4.
- **`regime`**: `"Coherent, ballistic, zero-bias only."` -- discussed in section 6.
- **`run_map`**: `"run1=(0.1,same) run2=(0.1,cross) run3=(0.6,same) run4=(0.6,cross)"` --
  already tabulated in section 1.
- **`limitations`**: quoted in full in section 6.

### 3. The energy warning -- read this before comparing across sequences

Root attr `energy_convention`, quoted verbatim:

> Egrid is RAW/absolute. Each record's grid is centred on THAT sequence's HOMO, so
> energy_reference_eV differs per record. WARNING: the reference is a composition proxy
> (AT-only vs GC-only sequences differ by 0.813 eV, 13.6 sigma, zero overlap), so
> comparing a fixed RELATIVE energy across sequences reintroduces a base-composition
> confound.

Concretely: `Egrid` for each `(sequence, run)` record is an ABSOLUTE energy axis in eV,
not a relative one. It is built as a window centred on that specific sequence's own HOMO
level (see `TransportSetup.py` above: HOMO +/- 1 eV, 200 points), and that HOMO position
-- recorded per record as `energy_reference_eV` -- shifts with the sequence's base
composition. Measured across this dataset, AT-only and GC-only sequences differ in
`energy_reference_eV` by 0.813 eV, a 13.6-sigma separation with zero distribution overlap
between the two groups.

The practical consequence: if you compute `E_rel = Egrid - energy_reference_eV` per
record and then compare `DOS(E_rel)` or `T(E_rel)` at a fixed value of `E_rel` across
sequences of different base composition, you are not holding energy fixed in any
absolute sense -- you are silently comparing across a roughly 0.8 eV shift that is a
near-deterministic function of base composition (AT vs GC content), not of whatever
electronic effect you intended to probe. Any cross-sequence comparison built on this
dataset should either work in absolute `Egrid` terms, explicitly control for base
composition, or report `energy_reference_eV` alongside any relative-energy comparison so
a reader can judge whether the composition confound explains the result.

### 4. Strand identity and `complementary_sequence`

Root attr `strand_identity`, quoted verbatim:

> PDB chainID is blank in these structures (the builder does not set it) and is
> therefore NOT exported. Strand identity comes from resseq: for a duplex of L base
> pairs, residues 1..L are the primary strand 5'->3' and residues L+1..2L are the
> complementary strand, also written 5'->3' and therefore antiparallel to the primary.
> This is why complementary_sequence is the REVERSE complement of sequence.

The source PDB files never populate the chainID column, so `atoms/chain` is not exported
(section 1); it cannot be used to separate strands and never could have been. Strand
membership must instead be read off `atoms/resseq`: for a sequence of length L (a duplex
of L base pairs, 2L residues total), residues with `resseq` 1..L belong to the primary
strand -- the one named by the `sequence` group -- written 5'->3', and residues L+1..2L
belong to the complementary strand, ALSO written 5'->3' in the file. Because both halves
of the file run 5'->3' in file order, the complementary strand is antiparallel to the
primary strand, exactly as in a real DNA duplex.

This is why `complementary_sequence` is the REVERSE complement of `sequence`, not a
position-wise (naive base-pairing) complement: sequence `aaac` has `complementary_sequence`
`gttt`, NOT `tttg` (verified directly against the residue order in
`tests/fixtures/dataset/aaac/aaac.pdb`: residues 1-4 are DA-DA-DA-DC and residues 5-8 are
DG-DT-DT-DT read 5'->3'). A reader who assumes `complementary_sequence[i]` base-pairs
with `sequence[i]` under a simple per-position complement will mis-map every base; the
correct pairing is `sequence[i]` with `complementary_sequence[L-1-i]`.

### 5. Contents

The archive contains 2077 records, covering 520 distinct DNA sequences of length 4 to 8
bases:

| sequence length (bases) | distinct sequences | records |
|---|---|---|
| 4 | 135 | 540 |
| 5 | 113 | 451 |
| 6 | 83 | 330 |
| 7 | 93 | 372 |
| 8 | 96 | 384 |

Records by run: run1 = 520, run2 = 519, run3 = 519, run4 = 519 (2077 total).

Atom counts (the length of `atoms/element`, i.e. the number of atoms in the
corresponding structure) range from 250 to 510 across the archive.

### 6. Limitations

Root attr `limitations`, quoted verbatim:

> Fock/Overlap/H0 are NOT included, so DOS/T cannot be independently reproduced from
> this archive alone.

The Gaussian Fock matrix, the AO overlap matrix, and the Lowdin-orthogonalized
Hamiltonian `H0 = S^-1/2 F S^-1/2` are NOT part of this archive -- only `gjf_text` (the
DFT job specification) and the final `Egrid`/`DOS`/`T`/`DOSAtom` transport results are
included. Consequences:

- `DOS`/`T` cannot be independently recomputed or verified from the files in this
  archive alone; doing so requires re-running the DFT calculation from `gjf_text` and the
  full NEGF pipeline described earlier in this README.
- Atomic-orbital-resolved quantities cannot be reconstructed; only atom-resolved
  `DOSAtom` is provided.

Root attr `regime`, quoted verbatim:

> Coherent, ballistic, zero-bias only.

Every transport result in this archive (`T`, `DOS`, `DOSAtom`) is computed in the
coherent, fully ballistic, zero-bias (linear-response) regime, under the wide-band-limit
contact model described in section 2. There is no inelastic or decoherent scattering, no
finite source-drain bias, and no explicit temperature broadening beyond whatever
numerical broadening the underlying Green's-function calculation itself uses.

### 7. Read examples

Python (`h5py`):

```python
import h5py

with h5py.File("g3nat_dna_transport.h5", "r") as h:
    print(list(h.attrs))                # the 14 physics-convention root attrs
    g = h["aaac/run1"]                  # one (sequence, run) group; replace with any
                                         # group path actually present in your file
    Egrid, DOS, T = g["Egrid"][:], g["DOS"][:], g["T"][:]
    left_atoms = g["left_atoms"][:]     # 1-BASED indices into g["atoms"] -- subtract 1
    coupling_eV = g.attrs["coupling_eV"]
    elements = [e.decode() for e in g["atoms/element"][:]]
```

MATLAB (`h5read` / `h5readatt`):

```matlab
file = "g3nat_dna_transport.h5";
Egrid = h5read(file, "/aaac/run1/Egrid");
DOS   = h5read(file, "/aaac/run1/DOS");
left_atoms = h5read(file, "/aaac/run1/left_atoms");   % 1-BASED, matches MATLAB indexing
coupling_eV = h5readatt(file, "/aaac/run1", "coupling_eV");
element = h5read(file, "/aaac/run1/atoms/element");
```
