# Analysis

This directory holds the scripts and outputs behind every number and figure-data
table in the paper: the design-of-experiments (DOE) statistics, the contact-invariance
metric, the per-sequence and per-duplex breakdowns, the parameter counts, and the
appendix capacity probe. Section numbers below are as of submission and may drift
in a later revision.

## Layout

```
analysis/
  scripts/          entry-point scripts, one per pipeline stage (run as `python scripts/<x>.py`)
  g3nat_analysis/   shared library code imported by the scripts
  tests/            unit and end-to-end tests for this package
  outputs/          shipped results (floor-25 only; see below)
```

`REPO` (the project root, i.e. the parent of this directory) is resolved as
`os.environ.get('G3NAT_ROOT', <computed from the script's own file location>)`.
Set `G3NAT_ROOT` if you run these scripts from outside a checkout of this repo.

## The floor

Every log10 of a DOS, per-site LDOS, or transmission value anywhere in this
pipeline -- in the reference data, the pickle loader, both models, and every
loss and metric derived from them -- is `log10(max(x, 1e-25))`. See
`g3nat/floor.py` for the physical justification. All shipped outputs in this
directory use that floor (hence "_floor25" in some of the source script names
you may see referenced in comments; the shipped files here have no separate
floor variants).

## What ships vs what you need to reproduce

**Runs directly from the shipped files in `outputs/`, no checkpoints or dataset
needed:**
- `scripts/doe_v3.py` -> `outputs/doe_v3.out` (every p-value, Benjamini-Hochberg
  count, best/worst cell, head-to-head comparison)
- `scripts/export_doe_runs.py` -> `outputs/doe_v3_{runs,cells}_{hamiltonian,direct}.csv`
- `scripts/ci_agg.py`, `ci_arms.py` -> stdout
  (Results 3.4 prose numbers)

**Needs the companion Zenodo checkpoints record** (extract into the repo root so
that `outputs_v3/` and `outputs_v5probe/` exist, DOI to be added):
- `scripts/posthoc_v3.py`, `scripts/contact_invariance_v3.py`,
  `scripts/best_epoch_extract.py`, `scripts/paramcount_v3.py`,
  `scripts/probe_capacity_v5.py`

**Needs the `geom_cache/` pickles** (`geometry_v2.pkl` for the training set,
`geometry_heldout_L12_L16.pkl` for the held-out set; not shipped here, built by
`build_geometry_cache` from the dataset PDB structures -- see the top-level
README):
- `scripts/geometry_spread.py` -> `outputs/geometry_spread.out` (mean/SD of every
  edge geometry feature over the training and held-out sets; the paper's "0.3 A"
  COM-distance figure is the backbone centroid-distance SD (0.273) rounded)

**Needs the companion Zenodo dataset record** (DOI to be added) restored with
`DNADataset/import_hdf5.py`:
- training pickles: `python DNADataset/import_hdf5.py transport.h5 pickle_files_v2 --split train`
- held-out pickles: `import_hdf5.py` writes every record from one call into a single
  directory and does not separate held-out records by duplex length. Import the
  `heldout` split into a scratch directory
  (`python DNADataset/import_hdf5.py transport.h5 <scratch_dir> --split heldout`),
  then move the 12-character-sequence records into `DNADataset/validation_L12/pickles/`
  and the 16-character-sequence records into `DNADataset/validation_L16/pickles/`
  (these are the paths `scripts/posthoc_v3.py` reads; see `HELDOUT_DIRS` there).

`scripts/build_heldout_geometry.py` additionally needs X3DNA-DSSR installed to
build the held-out geometry cache; its output is not shipped here (figures and
figure data are out of scope for this directory -- see below).

Figures and the CSV data files that feed them are not shipped in this
directory; only the numeric/statistical outputs listed here are.

## Pipeline

| stage | script | reads | writes | paper section |
|---|---|---|---|---|
| 0 | `build_heldout_geometry.py` | held-out structures, DSSR | geometry cache for held-out evaluation (not shipped) | -- |
| 1 | `posthoc_v3.py` | 120 checkpoints, training pickles, held-out pickles, geometry caches | `outputs/posthoc_v3_report.json` | Results 3.3, 3.5 |
| 2 | `contact_invariance_v3.py` | Hamiltonian checkpoints, training pickles, stage-1 report | `outputs/contact_invariance_v3.json` | Results 3.4 |
| 3 | `best_epoch_extract.py` | 120 checkpoints | `outputs/best_epoch_cache.json` | training-time section, Discussion |
| 4 | `doe_v3.py` | stages 1-3 | `outputs/doe_v3.out` | every p-value, BH count, best/worst cell, head-to-head |
| 5 | `export_doe_runs.py` | stages 1-2 | `outputs/doe_v3_{runs,cells}_{hamiltonian,direct}.csv` | figure-summary data |
| 6 | `persequence_v3.py` | 2 cells x 3 seeds checkpoints, held-out pickles | `outputs/persequence_v3_figcells_nogeom.out` | Results 3.5 |
| 9 | `ci_agg.py`, `ci_arms.py` | stages 1-2 | stdout | Results 3.4 |
| 10 | `paramcount_v3.py` | 6 checkpoints | stdout | model parameter counts |
| -- | `probe_capacity_v5.py` | capacity-probe checkpoints | `outputs/probe_capacity_v5.out`, `outputs/probe_capacity_v5_{runs,perseq,summary}.csv` | appendix capacity probe |
| -- | `geometry_spread.py` | `geom_cache/geometry_v2.pkl`, `geom_cache/geometry_heldout_L12_L16.pkl` | `outputs/geometry_spread.out` | Results: Geometry paragraph (feature spreads) |

(section numbers as of submission)

## The figure-pair `persequence_v3.py` invocation

`persequence_v3.py`'s default checkpoint pair is a diagnostic ("protocol")
pair, not the pair used in the paper's figure. To reproduce the shipped
`outputs/persequence_v3_figcells_nogeom.out`, run:

```
python scripts/persequence_v3.py \
  --ham-dir-tmpl <REPO>/outputs_v3/ham_ldosonly_n2_L4_nogeom_s{seed} \
  --blind-dir-tmpl <REPO>/outputs_v3/blind_dos_L2_nogeom_s{seed} \
  --blind-geometry false \
  --out-suffix figcells_nogeom
```

## Figure models

The two checkpoints used to generate the paper's figures are shipped in
`trained_models/`: `hamiltonian_ldosonly_n2_L4_nogeom_s3731635825.pth` and
`standard_dos_L2_nogeom_s3731635825.pth`, each with a `.json` copy of its
`resolved_config.json`. Both happen to use seed 3731635825; the selection rule
was to pick, independently for each model, the seed with the lowest sum of
three transmission-loss metrics (validation, 12 bp held-out, 16 bp held-out)
across the three seeds run for that cell. The summed values were
3.05 / 3.97 / 4.96 for the Hamiltonian model and 5.32 / 5.49 / 6.03 for the
direct (blind) model; the lowest sum in each list is seed 3731635825. The two
models landing on the same seed is a coincidence of the rule, not a shared
selection.

## Shipped outputs

`outputs/` contains only the floor-25 results:
`posthoc_v3_report.json`, `contact_invariance_v3.json`, `best_epoch_cache.json`,
`doe_v3.out`, `doe_v3_runs_hamiltonian.csv`, `doe_v3_runs_direct.csv`,
`doe_v3_cells_hamiltonian.csv`, `doe_v3_cells_direct.csv`,
`persequence_v3_figcells_nogeom.out`, `probe_capacity_v5.out`,
`probe_capacity_v5_runs.csv`, `probe_capacity_v5_perseq.csv`,
`probe_capacity_v5_summary.csv`, `geometry_spread.out`.

The COMPANION lines in `doe_v3.out` and the companion fields in
`contact_invariance_v3.json` are context copies of each run's transmission
losses taken before the 1e-25 floor was applied; no number in the paper is
taken from them.

Other variants that existed during development (alternate floors, the
protocol-pair per-sequence output, a TransformerConv variant, a loss-probe
variant, run logs) are not shipped.

## Running the tests

```
cd analysis
python -m pytest tests/
```
