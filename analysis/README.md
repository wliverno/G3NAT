# Analysis

This directory holds the scripts and outputs behind every number and figure-data
table in the paper: the design-of-experiments (DOE) statistics, the contact-invariance
metric, the per-sequence and per-duplex breakdowns, the parameter counts, the
appendix capacity probe, and the contact-free variant added during review. Paper
section, figure and table names below are as of submission unless marked
"revision"; the section "Revision (2026-11)" below covers the retrained campaign
behind the revised paper.

Run names use short supervision labels; the paper's names are: `dos` = DOS+T,
`ldos` = LDOS+DOS+T, `ldosonly` = LDOS+T, `tonly` = T-only. `ham_*` runs are the
Hamiltonian model and `blind_*` runs the direct model.

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
- `scripts/ci_agg.py`, `ci_arms.py` -> stdout (contact-drift numbers in "ANOVA
  Results", GNN Layers subsection, and the "Discussion"; both read only
  `outputs/contact_invariance_v3.json`)
- the in-distribution means quoted in the "Discussion" ("0.27 vs 0.46"), the mean
  validation transmission loss over all runs of each model:
  `python -c "import pandas as pd; [print(f, '%.4f' % pd.read_csv(f'outputs/doe_v3_runs_{f}.csv').val_transmission.mean()) for f in ('direct', 'hamiltonian')]"`
  (run from `analysis/`; prints `direct 0.2738` and `hamiltonian 0.4550`)

**Needs the companion Zenodo checkpoints record** (https://doi.org/10.5281/zenodo.22964130).
The current version holds `campaign.tar`, `capacity_probe.tar` and `exploratory.tar`;
extract them at the repository root to get `campaign/`, `capacity_probe/` and
`exploratory/`, which the revision scripts below read. The scripts for the original
submission, listed here, read the original campaign and capacity probe, which are in
version 1 of the record (https://doi.org/10.5281/zenodo.22964131: `campaign_v3.tar`
and `capacity_probe_v5.tar`, extracting to `outputs_v3/` and `outputs_v5probe/`):
- `scripts/posthoc_v3.py`, `scripts/contact_invariance_v3.py`,
  `scripts/best_epoch_extract.py`, `scripts/paramcount_v3.py`,
  `scripts/probe_capacity_v5.py`

**Needs the `geom_cache/` pickles** (`geometry_v2.pkl` for the training set,
`geometry_heldout_L12_L16.pkl` for the held-out set). Both ship in the Zenodo dataset
record as `geom_cache.tar`; extract it at the repository root
(`tar -xf geom_cache.tar`) to get `<repo>/geom_cache/`. Alternatively, rebuild them
with `build_geometry_cache` (see the top-level README), which needs X3DNA-DSSR and the
PDB structures:
- `scripts/geometry_spread.py` -> `outputs/geometry_spread.out` (mean/SD of every
  edge geometry feature over the training and held-out sets; the paper's "0.3 A"
  COM-distance figure is the backbone centroid-distance SD (0.273) rounded)

**Needs the companion Zenodo dataset record** (https://doi.org/10.5281/zenodo.22964052) restored with
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
| 1 | `posthoc_v3.py` | 120 checkpoints, training pickles, held-out pickles, geometry caches | `outputs/posthoc_v3_report.json` | every response behind "ANOVA Results" (Supervision, Base Orbital Assignment, GNN Layers, Geometry), "Final Model Selection" and Appendix "Full Factorial ANOVA" (Tables 3-4) |
| 2 | `contact_invariance_v3.py` | Hamiltonian checkpoints, training pickles, stage-1 report | `outputs/contact_invariance_v3.json` | contact drift in "ANOVA Results" (Table 3) and the "Discussion" |
| 3 | `best_epoch_extract.py` | 120 checkpoints | `outputs/best_epoch_cache.json` | none (feeds the training-time section of `doe_v3.out`, not reported in the paper) |
| 4 | `doe_v3.py` | stages 1-3 | `outputs/doe_v3.out` | "ANOVA Results" and Appendix "Full Factorial ANOVA" (Tables 3-4): every p-value, BH count, best/worst cell |
| 5 | `export_doe_runs.py` | stages 1-2 | `outputs/doe_v3_{runs,cells}_{hamiltonian,direct}.csv` | Appendix "Full Factorial ANOVA" summary figure data; "Discussion" in-distribution means |
| 6 | `persequence_v3.py` | 2 cells x 3 seeds checkpoints, held-out pickles | `outputs/persequence_v3_figcells_nogeom.out` | Figure 4(c,d) |
| 9 | `ci_agg.py`, `ci_arms.py` | `outputs/contact_invariance_v3.json` only | stdout | "ANOVA Results" (GNN Layers: median contact drift by depth and overall), "Discussion" |
| 10 | `paramcount_v3.py` | 6 checkpoints | stdout | "Hyperparameter Settings" (Table 1, parameter counts) |
| -- | `probe_capacity_v5.py` | capacity-probe checkpoints | `outputs/probe_capacity_v5.out`, `outputs/probe_capacity_v5_{runs,perseq,summary}.csv` | Appendix "Capacity probe" (Table 5) |
| -- | `probe_dnaonly.py` | final-cell checkpoints with and without `--dna_only_message_passing` (3 seeds each), training and held-out pickles | `outputs/probe_dnaonly.out`, `outputs/probe_dnaonly_{runs,perseq}.csv` | contact-free variant (added during review) |
| R1 | `select_v4_val.py` | 96 contact-free Hamiltonian checkpoints (`campaign/ham_*`), training pickles | `outputs/select_v4_val/<run>.json` | revision: validation T, DOS, per-residue LDOS and onsite distance (model selection, Table 3) |
| R2 | `score_heldout_v4.py` | 144 checkpoints (96 Hamiltonian, 24 sum-pooled and 24 mean-pooled direct), held-out pickles | `outputs/heldout_v4/<run>.json` | revision: held-out 12/16-bp T and DOS, per duplex (report only) |
| R3 | `score_newset_v4.py` | the same 144 checkpoints, untouched 16-bp pickles | `outputs/newset_v4/<run>.json` | revision: untouched test set (report only) |
| R4 | `build_report_v4.py` | stages R1-R2 | `outputs/posthoc_v4_report.json`, `outputs/best_epoch_cache_v4.json` | revision: inputs to stage R5 |
| R5 | `doe_v4.py` | stage R4 | `outputs/doe_v4.out` | revision: "ANOVA Results" and Appendix "Full Factorial ANOVA" |
| R6 | `anova_tables_v4.py` | stages R4-R5 | `outputs/anova_tables_v4.tex`, `outputs/new_tables_v4.tex` | revision: Tables 3-4 and the held-out/untouched tables |
| R7 | `newset_report_v4.py` | stage R3 | `outputs/newset_report_v4.out` | revision: untouched test set (per duplex, sign test, family fractions) |
| -- | `probe_splitpool.py`, `probe_splitpool_newset.py` | the DOS+T two-layer direct cell with mean, sum and split pooling (`exploratory/blind_dos_L2_nogeom_splitpool_s*`), the selected models; held-out and untouched pickles | `outputs/probe_splitpool*`, `outputs/probe_splitpool_newset*` | revision: exploratory split pooling (mean pool for the DOS head, sum pool for the transmission head), Discussion and the untouched-set table |
| -- | `probe_capacity_v6.py` | capacity-probe checkpoints (`capacity_probe/`) and the selected cells (`campaign/`) | `outputs/probe_capacity_v6.out`, `outputs/probe_capacity_v6_{runs,perseq}.csv` | revision: Appendix "Capacity probe" (Table 5) |
| -- | `probe_sumpool.py`, `probe_newset.py` | the first three sum-pooled direct runs | `outputs/probe_sumpool*`, `outputs/probe_newset*` | revision: early sum-pooling probe, disclosed in the selection record; not used for selection |
| -- | `geometry_spread.py` | `geom_cache/geometry_v2.pkl`, `geom_cache/geometry_heldout_L12_L16.pkl` | `outputs/geometry_spread.out` | "ANOVA Results", Geometry subsection (feature spreads) |

(paper section, figure and table names as of submission)

The dispersion and training-time (best-epoch) sections of `doe_v3.out` are not
reported in the paper.

## The figure-pair `persequence_v3.py` invocation

`persequence_v3.py`'s default checkpoint pair is the DOE-selected pair (best
cells on l12_transmission), not the pair used in the paper's Figure 4. To reproduce the shipped
`outputs/persequence_v3_figcells_nogeom.out`, run:

```
python scripts/persequence_v3.py \
  --ham-dir-tmpl <REPO>/outputs_v3/ham_ldosonly_n2_L4_nogeom_s{seed} \
  --blind-dir-tmpl <REPO>/outputs_v3/blind_dos_L2_nogeom_s{seed} \
  --blind-geometry false \
  --out-suffix figcells_nogeom
```

## Figure models

Revision. The models behind the revised paper's figures and tables are shipped in
`trained_models/`, each with a `.json` copy of its `resolved_config.json`:

- `hamiltonian_dos_n2_L4_nogeom_s2129768291` -- contact-free Hamiltonian model,
  DOS+T supervision, two orbitals per base, four GNN layers, no geometry;
- `standard_tonly_L4_nogeom_sumpool_s3731635825` -- direct model, T-only
  supervision, four layers, no geometry, sum pooling;
- `standard_dos_L2_nogeom_sumpool_s2129768291` -- direct model, DOS+T, two layers,
  no geometry, sum pooling; reported beside the T-only model wherever DOS is
  compared.

Selection used validation data only, the same rule for both families: the
configuration with the lowest 3-seed-mean validation transmission loss, then the
seed with the lowest validation transmission loss within it. The held-out 12/16-bp
duplexes and the untouched 16-bp set were never used for selection.

As submitted. The submitted paper used
`hamiltonian_ldosonly_n2_L4_nogeom_s3731635825` and
`standard_dos_L2_nogeom_s3731635825`, chosen by the lowest sum of validation, 12-bp
and 16-bp transmission losses (3.05 / 3.97 / 4.96 Hamiltonian, 5.32 / 5.49 / 6.03
direct). Both are in the first Zenodo checkpoints version.

The `git_sha` recorded in the `resolved_config.json` files (046bc55 for the 120
campaign runs and both figure models, 3efcc97 for the capacity probe) predates a
history rewrite. Their public equivalents are 8c2ec72 and 9b65bf0. Each pair differs
only in comments, docstrings, one argparse help string, one placeholder path in
`scripts/reconcile_dataset.py`, two removed job scripts and removed planning
documents; no model, training or data-loading code differs.

## Shipped outputs

`outputs/` contains only the floor-25 results:
`posthoc_v3_report.json`, `contact_invariance_v3.json`, `best_epoch_cache.json`,
`doe_v3.out`, `doe_v3_runs_hamiltonian.csv`, `doe_v3_runs_direct.csv`,
`doe_v3_cells_hamiltonian.csv`, `doe_v3_cells_direct.csv`,
`persequence_v3_figcells_nogeom.out`, `probe_capacity_v5.out`,
`probe_capacity_v5_runs.csv`, `probe_capacity_v5_perseq.csv`,
`probe_capacity_v5_summary.csv`, `geometry_spread.out`, `probe_dnaonly.out`,
`probe_dnaonly_runs.csv`, `probe_dnaonly_perseq.csv`.

Revision: `select_v4_val/`, `heldout_v4/`, `newset_v4/` (one JSON per run),
`posthoc_v4_report.json`, `best_epoch_cache_v4.json`, `doe_v4.out`,
`anova_tables_v4.tex`, `new_tables_v4.tex`, `newset_report_v4.out`,
`probe_capacity_v6.out`, `probe_capacity_v6_runs.csv`, `probe_capacity_v6_perseq.csv`,
`probe_sumpool.out`, `probe_sumpool_runs.csv`, `probe_sumpool_perseq.csv`,
`probe_newset.out`, `probe_newset_runs.csv`, `probe_newset_perseq.csv`, `probe_splitpool.out`, `probe_splitpool_runs.csv`, `probe_splitpool_perseq.csv`, `probe_splitpool_newset.out`, `probe_splitpool_newset_runs.csv`, `probe_splitpool_newset_perseq.csv`.

## Revision (2026-11)

The revised paper uses a retrained campaign:

- Hamiltonian family: all 32 configurations (supervision x orbitals x layers x
  geometry) x 3 seeds = 96 runs, retrained contact-free
  (`--dna_only_message_passing`, now the default), in `campaign/ham_*`. Same recipe as
  below, with the supervision weights, `--n_orb`, `--num_layers` and the geometry
  flag set per configuration.
- Direct family: the 24 submitted mean-pooled runs (`campaign/blind_*` without a pooling
  suffix) plus 24 new runs with `--readout_pooling sum` (`campaign/blind_*_sumpool_*`), so pooling is
  a design factor: 16 configurations x 3 seeds.
- The contact-invariance response is zero by construction for contact-free models
  and is replaced in the ANOVA by the validation per-residue LDOS loss.
- The untouched 16-bp test set (8 duplexes x 4 contact configurations, 32 records) is
  `untouched_L16.h5` in version 2 of the dataset record (https://doi.org/10.5281/zenodo.22964052). Its scoring plan was written down before
  any of its DFT results existed.

## The contact-free variant

(As first run during review, on the submitted final cell; the revision retrains the
whole Hamiltonian family this way.) The final Hamiltonian cell retrained with the GNN layers restricted to the DNA
subgraph: the two contact nodes and their edges are removed from message passing,
while GammaL/GammaR are still read from the contact edges, so one learned H serves
all four contact configurations of a sequence. Everything else matches the campaign
cell `ham_ldosonly_n2_L4_nogeom` (same seeds, split seed 42, 10,000 epochs):

```
python scripts/train.py --data_source pickle --data_dir pickle_files_v2 \
  --model_type hamiltonian --hidden_dim 256 --num_heads 4 --conv_type gat \
  --ldos_target residue --optimizer adam --weight_decay 1e-5 --learning_rate 1e-3 \
  --batch_size 32 --num_epochs 10000 --split_seed 42 --num_layers 4 \
  --loss_a 1.0 --loss_b 1.0 --loss_c 1.0 --init_seed <seed> \
  --solver_type complex --log_floor 1e-38 --floor_mode smooth --complex_eta 1e-12 \
  --use_log_outputs True --enforce_hermiticity True --n_orb 2 \
  --dna_only_message_passing \
  --output_dir exploratory/ham_ldosonly_n2_L4_nogeom_dnaonly_s<seed>
```

The three checkpoints ship in `exploratory.tar` of the Zenodo checkpoints record
(`exploratory/ham_ldosonly_n2_L4_nogeom_dnaonly_s*`), together with the three split-pooled
direct runs (`exploratory/blind_dos_L2_nogeom_splitpool_s*`, read by
`scripts/probe_splitpool.py`). `scripts/probe_dnaonly.py` and `scripts/probe_sumpool.py`
also read the original contacts-in-graph cell from version 1 of the record (`outputs_v3/`).

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
