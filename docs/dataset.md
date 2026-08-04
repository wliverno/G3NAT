# The DFT transport dataset: inventory, conventions, and open questions

> **The DFT data is released with the preprint, not before** (willll, 2026-08-03). The
> format, conventions and inventory are documented openly -- see
> `DNADataset/README.md` -- but no archive file is distributed yet, and none has ever been
> committed to this repository on any branch.

Working notes toward a dataset to ship with the paper. Everything here was
measured on 2026-07-24 (`scripts/reconcile_dataset.py`), not
recalled. Where a claim is unverified it says so.

Scope decision (willll, 2026-07-24): the dataset, when released, will carry **transport
observables plus structure**. It does NOT carry the DFT Fock/overlap matrices. Those exist upstream
(523 `_Fock.mat`) and are a separate publication decision; they are also already ruled out
as a training target for this project.

## Inventory (verified)

**Which directory a number refers to matters** -- `pickle_files` is the original set and
`pickle_files_v2` is the regenerated one, and they differ. Recounted 2026-08-01:

| directory | files | unreadable | unique sequences | by length (4/5/6/7/8) |
|---|---|---|---|---|
| `pickle_files` (v1) | 2058 | 1 | **515** | 135 / 112 / 82 / 90 / 96 |
| `pickle_files_v2` | 2077 | **0** | **520** | 135 / 113 / 83 / 93 / 96 |

v2 is a strict superset: it adds `CGTAT`, `CGTTCCT`, `GCCTGG`, `TCGCTCC`, `TTAAAAG` and loses
nothing. It also repaired the truncated `gaaac_run2.pkl`, so v2 loads 2077 of 2077 with no
skipped file -- the "Ran out of input" warning that appeared in every v1 run log is gone.

The table below is the ORIGINAL v1 inventory, retained because the reconciliation arithmetic
refers to it.

| quantity | count |
|---|---|
| pickle files | 2058 |
| unique sequences in pickles | 515 |
| sequence dirs under `asyed4/DNADataSet` | 583 |
| dirs with at least one DOS `.mat` | 522 |
| dirs with a PDB | 581 |
| DOS `.mat` files | 2081 |
| Tran `.mat` files | 2086 |

Reconciliation: `2058 = 514 sequences x 4 variants + 1 sequence x 2`. The 4 variants are
`{contact_type: same, cross} x {coupling: 0.1, 0.6 eV}`, verified over the full dataset
(same/0.1 515, cross/0.1 514, same/0.6 514, cross/0.6 514).

**Coverage for LDOS work is complete**: zero sequences in the pickles lack a DOS `.mat`,
zero lack a PDB. 515 usable sequences.

Sequence lengths: 135 / 112 / 82 / 90 / 96 sequences at lengths 4 / 5 / 6 / 7 / 8.

Composition subsets that matter for testing conventions: **20 AT-only** sequences (no G or
C) and **16 GC-only** (no A or T).

## Defects to fix on regeneration

- **`gaaac_run2.pkl` is 0 bytes** -- an empty file from a truncated write, not corrupt
  content. Every training run logs "Error loading ... Ran out of input" and skips it, so
  2057 of 2058 files actually load. **FIXED in v2**, which loads 2077 of 2077.
- **`gjf_text` is redundant.** It carries the full Gaussian input deck including 3D
  coordinates, and the loader discards it. Verified: the gjf coordinates and the PDB
  coordinates are the same geometry to **0.0000 A** over all 253 atoms of `aaac`. The
  published record should carry the PDB and drop the deck -- it also shrinks every record
  substantially.
- **7 sequences had DFT results but no pickle**: `cactc`, `cgtat`, `cgttcct`, `gcctgg`,
  `tcgctcc`, `tggaa`, `ttaaaag`. **5 of the 7 were recovered in v2.** The remaining two are
  not recoverable and should be dropped from the todo:
  - `cactc` and `tggaa` have upstream `DOS_*.mat` files of **exactly 131072 bytes** (128 KiB,
    a buffer boundary) against ~491 KB for a healthy one -- the upstream write was truncated.
    `scipy.io.loadmat` fails with "Did not read any bytes", and the converter drops them
    correctly. Their remaining runs have a `Tran_*.mat` but no `DOS_*.mat` at all.
  - `tggaa` additionally has no `.pdb`, `.gjf` or `.log`, so it has no structure either.
  Recovering them means re-running the upstream transport calculation, not re-parsing.
- **The energy convention is undocumented** (see below). A published dataset cannot leave
  each consumer to infer it.

## Transmission dynamic range, and what it does to every loss number (2026-08-01)

Measured over all 417,477 energy points in 2077 v2 records.

| percentile | T |
|---|---|
| p50 | **1.709e-08** |
| p90 | 5.04e-04 |
| p99 | 0.1509 |
| p99.9 | 0.7932 |
| max | 1.002 |

**Half of every spectrum is deep tunneling at ~1e-8, carrying no appreciable current**, and
the training loss weights all 201 energy points equally in log10 space. So roughly half the
error budget of every number this project has ever quoted is spent on a region no transport
measurement would resolve. This is not an argument for changing the loss -- a uniform log-space
fit is a defensible choice -- but any model comparison should be reported BOTH whole-window and
restricted to where T is appreciable. A model that wins on the tail and loses at the resonances
would look better on the whole-window metric while being worse for transport.

**The rank-1 unitarity ceiling is not a binding constraint.** The model applies its
self-energy through rank-1 contacts, so its `T = Tr(Gamma_L G^r Gamma_R G^a) <= 1`, whereas the
DFT reference couples every atomic orbital of each terminal residue (rank ~330/349 for `aaac`)
and could in principle exceed 1. It essentially never does: **1 point in 417,477 is above
T = 1, at 1.002**. Recorded so the hypothesis that this structurally floors the model's error
is not re-derived -- it explains nothing.

**Reference smoothness**, mean |2nd difference| of log10: **T 0.2613, DOS 0.1704**. DOS and T
generated from a finite Hermitian H with fixed Gamma are sums of Lorentzians and therefore
smooth. This is the target for any comparison against a model that emits 201 independent
values with no constraint tying neighbouring energies.

## Energy convention

Raw `Egrid` is 201 points spanning exactly 2 eV, e.g. `aaac` is `[-6.3031, -4.3031]` with
mean `-5.3031`. The loader (`g3nat/data/pickle.py:50`) subtracts the mean, giving `[-1, 1]`.

**The grid is centred on the HOMO**, so the window is HOMO +/- 1 eV and the reference is
**per-sequence**. Confirmed by willll 2026-07-24: this is how the transmission and DOS were
generated, i.e. it is a property of the generation pipeline, not an inference.

Corroborated independently by the composition measurement below: `mean(Egrid)` separates
AT-only from GC-only sequences by 0.813 eV at 13.6 sigma with zero overlap, in the direction
G's low ionization potential predicts.

Consequences:

1. The `[-1, 1]` window is centred on the frontier orbital, not an arbitrary slice. All 201
   grid points lie inside it; there is no DFT data outside the window at all. So questions
   about out-of-window model behaviour **cannot be tested against this dataset**.
2. The published record MUST store the **absolute reference energy** per sample. As it
   stands the centring is irreversible once the raw grid is dropped, and nobody downstream
   could recover physical energies or compare across sequences.

## RESOLVED: HOMO centring does make the G result convention-driven (2026-07-24)

Tested with `scripts/homo_composition.py` over all 515 sequences. The hypothesis below was
**confirmed**, and a second measurement overturned the reason we had for worrying about it.

**Composition drives the HOMO, decisively:**

| class | n | E_HOMO (eV) |
|---|---|---|
| AT-only | 20 | -5.7328 +/- 0.0591 |
| mixed | 479 | -5.0356 +/- 0.1242 |
| GC-only | 16 | -4.9194 +/- 0.0605 |

GC-only minus AT-only = **+0.8134 eV at 13.6 sigma**, and the classes are **completely
disjoint** (AT max -5.6258 < GC min -5.0491). correlation(GC fraction, E_HOMO) = +0.69.

It behaves as a **step function, not a gradient**: GC=0.00 gives -5.7328, GC=0.12 gives
-5.1109 -- a single GC pair moves the HOMO 0.62 eV -- and the entire remaining range from
GC=0.12 to GC=1.00 adds only 0.19 eV. Exactly what you expect if the HOMO is *the G level
whenever any G is present*.

**Consequence:** for 495 of 515 sequences, per-sequence centring pins a G-derived level at
~0 by construction. The learned table's "G on top at -0.295, everything else 0.7-1.0 below"
is substantially the reference convention. `docs/model-results.md` section 4b now carries
this caveat. A single per-base table must reconcile "G at 0" (495 sequences) with "A at 0"
(20 AT-only sequences), and at 495-to-20 the G constraint dominates.

**A shared absolute axis IS available** (this corrects an earlier claim that it was not):
E_HOMO spans 1.035 eV across all sequences against 2.0 eV windows, so the intersection of
all 515 windows is **0.965 eV**, rising to **1.63 eV** if 90% of sequences are kept. Universal
absolute parameters would be learnable on that axis.

**DECISION (willll, 2026-07-24): stay HOMO-referenced.** The Fermi energy sits near the HOMO,
transport happens at E_F, and we are modelling NEGF transport -- so the HOMO-referenced window
is the physically meaningful one. Re-referencing to a common absolute axis would misalign the
transport window across sequences, optimising for parameter extraction at the cost of the
observable we actually model. Universal absolute parameters are **out of scope**, noted as
future work. The published records still store the absolute reference so others can make the
other choice.

**What "interpretable H" now means for this project:** not a universal absolute per-base
table -- that framing is retired. It means (a) relative level structure within a sequence,
which is reference-free, and (b) whether H puts spectral weight in the right places, which is
what LDOS measures. (b) is now the primary interpretability handle rather than a supporting
one.

---

### Original hypothesis, kept for the reasoning trail

Every duplex here contains G unless it is AT-only, because every C pairs with a G. If the
HOMO of a G-containing duplex is a G-derived level, then centring puts G at ~0 **by
construction**. Our learned per-base table (alpha=1.0) puts G on top at `-0.295` with A, T
and C 0.7-1.0 eV below, and G was the one base whose position replicated across seeds while
C scattered with std 0.52 (see `docs/model-results.md`, "Replication across seeds"). That
pattern is what a HOMO-referenced frame would produce whether or not the model learned
anything about G.

The measured spread supports the mechanism being real: `mean(Egrid)` varies by ~0.36 eV
across `gggg`/`acgt`/`aaac`, and per-sequence centring removes exactly that variation --
which is the base-composition signal.

**Test (cheap, no training required).** GC-only sequences (16 available) must have a
G-derived HOMO. AT-only sequences (20 available) contain no G at all, so their HOMO must be
an A- or T-derived level. Compare `mean(Egrid)` between the two classes. If it shifts
systematically, the centring carries composition information, and a single per-base onsite
table cannot satisfy both classes simultaneously.

If confirmed, this reframes the onsite investigation: the quantity that most distinguishes
the bases -- their level positions -- would have been normalised away per sequence before
the model ever saw it. The "G is resolved" finding would need a caveat, and training on
absolute rather than HOMO-referenced energies would become worth considering.

## Per-record schema (draft, not yet settled)

Verified facts the schema rests on:
- `DOSAtom` is `[n_atoms, 201]` and its row order matches PDB atom order exactly
  (253 == 253 for `aaac`).
- `sum_atoms(DOSAtom) / DOS = 1.0000` at every energy -- the atom-resolved LDOS is an
  **exact** decomposition of the total DOS, not an approximation.
- Residue count equals model site count (8 residues == 8 sites for `aaac`, both strands,
  `n_orb=1`), so residues map 1:1 onto tight-binding sites.

Draft fields: `sequence`, `complementary_sequence`, `energy_grid` (centred) plus
`energy_reference_eV` (absolute, so centring is invertible), `dos`, `transmission`,
`ldos_residue` `[n_res, n_E]`, `ldos_base_only` `[n_res, n_E]`, `atom_to_residue`,
`residue_to_site` (the mapping whose failure mode is silent), `contact_type`,
`coupling_eV`, `contact_sites`, `pdb` (structure), and provenance (method, basis,
generation date, upstream path).

## Dataset v2 (2026-07-26): per-atom LDOS

Regenerated from the existing DFT and transport output -- pure parsing, nothing recomputed.
The v2 records add `DOSAtom`, the atom-resolved density of states, shape
`[n_atoms, n_energy]`.

**2077 records across 520 sequences.** Output in `pickle_files_v2/`, both gitignored. An
unreleased local archive `g3nat_dna_transport.h5` is built from it by `export_hdf5.py`.

| sequence length | sequences | records |
|---|---|---|
| 4 | 135 | 540 |
| 5 | 113 | 451 |
| 6 | 83  | 330 |
| 7 | 93  | 372 |
| 8 | 96  | 384 |

Per run: run1 = 520, run2 = 519, run3 = 519, run4 = 519.

`pickle_files/` (2058 records, no `DOSAtom`) is deliberately NOT replaced. Every recorded
model result in `model-results.md` was trained against it, so switching the training path to
`pickle_files_v2/` changes the dataset underneath the model and invalidates comparison
across that boundary. Re-baseline deliberately rather than swapping in place.

### Properties verified across the full set

- `DOSAtom.sum(axis=0)` reproduces `DOS` to a worst-case relative deviation of **3.6e-15**
  across all 2077 records, and **4.2e-15** read back out of the HDF5. Float64 noise. On the
  single `aaac` fixture it was 1.4e-15, so the invariant holds across all five duplex
  lengths and 250-510 atoms, not only where it was first measured. This is what makes the
  per-atom LDOS usable as a training signal: it is exactly consistent with the DOS the model
  already fits, not a second and disagreeing target.
- The per-record HOMO reference is composition-driven, as expected: `aaat` (pure AT) sits at
  -5.8120 eV while `aaacgacg` (GC-rich) sits at -4.8709 eV -- a 0.94 eV spread across five
  consecutive records, consistent with the 0.813 eV / 13.6 sigma AT-vs-GC separation
  measured earlier. This is why the archive's own root attrs warn against comparing a fixed
  relative energy across sequences.
- Element-to-orbital mapping, measured over 55 sequences / 22781 atoms and one-to-one with
  no exceptions: `H:5, C:15, N:15, O:15, P:19`, as expected for B3LYP/6-31G(d,p) with
  Cartesian polarization functions. `validate_record` checks every atom against this map.
- `resseq` is monotonic non-decreasing in every sequence, so the atom table is always
  grouped by residue in PDB file order -- which is also `DOSAtom` row order.
- Full residue-numbering verification, run over all 2077 records (762,410 atoms) on
  2026-07-26:

  | check | result |
  |---|---|
  | sorted unique `resseq` == contiguous `1..2L` | 2077 pass, 0 fail |
  | residues `1..L` resname match `sequence` | 2077 pass, 0 fail |
  | residues `L+1..2L` resname match `complementary_sequence` | 2077 pass, 0 fail |
  | `resseq` non-decreasing down the atom table | 2077 pass, 0 fail |

  Distinct resnames across all atoms: exactly `DA`, `DC`, `DG`, `DT` -- no terminal variants
  (`DA5`/`DA3`), no waters, no ions. This is the basis for `H index = resseq - 1`, used by
  `aggregate_by_residue` in `g3nat/data/ldos.py`.

## Identifiability limit: A/T and G/C onsite terms are confounded by construction (2026-07-30)

Raised by willll. **Every sequence in the dataset is a perfect Watson-Crick duplex in a
single geometry.** Verified over all 2077 v2 records: residues `L+1..2L` are exactly the
reverse complement of residues `1..L`, with only four resnames present (`DA/DC/DG/DT`), no
mismatches, no single strands, no alternative pairings.

So **every A is accompanied by a T, and every G by a C, in identical geometry.** There is no
record anywhere in the dataset in which A appears without T. For a per-base onsite table this
is a structural confound: only a combination of the A and T onsite terms is identifiable from
these observables, never the split between them. The same holds for G and C.

**Consequence for the learned table.** A model that lands on A ~ T is responding correctly to
an unidentifiable parameter, not failing to resolve one. Measured in the Phase O structured
onsite runs: at `b = 0.5` the A-T separation is 0.0085 against a pooled cross-seed std of
0.0298 -- i.e. indistinguishable, and now tightly so. Any A-vs-T comparison against literature
values (Roche has A -0.49, T -1.39, a 0.90 eV split) is therefore a comparison the data cannot
support in either direction, and should not be presented as agreement OR as disagreement.

**Why G and C are not equally degenerate.** They are confounded by the same argument, but the
energy convention supplies an extra constraint that A/T do not get: the grid is centred on the
HOMO, and the HOMO is a G-derived level whenever any G is present (495 of 515 sequences, see
the composition section above). That pins G near zero externally and leaves C free to take the
remaining freedom. The G/C split is therefore set partly by the reference convention rather
than by the fit, which is the same caveat already recorded for the G column in
`scripts/extract_tb_params.py`.

**What would break the confound.** Data containing the bases in geometries that do not pair
them one-to-one -- mismatched pairs, single strands, non-Watson-Crick pairings, or homoduplex
constructs -- would separate the A and T contributions. None of that exists in this dataset,
so it is a limit of the data, not of the model or the loss. Worth stating plainly in the
discussion rather than leaving the A/T degeneracy to look like a shortcoming of the method.

## States inside the transport window, and a heavy tail worth investigating (2026-07-30)

Measured by trapezoid-integrating the DFT `DOS` over the 2 eV window for all 2077 v2 records.

**CORRECTION (2026-07-30): this integral is NOT a level count, and the section originally
said it was.** The identity "integral of DOS = number of levels" holds over ALL energy, where
it gives the basis size `N`. Over a finite window it requires every in-window level to be
resolved by the broadening, and in this dataset that fails badly in both directions, because
`eta = 0` leaves contact `Gamma` as the only imaginary part. Sampled against direct
eigenvalue reads, the integral runs from ~0.35x the in-window level count to more than 2000x
it, continuously, with no clean separation. Treat the numbers below as **a property of the
integrated DOS**, useful for spotting sick records, and not as a state count. Anything built
on them as a state count is retracted -- see `docs/model-results.md` section 7a.

**Medians** (the mean is destroyed by the tail described below):

| L | 2L (model sites) | records | median states in window | ratio to 2L |
|---|---|---|---|---|
| 4 | 8 | 540 | 15.51 | 1.94 |
| 5 | 10 | 451 | 22.20 | 2.22 |
| 6 | 12 | 330 | 40.03 | 3.34 |
| 7 | 14 | 372 | 86.26 | 6.16 |
| 8 | 16 | 384 | 217.18 | 13.57 |

Pooled ratio percentiles: p1 0.67, p5 0.86, p25 1.52, **p50 2.69**, p75 10.15, p95 147.70,
p99 608.39.

### The true level counts, and why the integral looked like it agreed with them

Counted properly from `asyed4/DNADataSet/<seq>/<seq>_eigen.mat`. **That file is in Hartree and
is unsorted** (identified by willll); multiply by 27.211386 and sort, and its first `n_occ`
entries reproduce `occ.mat` to 7.5e-4 eV and its HOMO reproduces `energy_reference_eV` to
2e-6 eV. Over 189 sampled records:

| L | median true levels in window | levels / 2L | median integral / level |
|---|---|---|---|
| 4 | 11.0 | 1.375 | **1.07** |
| 5 | 15.0 | 1.500 | 1.27 |
| 6 | 18.0 | 1.500 | 3.40 |
| 7 | 22.0 | 1.571 | 4.31 |
| 8 | 24.5 | 1.531 | **10.44** |

The true count per site is **flat at about 1.5**, not the 1.94-to-13.57 gradient the integral
suggested. The apparent gradient was the integral's own error growing with length.

This also explains why the original cross-check looked convincing. It compared a Gaussian-log
count for `aaac` (14 in window, ratio 1.75) against the L=4 integral (15.51, ratio 1.94) and
concluded "two methods, same answer". At L=4 the integral genuinely is a decent proxy --
integral/level is 1.07 there. The check was valid, and it generalized to nothing: by L=8 the
same comparison is off by a factor of ten. A single-length agreement was read as validating
the method at all lengths.

### The tail IS a numerical artifact, and the mechanism is identified (2026-07-30)

**7.1% of records (147 of 2077) have a ratio above 100**, i.e. more than 800 states inside a
2 eV window. The worst are `attggctg_run2` (6423x), `taagtaa_run2` (5807x), `taagtaa_run4`
(4309x), `atgccaca_run2` (2185x). The largest single DOS value anywhere in the dataset is
**1.028e7**.

**Proof that this is not physical, requiring no eigenvalue interpretation at all.** The
integral of DOS over ALL energy equals the basis size `N`. An integral over a 2 eV
sub-window therefore cannot exceed `N`. For the worst records it does, by more than an order
of magnitude (`N` = dimension of `<seq>_Fock.mat`; integral by trapezoid over `Egrid`):

| record | N | 2 eV window integral | ratio to N |
|---|---|---|---|
| `attggctg_run2` | 5806 | 102772.5 | **17.7** |
| `taagtaa_run2` | 5083 | 81299.8 | 16.0 |
| `taagtaa_run4` | 5083 | 60318.3 | 11.9 |
| `atgccaca_run2` | 5806 | 34954.0 | 6.0 |

Reproduced independently by two separate implementations to the digit. The DOS in these
records is not counting states.

**Mechanism.** `eta = 0` in the generator (verified in `Parameters.txt`), so contact `Gamma`
is the only imaginary part anywhere. The energy grid is constructed to place a point exactly
on the HOMO of every record -- confirmed, `max(occ.mat)` equals `energy_reference_eV` to
3e-7 eV for `aaac` and 7.7e-8 eV for `attggctg`. When the HOMO eigenvector additionally has
negligible weight on the contacted terminal residues (measured Mulliken weight 1e-6 to 1e-7
for tail records, 3-4 orders below healthy controls), nothing broadens that pole and `-Im G^r`
at that grid point diverges. Every one of the 147 flagged records peaks at exactly the HOMO
grid index. Note that healthy records frequently do too -- peaking at the HOMO is necessary
for the artifact, not sufficient, and is not on its own a diagnostic.

**This is the same tail seen in the LDOS distribution**: max 7.80e6 against a p99 of 9.76,
with only 4 values out of 4,852,542 above 3e6.

Still open, and worth settling before these records are trusted:

- **Their effect on window-averaged metrics is negligible -- measured, not assumed.**
  Recomputing the held-out DOS and LDOS bias for both `n_orb` arms with all 147 excluded
  moves the DOS bias by 0.0004 decades (`n_orb=1`: -0.4762 all, -0.4758 clean) and the LDOS
  bias by 0.007. The reason is that the artifact is a spike at a *single* grid point -- the
  HOMO index -- out of 201, so any metric averaged across the window sees about 1/201 of it.
  An earlier estimate that contamination might account for ~0.36 decades of the DOS bias
  assumed the corruption was spread across the window; it is not.
- They are therefore safe to leave in for window-averaged training and evaluation, and unsafe
  for anything evaluated AT or NEAR the reference energy, which is where transport actually
  happens. Any per-energy analysis at E_F must exclude them.
- `ratio > 100` is a threshold on a continuum, not a boundary between two populations. Sampled
  records run 0.35, 1.9, 7.9, 26.6, 88.2, ... with no gap. Any cut is arbitrary.
- The right fix is upstream: a small finite `eta` would regularize every one of these. That is
  a regeneration decision, not something to patch in the loader.
