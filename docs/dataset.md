# The DFT transport dataset: inventory, conventions, and open questions

Working notes toward a published dataset that ships with the paper. Everything here was
measured on 2026-07-24 (`/mmfs1/gscratch/anantram/willll/reconcile_dataset.py`), not
recalled. Where a claim is unverified it says so.

Scope decision (willll, 2026-07-24): the published dataset carries **transport observables
plus structure**. It does NOT carry the DFT Fock/overlap matrices. Those exist upstream
(523 `_Fock.mat`) and are a separate publication decision; they are also already ruled out
as a training target for this project.

## Inventory (verified)

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
  2057 of 2058 files actually load.
- **`gjf_text` is redundant.** It carries the full Gaussian input deck including 3D
  coordinates, and the loader discards it. Verified: the gjf coordinates and the PDB
  coordinates are the same geometry to **0.0000 A** over all 253 atoms of `aaac`. The
  published record should carry the PDB and drop the deck -- it also shrinks every record
  substantially.
- **7 sequences have DFT results but no pickle**: `cactc`, `cgtat`, `cgttcct`, `gcctgg`,
  `tcgctcc`, `tggaa`, `ttaaaag`. Free additions.
- **The energy convention is undocumented** (see below). A published dataset cannot leave
  each consumer to infer it.

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
