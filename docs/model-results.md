# Model Results (measured)

Running log of measured training results, so model-selection decisions are backed by
recorded numbers rather than memory. Val loss = final validation loss reported by the
training loop (log10 DOS + log10 transmission MSE, see `g3nat/models/hamiltonian.py`).

## Graph convolution type: GAT vs Transformer

**Decision: `--conv_type` default is `gat`** (set in `scripts/train.py`). GAT is the best
DFT-fitting convolution on record. The winner is dataset-dependent (see table), and the
current and upcoming work (X3DNA edge geometry, Plan 2) is on the DFT/pickle data, where
GAT wins decisively.

| Dataset            | conv        | final val loss | model file                                        | training log        |
|--------------------|-------------|----------------|---------------------------------------------------|---------------------|
| DFT (pickle)       | **gat**     | **0.5469**     | `outputs_pickle_gat/hamiltonian_pickle_model.pth` | `slurm-37393473.out`|
| DFT (pickle)       | transformer | 1.4197         | `outputs_pickle/hamiltonian_pickle_model.pth`     | `slurm-37391502.out`|
| TB synthetic (regen)| transformer | 0.0381         | `outputs_regen_transformer/hamiltonian_tb_model.pth`| `slurm-37375162.out`|
| TB synthetic (regen)| gat        | 0.4775         | `outputs_regen_gat/hamiltonian_tb_model.pth`      | `slurm-37373544.out`|

Reading: on DFT data GAT is ~2.6x lower val loss than Transformer (0.547 vs 1.42). On the
synthetic tight-binding data the ordering flips (Transformer 0.038 vs GAT 0.477). The
default is chosen for the DFT line of work.

All four runs are base-aware (Hamiltonian coupling depends on the two endpoint bases;
`g3nat/models/hamiltonian.py:164`, introduced in commit 69ef4d6). "base-aware" is intrinsic
to the current model, not a toggle.

## Best DFT model of record

- **File:** `outputs_pickle_gat/hamiltonian_pickle_model.pth`
  (also copied to `trained_models/hamiltonian_DFT_gat_baseaware.pth` as the canonical artifact)
- **Val loss:** 0.5469 (~0.547)
- **Config:** GAT + base-aware; hidden_dim=256, num_layers=4, num_heads=4, n_orb=1,
  lr=1e-3, batch=64, epochs=5000, data_source=pickle (DFT).
- **Provenance:** trained in `slurm-37393473.out`; config echoed in `slurm-37393449.out`
  (`conv=GAT ... params=665,346`).

Note: `--hidden_dim 256` was passed on the command line for these runs; the `train.py`
default remains 128.

## X3DNA edge geometry (Plan 2)

The geometry model-integration work defaults to `conv_type='gat'` (this result). See
`docs/superpowers/specs/2026-07-20-x3dna-edge-geometry-design.md`.

### Plumbing-check run: geometry ON vs OFF (DFT, GAT, 5000 epochs)

| run                 | final val | model file                                          | training log        |
|---------------------|-----------|-----------------------------------------------------|---------------------|
| geometry OFF (base) | 0.5469    | `outputs_pickle_gat/hamiltonian_pickle_model.pth`   | `slurm-37393473.out`|
| geometry ON         | 0.5383    | `outputs_pickle_gat_geom/hamiltonian_pickle_model.pth` | `slurm-37408577.out`|

Same config (GAT + base-aware, hidden=256, batch=64, 5000 epochs, DFT pickle), the
only difference is `--use_geometry`. **Interpretation: indistinguishable, NOT an
improvement.** The 0.009 gap is smaller than each run's own late-epoch wobble
(geom-ON 0.538-0.556, baseline 0.527-0.547 over their last 50 epochs), and the
geometry on this dataset is near-constant (idealized fiber B-DNA: rise 3.375 +/- 0.005,
twist 35.9 +/- 1.0, h-bond stagger exactly 0), so it carries no predictive signal.
What this run confirms is the plumbing: `--use_geometry` runs end-to-end on the full
dataset, trains stably (no NaN despite the near-constant features + 1e-6 std floor),
and the added geom_encoder neither helps nor hurts. Real "geometry helps" needs varied
structures (MD / crystal / predicted), which this branch is built to receive.

## Onsite/spectrum window constraint -- NEGATIVE RESULT (2026-07-23)

Goal: force the learned reduced Hamiltonian to be physical (energies inside the transmission
window). Branch `constrain-onsite-window` (DEAD, not merged).

**Physicality of existing models** (onsite = diag(H); eig = eigvalsh(H); window [-1,1] eV):

| model | val loss | onsite [min,max] | eig [min,max] | eig in-window | verdict |
|-------|----------|------------------|---------------|---------------|---------|
| GAT-DFT (baseline) | 0.547 | [-32.5, -0.30] | [-33.4, -0.01] | 59% | UNPHYSICAL |
| transformer-DFT | 1.42 | [-0.71, 2507] | [-0.81, 2651] | 25% | worse |
| GAT-TB (synthetic) | 0.477 | [-1.41, 0.02] | [-1.59, 0.03] | 100% | physical (matches Roche) |
| transformer-TB (synthetic) | 0.038 | [-1.40, 0.01] | [-1.62, 0.10] | 100% | physical |

Key finding: **physicality tracks the DATA, not the architecture.** When the ground truth is a
physical per-base TB (synthetic), both convs recover it. On real DFT both distort; the low-loss
GAT "win" (0.547) is an unphysical H. Likely a 1-orbital-per-base TB cannot represent full DFT
transport.

**Soft penalty attempts** (GAT + base-aware, hidden=256, 5000 epochs, W=10):

| constraint | final val | onsite result | eig in-window |
|------------|-----------|---------------|---------------|
| diagonal penalty | 1.87 | collapsed to -0.97 | (n/a) |
| eigenvalue penalty | 2.19 | collapsed to -0.63 | 100% |

Both failed: a penalty enforces a RANGE but not STRUCTURE, so the optimizer collapses every
onsite to one value (in-window, degenerate, useless). Penalty route exhausted.

**Next (planned, fresh branch off main):** structured onsite head -- tie onsite to base identity
(residual base-baseline + limited context correction) so physicality comes from the
parameterization, not a penalty. See the `g3nat` skill "Active investigation" for the reasoning.

## Clean (grouped) split: the honest baseline (2026-07-24)

Every DFT val number above this line was computed under a LEAKING flat-index split
(`train_test_split(range(len(dataset)))` over ~2057 samples that are 515 sequences x 4
contact variants, so the same sequence appears in train and val). Fixed in commit c7feebd
(`GroupShuffleSplit` on the sequence string).

The leaking run scores 0.547; the grouped-split run scores **0.6054**
(`outputs_onsite_sweep_a0_s42`, which is alpha=0 == the unmodified model). Treat 0.605, not
0.547, as the free-model reference. Numbers above the line are not re-run, remain optimistic,
and should not be compared across the line.

**Do not attribute the whole 0.058 gap to the leak.** The two runs differ in more than the
split: the 0.547 record used `batch=64`, while the sweep runner
(`scripts/run_onsite_sweep.sh:69`) uses `--batch_size 32`. The gap therefore confounds split
with batch size, and it is a single seed on each side. The direction (clean split is worse)
is what the leak predicts and is not in doubt; the magnitude is not established. A clean
attribution needs the leaking config re-run at batch=32, or the grouped split at batch=64.

## Structured onsite head: alpha sweep (2026-07-24)

Config: GAT + base-aware, hidden=256, num_layers=4, num_heads=4, n_orb=1, lr=1e-3,
5000 epochs, DFT pickle, **grouped split**, `--split_seed 42`. Runner:
`scripts/run_onsite_sweep.sh`. Collector: `scripts/collect_onsite_sweep.py`.
Raw output: `slurm-37620671.out`. Physicality measured over a fixed sample of 400 unique
DFT sequences (identical across cells). Window = [-1, 1] eV.

| alpha | val_loss | tail_mean | slope/ep  | ons_in_win | eig_in_win | coup_bw | eta2  | distinct |
|-------|----------|-----------|-----------|------------|------------|---------|-------|----------|
| 0     | 0.6054   | 0.6033    | -7.32e-05 | 0.381      | 0.396      | 0.94    | 0.028 | 0.000    |
| 0.25  | 0.6312   | 0.6207    | -2.86e-04 | 0.457      | 0.443      | 1.67    | 0.001 | 0.011    |
| 0.5   | 0.5734   | 0.5756    | -1.76e-05 | 0.416      | 0.430      | 0.83    | 0.001 | 0.020    |
| 0.75  | 0.5881   | 0.5842    | -1.63e-04 | 0.445      | 0.450      | 2.93    | 0.003 | 0.006    |
| 0.9   | 0.6928   | 0.7042    | -2.17e-04 | 0.576      | 0.573      | 0.28    | 0.745 | 0.494    |
| 1.0   | 0.7034   | 0.6592    | +1.73e-04 | 0.244      | 0.484      | 0.62    | 1.000 | 0.061    |

All six cells satisfy |slope| <= 3e-4/epoch, so none is under-converged by that criterion.
alpha=1.0 is nonetheless the least settled: it is the only cell with a *positive* tail slope
(+1.7e-4) and the only one whose final epoch (0.7034) is worse than its own tail mean
(0.6592). Read its headline number as ~0.66-0.70, not 0.7034.

**Learned per-base baselines** (eV, `onsite_baseline`, n_orb=1):

| alpha | A       | T       | G       | C       | spread | min pairwise gap |
|-------|---------|---------|---------|---------|--------|------------------|
| 0.9   | -1.1252 | -0.6315 | +0.2508 | -1.6306 | 1.882  | 0.494            |
| 1.0   | -1.2725 | -1.1955 | -0.2950 | -1.3340 | 1.039  | 0.061 (C-A)      |

### CORRECTION: the alpha sweep does not measure what it was designed to measure

The sweep was pre-registered as a discriminator for "how much context does the DFT onsite
actually need." **It cannot answer that**, because for every `alpha < 1` the mixing is a
vacuous reparametrization, not a constraint.

`_mix_onsite` (`g3nat/models/hamiltonian.py:154-171`) computes
`onsite = a*baseline[base] + (1-a)*onsite_proj(h)`, and `onsite_proj`
(`hamiltonian.py:84-88`) is an unbounded MLP ending in a free `Linear`.

For any `a < 1`, the free model's onsite function `f(h)` is reproducible: collapse the
baseline table to a common constant `c` (any value, not necessarily 0), scale the final
`Linear`'s weight by `1/(1-a)`, and set its bias to `(b2 - a*c)/(1-a)`. Then
`a*c + (1-a)*context_new(h) = f(h)` identically. The baseline term must be absorbed as well
as the scale -- stating only "rescale the last layer" is incomplete, because an uncancelled
`a*baseline[base]` would remain. **The hypothesis class is identical for all alpha in [0,1)**
(and the collapse-to-a-constant that the recipe requires is exactly what the measured
`distinct ~ 0` column shows). Only `alpha = 1.0` deletes the context term and changes the
class.

So alpha cannot be a structural constraint. It is a reparametrization.

**RETRACTED (2026-07-24, adversarial review): the "needs 10x weights" mechanism.** An earlier
version of this section claimed the trained outcomes differ because reaching the free
solution at `a=0.9` requires ~`1/(1-a)` = 10x larger head weights, which `weight_decay=1e-5`
(`g3nat/training/trainer.py:38-43`, applied uniformly with no per-parameter exclusions)
resists. That story is **falsified by the checkpoints themselves.** Measured final-layer
norms of `onsite_proj`:

| alpha    | 0     | 0.25   | 0.5   | 0.75   | 0.9   | 1.0   |
|----------|-------|--------|-------|--------|-------|-------|
| \|\|W2\|\|   | 2.164 | 4.063  | 2.218 | 6.458  | 3.090 | 0.000 |
| \|\|W0\|\|   | 9.386 | 12.491 | 9.355 | 14.166 | 5.860 | 0.000 |

The predicted growth (~2.2, 2.9, 4.3, 8.7, 21.6, inf) does not appear: alpha=0.9's norm
(3.09) is *smaller* than alpha=0.75's (6.46) and only ~1.4x alpha=0's. (alpha=1.0 at exactly
0 is expected -- the context head gets zero gradient there and decays away.)

What stands: the hypothesis classes are equal, so alpha does not restrict what the model can
express, and the sweep therefore cannot answer "how much context does the data need."
**What is NOT established: why the trained outcomes differ across alpha.** Weight-norm cost
is ruled out as the mechanism. An optimization-landscape explanation (e.g. at high alpha the
baseline receives gradient proportional to alpha and captures the base-identity signal
first, leaving the context head to fit residuals) is plausible but untested -- do not repeat
it as fact. Adam's approximate per-parameter scale invariance is a reason to expect gradient
rescaling *not* to translate into weight growth, which is consistent with the table above.

Consequences for interpretation:
- alpha = 0 and alpha = 1.0 remain genuinely distinct models; their numbers stand.
- alpha = 0.25 / 0.5 / 0.75 are **the free model under different implicit regularization**.
  Do not describe them as "partially structured."
- `eta2 = 1.000` at alpha = 1.0 is tautological (onsite *is* the per-base table there), not
  a finding.
- `ons_in_win = 0.244` at alpha = 1.0 is just the fraction of bases that are G: the four
  baselines are -1.27/-1.20/-0.295/-1.33 and only G falls inside [-1,1]. At alpha=1.0 the
  in-window metric measures base composition and nothing else.

### The [-1,1] "window" is the supervision range, not a physicality criterion

The energy grid is 201 points on [-1, 1] (confirmed from checkpoint `energy_grid`), so DOS
and transmission are supervised **only** there. Eigenvalues outside that range are
unconstrained by the loss -- which is both why the old free model could run to -33 eV, and
why out-of-window states are not per se a defect. Real DNA has states outside any window we
choose. `frac_in_window` should be read as a coarse sanity check against runaway values
(-33 eV is pathological), never as the success criterion. willll's call, 2026-07-24, and it
reinforces the standing decision to keep supervision on transport observables only.

### What is actually resolved by the fit

At alpha = 1.0, referenced to G: `C -1.039 < A -0.978 < T -0.901 < G 0`.

**PROVISIONAL -- n = 1.** The pattern below comes from a single training run (`--split_seed
42`) with no error bars, and `collect_onsite_sweep.py` has no cross-seed aggregation. Seeds
43 and 44 were already running when this was first written; block on them before treating
any of it as established. The neighbouring alpha=0.9 cell shows how much these numbers can
move (baseline spread 1.88 vs 1.04; T shifts -0.63 -> -1.20; G changes sign), so a 0.061 eV
gap quoted to three significant figures is not yet meaningful.

Provisional reading: **G looks resolved and A/T/C do not.** A, T and C sit within 0.14 eV of
each other (min pairwise gap 0.061 eV, C-A) while G stands ~0.9 eV above all three. A gap
should only be called "resolved" once it clearly exceeds the cross-seed spread; that test is
pending.

Comparisons:
- Roche et al. 2003 (G 0, A -0.49, T -1.39, C -1.12; `g3nat/utils/physics.py:24`,
  doi:10.1103/PhysRevLett.91.228101): matches on G-highest; the A/T/C ordering is inside our
  resolution, so it is not evidence either way.
- **Vertical-IP ordering: attribution UNRESOLVED, do not cite as "Caruso".** The values in
  circulation for this project (bases G 7.91 / A 8.30 / C 8.74 / T 9.05 eV; pairs GC 7.28 /
  AT 7.86 eV) could not be traced to any paper by that author (verification pass 2026-07-24).
  Nearest genuine sources found, neither matching digit-for-digit: Faber, Attaccalite,
  Olevano, Runge, Blase, "First-principles GW calculations for DNA and RNA nucleobases,"
  Phys. Rev. B 83, 115123 (2011), doi:10.1103/PhysRevB.83.115123 -- GW vertical IPs G 7.81 /
  A 8.22 / C 8.73 / T 9.05, no base-pair data; and Khan, Comput. Theor. Chem. 1013, 136-139
  (2013) -- GC 7.29 / AT 7.88 eV. Source these six numbers properly before publication.
  Whatever the source, the qualitative content we rely on (G has the lowest IP, hence the
  highest hole on-site energy) is not in dispute and our fit reproduces it.
- Absolute values are NOT comparable to literature TB parameters: different energy reference,
  and ours are effective one-orbital-per-base fits to transport observables rather than
  computed dimer integrals. Compare orderings and ratios only.

Subject to replication, this would be a *parameter-specific* identifiability result: the data
pins G and does not pin A/T/C. That distinction matters more than a global "is H
identifiable" verdict.

### Gauge audit: already fixed, and it is not the mechanism

Checked against the construction actually used: `forward` (`hamiltonian.py:862`) calls
`construct_hamiltonian_from_graph` (`hamiltonian.py:328-461`) at line 940.
(`_construct_hamiltonian_reference`, lines 189-326, is kept for test parity and is NOT the
live path -- earlier revisions of this section cited it by mistake.)

- H is built only on graph edges; zero-initialized and filled per edge pair. Already
  edge-banded, never dense.
- Already real (`float32`, ~394) and symmetric (onsite blocks symmetrized ~392; off-diagonal
  blocks written as a symmetric pair ~454).
- Energy reference is pinned by the data: DOS/T are supervised on a fixed 201-point grid
  (`energy_grid` in the checkpoints is exactly `linspace(-1,1,201)`), so a global shift of
  `diag(H)` shifts every eigenvalue and is observable by the loss.
- Residual freedom at n_orb=1 is per-site orbital sign flips. These leave `diag(H)`
  **exactly unchanged** (`H_ii -> s_i^2 H_ii = H_ii`), independent of graph topology.

**Correction on how far the sign gauge extends.** The graph is NOT a tree: backbone edges
along each strand plus hydrogen-bond rungs between paired positions
(`g3nat/graph/construction.py`, ~209-265) create a 4-cycle for every pair of adjacent paired
positions. The sign-flip gauge group has `N-1` effective degrees of freedom (a global flip
is trivial) while there are `E > N-1` edges, so the products of hopping signs around each of
the `E-N+1` independent cycles are gauge-**invariant**, i.e. physical. So "hopping signs are
gauge" is too strong: only `diag(H)` is provably gauge-invariant, individual hopping signs
are not freely resettable once a node has degree > 1 (every non-terminal base), and loop
holonomies carry real information. Correct statement: compare hopping **magnitudes** and
loop holonomies across fits; do not compare individual hopping signs.

Gauge freedom still cannot explain the onsite behaviour, since none of it touches `diag(H)`.
What remains is *soft* under-determination -- many different banded H reproduce DOS/T on a
201-point grid to within the loss -- which is a flat valley, not a group orbit, and cannot
be quotiented away. It can only be removed by adding observables that constrain
eigen**vectors**.

### Dataset facts established while investigating

- The 4 pickle variants per sequence are `{contact_type: same, cross} x {coupling: 0.1, 0.6}`
  (verified over 100 sequences: exactly 4 cells, 100 each). So we **already** train across
  two contact couplings and two contact geometries per sequence, and that did not resolve
  the under-determination -- expected, since all of it constrains eigenvalues plus one
  contact-to-contact matrix element of G.
- Nothing ties the DNA block of H to be identical across a sequence's 4 contact variants,
  though physically it must be: the on-site energies of the bases are a property of the DNA,
  not of the electrodes attached to it. **Measured** (8 sequences having all 4 variants,
  mean per-site spread of `diag(H)` across the variants):

  | model                 | drift across contact variants | within-sequence onsite span |
  |-----------------------|-------------------------------|-----------------------------|
  | alpha=0 (free)        | 0.933 eV                      | 7.10 eV                     |
  | alpha=0.9             | 0.117 eV                      | 1.86 eV                     |
  | alpha=1.0 (per-base)  | 0.000 eV                      | 0.92 eV                     |

  alpha=1.0 is exactly 0 by construction (onsite is the per-base table, which cannot see the
  contacts) -- a sanity check on the probe, not a result. The informative cell is the free
  model: **0.933 eV of on-site drift purely from changing the electrodes**, which is
  comparable to the entire base-identity spread (1.039 eV). Physically it should be ~0.
  Note the ratio-to-span framing originally built into the probe is misleading here, because
  the free model's 7.10 eV span is itself the pathology; against the physical scale the
  drift is large, not small. Tying the DNA block across a sequence's contact variants is
  therefore a real candidate constraint, and it costs no new data.
- Sequence lengths are 4-8 bases (540/446/328/360/384 files at lengths 4/5/6/7/8). With 8-16
  DNA nodes on a ladder, **4 GAT layers reach the entire molecule** -- that is global mixing,
  not nearest-neighbour smearing. The physics argues for a 1-2 hop receptive field.
  `num_layers` is therefore a live experimental knob, not a fixed choice.
- Residue-resolved DFT local DOS is available at essentially full coverage:
  `asyed4/DNADataSet/<seq>/run{1..4}/DOS_<seq>_gammaL_*_gammaR_*.mat`, 2081 files over 523
  sequences (1043 at gamma 0.1, 1038 at gamma 0.6), plus matching `Tran_*.mat`.
  `scripts/dos_map.py:66-87` already maps DOSAtom rows to PDB atoms and groups by residue.

### DOS-map before/after (sequence AAAC)

Learned onsite (`diag H`) overlaid on the DFT residue-resolved local DOS:

| model                        | onsite range (eV) |
|------------------------------|-------------------|
| free, leaking split (old)    | to -33            |
| free, grouped split (alpha=0)| -10.5 to -0.7     |
| structured alpha=0.9         | -1.69 to 0.00     |
| structured alpha=1.0         | -1.33 to -0.30    |

The clean split alone does not fix the runaway (-10.5 eV); constraining the parameterization
is what brings onsite into the DFT band.

### Next

1. LDOS discriminator (no training change): compare the model's per-site spectral weight
   against the DFT residue-resolved LDOS. This is the first constraint on eigen**vectors**
   rather than eigenvalues, and it may pin the onsite energies without any parameterization
   constraint at all -- in which case the structured head is unnecessary and should be
   dropped. Test that before adding machinery.
2. `num_layers` sweep {1,2,3,4}: settle receptive field empirically.
3. Only if 1 and 2 leave onsite unpinned: replace alpha with a knob that actually constrains
   the class, `onsite_i = (1-kappa)*b[base_i] + kappa*s*tanh(context_i)`, with `s` fixed (NOT
   learned -- a free `s` grows to cancel `kappa` and reopens exactly the evasion diagnosed
   above). `kappa` stays dimensionless ("fraction of the onsite budget delegated to
   context"); kappa=0 reproduces alpha=1.0 exactly.

   **Do not size `s` from the 1.039 eV baseline spread** (an earlier revision did). That
   number is the spread *between* the four per-base constants, whereas `s` must bound
   *within-base, context-driven* variation -- two different quantities with no reason to be
   equal. Measured: the free model uses ~9.8 eV of onsite span on AAAC (range -10.50 to
   -0.70) versus 1.04 eV at alpha=1.0. Capping the whole context term at `kappa*1.039` would
   leave even kappa=1 about 9x too tight to reach the free model's regime, so every cell
   would sit at a high val_loss and the sweep would measure "how bad is a tight cap" rather
   than "how much context does the data need" -- a milder rerun of the dead
   `constrain-onsite-window` failure. Size `s` from the free model's own within-base onsite
   spread (`scripts/probe_onsite_dilution.py` measures exactly this), or sweep `s` as a
   second axis, and state before running what kappa=1 is expected to recover.

   Also report the fraction of `tanh` outputs near +/-1 across the sweep. A hard range with a
   saturating nonlinearity can collapse to a near-binary signal -- structurally the same
   failure as the range penalty that collapsed onsite to a single value -- and val_loss alone
   will not reveal it.
4. No coupling lookup table in the model (willll, 2026-07-24): the model must stay
   compatible with arbitrary geometry (fraying, flipped/twisted bases must produce different
   onsite and coupling). Generate a table *from* the trained model post hoc instead.
