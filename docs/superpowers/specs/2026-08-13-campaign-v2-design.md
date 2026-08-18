# Campaign v2: the clean training factorial -- design spec

Date: 2026-08-13, revised 2026-08-15 after a two-reviewer pass (adversarial +
constructive) and PI review. Status: decisions final as listed; next step is the
implementation plan. Review artifacts: private notes tree,
reviews/2026-08-13-campaign-v2/ (four architecture reviews, two spec reviews, the
broadening literature pass). Companion history: docs/model-results.md sec. 14-17,
docs/doe-methods.md.

## 1. Why a from-scratch campaign

Three findings, 2026-08-11..13, each independently sufficient:

1. **Training is not reproducible from its seeds.** MECHANISM (code-verified): the batch
   sampler builds batches from an unseeded `np.random.default_rng()` every epoch
   (`g3nat/training/utils.py:64,71`), and `set_init_seed` seeds only initialization.
   Illustrative replica pair (n=2): same-command runs diverge from epoch 0, best-val
   spread 0.053, best-epoch shift 89. No RNG state is checkpointed.
2. **73 of 84 published runs hold non-optimal weights** (11 sit exactly at their optimum;
   58 have a gap above 0.005; median gap 0.0124 over all 84). Checkpoint-cadence bug,
   fixed in `14f65ca`; see model-results.md sec. 16.
3. **Config illegibility.** The campaign ran n_orb=2 via a runner override while
   `train.py --help` said 1. The PI must be able to read a run's exact parameters from
   its artifacts.

A four-review adversarial audit (2026-08-13; artifacts above) verified the physics core:
solver matches the float64 reference at n_orb 1 and 2 (max rel err <= 2.5e-5), vectorized
assembly is bit-exact against reference implementations, loss composition is correct in
every branch, and the v2 dataset passes full-set integrity scans (all 2077 records).
Every serious finding lands in the scaffolding and is fixed in Phase 1.

## 2. The design

Four factors, complete crossing, 24 cells x 3 init seeds (42/43/44) = **72 runs**:

| Factor | Levels | Notes |
|---|---|---|
| SUPERVISION | DOS-only (a=1,b=0,c=1) / DOS+LDOS (a=1,b=0.5,c=1) / T-only (a=1,c=0) | One categorical factor; b x c crossing is degenerate at c=0. |
| N_ORB | 1 / 2 | Gamma mapping convention set by the sec. 5.1 test. |
| NUM_LAYERS | 2 / 4 | Receptive-field / locality factor. |
| GEOMETRY | off / on | X3DNA edge features; v2 cache, hard-fail on miss (B10). |

**Budget:** fixed **15000-epoch cap, NO early stopping**. An earlier patience-2000 stop on
`val_dos_t_unweighted` was found by review to be a double confound: the stopping metric
is the DOS-only cells' exact training objective, half of DOS+LDOS's, and fully held-out
for T-only (differential truncation on SUPERVISION), and post-optimum drift is
capacity-dependent -- it is on record as having inverted the num_layers ordering at
final-epoch (train.py:303-309) -- so fixed patience also truncates differentially on
NUM_LAYERS. The recorded best-epoch distribution (sec. 16b: min 123 / median 1730 /
max 14600; 18 of 58 recovery-scope runs needed >5000) means patience-2000 would bind
often. The fixed cap removes the confound entirely; every response is read at its own
curve's optimum from metric_history (sec. 4), so late optima are never lost. Cost is the
already-budgeted ceiling below. Early stopping is NOT implemented (review confirmed: no
such code exists) and under this decision never will be for this campaign.

**Budget scope:** the ~470 GPU-h ceiling (72 x 6.5 h, MEASURED at one cell corner,
sec. 16d) covers the factorial only. Pilots + alpha epilogue add AT MOST ~30 GPU-h: that
figure was costed when this section still carried a broadening-eta pilot, which has since
been dropped (the surviving GPU cost is the optimizer pilot and the reproducibility gate),
so read it as an upper bound, not an estimate.
Pilots record per-corner throughput so the ceiling
becomes MEASURED across (N_ORB, NUM_LAYERS).

**Held fixed** (evidence tier per value): complex solver [MEASURED: Frobenius path is
silently wrong at resonances and ignores eta -- review 2; hard-failed in code for
campaign runs, not just prose]; log outputs on [data are log10]; hermiticity on
[hard-fail if off at n_orb>1 -- MEASURED, review 2]; GAT conv [MEASURED-tie with
transformer on best-val, sec. "conv_type"]; hidden 256, heads 4, lr 1e-3, batch 32
[inherited defaults from prior sweeps -- FOLKLORE, no factorial evidence, held for
continuity]; huber delta 1.0 [inherited -- FOLKLORE; exposed in recorded config];
ldos_target=residue [DOMAIN: the TB site is a base+backbone residue; base_only is
future work]; split_seed=42, GroupShuffleSplit by sequence string [MEASURED: leak-free,
reviews 1 and 4]; dataset pickle_files_v2 + geometry_v2.pkl [MEASURED coverage,
review 4]; geometry z-score stats computed on the TRAINING SPLIT only (fixes a
train+val leak found in review 3). Optimizer/weight_decay: pinned by pilot 5.2.

**Alpha is not a factor.** The structured-onsite head becomes a BOOLEAN (per-base
baseline on/off); the fractional alpha machinery is removed in a dedicated commit
(rationale: measured answer was alpha=0; the per-base-table payoff is largely fixed by
the HOMO-referenced energy convention). Off in all 72 runs; ONE post-factorial epilogue
run with it on at the winning cell. The geom/init RNG-order fix (B9) exists for this
epilogue's sake; it is a no-op for the 72 runs (nothing draws RNG after the geometry
encoder once alpha is off).

NO reserve slice. This is a considered trade, not a corner cut: the old
reserve's job is now done structurally by the protocol -- plus, at zero cost, the claim
list is PRE-REGISTERED before any analysis runs (sec. 6), and seed 44 may be designated
a confirmatory replicate for headline claims (fit on 42/43, confirm on 44) at analysis
time if a headline claim emerges.

## 3. Phase 1: pre-campaign work (all TDD)

**Bug fixes (blocking):**

| # | Fix | Anchor / test seed |
|---|---|---|
| B1 | Seed the batch sampler: generator passed EXPLICITLY at construction (samplers are built before set_init_seed runs -- train.py:230-238), plus `set_epoch(absolute_epoch)` called from the fit loop so the seed survives requeue. Acceptance test = the Phase-2 reproducibility gate (sec. 5.3). | utils.py:64,71 |
| B2 | Best selection on the unweighted metric; weighted-optimum checkpoint optionally kept, never compared across cells. | trainer.py:400, train.py:344 |
| B3 | Resume guard: assert stored args match; delete stale checkpoint_best ONLY when no checkpoint_latest exists (unconditional deletion would destroy the best weights on every preemption -- review finding); unique dir per cell x seed. Test: write latest+best, restart, assert best survives unchanged. | train.py:355-388 |
| B4 | Best state_dict held in memory, updated EVERY epoch against the selection metric, serialized at checkpoint cadence and at exit -- so saved weights are exactly the epoch the metric responses are read at (removes the cadence-10 offset). checkpoint_frequency recorded in config. | train.py:337-343,403 |
| B5 | isfinite guard on loss and grad norm before optimizer.step; skip-and-count; the count is a recorded metric. | trainer.py:342-366 |
| B6 | Smooth log floor: log10(x + eps), eps = 1e-38 (the float32 representability edge -- at L=16+ true T reaches 1e-32, so any larger floor binds on the physics being extrapolated to), recorded, single value for DOS and T, identical train/eval. Rationale: the old 1e-16 clamp binds INSIDE the target range (targets reach 6.7e-19) with zero gradient exactly in the deep tail; a hard clamp at any lower value would keep the zero-gradient region and inflate clamped Huber terms; the smooth floor keeps gradient everywhere. Per-epoch clamped/floored-fraction recorded in metric_history. | hamiltonian.py:578,606,670,682 |
| B7 | enforce_hermiticity=False hard-fails at n_orb>1; --solver_type frobenius hard-fails for campaign runs. | hamiltonian.py:419-420 |
| B8 | Inference paths rebuild geometry when the checkpoint used it (predict_sequence, dos_map). | inference.py:186 |
| B9 | Geom/init RNG-order decoupling (epilogue-relevant; no-op for the 72 runs -- see sec. 2). | hamiltonian.py:139,163 |
| B10 | --geom_cache defaults to geometry_v2.pkl; cache miss is a hard error. | train.py:110, datasets.py:256 |
| B11 | Config legibility: runner is the single source of truth; every run writes resolved-config JSON + git sha; argparse defaults match campaign reality. | train.py |
| B14 | n_orb-aware H readouts: onsite = eigenvalues of each n_orb x n_orb onsite block; coupling bandwidth = inter-site block norms; every diag(H) consumer asserts n_orb. (evaluation/physicality.py currently misreads n_orb=2 blocks -- its response would be a different object at each N_ORB level.) | evaluation/physicality.py:27,42 |
| B15 | Energy grid registered as a buffer (currently re-built and re-transferred host->device EVERY forward -- per batch, ~50 x 15000 x 72 times). | hamiltonian.py:540,647,664 |

**Features (blocking, but new capability -- not bugs):**

| # | Feature |
|---|---|
| F1 | Alpha booleanization (own commit with rationale). |

**Metric key set freeze (blocking, launch-irreversible):** anything not recorded
per-epoch in metric_history is unavailable at any epoch but one -- re-deriving means
re-running all 72 jobs. Frozen set at launch = current keys + the floored-point fraction
(B6) + the NaN-skip count (B5).

(A threshold-restricted transmission key was in this freeze and was removed 2026-08-18:
its threshold was the old numerical clamp value, not a physical quantity, and it
discarded the deep tail the length-extrapolation claim depends on. No replacement
threshold is introduced -- see docs/metrics.md sec. 8a.)

**SHOULD (never gates a phase transition):** nanargmin for best_val_epoch + a NaN scan
of the 84 old metric_histories (does the argmin bug touch any quoted best-epoch number);
sample-weighted epoch averaging or a measured skew statement; expose max_grad_norm /
warmup / delta in recorded config; DOS/T positivity guard mirroring LDOS's
[prophylactic -- the full-set scan shows zero current violations]; fix the two stale
analysis scripts (reversed stacking key; diag(H) asserts are B14); document v2 coverage
gaps (CGTAT 3/4, GCCTGG 2/4) and the GAAAC backfill in docs/dataset.md.

## 4. Responses

All read from metric_history AT EACH RESPONSE'S OWN CURVE OPTIMUM (no shared stopping
epoch exists under the fixed cap; the selection metric in B2/B4 governs only which
WEIGHTS are saved, stated in methods): val_dos_t_unweighted, val_transmission,
val_dos, val_ldos_residue, ldos localization gap,
best_epoch [DESCRIPTIVE], floored-point fraction [DESCRIPTIVE].
Post-hoc on saved best weights (B4 makes these the same epoch as the selection metric's
optimum): length extrapolation against the 16 held-out L=12 DFT records (sec. 15 set --
not the synthetic decay slopes), supervision-window membership [DESCRIPTIVE, renamed
from "H physicality" per the module's own retraction header -- never a success
criterion], substitution response, base-identity variance ratio (the record's "eta2";
the one response NUM_LAYERS is documented to move -- probe_onsite_dilution machinery,
with its on-record caveat).

## 5. Phase 2: pre-launch gates and pilots (concurrent where independent)

1. **Gamma convention (zero-cost inference test only; no retraining arm).**
   The fit-level comparison is valid regardless -- the
   convention is part of the n_orb=2 forward operator and the network compensates
   through training. What cannot wash out: a Hermitian H cannot cancel imaginary
   broadening directly; compensation happens by redistributing wavefunction weight off
   the contact sites, i.e. by distorting exactly the H the paper publishes (the same
   channel as the historical onsite pathology, milder). The stake is H interpretability
   near contacts and effect attribution, not fit validity. The frozen-H inference test
   measures the raw mechanism size for free; if small, the convention stands with one
   methods sentence; if large, the mapping is chosen with the numbers in hand.
   Review reframing: at n_orb=2 the contact is not just 2x the trace,
   it changes RANK (rank-1 -> rank-2 per contact site: the coupling is broadcast to both
   orbitals). The halved-coupling retraining pilot would only measure whether the
   network can compensate (targets stay fixed while the contact condition changes).
   Replacement, zero GPU cost: (a) hold trained H fixed, recompute DOS/T under raw vs
   1/n_orb-normalized Gamma -- isolates the mechanism exactly, no seed noise; (b) score
   existing n_orb=2 checkpoints separately on the coupling=0.1 eV vs 0.6 eV record
   subsets (a 6x matched-target coupling range already in the data). The open
   convention question: what should a physical rank-1 contact map to in an n_orb-orbital
   site
   basis -- full coupling on every orbital (raw), 1/n_orb-normalized trace, or coupling
   on one designated orbital (preserves rank)? The inference tests inform the choice.
2. **Optimizer pilot** (Adam wd=1e-5 vs AdamW, 3 seeds each). Pre-stated rule: pick the
   better unweighted best-val mean ONLY if |difference| exceeds the pooled 3-seed range
   (seed noise on best-val at fixed config is ~0.05); otherwise AdamW is chosen on
   EVIDENCED grounds (Loshchilov & Hutter, ICLR 2019) and recorded as such, not as a
   measured win.
3. **Reproducibility gate** (B1's acceptance test): two replicas of one command, 200
   epochs; report max |delta val_dos_t_unweighted| and first divergence epoch.
   Pre-stated tolerance: elementwise-identical val curves (CPU-reduction determinism
   permitting; if GPU nondeterminism leaves a residual, the measured residual must be
   << the 0.017 meaningful-difference bar and is documented as the irreducible floor).
   The campaign does not launch on a failed gate -- reproducibility is sec. 1's premise.

**Parallelization:** pilots 1-2 are mutually independent (1 costs no GPU), as is gate 3. The
**n_orb=1 half of the design (36 runs) does not depend on
the gamma convention and launches as soon as Phase 1 + gate 3 pass**, concurrent with
the pilots. The n_orb=2 half launches when 5.1's convention is decided.

## 6. Analysis protocol (extends docs/doe-methods.md)

- Model ENUMERATED from the factors: SUPERVISION (2 dof) x N_ORB x NUM_LAYERS x
  GEOMETRY, full crossing = 23 dof, n=72, residual df 48. The location model saturates
  the 24 cell means: residual is pure replicate error (no lack-of-fit term exists --
  state this). Occupancy is trivially complete; print the entered-levels line anyway.
- **Sum-to-zero (deviation) coding mandated** for all factors (2-level coded +/-0.5;
  SUPERVISION on 2 orthogonal deviation contrasts) so drop-one F-tests are Type III and
  coding-independent. Asserted in the analysis script.
- **Pre-registered SUPERVISION contrasts** (1 df each, more power than the omnibus):
  C1 = T-only vs {DOS-only, DOS+LDOS} (the sec. 17 claim, generalized off its single
  config); C2 = DOS-only vs DOS+LDOS (the b question, re-asked cleanly).
- **Circularity, stated correctly:** NO response is neutral across SUPERVISION -- the
  factor changes the objective; that is what it measures. val_transmission is the
  common-objective response (weight a=1 in every cell). val_dos_t_unweighted is the
  DOS-inclusive response (T-only cells' held-out reading -- the fairest available
  cross-check, not "fair"). Everything else is in-loss for at least one level; labeled.
- Dispersion channel: mains + two-ways only (14 dof on 24 cells, df 9 -- fixed in
  advance), BOTH raw and log10 scales with Shapiro-Wilk on raw spreads, per
  doe-methods.md. df 9 is nearly powerless: a dispersion null is not evidence of absence.
- BH at q=0.05 over the family actually tested, m printed (expected m ~ 460 across ~9
  responses; the old campaign at m=264 returned 2 survivors, 1 circular). **Minimum
  detectable effects computed per response from the historical residual SD at df 48
  before launch and printed in the analysis header** -- the referee question answered
  in advance.
- Best-cell table first-class (best cell, seed range, worst cell, gap/residual-SD).
- **The claim list is pre-registered** (written to the private notes tree) before any
  analysis script runs.
- Single-seed results are not results; single-config results are not results.

## 7. Run infrastructure

Runner generates all jobs from the design table; no hand-edited variants. Unique output
dir per cell x seed; resolved-config JSON + git sha at start; metric_history complete in
every checkpoint (write-tmp + os.replace is already crash-safe). Monitors check output
validity (metric_history advancing, NaN-skip counter, floored fraction), not exit codes.
ckpt partitions with --requeue; resume correctness is B1/B3/B4's tested domain.
--exclude carries known-bad nodes at launch.

## 8. Non-goals

Universal absolute per-base parameters (HOMO-referencing decision). Rescuing old-campaign weights
(impossible, sec. 1). base_only LDOS targets (follow-up).

## 9. Open at review time

- The n_orb=2 contact mapping convention: decided after the sec. 5.1 inference test
  (small effect = raw convention stands with a methods sentence).
- Which solver path produced the sec. 15 L=12 numbers (verification queued; affects
  their description, not this campaign).

## 10. Key citations (verified; full list + quotes in the review archive, merged into
docs/references.md during Phase 1)

Loshchilov & Hutter, ICLR 2019 (AdamW) -- EVIDENCED. Nelder 1977 / Peixoto 1990 -- effect
hierarchy (already in references.md).
