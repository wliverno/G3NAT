# Structured onsite head: physicality from parameterization (not penalty)

> **SUPERSEDED 2026-07-24 -- historical record, do not build on it.** This spec's goal was to
> "extract the per-base tight-binding parameters for comparison to literature" and to make
> onsite "physical by construction" judged by membership of the [-1,1] window. Both premises
> are retired: the energy grid is centred per sequence on the HOMO, so absolute per-base
> values are not recoverable and G is pinned near 0 by construction for 495/515 sequences;
> and the window is the supervision range, not a physicality criterion. The alpha sweep this
> spec designed also cannot discriminate what it claims, since every alpha < 1 shares one
> hypothesis class. The implementation and its measurements are sound and are recorded in
> `docs/model-results.md`; only the framing is wrong. See `docs/dataset.md`.

Status: design approved (brainstorm); revised after adversarial review. Pre-implementation.
Branch: `structured-onsite-head` (off `main`). Date: 2026-07-23.

## Goal

Make the learned per-base onsite energies physical by construction, MEASURE how much context
the DFT data actually needs, and EXTRACT the per-base tight-binding parameters for comparison
to literature. Replace the free onsite head with a convex mix of a per-base baseline and the
existing context head, controlled by a mixing factor `alpha`. Physicality comes from the
PARAMETERIZATION; supervision stays on DOS/T only.

## Background (why this, why now)

- The lowest-loss DFT model (GAT, val 0.547) is UNPHYSICAL: onsite/eigenvalues run to -32..-33
  eV, only ~59% of eigenvalues in the [-1,1] window. DOS+transmission UNDER-DETERMINE H, so the
  free head dumps states outside the window at no fit cost.
- The penalty route is DEAD (`constrain-onsite-window`): an out-of-window penalty on diag(H)/
  eigvalsh(H) enforces a RANGE not STRUCTURE, collapsing every onsite to one degenerate value.
- Correction (2026-07-23): "message passing dilutes base identity" is NOT the mechanism. The
  same GAT arch recovers tight per-base onsite on synthetic TB data (where the data HAS per-base
  onsite), so the architecture CAN anchor; the DFT spread is a DATA property. This is a TEST,
  not a bug-fix.
- Ruled out (willll): do NOT match/regress H to the DFT Fock matrix. Supervision stays on DOS/T.

## The onsite mixing head

Current onsite (vectorized `construct_hamiltonian_from_graph`, hamiltonian.py:343):
`onsite_i = onsite_proj(dna_features_i)` -- free, context-only, near-zero init.

New:
```
onsite_i = alpha[base_i] * baseline[base_i] + (1 - alpha[base_i]) * onsite_proj(dna_features_i)
```
- `baseline`: learned per-base onsite. n_orb=1 -> 4 scalars. Indexed by a DIFFERENTIABLE
  soft-matmul, not argmax: `base_onsite = original_node_features[dna_mask] @ baseline_table`
  (one-hot @ table == the right per-base value, but keeps a gradient path for soft/generator
  features). Near-zero init (keeps early `(E*I - H)` well-conditioned).
- `onsite_proj`: the EXISTING context head, unchanged.
- Symmetrize the onsite block as today (trivial for 1x1; general for n_orb>1).
- Endpoints: alpha=0 is EXACTLY the current model; alpha=1 is pure per-base (4 DISTINCT values
  -- distinctness is MEASURED, not assumed; see Metrics).

## alpha configurations (2 orthogonal toggles)

- granularity: `global` (1 shared alpha) or `per-base` (4 alphas, one per base identity).
- fit mode: `fixed` (constant, for the sweep) or `learned`.
  - FIXED: store alpha DIRECTLY as a constant buffer set to `alpha_value` (NO logit/sigmoid
    round-trip -- that would miss exact 0.0/1.0 under eps-clamping and break the sweep endpoints).
  - LEARNED: BILEVEL, DARTS-style -- update alpha on a held-out validation minibatch, not on
    train loss. (A train-loss-optimized alpha is near-tautological: the context MLP always
    out-fits 4 scalars on train, so alpha collapses regardless of the science. Val-based update
    makes its resting point actual evidence about generalization.)
- `per_base + fixed` with a single scalar is degenerate (== global+fixed): reject it, or accept
  4 comma-separated values.

## Intervention points in code (correctness requirements)

- Patch the VECTORIZED `construct_hamiltonian_from_graph` (hamiltonian.py:282; the path `forward`
  uses at :894). base ids/soft-matmul from `original_node_features[dna_mask]` -- SAME masked
  order as `dna_features = node_features[dna_mask]`.
- Apply the same mixing in the non-vectorized reference construct (~:148) for consistency/tests.
- RNG-SAFE new state: create `onsite_baseline`/`onsite_alpha` ONLY inside `if structured_onsite:`
  AND after all existing layers (mirror `use_geometry`, hamiltonian.py:104-125). Adding params
  earlier shifts the init RNG stream and silently changes the "unchanged" model's weights for the
  same seed. Both are real `nn.Parameter`/buffer (move with `.to(device)`), never bare tensors.
- Register the four new args (`structured_onsite`, `alpha_granularity`, `alpha_mode`,
  `alpha_value`) in EVERY model-reconstruction site (strict `load_state_dict` else crashes):
  `g3nat/evaluation/inference.py:58-73`, `scripts/analyze_learned_hamiltonian.py`,
  `scripts/probe_onsite_dilution.py`, `scripts/ablate.py`. Needed for post-hoc extraction.
- New flags on model + `scripts/train.py`: `--structured_onsite` (default OFF), `--alpha_granularity
  {global,per_base}`, `--alpha_mode {fixed,learned}`, `--alpha_value <float|4 floats>`.

## Backward compatibility

- Default (`--structured_onsite` off): onsite head untouched, model byte-identical. TEST by
  SEEDING ONCE, building off vs on, and asserting `torch.equal` on every PRE-EXISTING parameter
  (a state-dict-load equality test would pass even with a broken RNG stream -- do not rely on it).
- `--structured_onsite --alpha_mode fixed --alpha_value 0`: numerically the current model.

## Data split (CORRECTED -- the old split leaked)

- The dataset is 515 unique sequences x ~4 contact-variants = ~2057 samples.
  `train_test_split(range(len(dataset)), random_state=42)` splits by FLAT INDEX, so the same
  sequence appears in train AND val. The free context head can memorize per-sequence quirks that
  reappear in val; the 4-scalar baseline cannot -- biasing the discriminator toward "context
  wins" independent of the physics. (This taints prior val numbers project-wide, incl. the 0.547
  baseline and the geometry ablation.)
- FIX (prerequisite): group-split by sequence string (`GroupShuffleSplit`/`GroupKFold`). Outer
  grouped held-out TEST set; inner grouped k-fold CV on the rest for alpha SELECTION. Never report
  the selection-set loss as the headline -- report the held-out test loss of the selected alpha.
- Re-run the reference (alpha=0 / current) model under the clean grouped split so all numbers are
  comparable. Flag the leak as a separate project-wide cleanup too.

## Experiment (the discriminator) -- all judged under the grouped split

Shared setup: GAT, hidden=256, DFT/pickle data, best-val-checkpoint (or last-N rolling mean), NOT
final-epoch loss. Report mean +/- std across CV folds/seeds. The sweep reuses the same grouped
splits; queue as sbatch jobs.

1. Fixed global sweep: alpha in {0, .25, .5, .75, .9, 1.0}, each across the CV folds. Headline
   `val_loss(alpha)` +/- std and `physicality(alpha)`. Also report TRAIN-loss convergence per
   alpha (to rule out "high-alpha just under-converged" vs a true capacity ceiling). Interpret the
   MODERATE-alpha region, not just alpha=1 (alpha=1 = only 4 numbers; its degrading merely proves
   "4 numbers is not enough," which is not in dispute).
2. Calibration / negative control: the IDENTICAL sweep on the synthetic-TB data (known: no context
   needed). Its curve is the reference for what "does not need context" looks like (should be
   ~flat). Interpret the DFT curve's shape relative to it -- not by eyeball.
3. Standalone 4-scalar control: a per-base onsite head OUTSIDE the mixing formalism, to separate
   the mixing-form confound from the capacity confound.
4. Learned global alpha (bilevel/val-based): read where alpha settles + its trajectory.
5. Per-base nested test: learned per-base alpha (4 values). Does it beat global on HELD-OUT loss
   (do 4 params earn their keep)? PRE-REGISTER the full predicted drift ranking to a dated commit
   BEFORE running (physical prior: G driftiest -- lowest IP, carries the hole, neighbor-modulated;
   T most stable). Judge the observed ranking with a permutation p-value across seeds, not one run.

## Metrics (per run)

- Held-out DOS/T loss (mean +/- std), comparable to a re-run clean-split baseline.
- Physicality, and a "win" requires onsite AND eig to move together: % onsite in [-1,1]; %
  eigenvalues(H) in-window; onsite range; PLUS a coupling-magnitude/bandwidth diagnostic (guards
  the couplings escape valve -- if onsite-in-window improves while eig-in-window/bandwidth do not,
  report "shifted, not fixed").
- Baseline DISTINCTNESS (guards the collapse failure): min pairwise |baseline_b - baseline_b'| and
  the eta^2/between-vs-within machinery from `scripts/probe_onsite_dilution.py`. Near-degenerate
  high-alpha baselines are a FAILURE, not a pass.
- alpha: fitted value(s) + trajectory (learned runs).

## Extracting the TB parameters (deliverable)

- After training, `model.onsite_baseline` is the 4 per-base onsite scalars -- read directly.
- Extraction script: dump the 4 values (+ distinctness/eta^2), GAUGE-CORRECT (subtract the model's
  own G, matching Roche's G=0 reference), and compare ORDERING + gauge-corrected magnitudes to
  Roche (A -0.49, T -1.39, G 0.00, C -1.12) AND >=1 other set (Voityuk/Rosch, Senthilkumar).
  Robust agreement across sets is the claim; proximity to one table is not.

## Non-goals

- NOT structuring couplings (measure the escape valve; do not close it here). Future extension.
- NOT matching/regressing to the DFT Fock matrix (ruled out).
- NOT a new penalty term (the bound is structural).
- NOT changing NEGF, contacts/gamma, edge features, or batching.

## Testing

- Backward-compat: seed once, build off vs on, `torch.equal` on every pre-existing param.
- alpha=0 fixed reproduces flag-off H (all-close); fixed 0.0/1.0 are EXACT (no logit round-trip).
- alpha=1 fixed: onsite == baseline[base] for every DNA node; same-base sites share one value.
- Baseline indexing on a BATCHED heterogeneous graph set (>=2 equal-DNA-count graphs whose base
  orders differ, e.g. local idx 0 = 'G' in one, 'A' in the other) -- catches index-by-position bugs.
- Checkpoint round-trip: save a structured-onsite model, reconstruct via each reconstruction site,
  `load_state_dict` succeeds.
- Gradient flow to baseline and alpha (learned mode) non-zero; soft-matmul preserves gradient to
  soft features (generator path) -- or document that `--structured_onsite` breaks it.
- Vectorized vs reference construct agree with mixing (all-close). Existing ~85-test suite passes.

## Risks / confounds (explicitly tracked)

- Selection-on-val: use the grouped held-out TEST for the final number, not the CV-selection set.
- Convergence vs capacity vs mixing-form (see Experiment 1/3): don't conflate.
- Couplings escape valve (see Metrics co-gate).
- Small-data noise: CV folds + seeds; report effect sizes, not single-run anecdotes.

## Open questions

- Compute matrix sizing: #alpha x #folds/#seeds x {DFT, TB-calibration, controls} could be ~50-80
  runs. Measure per-run wallclock in the smoke run, then size (and possibly trim alpha points).
- CV granularity on ~515 grouped sequences: k for the inner CV vs a single grouped held-out; pick
  during smoke based on fold-to-fold variance.

## Run (staging)

0. Smoke: build head, one short grouped-split run at alpha=1; verify plumbing + baselines move
   toward sane/DISTINCT values + TIME a run. De-risk before spending compute.
1. LOAD-BEARING (do first -- answers the core question):
   a. Re-run the reference model (alpha=0) under the clean grouped split -> the honest baseline.
   b. Fixed global DFT sweep {0,.25,.5,.75,.9,1.0} under grouped CV, best-checkpoint metric ->
      headline val_loss(alpha) +/- std, physicality co-gate (onsite AND eig), baseline distinctness.
   c. Extract the per-base baseline params at the informative alpha (the interpretable payoff).
2. FULL COMPARISON (after 1): TB-calibration sweep (interpret the DFT curve against it) -> standalone
   4-scalar control -> learned bilevel alpha -> per-base nested test (pre-registered ranking) ->
   literature comparison across >=2 parameter sets.
