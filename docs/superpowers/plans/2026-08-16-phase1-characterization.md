# Phase 1 characterization: what did our fixes actually change?

Status: DESIGN rev 2, revised 2026-08-16 after adversarial review. Replaces the toy
two-run determinism check that was Task 16 step 3 as the real Phase 1 exit gate.

## Why

Phase 1 changed the training loop, the solver's log floor, checkpoint selection, resume
semantics, geometry handling and several defaults. 275 unit tests pass and NOT ONE
TRAINING RUN has executed against any of it. That is the posture the project was in
before a previous fix left training on the complex solver while inference used Frobenius,
which made log_floor a dead knob at eval and invalidated the section-12a length curves
(recorded at `g3nat/evaluation/inference.py:81-88`). This session reproduced that class
twice: the log_floor default silently shifted the LDOS floor 22 decades (R1), and floor
semantics were unrecorded so legacy checkpoints re-evaluated differently (R2). Both are
now fixed (7950f99); the point stands.

The question is not "does it run". It is: **every difference between old and new must be
explainable by a change we made deliberately. Anything unexplained is a bug we introduced
and have not found.**

## Preconditions (BLOCKING -- rev 2, review finding F3/I3)

1. **Clean, committed tree.** No uncommitted tracked changes. `runmeta` stamps
   `git_sha`/`git_dirty` into every artifact; running dirty produces a provenance hole
   and a non-reproducible result. Record the exact sha in the deliverable.
2. **Code freeze on `g3nat/` for the duration.** The plan's OPEN WORK list must be empty
   (R7, R8, R9 outstanding at rev 2). Any later commit under `g3nat/` RE-TRIGGERS Part 1.
3. **Node pinning is mandatory** for Parts 1 and 3 (`--constraint` or explicit gres type);
   record `nvidia-smi -L` and the node name. Not "ideally" -- TF32/reduction order differs
   across architectures at the magnitude being compared, and the eps=1e-38 subnormal
   question (R8) is per-device.
4. **`--geom_cache`, `--log_floor` and `--floor_mode` are passed EXPLICITLY on both sides,
   never defaulted** (I1, plus the R1-R6 agent's note that a model built without
   `floor_mode` now silently gets the legacy clamp).

## The design problem

The naive experiment -- train the same config on old and new code and diff the curves --
fails because old code's sampler is unseeded, so two old-code runs already differ by 0.053
best-val with an 89-epoch shift in the optimum.

BUT (review finding F1, the key insight): old code's sampler is unseeded **only when
`shuffle=True`**. At `shuffle=False` it draws no RNG at all. So a paired, deterministic,
epoch-by-epoch training comparison IS constructible. That becomes Part 1c and it carries
most of the weight, because a fixed-weights forward pass (Part 1a/1b) exercises only the
NEGF block and is blind to Tasks 1, 2, 3, 4, 12, 14, 15 -- the entire training loop, which
is the part that has never run.

Caveat for 1c: Task 1 also added `sorted(self.buckets.items())`, so bucket ORDER differs
between versions even at `shuffle=False`. Run 1c on a **single-length subset** (one
bucket), where bucket ordering is irrelevant and the pairing is exact.

## Part 0 (DONE): baseline artifact check

Cheapest deterministic check available, and already executed 2026-08-16: the golden
fixture `tests/baseline/outputs/model_hamiltonian.pkl` at 7950f99 versus its pre-floor-
change ancestor at c1dabc8^. Result: `dos` and `transmission` both **bitwise identical,
max_abs_diff 0.0**, verified by direct array comparison on a compute node (not by file
hash -- pickle metadata differs). This is what `floor_mode='clamp'` as the constructor
default predicts, and it is a narrow instance of Part 1a passing.

## Part 1a/1b (deterministic, forward pass only): fixed weights, fixed inputs

No training, no RNG, `torch.no_grad()`, eval mode, both sides.

- Old code: `git worktree add` at **`0157ea8`** = `origin/main`, the last pushed state and the
  pre-Phase-1 code tree. (This document originally named `1a992af`; that commit was removed
  from this history line by the 2026-08-17 rewrite and is NOT an ancestor of HEAD. The
  substitution was verified non-material: `git diff 1a992af 0157ea8 -- g3nat/` is EMPTY, i.e.
  byte-identical package trees, and 0157ea8's hamiltonian.py contains zero occurrences of
  `floor_mode` against 19 in HEAD's, confirming it is genuinely pre-Phase-1.) NEVER move HEAD
  on the main checkout. Data is gitignored, so the worktree starts empty -- symlink
  `pickle_files_v2/`, `geom_cache/`, and the checkpoint dirs in explicitly (I1).
- **USABLE-CHECKPOINT LIST (measured 2026-08-17, commit 90bf709 -- "loads" is NOT the
  same as "runs").** A golden forward-pass fixture found that one tracked checkpoint
  loads cleanly and then RAISES on the forward pass:
  - `hamiltonian_DFT_gat_baseaware.pth` -- USABLE (gat, 201-pt DFT grid)
  - `hamiltonian_2000x_4to10BP_5000epoch.pth` -- USABLE (transformer, 100-pt tb grid)
  - `standard_2000x_4to10BP_5000epoch.pth` -- USABLE (direct/blind baseline)
  - `hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth` -- **NOT USABLE**: base-blind-era
    coupling head is 1x hidden wide where the current base-aware model feeds 3x
    (`mat1 and mat2 shapes cannot be multiplied`). Not a loader bug; the endpoint-
    embedding weights were never trained. DO NOT use it in any arm of this experiment.
  Standing implication: a base-blind-era result cannot be regenerated without checking
  out period code. Nothing in docs/, scripts/, g3nat/ or the figures depends on one.
- **The flag assertions and the golden fixture cover DIFFERENT failures -- keep both.**
  Measured, not assumed: the `solver_type` flip does NOT move the golden numbers (short
  off-resonance sequences agree inside tolerance; the documented 3.2e-5 median gap has
  its heavy tail at near-resonance energies these fixtures do not sample), and
  `enforce_hermiticity` CANNOT move them (every tracked checkpoint is n_orb=1, where
  symmetrizing a 1x1 block is an exact no-op). A numeric fixture does not subsume
  per-flag assertions for the flag that has already caused a real campaign-scale error.
- Checkpoints (MANDATORY arms, review finding I4 -- `trained_models/` is all n_orb=1
  pre-campaign and would validate the wrong configuration):
  - one n_orb=1 checkpoint from `trained_models/`. **PICK FROM THE RUNNABLE THREE**
    (measured 2026-08-17, job 38618534): `hamiltonian_DFT_gat_baseaware.pth`,
    `hamiltonian_2000x_4to10BP_5000epoch.pth` and `standard_2000x_4to10BP_5000epoch.pth`
    load AND complete a forward pass under current code.
    `hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth` LOADS AND THEN RAISES on the
    first forward pass: it is from the base-BLIND coupling era, so its
    `coupling_proj.0` is 1*hidden wide while the current base-aware model feeds
    3*hidden (`mat1 and mat2 shapes cannot be multiplied (Nx768 and 256x128)`). That
    is an architecture boundary, not a serialization one -- the endpoint-embedding
    weights were never trained, so no loader fix can recover it, and any result from a
    base-blind-era model can only be regenerated by checking out the corresponding
    code. Pinned by
    `tests/baseline/test_baseline_legacy_checkpoints.py::test_baseblind_checkpoint_loads_but_cannot_run`.
  - one n_orb=2 checkpoint, e.g. `ckpt_ldos_B_b0.5_s42_n2/checkpoint_best.pth`
  - one n_orb=2 + geometry checkpoint, e.g. `ckpt_ldos_B_b0.5_s42_n2_geom/checkpoint_best.pth`
  Verify each loads on BOTH sides before comparing (energy_grid_t is persistent=False, so
  strict loading works in both directions -- confirmed by review).
- Inputs: 8 fixed sequences from `pickle_files_v2` spanning lengths and both contact
  variants. PRE-VERIFY every one is present in both geometry caches (I2) -- new code
  hard-fails on a cache miss where old code silently inserted None, which is a crash on
  one side rather than a comparison.
- Compare elementwise, reporting max and median absolute difference per channel: log10
  DOS, log10 transmission, per-site LDOS, H, GammaL/GammaR, and every loss term.

**1a "semantics held constant":** new code forced to old settings -- `log_floor=1e-16`,
`floor_mode='clamp'`, v1 geometry cache, complex solver, AND geometry norm stats computed
over the full cache rather than the train split (I2: Task 9's train-split filter changes
normalized edge features, hence H, at fixed weights -- it is deliberate, so it belongs in
the forced-settings list, not in the bug column).
EXPECTATION: bitwise equality for H and Gamma on CPU float64; <= 1e-12 relative on the
log10 outputs (M2). Any structural difference is an unintended change, i.e. a bug.

**1b "campaign defaults":** `log_floor=1e-38`, `floor_mode='smooth'`, v2 cache, train-split
norm stats. EXPECTATION: differences ONLY below the old 1e-16 clamp and in LDOS scale
where the floor binds. **Every nonzero difference must be attributed to a named commit.
An unattributed difference BLOCKS the campaign.**

## Part 1c (PRIMARY, deterministic): paired training on a fixed batch order

This is the arm that actually covers the training loop.

- Single-length subset (one bucket) of the dataset; `shuffle=False` on the train loader
  on both sides; `--init_seed` fixed and identical; 5-10 epochs; CPU.
- Old-code worktree vs new code, same config, `floor_mode='clamp'` and `log_floor=1e-16`
  forced so 1c isolates training-loop changes rather than floor semantics.
- Compare per-epoch train loss and every metric_history channel elementwise.
- EXPECTATION: identical to float noise. Differences must be attributable to a deliberate
  change (e.g. the epoch-mean denominator now excludes skipped batches -- with no NaN
  batches present, that is a no-op and the curves should match exactly).
- Then repeat with campaign settings to see the intended deltas.

## Part 2 (SMOKE TEST ONLY -- no comparative verdict): short new-code runs

Review finding F2: with 3 seeds a side against old code's ~0.037 sd, the minimum
detectable effect is **0.081, about 5x the project's own 0.017 meaningful bar**, and the
min/max envelope test flags a false displacement 80% of the time under a true null while
being biased toward passing because new code is tighter by construction. The old-code arm
is therefore DROPPED. **No comparative verdict may be drawn from Part 2 in either
direction, and a null here must never be reported as agreement.**

What it is for: 1-2 new-code runs, ~400 epochs (noting this is pre-convergence, so
converged scatter figures do not apply -- M3), asserting operationally:
`resolved_config.json` written; `checkpoint_best.pth` produced; metric_history carries
every frozen key including the new counters; the C2 no-best warning does NOT appear;
`nan_skipped_total == 0`; floored and negative fractions ~0 on all three channels.

## Part 3: determinism confirmation (split criteria, review finding I5)

- **CPU, BLOCKING:** two new-code runs, identical command incl. `--init_seed`, 20-30
  epochs. Criterion: bitwise identical val curves. Cheap and exact.
- **GPU, CHARACTERIZATION ONLY, no blocking threshold:** same comparison on the pinned
  GPU node. There is no `torch.use_deterministic_algorithms`, no cudnn determinism flag,
  and PyG message passing is scatter/atomic-based, so a residual is expected and can
  compound chaotically over epochs. Report the measured residual as the documented
  irreducible floor; do NOT block on it. Separately decide and record whether the campaign
  sets `use_deterministic_algorithms(warn_only=True)`.

## Carried into the campaign runner (review finding I6, R8; not part of this experiment)

**Per-run subnormal assertion, precise form.** At startup, on the run's OWN allocated
device (`dev`, not `cuda:0` by assumption), assert:

```
# eps is the RUN'S OWN configured floor, not a literal: probing 1e-38 while the run
# trains at some other log_floor tests a number the run never uses.
eps = float(args.log_floor)
target = math.log10(eps)          # -38.0 at eps=1e-38, and correct at any other eps
v = torch.log10(torch.zeros(1, device=dev) + eps)
# NOT `assert`: `python -O` strips assert statements outright, and this check is
# required to ABORT the run. A stripped guard fails open, which is the one outcome
# this check exists to prevent.
if not torch.isfinite(v).item():
    raise RuntimeError(f"non-finite subnormal probe on {dev} at eps={eps:g}: {v.item()}")
if abs(v.item() - target) >= 1e-3:
    raise RuntimeError(
        f"subnormal probe off-target on {dev} at eps={eps:g}: "
        f"{v.item()} vs expected {target}")
```

- Tolerance is `1e-3`, not exact equality: float32 rounding gives `-38.000003814697266`
  on measured hardware (P100, see R8 below), not exactly `-38.0`.
- The expected value is DERIVED from the run's `log_floor` (`log10(eps)`), so the check
  follows the config instead of silently going stale if the campaign's floor changes.
  It is only meaningful where the floor is actually subnormal-adjacent; at the legacy
  `log_floor=1e-16` with `floor_mode='clamp'` the probe is trivially satisfied, which is
  correct -- there is no subnormal to flush.
- **On failure, the run must ABORT before training starts**, not warn and continue. A
  flushed subnormal makes `log10(0)` return `-inf`, and the existing non-finite-loss guard
  then silently skips the optimizer step for every batch that hits the floor -- on a
  deep-tail-heavy arm that can mean skipping most steps while the run still exits 0. A
  warning is not enough because nothing downstream is watching for the consequence.
- Runs on `ckpt-all` land on whatever node is free; one validation node (R8 tested only a
  Tesla P100) proves nothing about the other 71 runs' nodes, which may include different
  GPU generations/driver stacks where an FTZ or fast-math path could reach this op. The
  assertion is the actual defense, not the R8 investigation's single-GPU spot check.
- See R8 (`docs/metrics.md`) for what was verified about mixed-precision reachability in
  the current codebase, and why no TF32/precision flag is being set as a substitute for
  this assertion.

## What blocks the campaign

- Any structural difference in Part 1a (beyond the stated per-channel tolerances).
- Any unattributed difference in Part 1b or 1c.
- Any operational assertion failing in Part 2.
- Part 3's CPU criterion failing.
- (Not blocking: Part 3's GPU residual, and anything in Part 2 that looks like a
  comparative difference -- it cannot resolve one.)

## Executor notes

- Part 0 is done. Do Part 1a, then 1c, and REPORT before starting 1b/2/3 -- if 1a or 1c
  is not clean, everything downstream is a waste of GPU time.
- Remove the worktree when finished.
- Old and new code must run in the SAME conda env, same partition, same pinned node.
- Audit before launch (from the R1-R6 report): every evaluation script that constructs a
  model directly now gets `floor_mode='clamp'` by default. Production training passes
  `'smooth'`, but ad-hoc analysis does not.
