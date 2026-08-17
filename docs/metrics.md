# Metric definitions

Exact formulas for every number tracked in `docs/model-results.md`, with where it is computed
and how to read it. Written because two metrics had already collided under the same word
("spread" meaning max-min in one table and standard deviation in a function of the same name).

If you add a number to a results table, define it here first.

---

## 1. `val_loss` -- the headline number

```
val_loss = mean over val batches of [ Huber(log10 DOS_pred, log10 DOS_target)
                                    + Huber(log10 T_pred,   log10 T_target) ]
```

- `g3nat/training/trainer.py:47` (`nn.HuberLoss()`, default delta = 1.0), summed at
  `trainer.py:183-185`, averaged over batches at `trainer.py:189`.
- **Huber, not MSE.** On log10 targets, delta = 1.0 means "one decade": residuals under a
  decade are quadratic, beyond it linear.
- It is a **sum of two equally weighted terms**. In the planned LDOS work this becomes
  `a*T + b*LDOS + (1-b)*DOS`, where today's loss is exactly `a=1, b=0`.
- Both targets are log10 of clamped quantities, so the metric is scale-free in the observable
  but sensitive to the clamp floor near zero.
### final-epoch vs best-val -- USE BEST-VAL

```
final_val = val_losses[-1]
best_val  = min(val_losses)          # recoverable from ANY stored checkpoint, no retraining
```

**Every model in this project trains ~3x past its optimum.** Measured over six identical
runs: best val is reached at epoch **549, 771, 1033, 1331, 1689, 1900 of 5000**, after which
the model overfits, ending a mean of **0.060** worse (max 0.115). Consequences:

- Final-epoch val measures *how far a run drifted past its own optimum* as much as it
  measures fit. Best-val has 3.4x lower run-to-run std (0.0084 vs 0.0286).
- The drift is **capacity-dependent**, so it penalises larger models more and can invert an
  ordering. It did: the `num_layers` trend looks non-monotonic at final-epoch (L2 worse than
  L1) and is cleanly monotonic at best-val. Overfit gaps by depth: L1 0.009, L2 0.070,
  L3 0.048, L4 0.051.
- `tail_mean` does **not** help (std 0.0295, no better than final-epoch). Do not use it as a
  stability-improved substitute.

Historical numbers quoted before 2026-07-24 are final-epoch. `scripts/collect_bestval.py`
recomputes any table at best-val from stored curves at no compute cost.

**But loss curves are recoverable and weights are not.** Runs before 2026-07-24 saved only
final-epoch weights, so anything measured *from the model* -- per-base baselines, eta2,
window fractions, LDOS -- was computed on overfit weights. `scripts/train.py` now also writes
`checkpoint_best.pth` and `<model>_best.pth`; prefer those for any model-derived quantity.

### 1a. Selection criterion changed again since the paragraph above was written -- READ BEFORE POOLING CHECKPOINTS

The paragraph above ("best among checkpointed epochs... saved only when the current epoch
is itself the running minimum") describes the state of the code **after the 2026-08-11 fix
and before commit 6a9c51b (2026-08-16)**. It is no longer how `checkpoint_best.pth` is
produced. Both the granularity and the metric changed:

| | before 6a9c51b (v1, 84 published runs) | after 6a9c51b (v2, the 72-run campaign) |
|---|---|---|
| granularity | best among **checkpointed** epochs only (the callback fires every `checkpoint_frequency`, default 10) | best held in memory and refreshed **every** epoch; the exact-epoch snapshot is what gets serialized |
| criterion | the loss_b-**weighted** total validation loss (`min(val_losses)`, i.e. `a*T_loss + b*LDOS_loss + (1-b)*DOS_loss`) | the **unweighted** metric `val_dos_t_unweighted` (`dos_loss + transmission_loss`, no `loss_a`/`loss_b`/`loss_c` scaling) -- see `g3nat/training/trainer.py:314` and `:509-514` |

Both changes were correct fixes on their own terms (see model-results.md sec. 16 for the
granularity bug, and sec. 16e for why the criterion also had to move off the weighted
total: it is scaled differently in every supervision cell, so "best" was not comparable
across arms). But the consequence is that **the 84 published v1 `_best.pth` files and the
v2 checkpoints were/will be selected by two different criteria at two different
granularities.** A table that pools v1 and v2 checkpoints is comparing weights chosen two
different ways, not just weights from different runs.

**The discriminator.** A v2 checkpoint's `_best.pth` (and `checkpoint_best.pth`) carries
`selection_metric` (`'val_dos_t_unweighted'`) and `selection_value` keys, populated at
`scripts/train.py:542-543` and republished into the final `_best.pth` at `:684-685`. A v1
checkpoint has neither key -- `bc.get('selection_metric')` on a v1 file returns `None`.
Check for these keys before trusting a cross-cohort comparison; their absence is exactly
the signal that a file predates 6a9c51b.

**Do not confuse `best_val`/`best_val_epoch` with the selection criterion.** Every
published `_best.pth`, v1 or v2, also stores `'best_val'` and `'best_val_epoch'`
(`scripts/train.py:679-680`, NaN-safe via `np.nanmin`/`np.nanargmin` since the same
commit). These are the global minimum of the **loss_b-weighted** validation curve --
i.e. the same quantity v1 selected on -- and they are stored on every file regardless of
what the weights were actually chosen by. On a v2 file, `best_val_epoch` is therefore
**not** the epoch the stored weights came from; that epoch is `saved_at_epoch` /
`selection_value`'s epoch. Quoting `best_val` as "what the weights were selected on" is
correct for v1 and wrong for v2. Always read `selection_metric`/`selection_value` when
present, and fall back to `best_val` (with a note that it is the weighted quantity) only
when they are absent.

The best checkpoint (v1 semantics) is the best among **checkpointed** epochs (every
`checkpoint_frequency`), and it is saved only when the current epoch is itself the running
minimum, so its stored weights always match its stored value. `best_val` / `best_val_epoch`
in that file are the true global minimum of the curve, which may be marginally lower.

## 2. Convergence and stability

```
tail        = val_losses[-50:]
tail_mean   = mean(tail)
tail_slope  = polyfit(arange(len(tail)), tail, 1)[0]        # units: loss per epoch
rel_gap     = (final - tail_mean) / tail_mean               # dimensionless
```

- `scripts/collect_all_runs.py::curve_stats`, `scripts/collect_onsite_sweep.py::convergence`.
- `|tail_slope| <~ 3e-4` means converged. A **positive** slope is the meaningful warning sign,
  not a large magnitude.
- `rel_gap > 5%` flags a cell that ended on a bad draw. Report both; neither alone is
  sufficient (a cell can have a small slope and still end badly).

## 3. Variance: three different things, do not mix them

```
init noise     = |val_A - val_B|   for identical config AND identical split_seed
split scatter  = std over split_seed values of val_loss
total scatter  = std over runs varying both
```

- **Model init is not seeded anywhere.** `--split_seed` controls only the train/val split
  (`g3nat/data/splits.py`). So two runs at the same `--split_seed` still differ in init, and
  any "cross-seed std" mixes split and init variance.
- **CORRECTED 2026-07-24 (this file previously said ~0.001 -- that was wrong).** The 0.001
  figure came from two same-config pairs that happened to agree, i.e. n=2. Six runs at
  identical config and identical `--split_seed 42` give:

  | metric | mean | std (ddof=1) | range |
  |---|---|---|---|
  | final-epoch | 0.6281 | **0.0286** | 0.076 |
  | best-val | 0.5679 | **0.0084** | 0.025 |

  So init noise is ~0.029 on final-epoch, not 0.001, and the alpha=0 cross-seed std of
  0.0099 at n=3 was a lucky draw rather than a tight measurement.
- **Use best-val as the yardstick: a difference must exceed ~2 x 0.0084 = 0.017 to mean
  anything.** On final-epoch the equivalent bar is ~0.057.
- **`std` here is `np.std` with default `ddof=0` (population).** At n=3 this understates the
  sample standard deviation by a factor `sqrt(3/2) = 1.22`. All quoted stds and the
  "resolved" verdicts below use the population form; the verdicts were re-checked and none
  flip under `ddof=1`, but new comparisons should say which convention they use.
- **RETRACTED:** an earlier version of this file blamed L4/s42's 0.6539 (vs 0.6042 on a clean
  node) on `g3070`'s uncorrectable ECC errors. With the real init spread known (range 0.076),
  0.6539 is an ordinary draw and the ECC attribution is unsupported. The ECC errors were real
  -- they killed two jobs outright -- but they do not explain that number. Still exclude a
  node throwing ECC errors; just do not attribute specific values to it without evidence.

## 4. "Resolved" -- when is a gap real?

```
gap_ij      = |mean_i - mean_j|                     across seeds, referenced to G
scatter_ij  = sqrt(std_i^2 + std_j^2)
RESOLVED    iff gap_ij > 2 * scatter_ij
```

- `scripts/collect_all_runs.py`.
- Referencing to G before comparing removes any per-run global energy offset, so a shift
  common to all four bases cannot masquerade as scatter.
- **This says a gap is larger than run-to-run noise. It does not say the value is physically
  meaningful.** G's position is pinned near 0 by the HOMO-centred energy convention for
  495/515 sequences, so G-vs-X gaps can be "RESOLVED" and still not be a fit result. See
  `docs/dataset.md`.

## 5. Per-base structure

```
range      = max(baseline) - min(baseline)                  # what results tables call "range"
spread     = std(baseline)                                  # what the CODE calls "spread"
min_pairwise = min over i<j of |baseline_i - baseline_j|
eta2       = SS_between / SS_total                          # correlation ratio
             SS_between = sum_b n_b * (mean_b - grand_mean)^2
             SS_total   = sum over all sites of (x - grand_mean)^2
```

- `range`/`spread`/`min_pairwise`: `g3nat/evaluation/physicality.py::baseline_distinctness`
  returns `{'min_pairwise', 'spread'}` where **`spread` is the std**, NOT the range. Results
  tables label the max-min column "range (max-min)" precisely to avoid this collision.
- `eta2`: `scripts/collect_onsite_sweep.py::eta2`. Fraction of the variance in per-site onsite
  explained by base identity. **eta2 = 1 means onsite is fully determined by which base it is;
  eta2 ~ 0 means context sets everything.** Free model measured 0.028.
- **eta2 = 1.000 at alpha=1.0 is tautological**, not a finding -- onsite *is* the per-base
  table there.
- `collect_onsite_sweep.py` groups per-base over the **primary strand only** (`range(len(seq))`,
  the first `len(seq)` DNA nodes). That is correct given node ordering, but it is a half-sample.

## 6. Window membership -- a sanity check, not a criterion

```
frac_in_window     = fraction of diag(H) inside [-1, 1]
frac_eig_in_window = fraction of eigvalsh(H) inside [-1, 1]
coupling_bandwidth = max |off-diagonal element of H|
```

- `g3nat/evaluation/physicality.py`.
- **`[-1,1]` is the SUPERVISION range**, i.e. HOMO +/- 1 eV per sequence, and there is no DFT
  data outside it. Eigenvalues outside are unconstrained, not wrong. Read these as a coarse
  guard against runaway values (onsite at -33 eV), never as a physicality verdict. The
  function formerly named `is_physical_win` is now `both_window_fracs_increased`.
- `coupling_bandwidth` exists to catch "badness relocated from onsite into the couplings".

## 7. Node and site ordering (the silent-failure risk)

```
graph node order = [left contact, right contact,
                    primary[0..L-1] IN ORDER,
                    complementary[0..L-1] IN ORDER]        # NOT reversed
H index i        = i-th DNA node (contacts excluded), n_orb=1 => diag(H)[i] is site i's onsite
```

- `g3nat/graph/construction.py:106-131`.
- Getting this wrong produces plausible numbers that are entirely wrong. A scratch script hit
  exactly this (used `complementary[::-1]`) and it was caught only because the alpha=1.0 model
  is a control whose per-base onsite must be constant, so any label scramble shows up as a
  nonzero shift where zero is guaranteed. **Any new per-site analysis should include that
  control.**

## 8. LDOS quantities

```
ldos_i(E)   = -(1/pi) * Im G_ii(E)                      per-site local DOS, linear units
DOS(E)      = sum_i ldos_i(E)                           exact, verified both sides
share_i     = mean over E of [ ldos_i(E) / sum_j ldos_j(E) ]
```

- Model side: `self.ldos` (`g3nat/models/hamiltonian.py`), the diagonal of the same Green's
  function whose trace already gives DOS. Verified to sum to DOS to 2.6e-7 on trained
  checkpoints, non-negative, all 201 energy points unclamped.
- DFT side: `sum_atoms(DOSAtom) / DOS = 1.0000` at every energy, so the atom-resolved LDOS is
  an **exact** decomposition of the same DOS the model trains on. No free scale factor.
- `share_i` averaged over energy is dominated by the gap/tunnelling region and by contact-site
  broadening. It is **not** a good statistic for asking where resonances live -- that needs an
  energy-resolved comparison. (A conclusion was drawn from an energy-averaged share earlier
  and withdrawn.)

## 8a. Transport-restricted transmission

```
val_transmission              = Huber(log10 T_pred, log10 T_target)   over ALL energy points
val_transmission_appreciable  = Huber(log10 T_pred, log10 T_target)   over points where
                                log10 T_target > APPRECIABLE_T_LOG10  (= -16.0)
```

- `g3nat/training/trainer.py` (`APPRECIABLE_T_LOG10`, accumulated in `_validate_epoch`).
- Half of every spectrum is deep tunnelling at ~1e-8 (docs/dataset.md, measured over
  417,477 energy points), so **roughly half the error budget of the whole-window number is
  spent where no transport measurement would resolve anything**. A model that wins on the
  tail and loses at the resonances looks better on `val_transmission` while being worse for
  transport. Report both; neither replaces the other.
- The threshold is on the **target**, not the prediction, so the point set is a property of
  the data and identical across models. Strict inequality: a target exactly at -16.0 does
  not qualify.
- Averaged over the batches that contained at least one qualifying point, not over all
  batches. `nan` when no point in the entire epoch qualified -- which is the honest value,
  not zero.
- This is a diagnostic only. The trained loss still weights all 201 energy points equally;
  a uniform log-space fit remains a defensible choice, and nothing here changes it.

## 8b. The frozen per-epoch metric schema

`Trainer._validate_epoch` appends one dict per epoch to `metric_history`, which is written
into every checkpoint. `EXPECTED_METRIC_KEYS` (module level in `trainer.py`) lists exactly
the keys that dict carries, and `_validate_epoch` raises if its entry differs -- naming
missing and extra keys separately. **Anything not recorded per epoch is unavailable at every
epoch but one, and re-deriving it means re-running the campaign**, so a metric quietly
appearing or disappearing mid-run has to fail loudly. Editing the set is allowed; doing it
by accident is not, and a run whose schema differs is not schema-comparable with the rest.

| key | meaning |
|---|---|
| `epoch` | absolute epoch number, already accounting for `start_epoch` on a resumed run -- align on this, not on list position |
| `val_dos` | Huber on log10 DOS, absolute magnitude (sec. 1) |
| `val_dos_shape` | same, after the shared median-residual offset (sec. 1) |
| `val_transmission` | Huber on log10 T over the whole window (sec. 8a) |
| `val_transmission_appreciable` | the same, restricted to appreciable targets (sec. 8a) |
| `val_dos_t_unweighted` | `val_dos + val_transmission`, no `loss_a/b/c` scaling -- **the checkpoint selection criterion** (sec. 1a) |
| `val_dos_t_shape_unweighted` | shape-variant counterpart of the above |
| `val_ldos_residue` | held-out LDOS Huber against the residue aggregation; `nan` unless `ldos_target='residue'` |
| `val_ldos_base_only` | same against the base-only aggregation; `nan` unless `ldos_target='base_only'` |
| `val_ldos_shape_residue` | shape variant, independently centered over (site, energy) -- see `_ldos_agreement` |
| `val_ldos_shape_base_only` | shape variant of the base-only aggregation |
| `val_ldos_localization_gap` | `J_pred - J_target`, the log-space localization measure (sec. 8). Positive: model more localized than DFT |
| `floored_frac_dos` | fraction of DOS points at genuine underflow (`0 <= x < eps`) in the last validation batch |
| `floored_frac_t` | same for transmission |
| `floored_frac_ldos` | same for per-site LDOS |
| `neg_frac_dos` | fraction of DOS values that are NEGATIVE -- unphysical, and a non-Hermitian-H signal rather than smallness. Kept separate from the floored fraction, which used to conflate the two |
| `neg_frac_t` | same for transmission |
| `neg_frac_ldos` | same for per-site LDOS |
| `nan_skipped_total` | cumulative count of training batches skipped for a non-finite loss or gradient |
| `nan_selection_metric_total` | cumulative count of epochs whose selection metric was non-finite. Equal to the epoch count means **no best checkpoint was ever written** |

The floor/negative fractions are last-batch spot readings, not epoch averages: they are read
off 0-dim tensors carried on the model, which costs one CUDA sync per epoch instead of one
per batch. They answer "is an arm living at the log floor", not "how often".

## 9. Composition shift (the HOMO reference test)

```
shift_b = mean(onsite of base b over G-containing sequences)
        - mean(onsite of base b over AT-only sequences)
E_HOMO  = mean(raw Egrid)                               per sequence, absolute eV
```

- `scripts/onsite_offset_test.py`, `scripts/homo_composition.py`.
- Sequence classes are length-matched so length is not a confound.
- At alpha=1.0, `shift_b` must be **exactly 0** by construction (onsite is a per-base
  constant). Use it as the mapping control.
