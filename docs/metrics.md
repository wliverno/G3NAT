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
- Unless stated otherwise, a quoted `val_loss` is the **final epoch**, not the best epoch.

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
- Measured 2026-07-24: init noise ~= **0.001** (L4/s42 0.6042 vs alpha=0/s42 0.6054; L4/s43
  0.5823 vs alpha=0/s43 0.5818). Split scatter at alpha=0 is 0.0099. So the cross-seed number
  is dominated by the split, and init contributes almost nothing.
- **`std` here is `np.std` with default `ddof=0` (population).** At n=3 this understates the
  sample standard deviation by a factor `sqrt(3/2) = 1.22`. All quoted stds and the
  "resolved" verdicts below use the population form; the verdicts were re-checked and none
  flip under `ddof=1`, but new comparisons should say which convention they use.
- A number obtained on a node with uncorrectable ECC errors is not a measurement. The L4/s42
  cell read 0.6539 on `g3070` and 0.6042 on a clean node -- an 8% error from a run that
  exited 0.

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
