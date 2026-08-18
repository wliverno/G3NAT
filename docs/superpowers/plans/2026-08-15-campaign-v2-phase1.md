# Campaign v2 Phase 1 (fixes + features) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land every blocking fix and the feature from the campaign-v2 spec so the
72-run factorial can launch on trustworthy code.

**Architecture:** Surgical changes to the existing training pipeline (sampler, trainer,
checkpoint flow, solver floors, model init order) plus one opt-in model feature
(boolean per-base onsite). Every change is default-off or
behavior-preserving unless the spec says otherwise; byte-identity tests enforce that.

**Tech Stack:** PyTorch + PyTorch Geometric, pytest, SLURM (klone).

**Spec:** `docs/superpowers/specs/2026-08-13-campaign-v2-design.md`

## OPEN WORK -- NOTHING LAUNCHES UNTIL THIS LIST IS EMPTY (2026-08-16)

Findings from the three independent reviews (private notes tree,
reviews/2026-08-16-phase1-code-reviews.md). Batch A and B are DONE (5d74d8c, a0e56ea).
Batch C and the doc item are NOT.

- [x] **R1 (DONE 7950f99) (CAMPAIGN-AFFECTING) LDOS floor never converted.** `site_ldos_log10`
  (hamiltonian.py ~line 39) still hard-clamps at `self.log_floor`, which the default
  change moved from 1e-16 to 1e-38. A floored per-site point now reads -38 instead of
  -16 and, in Huber's linear regime, contributes ~37 instead of ~15 to `ldos_loss` with
  ZERO gradient -- feeding `total` and therefore checkpoint selection in every DOS+LDOS
  cell. Convert to the same `log10(clamp_min(x,0)+eps)` form, fix the stale docstring
  ("nine decades above the 1e-16 default"), and record `last_floored_frac_ldos`.
- [x] **R2 (DONE 7950f99) (blocks any legacy re-evaluation) floor semantics are unrecorded.** With the
  additive form, `eps=1e-16` no longer clamps, it BIASES -- proven from the regenerated
  constants: 10^-14.81016 - 10^-14.83916 = 1.001e-16 exactly, at a point the old clamp
  never touched. Re-running a pre-2026-08-15 checkpoint returns different tail
  transmission than the recorded numbers, silently. Add `floor_mode` ('clamp' default
  for legacy, 'smooth' written by train.py) and select on it; at minimum make
  inference.py warn loudly when a checkpoint's args carry no `log_floor`.
- [x] **R3 (DONE 7950f99) energy_grid_t buffer is float32 while a float64 consumer exists**
  (tests/test_all.py negf consistency): 2.9e-8 eV grid error, currently silent. Register
  with the source dtype and let `.to(dtype=...)` downcast per forward.
- [x] **R4 (DONE 7950f99) reference `calculate_NEGF` still clips at 1e-16** (physics.py ~182), so any
  model-vs-reference tail figure shows a real tail against a plateaued reference.
- [x] **R5 (DONE 7950f99) floored-fraction diagnostic**: two CUDA syncs per forward INCLUDING in
  training (the hot loop Task 11 was optimizing) -- guard with `if not self.training` or
  keep as a tensor and sync once in `_validate_epoch`; and it conflates underflow with
  NEGATIVE values -- record a separate negative fraction, which is the diagnostic that
  would have caught the non-Hermitian arm without a code audit.
- [x] **R6 (DONE 7950f99) two more vacuous tests**: `test_energy_grid_buffer` asserts only a shape (it
  passes with a grid of zeros); `test_log_floor`'s "never binds" assertion says a boolean
  mean lies in [0,1], true by construction. Both need real assertions + mutation checks.
- [x] **R7 (DONE 8c66efb) (DOCS, blocks any pooled table) selection criterion changed and the docs still
  describe the old one.** `docs/metrics.md` sec 1 and `docs/model-results.md` sec 16 say
  best = best among CHECKPOINTED epochs on val loss. Since 6a9c51b that is false:
  selection is per-epoch on `val_dos_t_unweighted`. The 84 v1 `_best.pth` were selected
  on the loss-weighted total at 10-epoch cadence; the 72 v2 runs are selected on an
  unweighted metric at the exact epoch. ANY TABLE POOLING THEM COMPARES WEIGHTS CHOSEN
  TWO DIFFERENT WAYS. v2 files carry `selection_metric`, v1 files do not -- that is the
  discriminator and it must be written down.
- [ ] **R8 (ops) AMP/TF32 flush-to-zero would break the floor.** eps=1e-38 is a float32
  subnormal; verified alive on CPU and the campaign GPU, but any fast-math/FTZ path turns
  `log10(0+1e-38)` into -inf and silently converts every deep-tail point into a skipped
  optimizer step. Re-check if mixed precision is ever enabled.
- [ ] **R9 DEFERRED TO PHASE 2 (no runner exists yet).** The campaign runner is Phase 2
  work; this item is a REQUIREMENT ON IT, carried forward with the subnormal assertion
  from the characterization design. It does not block Phase 1 closing.
  Original: **the C2 warning prints but does not change the exit code** -- deliberate
  (non-zero exit interacts with --requeue policy). The campaign runner's monitors MUST
  key on that warning string, or a no-best-checkpoint run still looks successful.

REJECTED, with reasoning (do not "fix" these):
- Reviewer recommended dropping eps to 1e-30 to tame the log-derivative near zero. Its
  own justification is arithmetically wrong ("~7 decades below the L=16 true
  transmission ~1e-32" -- 1e-30 is two decades ABOVE 1e-32), so it would bind on exactly
  the extrapolation physics 1e-38 exists to protect; willll's stated floor was "at least
  1e-35". Mitigations for the derivative are already in place (Huber's bounded outer
  derivative, grad clipping, the non-finite skip) and the reviewer's own measured
  |dL/dH| maxed at 1e19, far from float32 overflow. STAYING AT 1e-38.

## Review step (added 2026-08-16 at willll's instruction -- applies to EVERY task)

Each task gets an independent code review before the next task starts. The executor
does NOT review its own work. After a task commits, the coordinator dispatches a
reviewer subagent per superpowers:requesting-code-review with the task's plan section
as the requirements and the commit range as the diff. Critical findings are fixed
before proceeding; Important findings before the phase ends; Minor findings are
recorded. Reviewers are asked explicitly to name any test they believe is vacuous --
this plan has already shipped one test that could not fail (Task 5's original
stationary-point gradient test) and one ordering test that needed a mutation check to
prove it discriminates (Task 8).

Tasks 1-11 and 15 were reviewed retroactively in three subsystem batches on 2026-08-16
(training loop, model/solver, data/eval); findings recorded in the private notes tree.

## Global Constraints

- ASCII only in all files.
- NEVER run python/pytest bare on a login node. Every test run in this plan means:
  `srun -A anantram-ckpt -p ckpt-all -c 4 --mem=16G -t 60 bash -lc 'cd /mmfs1/gscratch/anantram/willll/G3NAT && source ~/.bashrc && conda activate g3nat && <command>'`
  Referred to below as `SRUN '<command>'`.
- Full suite must pass after every task: `SRUN 'python -m pytest tests/ -x -q'`.
- Commit after every task; stage specific files, never `git add -A`. No references to
  the private notes repo in commits or code. End commits with the Claude co-author
  trailer used throughout this repo's history.
- Default-off invariants: with no new flags passed, model construction and one
  training epoch must be BYTE-IDENTICAL to current behavior EXCEPT where a task
  explicitly changes a default (Tasks 5, 9, 12 note their intentional changes).
- The historical `alpha` CLI flags are removed in Task 12; do not add
  back-compat shims -- old checkpoints that carry alpha args are loaded only by
  analysis code, not by train.py.

---

### Task 1: Seeded, epoch-aware batch sampler (spec B1)

**Files:**
- Modify: `g3nat/training/utils.py:43-83`
- Modify: `scripts/train.py:240-255` (sampler construction / seeding order)
- Modify: `g3nat/training/trainer.py:122` (fit loop calls set_epoch)
- Test: `tests/test_training/test_sampler_seeding.py` (create; also create empty `tests/test_training/__init__.py` if missing)

**Interfaces:**
- Produces: `LengthBucketBatchSampler(dataset, batch_size, shuffle=True, seed=None)`;
  method `set_epoch(epoch: int) -> None`. seed=None reproduces current unseeded behavior.
- Trainer calls `train_loader.batch_sampler.set_epoch(epoch)` if that attribute chain exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training/test_sampler_seeding.py
import torch
from g3nat.data import generate_tight_binding_data, create_dna_dataset
from g3nat.training import LengthBucketBatchSampler


def _tiny_dataset():
    seqs, comp, dos, trans, grid = generate_tight_binding_data(
        num_samples=24, seq_length=4, num_energy_points=8)
    return create_dna_dataset(sequences=seqs, dos_data=dos, transmission_data=trans,
                              energy_grid=grid, complementary_sequences=comp)


def _epoch_batches(sampler, epoch):
    sampler.set_epoch(epoch)
    return [list(b) for b in sampler]


def test_same_seed_same_epoch_identical_batches():
    ds = _tiny_dataset()
    s1 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    s2 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    assert _epoch_batches(s1, 0) == _epoch_batches(s2, 0)
    assert _epoch_batches(s1, 7) == _epoch_batches(s2, 7)


def test_different_epochs_differ_and_different_seeds_differ():
    ds = _tiny_dataset()
    s = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    assert _epoch_batches(s, 0) != _epoch_batches(s, 1)
    s_other = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=124)
    assert _epoch_batches(s, 0) != _epoch_batches(s_other, 0)


def test_seed_none_and_shuffle_false_paths_still_work():
    ds = _tiny_dataset()
    unseeded = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=None)
    assert len(list(iter(unseeded))) == len(unseeded)
    ordered = LengthBucketBatchSampler(ds, batch_size=4, shuffle=False)
    assert _epoch_batches(ordered, 0) == _epoch_batches(ordered, 5)


def test_set_epoch_survives_resume_semantics():
    # Epoch N's batches must depend only on (seed, N), not on iteration history --
    # that is what makes a requeue at epoch N reproduce the original epoch N.
    ds = _tiny_dataset()
    s1 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=99)
    for e in range(5):
        _epoch_batches(s1, e)
    fresh = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=99)
    assert _epoch_batches(s1, 5) == _epoch_batches(fresh, 5)
```

- [ ] **Step 2: Run to verify failure**

`SRUN 'python -m pytest tests/test_training/test_sampler_seeding.py -v'`
Expected: FAIL / TypeError ("unexpected keyword argument 'seed'").

- [ ] **Step 3: Implement**

In `g3nat/training/utils.py`, replace the class body's RNG handling:

```python
class LengthBucketBatchSampler(Sampler[List[int]]):
    """BatchSampler grouping indices by DNA-node count into uniform-size batches.

    seed=None reproduces the historical unseeded behavior (fresh OS entropy every
    epoch -- NOT reproducible; see determinism finding, 2026-08-13). With a seed,
    epoch N's batch composition is a pure function of (seed, N) via set_epoch, so a
    requeued run that calls set_epoch(N) regenerates the original epoch N exactly.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        buckets = {}
        for idx in range(len(dataset)):
            data = dataset[idx]
            num_dna = int(getattr(data, 'num_dna_nodes', data.x.size(0) - 2))
            buckets.setdefault(num_dna, []).append(idx)
        self.buckets = buckets
        self._batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self):
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng((self.seed, self._epoch))

    def _build_batches(self):
        rng = self._rng() if self.shuffle else None
        batches = []
        for _, indices in sorted(self.buckets.items()):
            indices = list(indices)
            if rng is not None:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        if rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self._batches = self._build_batches()
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)
```

Note the `sorted(self.buckets.items())`: dict order is insertion order, which is
deterministic here, but sorting makes bucket order independent of dataset iteration
order. One shared rng per rebuild (not one per bucket) is fine and simpler.

In `g3nat/training/trainer.py`, at the TOP of the epoch loop in `fit()` (right after
`for epoch in range(...)`):

```python
            sampler = getattr(train_loader, 'batch_sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
```

In `scripts/train.py`: move the `set_init_seed` block (lines 250-255) to ABOVE the
loader construction (before line 240), and pass the seed to the train sampler:

```python
    train_sampler = LengthBucketBatchSampler(train_dataset, args.batch_size,
                                             shuffle=True, seed=args.init_seed)
```

(val sampler stays shuffle=False, no seed needed.)

- [ ] **Step 4: Run tests** -- the new file, then the full suite. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/training/utils.py g3nat/training/trainer.py scripts/train.py tests/test_training/
git commit -m "fix(training): seed the batch sampler from init_seed, epoch-indexed

Batch composition was fresh OS entropy every epoch (default_rng with no seed),
so no run was reproducible from its seeds and a requeue could not regenerate its
own trajectory. Epoch N is now a pure function of (seed, N)."
```

---

### Task 2: NaN guard before the optimizer step (spec B5)

**Files:**
- Modify: `g3nat/training/trainer.py:342-366` (_train_epoch), `:71-73` (init), `:435-453` (metric entry)
- Test: `tests/test_training/test_nan_guard.py`

**Interfaces:**
- Produces: `Trainer.nan_skipped_total: int`; metric_history entries carry
  `'nan_skipped_total'` (cumulative count, float).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training/test_nan_guard.py
import math
import torch
import torch.nn as nn
from g3nat.training.trainer import Trainer
from g3nat.training.config import TrainingConfig


class _NanEveryOther(nn.Module):
    """Tiny stand-in model: forward returns (dos_pred, t_pred); every 2nd batch NaN."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        out = self.lin(batch.x_feat)  # [B, 2]
        dos = out[:, :1].expand(-1, 4)
        t = out[:, 1:].expand(-1, 4)
        if self.calls % 2 == 0:
            dos = dos * float('nan')
        return dos, t


class _Batch:
    def __init__(self, n):
        self.x_feat = torch.randn(n, 2)
        self.dos = torch.randn(n * 4)
        self.transmission = torch.randn(n * 4)

    def to(self, device):
        return self


def test_nan_batch_is_skipped_and_counted_and_params_stay_finite():
    torch.manual_seed(0)
    model = _NanEveryOther()
    trainer = Trainer(model, TrainingConfig(num_epochs=1, learning_rate=1e-2))
    loader = [_Batch(3) for _ in range(6)]
    loss = trainer._train_epoch(loader)
    assert trainer.nan_skipped_total == 3
    assert math.isfinite(loss)
    for p in model.parameters():
        assert torch.isfinite(p).all(), "a NaN step reached the optimizer"
```

- [ ] **Step 2: Run to verify failure**

`SRUN 'python -m pytest tests/test_training/test_nan_guard.py -v'`
Expected: FAIL -- `nan_skipped_total` does not exist, and params go NaN.

- [ ] **Step 3: Implement**

In `Trainer.__init__`, after `self.metric_history = []`: add `self.nan_skipped_total = 0`.

Rewrite the middle of `_train_epoch`'s batch loop:

```python
            losses = self._compute_losses(batch, dos_pred, transmission_pred)
            total_loss = losses['total']

            if not torch.isfinite(total_loss):
                self.nan_skipped_total += 1
                continue

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.nan_skipped_total += 1
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()

            train_loss += total_loss.item()
            n_used += 1
```

with `n_used = 0` initialized alongside `train_loss` and the epoch average becoming
`train_loss /= max(1, n_used)` (skipped batches must not drag the mean toward 0).

In `_validate_epoch`'s `entry` dict add:
```python
            'nan_skipped_total': float(self.nan_skipped_total),
```

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/training/trainer.py tests/test_training/test_nan_guard.py
git commit -m "fix(training): never step the optimizer on a non-finite loss or grad

One NaN gradient passes through clip_grad_norm_ unscaled and poisons Adam's
moments permanently -- the run then burns to the epoch cap producing garbage.
Skip-and-count instead; the count is recorded per epoch in metric_history."
```

---

### Task 3: Unweighted-metric best selection, exact-epoch weights, honest final save (spec B2 + B4, plus the nanargmin SHOULD)

**Files:**
- Modify: `g3nat/training/trainer.py` (fit: track best state in memory; final save epoch)
- Modify: `scripts/train.py:315-349` (checkpoint_cb), `:424-448` (best publication, nanargmin)
- Test: `tests/test_training/test_best_selection.py`

**Interfaces:**
- Produces: `Trainer.best_unweighted: dict` with keys `'value'` (float),
  `'epoch'` (int), `'state_dict'` (CPU state_dict copy). Updated every epoch on
  `metric_history[-1]['val_dos_t_unweighted']`.
- checkpoint_callback signature unchanged; callbacks read `trainer.best_unweighted`
  via a new optional kwarg `best_state=` passed by fit().

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training/test_best_selection.py
import copy
import torch
import torch.nn as nn
from g3nat.training.trainer import Trainer
from g3nat.training.config import TrainingConfig


class _Scripted(nn.Module):
    """Model whose val unweighted metric follows a scripted curve via a scale param."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        out = batch.base * self.scale
        return out, out


class _Batch:
    def __init__(self):
        self.base = torch.ones(2, 4)
        self.dos = torch.zeros(8)
        self.transmission = torch.zeros(8)

    def to(self, device):
        return self


def test_best_unweighted_tracks_the_true_minimum_epoch():
    model = _Scripted()
    cfg = TrainingConfig(num_epochs=5, learning_rate=0.0, warmup_epochs=0)
    trainer = Trainer(model, cfg)
    # Script the parameter per epoch: dip at epoch 2, rise after.
    values = [1.0, 0.5, 0.1, 0.7, 0.9]

    def progress(epoch, tr, vl):
        with torch.no_grad():
            model.scale.fill_(values[min(epoch + 1, 4)])

    with torch.no_grad():
        model.scale.fill_(values[0])
    trainer.fit([_Batch()], [_Batch()], progress_callback=progress)
    assert trainer.best_unweighted['epoch'] == 2
    saved_scale = trainer.best_unweighted['state_dict']['scale'].item()
    assert abs(saved_scale - 0.1) < 1e-6, "stored weights are not from the best epoch"


def test_final_checkpoint_reports_actual_last_epoch():
    model = _Scripted()
    cfg = TrainingConfig(num_epochs=3, learning_rate=0.0, warmup_epochs=0,
                         checkpoint_frequency=100)  # periodic save never fires
    trainer = Trainer(model, cfg)
    seen = {}

    def cb(model_, opt, epoch, tr, vl, metric_history=None, best_state=None):
        seen['epoch'] = epoch
        seen['best_state'] = best_state

    trainer.fit([_Batch()], [_Batch()], checkpoint_callback=cb)
    assert seen['epoch'] == 2  # the loop's actual last epoch, not num_epochs - 1 by luck
    assert seen['best_state'] is not None and 'epoch' in seen['best_state']
```

- [ ] **Step 2: Run to verify failure**

`SRUN 'python -m pytest tests/test_training/test_best_selection.py -v'`
Expected: FAIL -- `best_unweighted` does not exist; callback lacks `best_state`.

- [ ] **Step 3: Implement**

In `Trainer.__init__`: `self.best_unweighted = {'value': float('inf'), 'epoch': -1, 'state_dict': None}`.

In `fit()`, after `self.val_losses.append(val_loss)`:

```python
            metric = self.metric_history[-1].get('val_dos_t_unweighted', float('nan'))
            if metric == metric and metric < self.best_unweighted['value'] - 1e-12:
                self.best_unweighted = {
                    'value': float(metric),
                    'epoch': epoch,
                    'state_dict': {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()},
                }
```

(`metric == metric` is the NaN check.) Pass `best_state=self.best_unweighted` as an
extra kwarg in BOTH checkpoint_callback call sites, and capture the loop epoch:

```python
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.config.num_epochs):
            last_epoch = epoch
            ...
        if checkpoint_callback is not None:
            checkpoint_callback(self.model, self.optimizer, last_epoch,
                                self.train_losses, self.val_losses,
                                metric_history=self.metric_history,
                                best_state=self.best_unweighted)
```

In `scripts/train.py` `checkpoint_cb(...)`: accept `best_state=None`; the "best"
branch now saves `best_state['state_dict']` (when not None) instead of the live
model's current weights, keyed on `best_state['value']`:

```python
    def checkpoint_cb(model, opt, epoch, train_losses, val_losses,
                      metric_history=None, best_state=None):
        save_checkpoint(model, opt, epoch, train_losses, val_losses,
                        vars(args), energy_grid,
                        os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'),
                        metric_history=metric_history)
        if best_state and best_state.get('state_dict') is not None \
                and best_state['value'] < best_val['value'] - 1e-12:
            best_val['value'] = best_state['value']
            ckpt_model = {'model_state_dict': best_state['state_dict'],
                          'epoch': best_state['epoch']}
            torch.save({**ckpt_model,
                        'optimizer_state_dict': opt.state_dict(),
                        'train_losses': train_losses, 'val_losses': val_losses,
                        'args': vars(args), 'energy_grid': energy_grid,
                        'metric_history': metric_history,
                        'selection_metric': 'val_dos_t_unweighted',
                        'selection_value': best_state['value']},
                       os.path.join(args.checkpoint_dir, 'checkpoint_best.pth'))
```

(Direct torch.save here rather than save_checkpoint, because the weights are the
in-memory best snapshot, not the live model. Keep the atomic write pattern: write to
`checkpoint_best.pth.tmp` then `os.replace` -- copy the try/except from
`callbacks.save_checkpoint`.)

Also in `scripts/train.py`'s final-publication block, replace both
`int(np.argmin(val_losses))` and `min(val_losses)` with NaN-safe versions:

```python
        'best_val': float(np.nanmin(val_losses)),
        'best_val_epoch': int(np.nanargmin(val_losses)),
```

and in the two print lines. Also record `'selection_metric'`/`'selection_value'`
into the published `_best.pth` dict (copied from the best checkpoint).

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/training/trainer.py scripts/train.py tests/test_training/test_best_selection.py
git commit -m "fix(training): select best weights on the unweighted metric, exactly at its epoch

The weighted total is differently scaled in every supervision cell, and the
checkpoint cadence meant saved 'best' weights lagged the metric optimum by up
to 10 epochs. The best state_dict is now held in memory, updated every epoch
on val_dos_t_unweighted, and serialized as-is; the final save reports the
actual last epoch; best_val_epoch is NaN-safe."
```

---

### Task 4: Resume guard and gated stale-best cleanup (spec B3)

**Files:**
- Modify: `scripts/train.py:354-384` (resume block)
- Test: `tests/test_training/test_resume_guard.py`

**Interfaces:**
- Produces: module-level function `check_resume_args(stored: dict, current: dict) -> None`
  in `scripts/train.py` (raises ValueError on mismatch). Compared keys:
  `CONFIG_DEFINING_ARGS = ['data_source', 'data_dir', 'model_type', 'hidden_dim',
  'num_layers', 'num_heads', 'n_orb', 'solver_type', 'log_floor', 'complex_eta',
  'use_log_outputs', 'enforce_hermiticity', 'conv_type', 'use_geometry', 'geom_cache',
  'optimizer', 'weight_decay', 'split_seed', 'init_seed', 'loss_a', 'loss_b', 'loss_c',
  'ldos_target', 'shape_loss', 'batch_size', 'learning_rate', 'num_epochs']`
  (module-level constant next to the function).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training/test_resume_guard.py
import importlib.util
import os
import sys
import pytest

_spec = importlib.util.spec_from_file_location(
    "train_script",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def test_matching_args_pass():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    train_script.check_resume_args(dict(cur), cur)  # no raise


def test_mismatched_arg_raises_with_the_key_named():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    stored = dict(cur)
    stored['n_orb'] = 2
    with pytest.raises(ValueError, match='n_orb'):
        train_script.check_resume_args(stored, cur)


def test_missing_stored_key_raises():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    stored = dict(cur)
    del stored['solver_type']
    with pytest.raises(ValueError, match='solver_type'):
        train_script.check_resume_args(stored, cur)
```

(Note: importing scripts/train.py as a module must not execute main() -- it is already
guarded by `if __name__ == '__main__'`. The `exec_module` runs top-level code only,
which is imports + function defs; if this test reveals top-level side effects, that is
itself a finding -- fix by moving them under main().)

- [ ] **Step 2: Run to verify failure**

`SRUN 'python -m pytest tests/test_training/test_resume_guard.py -v'`
Expected: FAIL -- `CONFIG_DEFINING_ARGS`/`check_resume_args` do not exist.

- [ ] **Step 3: Implement**

In `scripts/train.py`, above `main()`:

```python
CONFIG_DEFINING_ARGS = [
    'data_source', 'data_dir', 'model_type', 'hidden_dim', 'num_layers', 'num_heads',
    'n_orb', 'solver_type', 'log_floor', 'complex_eta', 'use_log_outputs',
    'enforce_hermiticity', 'conv_type', 'use_geometry', 'geom_cache', 'optimizer',
    'weight_decay', 'split_seed', 'init_seed', 'loss_a', 'loss_b', 'loss_c',
    'ldos_target', 'shape_loss', 'batch_size', 'learning_rate', 'num_epochs',
]


def check_resume_args(stored: dict, current: dict) -> None:
    """A checkpoint may only resume the run that wrote it. Raises on any mismatch
    of a config-defining arg, naming the offending key -- resuming under different
    args silently republishes one config's weights under another's label."""
    problems = []
    for key in CONFIG_DEFINING_ARGS:
        if key not in stored:
            problems.append(f"{key}: missing from checkpoint args")
        elif stored[key] != current.get(key):
            problems.append(f"{key}: checkpoint={stored[key]!r} vs current={current.get(key)!r}")
    if problems:
        raise ValueError(
            "checkpoint_latest.pth was written by a DIFFERENT configuration; refusing "
            "to resume. Use a fresh --checkpoint_dir per run. Mismatches: "
            + "; ".join(problems))
```

In the resume block, right after `ckpt = torch.load(...)`:
```python
        check_resume_args(ckpt.get('args', {}), vars(args))
```

And BEFORE the resume block, the gated cleanup:
```python
    best_path_stale = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
    if not os.path.exists(checkpoint_path) and os.path.exists(best_path_stale):
        # No latest checkpoint means this is a FRESH run in a reused dir: a leftover
        # best would be republished under the new args. With a latest checkpoint
        # present we are resuming, and the best is this run's own -- keep it.
        os.remove(best_path_stale)
        print(f"Removed stale checkpoint_best.pth from a previous run in {args.checkpoint_dir}")
```

(`checkpoint_path` is already defined at line 360; move its definition above this block.)

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/test_training/test_resume_guard.py
git commit -m "fix(train): refuse to resume under different args; delete stale best only on fresh runs

Resume previously loaded any checkpoint_latest.pth in the dir with no args
check, and a leftover checkpoint_best.pth could be republished under a new
run's args. Deletion is gated on the absence of checkpoint_latest so a
preempted run never loses its own best weights."
```

---

### Task 5: Smooth log floor + floored-fraction metric (spec B6; INTENTIONAL DEFAULT CHANGE)

**STATUS: DONE (commit c1dabc8, executed ahead of order 2026-08-15).** Deviations from
the text below, recorded so no executor repeats them: (1) the deep-tail gradient test
as written used H=0, a stationary point of transmission (dT/dH=0 structurally, under
ANY floor) -- the landed test uses a weakly coupled 4-site chain with T in 1e-27..1e-20
instead; (2) two baselines encoding the old clamp's -16 plateau were regenerated
(test_ldos_exposure.py expected_t; baseline model_hamiltonian.pkl, transmission only);
(3) the MODEL constructor default is still 1e-16 (plan only changed the CLI) -- Task 16
step 3.5 below aligns it.

**Files:**
- Modify: `g3nat/models/hamiltonian.py:576-607` (Frobenius) and `:640-685` (complex)
- Modify: `scripts/train.py:92-94` (--log_floor default + help)
- Modify: `g3nat/training/trainer.py:435-453` (metric entry reads floored fractions)
- Test: `tests/test_models/test_log_floor.py`

**Interfaces:**
- Produces: model method `_log10_floored(x: Tensor) -> Tensor` implementing
  `log10(clamp_min(x, 0) + self.log_floor)`; model attributes
  `last_floored_frac_dos: float`, `last_floored_frac_t: float` (fraction of grid
  points with linear value below log_floor, set on every forward).
- DEFAULT CHANGE: `--log_floor` default 1e-16 -> **1e-38** and it is now a smoothing
  eps, not a clamp. Old checkpoints carry their own log_floor in args and keep it.
- metric_history gains `'floored_frac_dos'`, `'floored_frac_t'` (nan for models
  without the attributes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models/test_log_floor.py
import numpy as np
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def _model(log_floor):
    grid = np.linspace(-1.0, 1.0, 8)
    return DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                      energy_grid=grid, n_orb=1,
                                      solver_type='complex', log_floor=log_floor)


def test_deep_tail_keeps_gradient():
    m = _model(1e-38)
    H = torch.zeros(1, 4, 4, requires_grad=True)
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    # A target 1.5 decades below the OLD 1e-16 clamp must still pull gradient.
    loss = ((T - (-17.5)) ** 2).mean()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert H.grad.abs().max() > 0, "deep-tail gradient is dead"


def test_floor_never_binds_above_eps_and_fractions_are_recorded():
    m = _model(1e-38)
    H = torch.randn(1, 4, 4) * 0.1
    H = 0.5 * (H + H.transpose(-1, -2))
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert torch.isfinite(T).all() and torch.isfinite(DOS).all()
    assert 0.0 <= m.last_floored_frac_t <= 1.0
    assert 0.0 <= m.last_floored_frac_dos <= 1.0


def test_smooth_floor_matches_plain_log10_when_far_from_floor():
    m = _model(1e-38)
    x = torch.tensor([1e-3, 1.0, 10.0])
    out = m._log10_floored(x)
    assert torch.allclose(out, torch.log10(x), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

`SRUN 'python -m pytest tests/test_models/test_log_floor.py -v'`
Expected: FAIL -- `_log10_floored` missing; with the old clamp the deep-tail gradient
assertion fails at the default floor.

- [ ] **Step 3: Implement**

Add to the model (near the solver methods):

```python
    def _log10_floored(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth floor: log10(max(x,0) + eps). Unlike a hard clamp this keeps
        gradient at every point above zero -- the old clamp at 1e-16 sat INSIDE
        the transmission target range (targets reach 6.7e-19) and zeroed the
        gradient exactly in the deep tail the length-extrapolation claim lives
        in. eps (self.log_floor, default 1e-38) is a pure log10(0) guard: below
        every physical value in the data, above float32 underflow."""
        return torch.log10(torch.clamp_min(x, 0.0) + self.log_floor)
```

Complex path: replace `maybe_log10` + the two `torch.clamp` lines:

```python
        DOS_lin = (-1/np.pi) * torch.imag(torch.einsum('benn->be', Gr))
        ldos_lin = (-1/np.pi) * torch.imag(torch.diagonal(Gr, dim1=-2, dim2=-1))
        ...
        T_lin = torch.real(torch.einsum('benn->be', M))
        self.last_floored_frac_dos = float((DOS_lin < self.log_floor).float().mean())
        self.last_floored_frac_t = float((T_lin < self.log_floor).float().mean())
        DOS = self._log10_floored(DOS_lin) if self.use_log_outputs else torch.clamp_min(DOS_lin, 0.0)
        T = self._log10_floored(T_lin) if self.use_log_outputs else torch.clamp_min(T_lin, 0.0)
```

Frobenius path: same substitution at `:578-579` and `:606-607` (keep behavior parity
even though the path is prohibited for campaign runs -- Task 6).

`scripts/train.py:92`: default `1e-38`, help text: "Smoothing eps for log10 of
DOS/T: log10(max(x,0)+eps). Pure log10(0) guard -- never binds on physical values
(dataset T minimum is 6.7e-19). Recorded in args; must match at train and eval."

Trainer `_validate_epoch` entry additions (after the localization gap):

```python
            'floored_frac_dos': float(getattr(self.model, 'last_floored_frac_dos', float('nan'))),
            'floored_frac_t': float(getattr(self.model, 'last_floored_frac_t', float('nan'))),
```

(Last batch's value -- a per-epoch spot reading, cheap and sufficient to see whether
an arm is living at the floor.)

- [ ] **Step 4: Run tests** (file + full suite; expect existing baseline tests that
pinned the old clamp to fail -- READ each failure; update a baseline ONLY if its
assertion encodes the old clamp value, and say so in the commit).

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py scripts/train.py g3nat/training/trainer.py tests/test_models/test_log_floor.py
git commit -m "fix(negf): smooth log floor at 1e-38 -- the 1e-16 clamp sat inside the data

Transmission targets reach 6.7e-19, so the old clamp hard-capped predictions
1.5 decades above the deepest targets with zero gradient, exactly in the tail
the extrapolation claim lives in. log10(max(x,0)+eps) keeps gradient
everywhere; the per-forward floored fraction is recorded in metric_history."
```

---

### Task 6: Hard-fail invalid physics arms (spec B7)

**Files:**
- Modify: `g3nat/models/hamiltonian.py` (__init__ validation)
- Modify: `scripts/train.py` (frobenius refusal)
- Test: `tests/test_models/test_invalid_arms.py`

**Interfaces:**
- Produces: `DNATransportHamiltonianGNN(..., enforce_hermiticity=False, n_orb=2)`
  raises ValueError. `scripts/train.py --solver_type frobenius` exits with an error
  unless `--allow_frobenius` (new flag, default False) is passed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models/test_invalid_arms.py
import numpy as np
import pytest
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def test_non_hermitian_norb2_refused():
    grid = np.linspace(-1, 1, 8)
    with pytest.raises(ValueError, match='hermiticity'):
        DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=2,
                                   enforce_hermiticity=False)


def test_non_hermitian_norb1_still_allowed():
    grid = np.linspace(-1, 1, 8)
    DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                               energy_grid=grid, n_orb=1,
                               enforce_hermiticity=False)  # no raise (it is a no-op)
```

- [ ] **Step 2: Run to verify failure** -- expected: first test FAILS (no raise).

- [ ] **Step 3: Implement**

In `DNATransportHamiltonianGNN.__init__`, immediately after the flag is stored:

```python
        if n_orb > 1 and not enforce_hermiticity:
            raise ValueError(
                "enforce_hermiticity=False with n_orb>1 produces a non-Hermitian H "
                "(measured: max|H-H^T| ~ 0.8, negative DOS hidden by the log floor). "
                "At n_orb=1 the flag is a no-op, which made this arm look valid. "
                "There is no physically admissible use; refusing.")
```

In `scripts/train.py`: add `parser.add_argument('--allow_frobenius', action='store_true',
help='Escape hatch for legacy comparisons only; the Frobenius path is silently wrong at resonances and ignores complex_eta.')`
and in `main()` after parse:

```python
    if args.solver_type == 'frobenius' and not args.allow_frobenius:
        raise SystemExit(
            "--solver_type frobenius is disabled for training runs: its singular-matrix "
            "fallback is silently ~98% wrong at resonances and it ignores --complex_eta. "
            "Pass --allow_frobenius only for legacy comparisons.")
```

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py scripts/train.py tests/test_models/test_invalid_arms.py
git commit -m "fix(model): hard-fail physically invalid arms

enforce_hermiticity=False at n_orb>1 breaks Hermiticity (its no-op behavior at
n_orb=1 disguised that); the Frobenius solver is quietly wrong at resonances.
Both now fail loudly instead of producing plausible-looking numbers."
```

---

### Task 7: Geometry restored at inference (spec B8)

**Files:**
- Modify: `g3nat/evaluation/inference.py` (predict_sequence, ~line 152-215)
- Modify: `scripts/dos_map.py:99` (same call pattern)
- Test: `tests/test_evaluation/test_inference_geometry.py` (create dir/__init__ if missing)

**Interfaces:**
- Produces: `predict_sequence(..., geometry_cache: dict | None = None)`. When the
  loaded model has `use_geometry=True`: a missing cache or a cache miss for the
  sequence raises ValueError (silent amputation was the bug); when the model has
  `use_geometry=False`, the argument is ignored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation/test_inference_geometry.py
import numpy as np
import pytest
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.evaluation import inference


def _geom_model(tmp_path):
    grid = np.linspace(-1, 1, 8)
    stats = {'backbone': {'mean': np.zeros(7), 'std': np.ones(7)},
             'hbond': {'mean': np.zeros(7), 'std': np.ones(7)}}
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=1,
                                   use_geometry=True, geom_norm_stats=stats)
    path = tmp_path / 'geom_model.pth'
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'model_type': 'hamiltonian', 'hidden_dim': 16,
                         'num_layers': 1, 'num_heads': 2, 'n_orb': 1,
                         'use_geometry': True, 'solver_type': 'complex',
                         'log_floor': 1e-38, 'complex_eta': 1e-12,
                         'use_log_outputs': True, 'enforce_hermiticity': True,
                         'conv_type': 'gat'},
                'energy_grid': grid}, path)
    return path


def test_geom_model_without_cache_raises(tmp_path):
    path = _geom_model(tmp_path)
    model, grid = inference.load_trained_model(str(path))
    with pytest.raises(ValueError, match='geometry'):
        inference.predict_sequence(model, 'GATT', energy_grid=grid)


def test_geom_model_with_cache_miss_raises(tmp_path):
    path = _geom_model(tmp_path)
    model, grid = inference.load_trained_model(str(path))
    with pytest.raises(ValueError, match='GATT'):
        inference.predict_sequence(model, 'GATT', energy_grid=grid,
                                   geometry_cache={'aaaa': object()})
```

(Adjust `load_trained_model`'s exact return signature to the real one when writing the
test -- read `g3nat/evaluation/inference.py:80-150` first; the shape above is the
expected pattern, the assertion targets are the ValueErrors.)

- [ ] **Step 2: Run to verify failure** -- expected: predictions silently succeed with
amputated geometry (no raise), so both tests FAIL.

- [ ] **Step 3: Implement**

In `predict_sequence`, where the graph is built (currently
`sequence_to_graph(sequence, ...)` with no geometry): add the parameter and the guard:

```python
def predict_sequence(model, sequence, ..., geometry_cache=None):
    ...
    use_geom = bool(getattr(model, 'use_geometry', False))
    geometry = None
    if use_geom:
        if geometry_cache is None:
            raise ValueError(
                "this checkpoint was trained with use_geometry=True but no "
                "geometry_cache was supplied -- scoring it with the geometry "
                "channel silently deleted biases geometry effects toward null. "
                "Pass the cache (geom_cache/geometry_v2.pkl).")
        geometry = geometry_cache.get(sequence.lower())
        if geometry is None:
            raise ValueError(f"geometry cache has no entry for sequence {sequence!r}")
    data = sequence_to_graph(sequence, ..., geometry=geometry)
```

Mirror the same pattern in `scripts/dos_map.py:99` (it already knows the sequence; add
a `--geom_cache` argument defaulting to `geom_cache/geometry_v2.pkl`, load it only when
the checkpoint's args say use_geometry).

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/evaluation/inference.py scripts/dos_map.py tests/test_evaluation/
git commit -m "fix(inference): a geometry-trained model refuses to run without its geometry

predict_sequence and dos_map built graphs with no geometry argument, so a
geom-trained checkpoint was scored with all-zero geometry masks, silently,
biasing every script-based geometry evaluation toward null."
```

---

### Task 8: Optional modules stop perturbing shared init draws (spec B9)

**Files:**
- Modify: `g3nat/models/hamiltonian.py:135-172` (__init__ ordering)
- Test: `tests/test_models/test_init_rng_isolation.py`

**Interfaces:**
- Produces: no API change. Construction order becomes: all always-present modules
  first (convs, projections), THEN optional modules (geom_encoder, per-base baseline)
  last, in a fixed documented order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models/test_init_rng_isolation.py
import numpy as np
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def _core_weights(model):
    return {k: v.clone() for k, v in model.state_dict().items()
            if not k.startswith(('geom_encoder', 'onsite_baseline', 'onsite_alpha'))}


def _build(**kw):
    torch.manual_seed(1234)
    grid = np.linspace(-1, 1, 8)
    stats = {'backbone': {'mean': np.zeros(7), 'std': np.ones(7)},
             'hbond': {'mean': np.zeros(7), 'std': np.ones(7)}}
    if kw.get('use_geometry'):
        kw['geom_norm_stats'] = stats
    return DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                      energy_grid=grid, n_orb=1, **kw)


def test_geometry_flag_does_not_shift_core_init():
    off = _core_weights(_build(use_geometry=False))
    on = _core_weights(_build(use_geometry=True))
    assert off.keys() == on.keys()
    for k in off:
        assert torch.equal(off[k], on[k]), f"core param {k} differs when geometry toggles"
```

- [ ] **Step 2: Run to verify failure** -- expected: PASSES or FAILS depending on
whether any RNG consumer follows geom_encoder today. IMPORTANT: if it passes at
n_orb=1 without the structured head, ALSO parameterize with the Task-12 boolean head
once it exists. If the test passes as written against current code, verify why (the
review says onsite_baseline follows geom_encoder -- construct with the structured
head on and assert its baseline differs / core stays equal), then keep the stronger
variant. The test must FAIL against a deliberately wrong ordering (temporarily move
geom_encoder construction above the conv layers to see it fail, then revert) --
record in the commit message that this mutation check was done.

- [ ] **Step 3: Implement**

In `__init__`, move the ENTIRE `if use_geometry:` block (lines 139-156) to the very
end of `__init__`, after the structured-onsite block, with the comment:

```python
        # OPTIONAL modules are constructed LAST, after every always-present
        # parameter, so toggling them cannot shift the init RNG stream of the
        # shared core. Order among optional modules is fixed (structured onsite,
        # then geometry, then any future addition appended AFTER these) and is
        # part of reproducibility -- do not reorder.
```

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS. NOTE: this
intentionally changes which draws optional modules receive at fixed seed vs old
checkpoints -- weights LOAD fine (state_dict keys unchanged); only fresh-init
reproducibility of optional-module weights differs. Say so in the commit.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_init_rng_isolation.py
git commit -m "fix(model): optional modules init last so flags cannot shift core RNG draws"
```

---

### Task 9: geometry_v2 cache default, hard-fail on miss, train-split norm stats (spec B10 + review minor 17; INTENTIONAL DEFAULT CHANGE)

**Files:**
- Modify: `scripts/train.py:110-111` (default), `:195-205` (stats on train split), `:207-231` (dataset build order)
- Modify: `g3nat/data/datasets.py:250-260` (hard-fail on miss)
- Test: `tests/test_data/test_geom_cache_guard.py` (create dir/__init__ if missing)

**Interfaces:**
- Produces: `create_dna_dataset(..., geometry_cache=...)` raises KeyError naming the
  sequence when a geometry_cache is supplied but has no entry for a sequence
  (silent None was the bug). `--geom_cache` default becomes
  `geom_cache/geometry_v2.pkl`. `compute_norm_stats(cache, sequences=None)` gains an
  optional sequence filter (norm stats from the training split only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data/test_geom_cache_guard.py
import numpy as np
import pytest
from g3nat.data import create_dna_dataset


def test_cache_miss_is_a_hard_error():
    grid = np.linspace(-1, 1, 8)
    dos = [np.zeros(8)]
    trans = [np.zeros(8)]
    with pytest.raises(KeyError, match='gatt'):
        create_dna_dataset(sequences=['GATT'], dos_data=dos, transmission_data=trans,
                           energy_grid=grid, geometry_cache={'aaaa': object()})


def test_no_cache_still_fine():
    grid = np.linspace(-1, 1, 8)
    create_dna_dataset(sequences=['GATT'], dos_data=[np.zeros(8)],
                       transmission_data=[np.zeros(8)], energy_grid=grid,
                       geometry_cache=None)
```

- [ ] **Step 2: Run to verify failure** -- expected: first test FAILS (miss returns
None silently today).

- [ ] **Step 3: Implement**

`g3nat/data/datasets.py` (~line 256), replace the `.get()`:

```python
        if geometry_cache is not None:
            key = sequence.lower()
            if key not in geometry_cache:
                raise KeyError(
                    f"geometry cache has no entry for sequence {key!r} -- a silent "
                    "miss trains that graph with geometry deleted (mask 0), diluting "
                    "the geometry channel. Rebuild the cache or use geometry_v2.pkl.")
            geometry = geometry_cache[key]
        else:
            geometry = None
```

`scripts/train.py:110`: default `'geom_cache/geometry_v2.pkl'`, help text notes the
v1 cache covers only 515 sequences.

`g3nat/graph/geometry.py::compute_norm_stats`: add `sequences=None` parameter; when
given, compute stats only over `{s: cache[s] for s in sequences if s in cache}`.
In `scripts/train.py`, compute stats AFTER the split using train sequences only:
move the `compute_norm_stats` call below `grouped_split` (the cache LOAD stays where
it is; only the stats move) and call
`compute_norm_stats(geom_cache, sequences={seqs[i].lower() for i in train_indices})`.
The dataset build then stays where it is (it needs geom_cache, not the stats); the
model constructor already receives `geom_norm_stats` separately.

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/data/datasets.py g3nat/graph/geometry.py scripts/train.py tests/test_data/test_geom_cache_guard.py
git commit -m "fix(geometry): v2 cache by default, hard-fail on a miss, norm stats from train split only

A cache miss silently trained that sequence with geometry deleted; the default
cache covered only the 515 v1 sequences; z-score stats were fit on train+val."
```

---

### Task 10: n_orb-aware H readouts (spec B14)

**Files:**
- Modify: `g3nat/evaluation/physicality.py`
- Modify: `scripts/collect_bestval_runs.py:130`, `scripts/analyze_learned_hamiltonian.py:60`, `scripts/onsite_offset_test.py:87` (assert n_orb == 1)
- Test: `tests/test_evaluation/test_physicality_norb.py`

**Interfaces:**
- Produces: `onsite_block_eigs(H, n_orb) -> np.ndarray` (per-site onsite-block
  eigenvalues, shape [n_sites, n_orb]); `onsite_metrics(H_diag_or_eigs, window)`
  unchanged signature (feed it block eigs at n_orb>1); `coupling_block_bandwidth(H, n_orb) -> float`
  (max Frobenius norm over inter-site blocks). Existing `coupling_bandwidth(H)`
  redirects to `coupling_block_bandwidth(H, 1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation/test_physicality_norb.py
import numpy as np
from g3nat.evaluation.physicality import onsite_block_eigs, coupling_block_bandwidth


def test_block_eigs_reduce_to_diag_at_norb1():
    H = np.diag([0.1, -0.2, 0.3])
    eigs = onsite_block_eigs(H, n_orb=1)
    assert eigs.shape == (3, 1)
    assert np.allclose(sorted(eigs.ravel()), sorted([0.1, -0.2, 0.3]))


def test_block_eigs_at_norb2_use_the_full_block():
    # One site, two orbitals: onsite block [[0, 0.5], [0.5, 0]] -> eigs +/-0.5.
    # Naive diag(H) would report (0, 0) and misclassify 0.5 as a coupling.
    H = np.array([[0.0, 0.5], [0.5, 0.0]])
    eigs = onsite_block_eigs(H, n_orb=2)
    assert eigs.shape == (1, 2)
    assert np.allclose(sorted(eigs.ravel()), [-0.5, 0.5])


def test_coupling_block_bandwidth_excludes_intra_site_block():
    H = np.array([[0.0, 0.5, 0.1, 0.0],
                  [0.5, 0.0, 0.0, 0.1],
                  [0.1, 0.0, 0.0, 0.5],
                  [0.0, 0.1, 0.5, 0.0]])
    # n_orb=2, 2 sites: the 0.5s are INTRA-site, only the 0.1 blocks are couplings.
    bw = coupling_block_bandwidth(H, n_orb=2)
    assert abs(bw - np.linalg.norm(np.array([[0.1, 0.0], [0.0, 0.1]]))) < 1e-12
```

- [ ] **Step 2: Run to verify failure** -- expected: ImportError (functions missing).

- [ ] **Step 3: Implement** in `physicality.py`:

```python
def onsite_block_eigs(H, n_orb=1):
    """Per-site onsite levels: eigenvalues of each n_orb x n_orb diagonal block.
    At n_orb=1 this is diag(H). At n_orb>1 naive diag(H) reads intra-block diagonal
    entries and misclassifies the block off-diagonal as a coupling; the block
    eigenvalues are the basis-invariant onsite levels."""
    H = np.asarray(H)
    n_sites = H.shape[0] // n_orb
    out = np.empty((n_sites, n_orb))
    for s in range(n_sites):
        block = H[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb]
        out[s] = np.linalg.eigvalsh(0.5 * (block + block.T))
    return out


def coupling_block_bandwidth(H, n_orb=1):
    """Max Frobenius norm over INTER-site blocks (intra-site blocks are onsite)."""
    H = np.asarray(H)
    n_sites = H.shape[0] // n_orb
    best = 0.0
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            blk = H[i*n_orb:(i+1)*n_orb, j*n_orb:(j+1)*n_orb]
            best = max(best, float(np.linalg.norm(blk)))
    return best
```

Change `coupling_bandwidth(H)` body to `return coupling_block_bandwidth(H, 1)` and
note in its docstring it is n_orb=1-only. In each of the three scripts, at the point
where `diag(H)` is taken, insert:

```python
    n_orb = int(ckpt_args.get('n_orb', 1))
    assert n_orb == 1, (
        f"this script reads diag(H) assuming n_orb=1 but the checkpoint has n_orb="
        f"{n_orb}; use g3nat.evaluation.physicality.onsite_block_eigs instead")
```

(adapt the variable holding the checkpoint args to each script; all three load a
checkpoint dict with an 'args' key.)

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/evaluation/physicality.py scripts/collect_bestval_runs.py scripts/analyze_learned_hamiltonian.py scripts/onsite_offset_test.py tests/test_evaluation/test_physicality_norb.py
git commit -m "fix(evaluation): n_orb-aware onsite/coupling readouts; diag(H) scripts assert n_orb==1

At n_orb=2 diag(H) reads intra-block entries as sites and misclassifies block
off-diagonals as couplings -- the window-membership response would be a
different object at each N_ORB level of the factorial."
```

---

### Task 11: Energy grid as a registered buffer (spec B15)

**Files:**
- Modify: `g3nat/models/hamiltonian.py` (__init__ + both solver paths at :540, :647)
- Test: `tests/test_models/test_energy_grid_buffer.py`

**Interfaces:**
- Produces: buffer `self.energy_grid_t` (float32 tensor); `self.energy_grid` (the
  numpy array) remains for every existing consumer. Solvers use the buffer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models/test_energy_grid_buffer.py
import numpy as np
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def test_energy_grid_is_a_buffer_and_solver_output_unchanged():
    grid = np.linspace(-1, 1, 8)
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=1)
    assert 'energy_grid_t' in dict(m.named_buffers())
    H = torch.zeros(1, 4, 4)
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert T.shape == (1, 8) and DOS.shape == (1, 8)
```

- [ ] **Step 2: Run to verify failure** -- expected: FAIL (no such buffer).

- [ ] **Step 3: Implement**: in `__init__` (after energy_grid is stored):
`self.register_buffer('energy_grid_t', torch.tensor(np.asarray(energy_grid), dtype=torch.float32))`.
In the complex path replace `energy = torch.tensor(self.energy_grid, ...)` with
`energy = self.energy_grid_t.to(dtype=dtype_real)` (device is handled by the buffer
following the module) and the same at the Frobenius path's grid construction (:540
area). CAUTION: `load_state_dict` on old checkpoints will miss the new buffer key --
load them with `strict=False`? NO: instead register with `persistent=False`
(`self.register_buffer('energy_grid_t', ..., persistent=False)`) so it never enters
state_dict and old/new checkpoints stay mutually loadable.

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_energy_grid_buffer.py
git commit -m "perf(negf): energy grid as a non-persistent buffer -- was re-built and re-transferred every forward"
```

---

### Task 12: Alpha booleanization (spec F1; REMOVES the alpha CLI surface)

**Files:**
- Modify: `g3nat/models/hamiltonian.py` (init block :158-197)
- Modify: `scripts/train.py` (args :127-134, model construction :291-295, assert :150-151)
- Modify: `tests/test_models/test_structured_onsite.py` (rewrite to the boolean API)
- Test: same file

**Interfaces:**
- Produces: model kwarg `per_base_onsite: bool = False` replacing
  `structured_onsite/alpha_granularity/alpha_mode/alpha_value/alpha_init`. When True:
  `onsite = one_hot @ onsite_baseline` (pure per-base table, the old alpha=1). When
  False: `onsite = onsite_proj(features)` (the old alpha=0 / default, byte-identical
  to current default). CLI: `--per_base_onsite` flag; the five alpha flags are GONE.
- Consumes: Task 8's ordering rule -- `onsite_baseline` remains constructed in the
  optional-modules-last section.

- [ ] **Step 1: Rewrite the test file** -- keep any existing default-off byte-identity
assertions; the API tests become:

```python
# tests/test_models/test_structured_onsite.py (rewritten; keep the file's existing
# default-off/byte-identity tests, adapting constructor calls)
import numpy as np
import pytest
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def _grid():
    return np.linspace(-1, 1, 8)


def test_per_base_onsite_true_creates_baseline_param():
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=_grid(), n_orb=1, per_base_onsite=True)
    assert hasattr(m, 'onsite_baseline')
    assert m.onsite_baseline.shape == (4, 1)


def test_per_base_onsite_false_is_default_and_adds_no_params():
    m0 = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                    energy_grid=_grid(), n_orb=1)
    assert not hasattr(m0, 'onsite_baseline')


def test_old_alpha_kwargs_are_gone():
    with pytest.raises(TypeError):
        DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=_grid(), structured_onsite=True)


def test_mix_onsite_is_pure_table_when_on():
    torch.manual_seed(0)
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=_grid(), n_orb=1, per_base_onsite=True)
    feats = torch.randn(3, 16)
    onehot = torch.eye(4)[[0, 2, 1]]
    out = m._mix_onsite(feats, onehot)
    expected = onehot @ m.onsite_baseline
    assert torch.allclose(out, expected)
```

- [ ] **Step 2: Run to verify failure** -- expected: TypeError on `per_base_onsite`.

- [ ] **Step 3: Implement**

Constructor: replace the five params with `per_base_onsite: bool = False`; the init
block becomes:

```python
        # Boolean per-base onsite head (formerly a continuous alpha mix; see the
        # commit removing it). True: onsite is a learned 4-entry-per-base table --
        # the old alpha=1. False (default): fully context-dependent onsite -- the
        # old alpha=0, byte-identical to the historical default.
        self.per_base_onsite = per_base_onsite
        if per_base_onsite:
            self.onsite_baseline = nn.Parameter(torch.empty(4, n_orb * n_orb))
            nn.init.normal_(self.onsite_baseline, std=0.01)
```

`_onsite_alpha` is deleted; `_mix_onsite` becomes:

```python
    def _mix_onsite(self, dna_features, original_dna_onehot):
        if self.per_base_onsite:
            return original_dna_onehot @ self.onsite_baseline
        return self.onsite_proj(dna_features)
```

`scripts/train.py`: delete the five alpha argparse lines and the per_base+fixed
assert; add `parser.add_argument('--per_base_onsite', action='store_true',
help='Onsite = a learned per-base table (4 values shared across all A/T/G/C sites) '
'instead of the context head. Off in the campaign; one post-factorial epilogue run.')`;
model construction passes `per_base_onsite=args.per_base_onsite`. Add
`'per_base_onsite'` to `CONFIG_DEFINING_ARGS` (Task 4) and REMOVE any alpha keys
from it if present.

Search the package for other constructors of the model
(`grep -rn "structured_onsite" g3nat/ scripts/ tests/`) and update every call site --
notably `g3nat/evaluation/inference.py::load_trained_model` rebuilds the model from
checkpoint args: it must map OLD checkpoints
(`args.get('structured_onsite')` + `alpha_value`) onto the new API for loading:
`per_base_onsite = bool(old_args.get('structured_onsite')) and float(old_args.get('alpha_value', 0.0)) == 1.0`;
any old checkpoint with 0 < alpha < 1 or alpha_mode=learned cannot be represented --
raise ValueError telling the user to use the pre-boolean code for those (they are
historical alpha-sweep artifacts, not campaign models).

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py scripts/train.py g3nat/evaluation/inference.py tests/test_models/test_structured_onsite.py
git commit -m "feat(model)!: alpha becomes a boolean per-base onsite head

The continuous alpha mix is removed. Measured across the factorial, alpha=0
(pure context) was the answer everywhere alpha had an effect, alpha=1 was
pathologically slow to converge, and the intermediate values never produced a
result the endpoints did not. The HOMO-referenced energy convention further
deflates the per-base-table reading the mix existed to enable. What remains is
the question the head was built to ask, as a switch: can a 4-value-per-base
table fit what the context head fits? Off throughout the campaign; one
epilogue run with it on. Old alpha-sweep checkpoints with fractional alpha
must be read with pre-boolean code."
```

---

### Task 14: Transport-restricted transmission metric + schema freeze (spec sec. 4 / metric key freeze)

**Files:**
- Modify: `g3nat/training/trainer.py` (_validate_epoch)
- Test: `tests/test_training/test_metric_schema.py`

**Interfaces:**
- Produces: metric key `'val_transmission_appreciable'` -- Huber on the (pred, target)
  pairs where `transmission_target > APPRECIABLE_T_LOG10` with
  `APPRECIABLE_T_LOG10 = -16.0` a module-level constant in trainer.py (the
  docs/dataset.md "appreciable transmission" threshold: the half of the spectrum
  where current actually flows). nan when no points qualify in an epoch.
- Produces: module-level `EXPECTED_METRIC_KEYS` frozenset in trainer.py listing every
  key `_validate_epoch` writes; `_validate_epoch` asserts its entry matches EXACTLY.
  This is the launch-irreversibility guard: adding a metric later forces a deliberate
  edit of the frozen set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training/test_metric_schema.py
import torch
import torch.nn as nn
from g3nat.training import trainer as trainer_mod
from g3nat.training.trainer import Trainer, EXPECTED_METRIC_KEYS
from g3nat.training.config import TrainingConfig


class _Const(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        base = torch.zeros(2, 4) + self.p
        return base, base


class _Batch:
    def __init__(self, t_target):
        self.dos = torch.zeros(8)
        self.transmission = torch.full((8,), t_target)

    def to(self, device):
        return self


def test_entry_matches_frozen_schema_exactly():
    tr = Trainer(_Const(), TrainingConfig(num_epochs=1, warmup_epochs=0))
    tr._validate_epoch([_Batch(-3.0)], epoch=0)
    assert set(tr.metric_history[-1].keys()) == set(EXPECTED_METRIC_KEYS)


def test_appreciable_metric_masks_the_deep_tail():
    tr = Trainer(_Const(), TrainingConfig(num_epochs=1, warmup_epochs=0))
    tr._validate_epoch([_Batch(-3.0)], epoch=0)   # all points appreciable
    above = tr.metric_history[-1]['val_transmission_appreciable']
    assert above == above  # not nan
    tr2 = Trainer(_Const(), TrainingConfig(num_epochs=1, warmup_epochs=0))
    tr2._validate_epoch([_Batch(-20.0)], epoch=0)  # none appreciable
    below = tr2.metric_history[-1]['val_transmission_appreciable']
    assert below != below  # nan
```

- [ ] **Step 2: Run to verify failure** -- expected: ImportError on
EXPECTED_METRIC_KEYS.

- [ ] **Step 3: Implement**

In trainer.py, module level:

```python
# The half of the spectrum where current actually flows (docs/dataset.md): model
# comparisons must be reported both whole-window and restricted to this region.
APPRECIABLE_T_LOG10 = -16.0

# LAUNCH-FROZEN metric schema. Anything not recorded per-epoch is unavailable at
# any epoch but one -- re-deriving it means re-running the campaign. Editing this
# set after launch invalidates cross-run schema compatibility; do it deliberately.
EXPECTED_METRIC_KEYS = frozenset({
    'epoch', 'val_dos', 'val_dos_shape', 'val_transmission',
    'val_transmission_appreciable', 'val_dos_t_unweighted',
    'val_dos_t_shape_unweighted', 'val_ldos_residue', 'val_ldos_base_only',
    'val_ldos_shape_residue', 'val_ldos_shape_base_only',
    'val_ldos_localization_gap', 'nan_skipped_total',
    'floored_frac_dos', 'floored_frac_t',
})
```

In `_validate_epoch`'s batch loop, accumulate the masked metric:

```python
                t_target = batch.transmission.view(dos_pred.size(0), dos_pred.size(1))
                mask = t_target > APPRECIABLE_T_LOG10
                if mask.any():
                    agg_t_appreciable += torch.nn.functional.huber_loss(
                        transmission_pred[mask], t_target[mask]).item()
                    n_appreciable += 1
```

(initialize `agg_t_appreciable = 0.0; n_appreciable = 0` with the other
accumulators) and in the entry:

```python
            'val_transmission_appreciable': (agg_t_appreciable / n_appreciable
                                             if n_appreciable else float('nan')),
```

After building `entry`, before appending:

```python
        if set(entry.keys()) != set(EXPECTED_METRIC_KEYS):
            missing = set(EXPECTED_METRIC_KEYS) - set(entry.keys())
            extra = set(entry.keys()) - set(EXPECTED_METRIC_KEYS)
            raise AssertionError(
                f"metric_history schema drifted: missing={sorted(missing)}, "
                f"extra={sorted(extra)}. The schema is launch-frozen; edit "
                "EXPECTED_METRIC_KEYS deliberately if this is intentional.")
```

NOTE: Tasks 2 and 5 added keys ('nan_skipped_total', 'floored_frac_*') -- this task's
frozen set is written AFTER those land, which is why it is ordered last among the trainer tasks. If executing out of order, the frozen set is
the source of truth and earlier tasks' keys must match it.

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/training/trainer.py tests/test_training/test_metric_schema.py
git commit -m "feat(training): transport-restricted transmission metric; launch-frozen metric schema

val_transmission_appreciable reads the region where current actually flows
(target log10 T > -16), per the standing methods requirement. The schema
assert makes silent metric drift impossible mid-campaign."
```

---

### Task 15: Resolved-config JSON + git sha per run (spec B11)

**Files:**
- Create: `g3nat/utils/runmeta.py`
- Modify: `scripts/train.py` (call at start of main), `g3nat/utils/__init__.py` (export)
- Test: `tests/test_utils/test_runmeta.py` (create dir/__init__ if missing)

**Interfaces:**
- Produces: `write_run_metadata(output_dir: str, args_dict: dict) -> str` writing
  `<output_dir>/resolved_config.json` with keys `'args'` (the full vars(args)),
  `'git_sha'`, `'git_dirty'` (bool), `'hostname'`, `'timestamp'`, `'g3nat_version'`.
  Returns the path. Git info read via subprocess with a safe fallback ('unknown').

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils/test_runmeta.py
import json
import os
from g3nat.utils.runmeta import write_run_metadata


def test_writes_complete_metadata(tmp_path):
    path = write_run_metadata(str(tmp_path), {'n_orb': 2, 'learning_rate': 1e-3})
    assert os.path.basename(path) == 'resolved_config.json'
    with open(path) as f:
        meta = json.load(f)
    assert meta['args']['n_orb'] == 2
    for key in ('git_sha', 'git_dirty', 'hostname', 'timestamp', 'g3nat_version'):
        assert key in meta
```

- [ ] **Step 2: Run to verify failure** -- expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# g3nat/utils/runmeta.py
"""Per-run provenance: the resolved config and code identity, written at start.

The campaign's legibility requirement: a run's exact parameters must be readable
from its artifacts, not reconstructed from runner scripts and defaults."""
import json
import os
import socket
import subprocess
import time


def _git(cmd, cwd):
    try:
        return subprocess.run(['git'] + cmd, cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def write_run_metadata(output_dir: str, args_dict: dict) -> str:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sha = _git(['rev-parse', 'HEAD'], repo) or 'unknown'
    dirty = bool(_git(['status', '--porcelain'], repo))
    try:
        import g3nat
        version = getattr(g3nat, '__version__', 'unknown')
    except Exception:
        version = 'unknown'
    meta = {
        'args': args_dict,
        'git_sha': sha,
        'git_dirty': dirty,
        'hostname': socket.gethostname(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'g3nat_version': version,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'resolved_config.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path
```

Export from `g3nat/utils/__init__.py`. In `scripts/train.py` main(), right after the
makedirs calls:

```python
    from g3nat.utils.runmeta import write_run_metadata
    meta_path = write_run_metadata(args.output_dir, vars(args))
    print(f"Run metadata: {meta_path}")
```

- [ ] **Step 4: Run tests** (file + full suite). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g3nat/utils/runmeta.py g3nat/utils/__init__.py scripts/train.py tests/test_utils/
git commit -m "feat(train): resolved-config JSON + git sha written at run start

A run's exact parameters are now readable from its artifacts. The n_orb
illegibility (runner override vs argparse default) cannot recur silently."
```

---

### Task 16: Full-suite gate + default-behavior audit -- DONE 2026-08-17

**Files:** none created; this is the plan's exit gate.

- [x] **Step 1 (RE-RUN 2026-08-17):** `python -m pytest tests/ -q` -- **320 passed, 0
  failed, exit 0 (job 38618932, node n3294, 6m12s)**, against HEAD `f69f87a` plus the
  legacy-checkpoint test work committed alongside this line. THIS is the tick that
  counts.
  - SUPERSEDED, kept for the record: the earlier tick read "310 passed, 0 failed (job
    38617686), 323 with the Step 3.4 test". That run was against a tree containing
    `c027e42`, WHICH HAS SINCE BEEN REVERTED (`500ef7b`), and predated `f7aede7`,
    `265e2c2` and `f69f87a` -- including a required-argument signature change in
    `inference.py`. It certified a tree that no longer exists, so it could not gate
    anything. Do not quote 310/323.
  - The 320 includes the legacy-checkpoint coverage added 2026-08-17:
    `tests/test_evaluation/test_legacy_checkpoint_roundtrip.py` now 19 tests (was 13 --
    it claimed to exercise "every args-less fallback at once" while asserting only the
    floor pair and the alpha mapping) and the new
    `tests/baseline/test_baseline_legacy_checkpoints.py` (4). Mutation-checked on 8
    separate loader defaults, job 38618836, all 8 killed.
- [x] **Step 2 (DONE 2026-08-17):** Grep audit -- each must return nothing:
  - `grep -rn "alpha_value\|alpha_mode\|alpha_granularity\|structured_onsite" g3nat/ scripts/train.py` (Task 12 completeness; analysis scripts reading OLD checkpoints may keep references -- only g3nat/ and train.py must be clean)
    -- `scripts/train.py` clean. 8 JUSTIFIED survivors, all in
    `g3nat/evaluation/inference.py`, all inside the legacy-checkpoint
    alpha -> `per_base_onsite` mapping and its error messages: that mapping exists
    precisely to READ old args, so the names must appear there. No alpha surface
    remains in the model, the trainer or the CLI.
  - `grep -rn "default_rng()" g3nat/training/` (Task 1 completeness) -- one survivor,
    `utils.py:70`, the `seed is None` branch of `LengthBucketBatchSampler._rng`. That
    branch IS the documented "reproduce historical unseeded behavior" path; the seeded
    branch is `default_rng((seed, epoch))`. Not a regression.
  - `grep -rn "learn_eta\|eta_raw\|learned_eta\|Kilgour\|Henderson\|downfolding" g3nat/ scripts/ tests/ docs/`
    -- empty (exit 1). The feature removed on 2026-08-17 left no residue.
- [x] **Step 3.4 (DONE 2026-08-17, `tests/test_evaluation/test_legacy_checkpoint_roundtrip.py`,
  13 tests, job 38617709):** all three real Hamiltonian checkpoints in `trained_models/`
  load into current code. Their args predate `solver_type`, `log_floor`, `floor_mode`
  and the whole alpha surface, so they exercise every args-less fallback at once. The
  test is mutation-checked on three separate lines (job 38617853). ORIGINAL TEXT: (gap found during Task 11, which
  verified NO existing test covers this): save a model's state_dict, construct a fresh
  model, load it, and assert success -- proving the persistent=False energy_grid_t
  buffer keeps checkpoints mutually loadable across the change. Also load one REAL
  pre-2026-08 checkpoint from trained_models/ into a current model and assert it loads
  (that is the case the campaign will actually depend on).
- [x] **Step 3.5 CANCELLED 2026-08-17 -- DO NOT DO THIS, it would break R2.** The step
  below was written before R2 made `floor_mode='clamp'` the constructor default so that
  an args-less legacy checkpoint reproduces its recorded numbers exactly. Legacy
  reproduction requires `clamp` AND `log_floor=1e-16` together; changing the constructor
  value to 1e-38 while the mode stays `clamp` would clamp legacy checkpoints at a
  different point and silently void the guarantee R2 exists to provide. The CLI still
  defaults to 1e-38 + `smooth`, which is what the campaign runs, and `resolved_config.json`
  records both -- so the defaults no longer "disagree" in the n_orb sense; they encode two
  different intents (legacy reproduction vs campaign physics) and both are written down.
  ORIGINAL TEXT, superseded: Align the MODEL constructor's log_floor default (hamiltonian.py
  __init__ signature) from 1e-16 to 1e-38 to match the CLI (defaults that disagree are
  the n_orb lesson); run the log-floor and baseline tests after.
- [ ] **Step 3 SUPERSEDED 2026-08-17 -- DO NOT RUN AS WRITTEN.** The two-run
  `val_losses` diff below is replaced by the separate characterization gate, which
  tests determinism directly rather than as a smoke-test side effect. Left unticked
  because it was not executed here. ORIGINAL TEXT: Behavior audit against the spec's
  intentional-change list: run
  `SRUN 'python scripts/train.py --data_source tb --num_samples 24 --seq_length 4 --num_epochs 3 --num_energy_points 8 --init_seed 42 --output_dir /tmp/p1_smoke --checkpoint_dir /tmp/p1_smoke_ckpt'`
  twice and diff the two runs' `val_losses` (they must be IDENTICAL on CPU -- the
  first end-to-end determinism evidence; if they differ, STOP and investigate before
  declaring Phase 1 done).
- [x] **Step 4 (DONE 2026-08-17): Commit** anything the audit fixed; otherwise no
  commit. The audit fixed nothing -- it found no regression -- so the commit carries
  the new round-trip test and this bookkeeping only.

---

## Self-review record (per the writing-plans skill)

- Spec coverage: B1(T1) B2(T3) B3(T4) B4(T3) B5(T2) B6(T5) B7(T6) B8(T7) B9(T8)
  B10(T9) B11(T15) B14(T10) B15(T11) F1(T12); metric freeze +
  val_transmission_appreciable (T14); nanargmin SHOULD folded into T3. NOT in this
  plan (deliberate): the Phase 2 pilots/gates, the campaign runner, MDE computation,
  analysis scripts, docs/dataset.md coverage notes, references.md merge -- those are
  the Phase 2 plan, written after this plan lands.
- Type consistency: `best_unweighted` dict shape consistent between T3's trainer and
  train.py consumer; `set_epoch` name consistent T1 trainer/sampler; metric keys in
  T2/T5 all appear in T14's frozen set; `per_base_onsite` consistent T12
  model/CLI/CONFIG_DEFINING_ARGS.
- Placeholder scan: clean (every test and implementation is concrete code; the two
  read-before-writing notes in T7/T10 name exactly which lines to read and why).
