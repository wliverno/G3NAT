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
