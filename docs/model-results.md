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
