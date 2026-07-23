# Fray Probe: terminal-destacking sensitivity of the geometry-ON model (Stage 1)

**Date:** 2026-07-22
**Branch:** `x3dna-edge-geometry`
**Primary files:** new `scripts/fray_probe.py`, new `FrayProbeJob` (sbatch)

## Goal

Measure how the geometry-ON Hamiltonian GNN's predicted **terminal stacking coupling**
responds when we destack (fray) the 3' terminal base of the primary strand, and show how
far that morph sits outside the training geometry distribution. Pure model inference, no
DFT. This is Stage 1 of a two-stage experiment; Stage 2 (DFT on rebuilt coordinates) is a
separate spec, gated on what Stage 1 shows.

## Framing (why this is diagnostic, not confirmatory)

The model was trained on geometrically-uniform data (idealized fiber B-DNA), where the
backbone geometry barely varied (rise std ~0.005 A, distance std ~0.27 A from the cache).
So the model had almost no signal teaching it "destacking weakens the coupling." The
expected result is that the predicted coupling is nearly flat across the sweep (or
extrapolates arbitrarily once out-of-distribution), NOT that it decays like real physics.
Quantifying that flatness/gap is the deliverable: it is the concrete, honest evidence for
why varied-geometry training data is needed. Success = a clear read on the model's geometry
sensitivity, whatever it is.

## What is morphed (and what is not)

- **Topology is never changed.** Every node (bases + 2 contacts) and every edge (backbone,
  H-bond, contact) stays present with mask unchanged. No bond/edge is removed.
- **Only one edge's geometry is swept:** the terminal primary backbone (stacking) edge
  between the last two primary bases, positions `N-2` and `N-1`. Its 7-tuple is
  `[d, shift, slide, rise, tilt, roll, twist]`; we raise slot 0 (`d`) and slot 3 (`rise`)
  **together** by a shared destack amount `delta`, holding shift/slide/tilt/roll/twist and
  all other edges fixed. Moving `d` and `rise` together corresponds to a physical axial
  destacking motion (not independent number-fiddling).
- **Physical bound.** `delta` is capped so `d` stays at or below the extended-backbone limit
  (order ~6-7 A center-to-center; the exact ceiling is set from the real C1'-C1' extended
  geometry at implementation, not eyeballed). Beyond that a covalent backbone bond would have
  to break -- out of scope for a physical fray.
- **Honest limitation:** other edges (e.g. the terminal H-bond edge) are held fixed, so each
  sweep point is a controlled single-edge perturbation, not a fully re-relaxed 3D structure.
  This is the standard way to probe a model's response to one input; fully-relaxed physical
  coordinates are built in Stage 2 for DFT.

## Readout

The number watched is the terminal stacking coupling
`t_term = | model.H[i_term, i_neighbor] |`, where for `n_orb=1` the primary base at position
`k` maps to Hamiltonian local index `k`, so `i_term = N-1`, `i_neighbor = N-2`. `model.H` is
the DNA Hamiltonian exposed by `DNATransportHamiltonianGNN.forward` (`self.H`), taken for the
single-graph batch (`model.H[0]`).

## Design

`scripts/fray_probe.py`:

1. Load the geometry-ON model:
   `load_trained_model('outputs_pickle_gat_geom/hamiltonian_pickle_model.pth')`. This model
   was saved with `use_geometry=True` in its args, so the loader reconstructs it with the
   geometry encoder + norm buffers (the reload path added in Plan 2 Task 4).
2. Load the geometry cache `geom_cache/geometry.pkl`.
3. Choose 3-4 training sequences spanning terminal step types (e.g. a purine-purine, a
   pyrimidine-pyrimidine, and mixed terminal steps); exact sequences selected from the cache
   keys at implementation. For each sequence:
   a. Build the geometry-ON graph: `sequence_to_graph(primary, comp, geometry=cache[seq])`
      with default contacts (probe looks at an internal coupling, not transport).
   b. Locate the two directed terminal primary backbone edges (endpoints = primary nodes
      `N-2`, `N-1`) and record their row indices in `edge_geom`; record `d0 = edge_geom[row,0]`,
      `r0 = edge_geom[row,3]`.
   c. Sweep `delta` over `~25` points from 0 to `delta_max` (so `d` reaches the physical
      ceiling). At each point: set `edge_geom[rows,0] = d0 + delta`, `edge_geom[rows,3] =
      r0 + delta` on both directed copies; forward the model; record `t_term`.
4. Record the in-distribution band for the backbone `d` and `rise` columns (mean +/- 3*std
   from `compute_norm_stats`), so the plot can shade where in-distribution ends.
5. Write raw sweep data to `outputs_fray/fray_sweep.csv` (columns: seq, delta, d, rise,
   t_term) and the norm band to `outputs_fray/norm_band.json`.

Plotting (styled with the dataviz skill): `outputs_fray/fray_sweep.png` -- x = destack
distance `d`, y = `t_term`, one line per sequence, in-distribution `d` band shaded, an
annotation of the physical expectation (coupling should decay toward 0). Built from the CSV.

`FrayProbeJob` (sbatch, g3nat env, GPU or CPU, short): runs `scripts/fray_probe.py`.

## Error handling / edge cases

- Sequence missing from the cache, or its terminal step masked (geometry absent): skip with a
  warning; the probe needs a real terminal-edge geometry to perturb.
- `n_orb != 1`: assert `n_orb == 1` (the trained model uses it); the index mapping above
  assumes it.
- If `model.H` is not populated, call `forward` once first (it sets `self.H`).

## Testing

- Unit: on a tiny hand-built graph + a small `use_geometry=True` model, `fray_probe`'s
  edge-location helper returns the correct terminal backbone rows, and raising `delta`
  monotonically changes `edge_geom[row,0]` -- i.e. the perturbation lands on the right edge.
- Sanity (in the job log): at `delta=0` the recovered `t_term` equals the model's coupling
  on the unmorphed graph (the sweep starts from the real value).

## Non-goals

- No DFT (Stage 2, separate spec).
- No coordinate rebuilding or full structure relaxation in Stage 1 (bounded param sweep only).
- No claim that the model *should* track fraying -- the point is to measure whether/how much
  it does.
- No change to the model, the graph builder, or any Plan 2 code -- this is a read-only probe
  script on top of the shipped feature.

## Run

`sbatch FrayProbeJob`, then read `slurm-<jobid>.out` and `outputs_fray/`. `outputs_fray/` is
a new output dir (add to `.gitignore` alongside the other `outputs_*`).
