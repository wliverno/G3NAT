# Fray Probe: whole-Hamiltonian response to terminal destacking (Stage 1)

**Date:** 2026-07-22
**Branch:** `x3dna-edge-geometry`
**Primary files:** new `scripts/fray_probe.py`, new `FrayProbeJob` (sbatch)

## Goal

Perturb the geometry of ONE edge (destack the 3' terminal base of the primary strand) and
watch how the **entire predicted Hamiltonian** `model.H` responds -- not just the terminal
coupling. We do not know what the geometry-ON model learned, so we make no assumption about
where the change shows up: the deliverable is a map of *which part of the Hamiltonian moves
the most*, including any non-local ("other side of the bases") or cross-strand effects. Pure
model inference, no DFT. Stage 2 (DFT on rebuilt coordinates) is a separate spec.

## Framing (open-ended, not confirmatory)

The model was trained on geometrically-uniform data (fiber B-DNA; backbone rise std ~0.005 A,
distance std ~0.27 A). So it had almost no signal about how geometry maps to the Hamiltonian,
and its learned response is unknown. A local edge morph passes through the GNN's message
passing and could move onsite energies or couplings anywhere. We therefore observe the full
`|H(morphed) - H(unmorphed)|`, deliberately sweeping into deep out-of-distribution geometry,
and report the actual response pattern -- flat, localized, delocalized, or "something crazy."
The training structures were idealized textbook models, so physical realism of the morph is
explicitly NOT a requirement; mapping the model's response function is.

## What is morphed (and what is not)

- **Topology is never changed.** Every node (bases + 2 contacts) and every edge (backbone,
  H-bond, contact) stays present with mask unchanged. No bond/edge is removed.
- **One edge's geometry is swept:** the terminal primary backbone (stacking) edge between the
  last two primary bases, positions `N-2` and `N-1`. Its 7-tuple is
  `[d, shift, slide, rise, tilt, roll, twist]`; we raise slot 0 (`d`) and slot 3 (`rise`)
  together by a shared destack amount `delta`, holding the other five slots and all other
  edges fixed. This isolates cause (one edge) so any change elsewhere is a propagated effect.
- **Sweep range:** `delta` from 0 to ~5 A (so `d` runs ~3.9 -> ~9 A), ~35 points -- wide on
  purpose, spanning in-distribution to deep out-of-distribution. No physical cap (textbook
  structures; see Framing).
- Both directed copies of the terminal backbone edge get the same values.

## Readouts (whole-Hamiltonian, not just one element)

Let `H0 = model.H[0]` for the unmorphed graph and `Hd = model.H[0]` for each morphed graph
(`M x M`, `M = num_dna_nodes`, `n_orb=1`). `D(delta) = Hd - H0`. We report:

1. **Response heatmaps:** `|D(delta)|` as an `M x M` heatmap at a few delta values (a small
   in-distribution one and large out-of-distribution ones). Axes labeled by
   (strand, position, base): primary `b0..b_{N-1}` at indices `0..N-1`, complementary at
   `N..2N-1`. This is the primary "which part changes most" visual.
2. **Argmax tracking:** for each delta, the location `(i,j)` of `max|D|`, and whether it stays
   at the terminal stacking element or moves (does the hot spot migrate to the 5' end / the
   complementary strand?).
3. **Region decomposition vs delta:** the summed `|D|` over disjoint regions, so we can see
   where the response concentrates:
   - diagonal (onsite energies) vs off-diagonal (couplings)
   - terminal-local (any element touching base `N-1` or `N-2`) vs distal (everything else)
   - primary-primary vs complementary-complementary vs cross-strand blocks
   - plus the total `||D||_F` (Frobenius).
4. **Top-K changed elements** at max delta: a table of the largest `|D_ij|` with location, the
   two bases involved, and coupling type (onsite / stacking / H-bond / cross-strand).
5. **Reference curve:** the terminal stacking coupling `|Hd[N-1, N-2]|` vs `d` (the element we
   naively expected to move), plotted with the in-distribution `d` band shaded -- kept as one
   line, not the whole story.

Primary-base position `k` maps to Hamiltonian index `k` (`n_orb=1`), so terminal base = `N-1`,
its stacked neighbor = `N-2`. `model.H` is `self.H` set in `DNATransportHamiltonianGNN.forward`.

## Design

`scripts/fray_probe.py`:

1. Load the geometry-ON model:
   `load_trained_model('outputs_pickle_gat_geom/hamiltonian_pickle_model.pth')` (saved with
   `use_geometry=True`; the loader reconstructs the geometry encoder + norm buffers via the
   Plan 2 reload path).
2. Load `geom_cache/geometry.pkl`.
3. Choose 3-4 training sequences spanning terminal step types (a purine-purine, a
   pyrimidine-pyrimidine, and mixed terminal steps); exact sequences chosen from the cache
   keys at implementation. For each sequence:
   a. Build the geometry-ON graph: `sequence_to_graph(primary, comp, geometry=cache[seq])`
      with default contacts.
   b. Locate the two directed terminal primary backbone edges (endpoints = primary nodes
      `N-2`, `N-1`); record their `edge_geom` rows and `d0`, `r0`.
   c. `H0` = forward on the unmorphed graph.
   d. Sweep `delta` (35 points, 0 to ~5 A): set `edge_geom[rows,0]=d0+delta`,
      `edge_geom[rows,3]=r0+delta`; forward; store `Hd` and `D=Hd-H0`; record all readouts
      above.
4. Record the in-distribution band (mean +/- 3*std of backbone `d`, `rise`) from
   `compute_norm_stats`.
5. Write raw outputs to `outputs_fray/`: `sweep_metrics.csv` (per seq, delta: d, rise,
   term_coupling, argmax_i, argmax_j, ||D||_F, region sums), `Hmats.npz` (H0 and selected Hd
   per seq, for the heatmaps), `norm_band.json`.

Plots (dataviz skill), into `outputs_fray/`:
- `response_heatmaps.png`: `|D|` heatmaps per sequence at chosen deltas, axis-labeled by base.
- `region_curves.png`: region-decomposition sums + `||D||_F` vs `d`, in-dist band shaded.
- `terminal_coupling.png`: the reference terminal-coupling curve vs `d`.

`FrayProbeJob` (sbatch, g3nat env, short; GPU or CPU): runs `scripts/fray_probe.py`.

## Error handling / edge cases

- Sequence missing from cache, or terminal step masked: skip with a warning.
- Assert `n_orb == 1` (the trained model uses it; the index mapping assumes it).
- Ensure `model.H` is populated by calling `forward` before reading it.
- Heatmap labels handle the two strands; single-strand inputs are out of scope (dataset is
  duplex).

## Testing

- Unit: on a tiny hand-built graph + a small `use_geometry=True` model, the edge-location
  helper returns the correct terminal backbone rows, and raising `delta` changes only
  `edge_geom[row,0]` / `[row,3]` on those rows (perturbation lands on the intended edge).
- Sanity (job log): at `delta=0`, `D` is exactly zero (sweep starts from the unmorphed H).

## Non-goals

- No DFT (Stage 2, separate spec).
- No coordinate rebuilding / structure relaxation and no physical bound on the morph
  (textbook structures; we are mapping the model's response function).
- No claim the model *should* respond in any particular way -- we report what it does.
- No change to the model, graph builder, or any Plan 2 code -- read-only probe on top of the
  shipped feature.

## Run

`sbatch FrayProbeJob`, then read `slurm-<jobid>.out` and `outputs_fray/`. Add `outputs_fray/`
to `.gitignore` alongside the other `outputs_*`.
