# Invariant Geometry on GNN Edges (X3DNA + atom distances)

**Date:** 2026-07-20
**Branch:** `x3dna-edge-geometry`
**Primary files:** `g3nat/graph/construction.py`, `g3nat/models/hamiltonian.py`, new `g3nat/graph/geometry.py`

## Summary

Attach an SE(3)-invariant description of the local geometry to the graph's intra-molecular
edges as a separate geometry channel, gated by a `use_geometry` toggle. Every edge carries
the same 7-number schema: one atom-based center-to-center distance plus the X3DNA relative
rigid-body transform (3 translations + 3 rotations, propeller included). The schema is
uniform across edge types (and future base-metal edges), so a single encoder handles all of
them. Geometry is extracted once (offline) from existing PDB structures (X3DNA params via
DSSR; centroid distances directly from coordinates), cached, and read at graph-build time.
When the toggle is off, the model is byte-for-byte identical to the current one.

This branch is **infrastructure plus provable correctness**, not a results-producing
experiment. See "Key finding" for why.

## Motivation

Today the electronic coupling between stacked bases is set purely by base identity. The
synthetic tight-binding generator makes this explicit: `nn_energies`
(`g3nat/utils/physics.py:52-69`) assigns a fixed stacking coupling per base-step (e.g. GC
is always 0.110 eV) with no conformational dependence. Physically, the stacking coupling
depends on the stacking geometry (twist, rise, slide, roll). Encoding geometry on edges
lets the GNN learn `coupling = f(bases, geometry)` instead of `f(bases)` alone. Keeping all
geometry on edges (never on nodes) is a deliberate invariance choice -- see "Invariance".

## Key finding (scope-defining)

The training structures are idealized canonical B-DNA fiber models built by NAB
`fd_helix("abdna", seq, "dna")` (`DNADataset/dnabuilder`, `nuc.nab`). The sugar-phosphate
backbone frame is placed identically for every sequence.

Evidence (parsing `gjf_text` coordinates out of the pickles for three distinct length-4
sequences `aaac`, `aaat`, `aaca`): every phosphorus atom sits at bit-identical coordinates
across sequences (e.g. `[4.728, -7.626, 1.239]`, `[8.308, -3.390, 4.619]`). Only the base
atoms differ.

Consequence: geometry computed from these structures (X3DNA params and centroid distances)
is near-constant across the dataset. On this data, geometry is a near-deterministic function
of sequence and largely redundant with the base identity the model already has on its nodes.
A training run cannot demonstrate the model *uses* geometry, because there is no geometric
variation to respond to.

Therefore the geometry signal must come from varied geometry later (MD / crystal /
sequence-dependent predicted structures / intercalated species). This branch builds the
architecture and proves it is correctly wired; it does not attempt a "geometry helps" metric
on the current data.

## Invariance requirement (the featurization criterion)

The prediction targets (transmission, DOS) are invariant under global rigid-body motion of
the molecule, so every geometric feature must be invariant under global rotation and
translation. Two consequences shape the whole design:

1. **All geometry lives on edges, never on nodes.** A single base frame has no
   rotation-invariant orientation content -- rotate the molecule and the frame rotates, so
   any function of one frame alone either changes or is constant. Invariant orientation is
   irreducibly *relational* (between two frames). So "angles across complementary bases"
   (propeller, buckle, opening) can only be an edge feature, not a per-base node feature.
2. **The invariant quantities are relative transforms + distances.** X3DNA parameters are
   relative frame-to-frame transforms (each base carries a local frame; the params are the
   transform between two frames), and a Euclidean distance is invariant by definition. Under
   a global rotation R and translation t every frame maps as `origin -> R.origin + t`,
   `axes -> R.axes`, so relative transforms and distances are unchanged.

Verified empirically on the actual tool and data: a random proper rotation + translation
(det = +1) applied to `aaac.pdb` leaves DSSR params unchanged to within ~0.01-0.02 (rounding
noise from writing rotated coordinates back at PDB's 3-decimal precision); centroid distances
are exactly unchanged. This becomes a regression test.

### Two discrete conventions that are NOT global SE(3) (handled explicitly)

1. **Edge direction.** The graph builds each backbone edge in both directions (`i->j`
   directionality +1, `j->i` directionality -1). A step read 5'->3' vs 3'->5' flips the sign
   of some X3DNA translations/rotations by convention (the `d_centroid` distance is
   inherently direction-symmetric). Handling: compute each edge's 7-tuple once per undirected
   edge and place identical values on both directed copies; the existing `directionality`
   feature (edge_attr index 3) carries orientation. The Hamiltonian is built from one edge
   per undirected pair and forced Hermitian, so symmetric placement cannot introduce a sign
   bug.
2. **Strand / pair order.** Base-pair params depend on which base is "strand I." DSSR fixes
   this by convention; we inherit it and place symmetric values on both directed H-bond
   edges.

## Constraints

- The working NEGF pipeline must not regress. Contact detection reads `edge_attr[:, 2]`
  (contact one-hot) and contact coupling reads `edge_attr[:, 4]`
  (`g3nat/models/hamiltonian.py:704-707`). These indices and the 5-dim topological edge
  vector must remain untouched. Geometry lives in a separate tensor.
- `use_geometry=False` must reproduce the current model exactly and load existing
  checkpoints.
- Migration must leave a working system at every step.
- Synthetic data carries no geometry (explicit decision): synthetic graphs pass
  `geometry=None` -> zeros + mask 0.

## Design

### Geometry parameterization (uniform 7-slot schema)

Every intra-molecular edge -- and every future base-metal edge -- carries the SAME 7-number
description of the relative geometry between the two frames it connects:

```
[ d_centroid,   t1, t2, t3,    r1, r2, r3 ]
  distance      translations   rotations
```

- **Backbone (stacking) edge**, between base `i` and base `i+1` on one strand:
    - `d_centroid` = distance between the two bases' ring-atom centroids (per strand; ~3.7 A).
    - `t1,t2,t3` = X3DNA step translations `shift, slide, rise`.
    - `r1,r2,r3` = X3DNA step rotations `tilt, roll, twist`.
- **Hydrogen-bond (pair) edge**, between base `i` and its Watson-Crick partner:
    - `d_centroid` = distance between the two paired bases' ring-atom centroids (~6.0 A).
    - `t1,t2,t3` = X3DNA base-pair translations `shear, stretch, stagger`.
    - `r1,r2,r3` = X3DNA base-pair rotations `buckle, propeller (r2), opening`.
- **(future) base-metal edge**: `d_centroid` = metal-to-base-centroid distance; `t,r` = the
  relative transform between the metal frame and the base frame. Same 7 slots.

The slots are **semantically aligned** across edge types: slot 0 is always an atom-centroid
distance, slots 1-3 are always translations, slots 4-6 always rotations. One shared encoder
can therefore process all edge types; the topological edge-type one-hot (already in
`edge_attr[:,0:2]`) plus per-edge-type normalization handle the different numeric regimes.

**Why an atom distance, not a frame-origin distance -- and why this is one more than the
minimal 6.** Complementary and neighboring edges are the same *type* of object (an invariant
relative frame transform) but live in very different regimes. Measured on `aaac.pdb`:

| relationship         | base-centroid | C1'-C1' | frame-origin `sqrt(t1^2+t2^2+t3^2)` |
|----------------------|---------------|---------|-------------------------------------|
| complementary (pair) | ~6.0 A        | 10.70 A | **0.09 A  (degenerate)**            |
| neighboring (stack)  | 3.72 A        | 4.93 A  | ~3.4 A  (fine)                      |

X3DNA places the two base frames of an ideal Watson-Crick pair with nearly coincident origins
(shear/stretch/stagger ~ 0), related by a ~180-degree flip. So a frame-origin distance for
the pair edge is a degenerate ~0.09 A -- useless as "the characteristic distance between
H-bonded bases." The meaningful pairing distance is atom-based (centroid ~6.0 A / C1'-C1'
10.7 A). The stacking edge does not have this problem, but for a uniform, comparable schema we
use the atom-centroid distance for both.

So the 7-tuple is the reconstruction-complete X3DNA 6-tuple PLUS one atom-centroid distance --
one redundant coordinate per edge (given base identity, the centroid distance is largely
determined by the transform). Accepted deliberately: a strictly minimal distance-explicit form
is degenerate for the pairing regime, and the redundancy buys a non-degenerate,
directly-comparable, metal-ready schema. Minimality was a simplicity preference, not a hard
constraint, and the uniform single-tensor schema is simpler in practice.

All seven numbers are SE(3)-invariant (Euclidean distance + relative frame transforms).

### Edge-type to relationship mapping

- Pair `k` (bp params + pair centroid distance) -> the **hydrogen-bond edge** for pair `k`.
- Step `k` (step params) -> **both backbone edges** at that level (primary `i->i+1` and its
  complementary partner). The X3DNA `t/r` are shared per level, but each backbone edge's
  `d_centroid` is its own per-strand base-to-base centroid distance -- so the two backbone
  edges differ in slot 0 (strand-specific stacking) while sharing slots 1-6. Strand-specific
  stacking signal for free.
- Contact edges and single-stranded / overhang edges: zeros, mask 0.

Base-pair index `k` corresponds to primary-strand position `k`. DSSR pairs residue `i` of
strand 1 with residue `(N-i+1)` of strand 2, matching the graph's H-bond construction
(`construction.py:224-238`).

### New tensors on the graph

- `edge_geom`: `[num_edges, 7]`, float. Slots as above (edge-type-dependent meaning, aligned
  by kind: distance / translations / rotations).
- `edge_geom_mask`: `[num_edges, 1]`, float in {0, 1}. 1 where geometry is present.

Both are edge-aligned (first dim = num_edges), so PyTorch Geometric concatenates them
correctly under batching with `edge_index` (default `__cat_dim__` = 0).

### Normalization (per edge type)

Separate `(mean, std)` over the 7 columns for backbone edges and for H-bond edges (and later
base-metal edges), computed once over the cache and stored with the checkpoint. Per-type is
**required** because the regimes differ (stacking distance ~3.7 vs pairing ~6.0; twist ~36 vs
opening ~0); a global z-score would blur them. Masked entries are excluded from the stats and
set to 0 after normalization. Angles are in degrees; none wrap near +/-180 for B-form, so
plain z-scoring is safe (switch a column to a sin/cos pair only if varied geometry later
pushes it toward the wrap boundary).

### Model integration (shared encoder + edge-type signal)

In `DNATransportHamiltonianGNN`:

- Convolution default: `conv_type='gat'`. GAT is the best DFT-fitting convolution on record
  (val 0.547 vs transformer 1.42 on the pickle data; see `docs/model-results.md`). Geometry
  work runs on DFT data, so Plan 2 trains with GAT unless explicitly overridden. (`train.py`
  default was flipped to `gat` accordingly.)
- New constructor args: `use_geometry: bool = False`, `geom_dim: int = 7`.
- New `geom_encoder = nn.Linear(geom_dim, hidden_dim)` shared across edge types (2-layer MLP
  optional; start linear), final layer initialized near zero.
- In `forward`, after `edge_attr = self.edge_proj(edge_attr)`:
  ```
  if self.use_geometry:
      g = self.geom_encoder(data.edge_geom) * data.edge_geom_mask
      edge_attr = edge_attr + g
  ```
  The edge-type one-hot enters through `edge_proj(edge_attr)` (the topological 5-dim vector),
  so the summed edge embedding carries both per-type geometry and type identity, and the
  encoder can specialize per type through that additive signal. (If specialization proves too
  weak, concatenate the 2-dim edge-type one-hot into the encoder input -> `Linear(9,
  hidden_dim)`; noted, not needed to start.)
- Geometry reaches both message passing and, via the post-conv edge embeddings,
  `coupling_proj` (the edge -> Hamiltonian-block head), so learned coupling vs. geometry stays
  inspectable.
- Init consistent with the existing near-zero init of `coupling_proj` / `onsite_proj`
  (`hamiltonian.py:91-94`).
- Persist `use_geometry` and the per-type normalization stats in the checkpoint dict.

`use_geometry=False` skips the block entirely -> identical to current forward pass.
`standard.py` receives the same flag for parity (lower priority; can land last).

### Offline preprocessing (`g3nat/graph/geometry.py`)

- `run_dssr(pdb_path, dssr_bin) -> out_text`: subprocess call `dssr_bin -i=<pdb> --more
  -o=<out>`. `dssr_bin` resolves from arg, then `$X3DNA_DSSR`, then a default path.
- `parse_dssr_out(out_text) -> {"bp_pars": [Npair,6], "step_pars": [Nstep,6]}`: regex on
  `bp-pars:` (Shear/Stretch/Stagger/Buckle/Propeller/Opening) and `step-pars:`
  (Shift/Slide/Rise/Tilt/Roll/Twist) lines. DSSR prints step-pars in more than one section;
  the parser targets the base-pair step-parameters block and de-duplicates so exactly `Nstep`
  rows are returned. Developed against the existing `aaac.out`.
- `base_centroids(pdb_path) -> {residue_key: xyz}`: ring-atom (non-backbone, non-hydrogen)
  centroid per residue, straight from the PDB coordinates. From these, `d_centroid` per edge
  (backbone: consecutive same-strand bases; H-bond: WC partners). No DSSR needed for
  distances.
- `assemble_edge_geometry(...) -> per-edge 7-tuples` for backbone and H-bond edges (X3DNA
  params + the matching centroid distance in slot 0).
- `build_geometry_cache(dataset_dir, out_path)`: iterate `<seq>/<seq>.pdb`, run + parse +
  assemble, store per-sequence edge geometry keyed by edge identity. Missing/failed structures
  recorded as absent (a warning), not fatal.
- `compute_norm_stats(cache) -> {"backbone": {"mean":[7],"std":[7]}, "hbond": {...}}`.

### Data-source decision

Real params come from the existing PDBs at
`/mmfs1/gscratch/anantram/asyed4/DNADataSet/<seq>/<seq>.pdb` (verified: 1165 dirs, full
coverage of the 515 training sequences; residues `DA/DC/DG/DT`, TER-separated strands, DSSR
runs cleanly). `gjf_text` in the pickles is elements + coordinates only (no atom/residue
names) so DSSR cannot identify bases from it; the PDBs are required. The parsed cache is the
durable artifact -- training reads only the cache, so the external directory is a one-time
preprocessing dependency.

Cache location: `geom_cache/` in the repo root, keyed by sequence.

## Error handling and edge cases

- Missing PDB or DSSR failure for a sequence: geometry treated as absent for that sequence
  (all its edges masked 0), warn, continue. System still trains.
- Unpaired / overhang bases: their edges masked 0.
- DSSR binary absent: hard error only in the preprocessing step, never at training time.
- Length / count mismatch between cached geometry and graph edges: assertion in
  `sequence_to_graph` with the sequence name, so a bad cache entry fails loudly at build time
  rather than silently misaligning.

## Testing

- **Invariance regression:** rotate+translate a structure; DSSR params unchanged within
  tolerance and centroid distances exactly unchanged. (Encodes the verification already run.)
- **Functional (wiring proof):** one graph, two different `edge_geom` values on one edge,
  `use_geometry=True` -> the predicted transmission/DOS differ. Proves geometry flows to the
  output. Primary "it works" test given the uniform-data limitation.
- **Backward compatibility:** `use_geometry=False` reproduces current outputs bit-for-bit on a
  fixed seed; an existing checkpoint loads and runs.
- **Parser unit:** `parse_dssr_out(aaac.out)` returns the expected 4 bp rows and 3 step rows
  with the known values.
- **Geometry-assembly unit:** for a small sequence, backbone edges get `[per-strand centroid
  dist, shift, slide, rise, tilt, roll, twist]`; H-bond edges get `[pair centroid dist, shear,
  stretch, stagger, buckle, propeller, opening]`; contacts masked; slot semantics correct.
- **Degeneracy regression:** assert the H-bond `d_centroid` is the atom-centroid distance
  (~6 A), NOT the degenerate frame-origin distance (~0.09 A) -- guards against reintroducing
  the frame-origin bug.

## Migration sequence (working system at every step)

1. Add `g3nat/graph/geometry.py` (DSSR runner + parser + centroid distances + assembly) and
   its unit tests. No pipeline change.
2. Build the geometry cache offline. Artifact only.
3. Add `edge_geom` / `edge_geom_mask` to `sequence_to_graph`, defaulting to zeros / mask 0
   when no geometry is supplied. Existing behavior unchanged (extra ignored tensors).
4. Add `use_geometry` (default False) + `geom_encoder` to `DNATransportHamiltonianGNN`; fuse
   only when True. Default off = identical model. Backward-compat test.
5. Thread the cache through `datasets.py` / `pickle.py`; add `--use_geometry` / `--geom_cache`
   to `train.py` and `config.py`. Flag off = unchanged. Synthetic path passes `geometry=None`.
6. Turn the flag on for a run; functional + invariance tests confirm wiring.
7. (Optional, last) mirror the flag into `standard.py`.

## Non-goals

- No synthetic geometry (explicit decision): the tight-binding generator stays sequence-only.
- No "geometry helps" metric on the current geometrically-uniform data.
- No varied-geometry data generation (MD / crystal / predicted) -- future work this branch
  prepares for.
- No intercalated-metal nodes/edges yet: the uniform 7-slot schema is designed so a base-metal
  edge slots in later as another edge type (its own normalization block), but building metal
  nodes/edges is a separate future project.
- Helical parameters (heli-pars) are not included: invariant but redundant with step params
  (heli-twist 36.00 tracks step-twist 35.90). Trivial to add later.
- Per-base node geometry is intentionally excluded (breaks invariance -- see "Invariance").
  Per-base *internal torsions* (chi, sugar pucker) ARE invariant node features and are the
  natural node-side extension if ever needed, but are out of scope here.

## Open questions

- Cache format: a single pickle/npz vs. per-sequence files. Proposed: one npz/pickle keyed by
  sequence for simplicity; revisit if it grows.
- Distance definition: base ring-atom centroid (chosen, uniform across edge types) vs. C1'-C1'
  (more standard for pairs). Proposed: centroid; easy to switch.
- Encoder: shared `Linear(7, hidden)` with edge-type via the additive topological channel
  (chosen) vs. concatenating the edge-type one-hot into the encoder input `Linear(9, hidden)`
  vs. a 2-layer MLP. Proposed: start with the simplest; upgrade only if needed.

## Research appendix

- X3DNA / DSSR: Lu, X.-J. and Olson, W. K., "3DNA: a software package for the analysis,
  rebuilding and visualization of three-dimensional nucleic acid structures", Nucleic Acids
  Research 31(17):5108-5121 (2003). DSSR: Lu, Bussemaker, Olson, NAR 43(21):e142 (2015). Tool
  in use: DSSR v2.8.0 (2026feb18), installed at `/mmfs1/gscratch/anantram/asyed4/x3dna-dssr`;
  AmberTools 24.7 `cpptraj nastruct` (3DNA conventions) available as a cross-check.
- Base-pair / step / helical parameter conventions and the standard base reference frame
  (which places ideal WC base origins nearly coincident): the Tsukuba (2001) EMBO
  recommendation on nucleic-acid base-pair geometry nomenclature.
- Tight-binding parameters currently in the codebase: onsite energies from Roche et al., PRL
  91, 228101 (2003); H-bond and stacking couplings from Voityuk et al., J. Chem. Phys. 114,
  5614 (2001) (`g3nat/utils/physics.py:24-69`).
- Empirical checks performed for this spec (2026-07-20): (a) phosphorus coordinates
  bit-identical across sequences `aaac`/`aaat`/`aaca` (uniform helix); (b) DSSR params
  invariant under a random SE(3) transform of `aaac.pdb` to within rounding noise; (c) on
  `aaac.pdb`, complementary base-centroid ~6.0 A / C1'-C1' 10.70 A vs. frame-origin distance
  0.09 A (degenerate), and neighboring base-centroid 3.72 A -- motivating the atom-based
  distance.
