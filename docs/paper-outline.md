# Paper outline (working) -- for venue matching

**Working title:** Learning reduced tight-binding Hamiltonians for nanoscale charge
transport with a differentiable-NEGF graph neural network

**One-line contribution:** A graph neural network that constructs a compact, interpretable
effective tight-binding Hamiltonian (one site per base) from molecular sequence/structure and
solves it through a differentiable NEGF layer, trained end-to-end to reproduce DFT charge-
transport observables (DOS, transmission). This is a physics-based dimension reduction
(downfolding) of a full all-atom DFT electronic structure into a small, physical model.

## What kind of paper this is (honest scope)

- **Primary novelty:** the *physics-based Hamiltonian reduction* -- using a differentiable
  NEGF physics layer as the training bridge so the learned object is a real Hermitian
  Hamiltonian you can extract and interpret, not a black-box regressor. Transport is the
  observable we fit, not the point.
- **ML character:** APPLIED, not groundbreaking ML machinery. The conv backbone (GAT/
  Transformer message passing) is standard; the novelty is the physics-integrated
  architecture (GNN -> Hermitian H -> differentiable NEGF -> spectra) and the reduction
  framing. So it is NOT a fit for flagship ML venues that reward new ML methods.
- **Size/impact:** one method + one demonstration system (DNA), a moderate DFT dataset,
  solid-but-not-SOTA-shattering accuracy, with interpretability as the selling point. A
  focused methods paper. Realistically mid-tier Q1 applied computational science -- NOT a
  flagship (Phys Rev B / NeurIPS-main / npj-flagship) scale.

## Method

1. Sequence/structure -> graph: nodes = bases (+2 contact nodes), typed edges (backbone/
   stacking, hydrogen-bond, contact).
2. GNN maps nodes -> onsite blocks and edges -> coupling blocks; assembles a Hermitian
   tight-binding Hamiltonian H (dimension = number of base sites). Base-aware couplings
   (coupling depends on the two endpoint bases). Size-agnostic (variable-length sequences).
3. Differentiable NEGF layer: wide-band contact self-energies -> retarded Green's function ->
   DOS and transmission over an energy grid (log10-safe outputs). Two solver variants.
4. Trained end-to-end to match DFT-derived DOS/transmission.

## Data

~515 DNA duplexes built as idealized fiber B-DNA (NAB) -> single-point DFT (Gaussian) ->
orthogonalized Fock/overlap -> ballistic NEGF ground-truth DOS/transmission; 2058 samples
(4 contact-type/coupling variants). Physical tight-binding priors from the literature
(onsite: Roche 2003; couplings: Voityuk 2001).

## Results

- DFT-fit accuracy: val ~0.55 on log10(DOS)+log10(transmission) MSE (best model: GAT conv +
  base-aware coupling).
- Ablations: GAT vs Transformer conv (dataset-dependent), base-aware vs base-blind coupling.
- The learned effective H is extractable and comparable to literature tight-binding
  parameters (interpretability).

## Discussion / limitations

What is reduced (all-atom DFT electronic structure -> per-base tight-binding) and what is
preserved (transport spectra); interpretability of the learned parameters; limitation that
current training structures are geometrically uniform (idealized fiber models).

## Future work

An SE(3)-invariant geometry channel (X3DNA edge features) is built and wired but needs
structures with real conformational variation (MD-minimized / crystal / predicted); on
uniform-geometry training the geometry channel extrapolates poorly out-of-distribution.

## Constraints for venue matching

- Tier: Q1 but MID-TIER (not flagship). Applied ML for physical/chemical science / computational
  methods. Interpretable-physics angle.
- Cost: free-to-publish strongly preferred (subscription route with no APC, or diamond OA).
  Note APC if the scope fit is excellent.
- Audience: computational chemistry / condensed-matter / ML-for-science readers who will grasp
  "differentiable-NEGF effective-Hamiltonian downfolding."
