# References

Verified bibliographic records for everything this project depends on. **Rule: nothing goes
in this file unless it was checked against a real source (publisher page, DOI resolver,
arXiv, or the software's own citation page).** No reconstructing citations from memory --
a wrong citation is worse than no citation.

Last verification pass: 2026-07-24.

## Tight-binding parameters

Both are already used in `g3nat/utils/physics.py` and were carried in the source as DOIs.

- **Onsite energies** (A -0.49, T -1.39, G 0.00, C -1.12 eV) -- `physics.py:24`
  Roche, S. et al. *Phys. Rev. Lett.* **91**, 228101 (2003).
  doi:[10.1103/PhysRevLett.91.228101](https://doi.org/10.1103/PhysRevLett.91.228101)

- **Hydrogen-bond and nearest-neighbour stacking couplings** -- `physics.py:33`
  Voityuk, A. A. et al. *J. Chem. Phys.* **114** (2001).
  doi:[10.1063/1.1352035](https://doi.org/10.1063/1.1352035)

## Transport formalism

- **NEGF for molecular/nanoscale electronics**
  Datta, S. *Quantum Transport: Atom to Transistor.* Cambridge University Press (2005).
  ISBN 978-0-521-63145-7. doi:[10.1017/CBO9781139164313](https://doi.org/10.1017/CBO9781139164313)
  (Distinct from Datta's earlier and more general *Electronic Transport in Mesoscopic
  Systems*, 1995.)

## Model architecture

- **Graph attention (GATConv)** -- the default `--conv_type gat`
  Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y.
  "Graph Attention Networks." *ICLR* (2018). arXiv:[1710.10903](https://arxiv.org/abs/1710.10903)

- **TransformerConv** -- the `--conv_type transformer` alternative; this is the paper
  PyTorch Geometric cites for that layer
  Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., Sun, Y. "Masked Label Prediction:
  Unified Message Passing Model for Semi-Supervised Classification." *IJCAI-21*, 1548-1554
  (2021). doi:[10.24963/ijcai.2021/214](https://doi.org/10.24963/ijcai.2021/214),
  arXiv:[2009.03509](https://arxiv.org/abs/2009.03509)

- **Sequence generator** -- already cited in `g3nat/models/generator.py:6`
  Linder, J. & Seelig, G. "Fast activation maximization for molecular sequence design."
  *BMC Bioinformatics* **22**, 510 (2021).
  doi:[10.1186/s12859-021-04437-5](https://doi.org/10.1186/s12859-021-04437-5)

## Structure generation and analysis

- **AmberTools** -- the environment used for structure building and cross-checks
  Case, D. A. et al. "AmberTools." *J. Chem. Inf. Model.* **63**(20), 6183-6191 (2023).
  doi:[10.1021/acs.jcim.3c01153](https://doi.org/10.1021/acs.jcim.3c01153)

- **NAB (Nucleic Acid Builder), containing `fd_helix`** -- how the idealized fiber-helix
  structures in this dataset were built (see `nuc.nab`)
  Macke, T. J. & Case, D. A. "Modeling Unusual Nucleic Acid Structures." In *Molecular
  Modeling of Nucleic Acids*, Leontis, N. B. & SantaLucia, J. Jr., eds., ACS Symposium
  Series **682**, 379-393 (1998).
  doi:[10.1021/bk-1998-0682.ch024](https://doi.org/10.1021/bk-1998-0682.ch024)

- **cpptraj** (used with `nastruct` as the X3DNA cross-check)
  Roe, D. R. & Cheatham, T. E. III. "PTRAJ and CPPTRAJ: Software for Processing and Analysis
  of Molecular Dynamics Trajectory Data." *J. Chem. Theory Comput.* **9**(7), 3084-3095
  (2013). doi:[10.1021/ct400341p](https://doi.org/10.1021/ct400341p)

- **3DNA** -- base-pair and base-step rigid-body parameters
  Lu, X.-J. & Olson, W. K. "3DNA: a software package for the analysis, rebuilding and
  visualization of three-dimensional nucleic acid structures." *Nucleic Acids Research*
  **31**(17), 5108-5121 (2003). doi:[10.1093/nar/gkg680](https://doi.org/10.1093/nar/gkg680)

- **DSSR** -- the specific tool invoked (X3DNA-DSSR v2.8.0)
  Lu, X.-J., Bussemaker, H. J. & Olson, W. K. "DSSR: an integrated software tool for
  dissecting the spatial structure of RNA." *Nucleic Acids Research* **43**(21), e142 (2015).
  doi:[10.1093/nar/gkv716](https://doi.org/10.1093/nar/gkv716)

## Machine learning: training protocol and architecture

Verified 2026-07-25. Each entry is tagged EVIDENCED (empirical result), FOLKLORE (common
practice, no strong evidence found), or NOT EVIDENCED FOR OUR SETTING (real paper, wrong
regime). This project's ML decisions must cite from here rather than be invented.

### Learning-parameters-inside-a-differentiable-solver -- the closest precedent to G3NAT

- **von Strachwitz, Alaa El-Din, Dutra, Vinko. "Data-efficient learning of exchange-correlation
  functionals with differentiable DFT." *Mach. Learn.: Sci. Technol.* **7**(2), 025001 (2026).**
  doi:[10.1088/2632-2153/ae3c5a](https://doi.org/10.1088/2632-2153/ae3c5a)
  **The key reference for our identifiability problem.** A network parameterizes the XC
  functional inside a differentiable Kohn-Sham solver, with loss on downstream observables --
  structurally the same pattern as GNN -> H -> NEGF -> DOS/T. States directly that "with
  insufficient constraints, the optimization problem can become underdetermined, allowing
  multiple distinct solutions to satisfy the training targets." Tested mitigations:
  (1) initialize near a physically-motivated functional (2-15% deviation) so optimization lands
  in the right basin; (2) **add a second, more local/direct loss term alongside the integrated
  one** to break degeneracies; (3) more/more-diverse data; (4) caution that the added term can
  itself induce instabilities if weighted badly.
  **This makes the planned LDOS loss EVIDENCED by close analogy rather than our own idea**, and
  independently motivates willll's `b` weighting.

- **Zhou, Chen, Zhang, Wang, Wang, Guo. "AD-NEGF: An End-to-End Differentiable Quantum Transport
  Simulator." *Phys. Rev. B* **108**, 195143 (2023).** arXiv:[2202.05098](https://arxiv.org/abs/2202.05098)
  Closest full-pipeline precedent: optimizes Hamiltonian parameters against a target
  transmission spectrum through a differentiable NEGF solve. **Does NOT discuss gauge freedom,
  degenerate parameterizations, or ill-conditioning from backpropagating through the Green's
  function inversion.** Their only acknowledged non-uniqueness runs the other way
  (under-parameterization). So our identifiability concern is genuinely under-examined in the
  ML-Hamiltonian+NEGF niche -- a gap, and an opportunity for the paper.

- **Um, Brand, Fei, Holl, Thuerey. "Solver-in-the-Loop." *NeurIPS* (2020).** Canonical reference
  for training through a fixed differentiable solver's loop.

- **Gutenkunst, Waterfall, Casey, Brown, Myers, Sethna. "Universally Sloppy Parameter
  Sensitivities in Systems Biology Models." *PLOS Comput. Biol.* **3**(10):e189 (2007).**
  doi:[10.1371/journal.pcbi.0030189](https://doi.org/10.1371/journal.pcbi.0030189)
  "Sloppy models": Hessian/sensitivity eigenanalysis distinguishes directions the data barely
  constrains from exact symmetries. Usable diagnostic for us -- a TRUE gauge appears as a hard
  zero mode, mere sloppiness as a small-but-nonzero one.

### Optimizer and training protocol

- **Loshchilov & Hutter. "Decoupled Weight Decay Regularization" (AdamW). *ICLR* (2019).**
  arXiv:[1711.05101](https://arxiv.org/abs/1711.05101). EVIDENCED and directly actionable:
  plain `Adam(weight_decay=...)` is NOT true weight decay -- it interacts with Adam's
  per-parameter adaptive rates. **We use `torch.optim.Adam(weight_decay=1e-5)`
  (`trainer.py:40-43`), so our nominal regularization is weaker/different than it appears.**
- **Prechelt. "Early Stopping -- But When?" in *Neural Networks: Tricks of the Trade*, LNCS
  1524, Springer (1998).** EVIDENCED, 1,296 runs over 12 problems x 12 architectures x 14
  criteria. Frames early stopping as regularization alongside weight decay; patient criteria
  buy ~4% generalization for ~4x training cost.
- **Bishop. "Regularization and Complexity Control in Feed-forward Networks." *ICANN* (1995).**
  Relates architecture selection, weight decay, early stopping and training noise as
  complexity-control mechanisms. Notably **prefers explicit regularization when available** --
  supports willll's instinct to fix rather than route around.
- **Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever. "Deep Double Descent." *ICLR* (2020).**
  arXiv:[1912.02292](https://arxiv.org/abs/1912.02292). EVIDENCED: epoch-wise double descent is
  real -- "training longer can correct overfitting." **Falsifies "validation loss must always
  decrease" as a universal law.** Scoped to noisy-label image classification near the
  interpolation threshold; applicability to our regime is NOT VERIFIED either way.
- **Heckel & Yilmaz. "Early Stopping in Deep Networks: Double Descent and How to Eliminate It."
  *ICLR* (2021).** arXiv:[2007.10099](https://arxiv.org/abs/2007.10099). Precedent for
  diagnosing a specific mechanism (per-layer learning-rate mismatch) and removing the
  validation bump, rather than only checkpointing around it.

### Dropout and GNN-specific variants -- evidence leans AGAINST for our setting

- **Singh, Jiang, Paige, Toni. "Effects of Dropout on Performance in Long-range Graph Learning
  Tasks." arXiv:[2502.07364](https://arxiv.org/abs/2502.07364) (2025, preprint).** The most
  directly relevant result. Tests Dropout/DropEdge/DropNode/DropAgg/DropGNN/DropMessage on
  17-39 node graphs, 188-1113 graphs -- our scale. **Insignificant-or-negative in ~62% of
  small-graph classification combinations**, versus ~74% positive on homophilic node
  classification. Test dropout; do not assume it helps.
- **DropEdge** (Rong et al., *ICLR* 2020, arXiv:1907.10903), **DropNode** (Feng et al. GRAND,
  *NeurIPS* 2020), **DropMessage** (Fang et al., *AAAI* 2023, arXiv:2204.10037): all real, all
  validated **only on large single-graph node classification**. NOT EVIDENCED FOR OUR SETTING.
- Dropout in **GAT** (Velickovic et al., *ICLR* 2018) used p=0.6 for small training sets, but on
  citation-graph node classification -- and the paper notes it was unnecessary on the larger
  inductive PPI task. Different regime from ours.

### Depth, oversmoothing, and graph size

- **Alon & Yahav. "On the Bottleneck of GNNs and its Practical Implications." *ICLR* (2021).**
  arXiv:[2006.05205](https://arxiv.org/abs/2006.05205). Depth >= diameter is required to
  exchange information across k hops; also introduces over-squashing, and finds GAT (learned
  aggregation) less bottleneck-prone than GCN/GIN. Directly relevant: our graphs have diameter
  ~2x sequence length, so shallow models under-reach.
- **Gilmer, Schoenholz, Riley, Vinyals, Dahl. "Neural Message Passing for Quantum Chemistry."
  *ICML* (2017).** arXiv:[1704.01212](https://arxiv.org/abs/1704.01212). Hyperparameter search
  on QM9 (up to 29 nodes -- our order of magnitude) constrained 3 <= T <= 8, finding any T>=3
  works. **Best evidence at comparable graph scale that 1-2 layers is insufficient.**
- **Epping, Rene, Helias, Schaub. "GNNs Do Not Always Oversmooth." *NeurIPS* (2024).**
  arXiv:[2406.02269](https://arxiv.org/abs/2406.02269). Proves a non-oversmoothing phase exists
  at arbitrary depth given large enough initial weight variance. Contradicts "more layers always
  worse"; our hidden_dim=256 plausibly sits in that regime.
- Oversmoothing foundations -- **Li, Han, Wu** (*AAAI* 2018, arXiv:1801.07606), **Oono & Suzuki**
  (*ICLR* 2020, arXiv:1905.10947), **Chen et al.** (*AAAI* 2020, arXiv:1909.03211): all real, all
  **node classification on graphs 3-4 orders of magnitude larger than ours**. Oono & Suzuki had
  to artificially densify Cora/Citeseer/Pubmed for the predicted decay to appear. Do not apply
  this folklore to 8-16 node graphs.

### Capacity vs dataset size -- FOLKLORE, no rule survives checking

- The "10x samples per parameter" heuristic traces to **Peduzzi, Concato, Kemper, Holford,
  Feinstein**, *J. Clin. Epidemiol.* **49**(12):1373-1379 (1996),
  doi:[10.1016/S0895-4356(96)00236-3](https://doi.org/10.1016/S0895-4356(96)00236-3) -- events
  per variable in **logistic regression**, not neural nets -- and was relaxed by **Vittinghoff &
  McCulloch**, *Am. J. Epidemiol.* **165**(6):710-718 (2007),
  doi:[10.1093/aje/kwk052](https://doi.org/10.1093/aje/kwk052).
  **No verified rule supports "404 params/sample is too many" for GNN regression.** Comparable
  molecular-regression setups at 1,400-8,000 graphs (Hu et al., *ICLR* 2020, arXiv:1905.12265)
  routinely use comparable-or-larger models. Do not shrink the model on this reasoning.
- **Zhang, Bengio, Hardt, Recht, Vinyals. "Understanding Deep Learning Requires Rethinking
  Generalization." *ICLR* (2017).** arXiv:[1611.03530](https://arxiv.org/abs/1611.03530).
  Negative result about classical complexity measures, not a positive sizing rule.

## UNRESOLVED -- do not cite until sourced

- **Vertical ionization potentials of DNA bases and base pairs.** The values in circulation
  for this project (bases G 7.91 / A 8.30 / C 8.74 / T 9.05 eV; pairs GC 7.28 / AT 7.86 eV)
  have been attributed in conversation to "Caruso". A verification pass on 2026-07-24 found
  **no paper by that author reporting these values**, and found that search engines
  repeatedly misattribute the Faber et al. GW paper below to a "Caruso" (there is a
  well-known GW physicist Fabio Caruso, co-author of the GW100 benchmark, but no DNA-base IP
  table by him was located).

  Nearest genuine sources, neither an exact match:
  - Faber, C., Attaccalite, C., Olevano, V., Runge, E., Blase, X. "First-principles GW
    calculations for DNA and RNA nucleobases." *Phys. Rev. B* **83**, 115123 (2011).
    doi:[10.1103/PhysRevB.83.115123](https://doi.org/10.1103/PhysRevB.83.115123)
    GW vertical IPs: G 7.81, A 8.22, C 8.73, T 9.05 eV. Same ordering, close magnitudes,
    **no base-pair data at all**.
  - Khan, A. "Reorganization, activation and ionization energies for hole transfer reactions
    through IC, ApT, AT, and GC base pairs." *Comput. Theor. Chem.* **1013**, 136-139 (2013).
    doi:[10.1016/j.comptc.2013.03.007](https://doi.org/10.1016/j.comptc.2013.03.007)
    Vertical IEs: GC 7.29, AT 7.88 eV.

  Action required before publication: either locate the true source of the six numbers as
  quoted, or replace them with the numbers above cited to their actual authors. The
  qualitative claim we actually lean on -- G has the lowest IP and therefore the highest hole
  on-site energy -- holds in every source found and is not at risk.

## Published-dataset file format (EVIDENCED, retrieved 2026-07-26)

Decision: publish the archive as HDF5, keep pickles as the internal working format.

Reference class -- the closest published datasets in this sub-field all ship HDF5; the search
found no published quantum-chemistry dataset distributed as pickle:

- Smith, J.S., Isayev, O., Roitberg, A.E. (2017). "ANI-1, A data set of 20 million calculated
  off-equilibrium conformations for organic molecules." *Scientific Data* 4, 170193.
  doi:10.1038/sdata.2017.193 -- distributed as HDF5.
  Title/authors/venue/year independently verified via the Crossref API, 2026-07-26.
- Smith, J.S., Zubatyuk, R., Nebgen, B., Lubbers, N., Barros, K., Roitberg, A.E., Isayev, O.,
  Tretiak, S. (2020). "The ANI-1ccx and ANI-1x data sets, coupled-cluster and density
  functional theory properties for molecules." *Scientific Data* 7, 134.
  doi:10.1038/s41597-020-0473-z -- distributed as HDF5.
- Eastman, P., Behara, P.K., Dotson, D.L., Galvelis, R., Herr, J.E., Horton, J.T., Mao, Y.,
  Chodera, J.D., Pritchard, B.P., Wang, Y., De Fabritiis, G., Markland, T.E. (2023).
  "SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning
  Potentials." *Scientific Data* 10, 11. doi:10.1038/s41597-022-01882-6 -- single HDF5 file,
  one top-level group per molecule. Title/authors/venue/year independently verified via the
  Crossref API, 2026-07-26.

Why not pickle, from the format owner rather than folklore:

- Python Software Foundation, `pickle` module documentation: "The `pickle` module is not
  secure. Only unpickle data you trust. It is possible to construct malicious pickle data
  which will execute arbitrary code during unpickling."
  https://docs.python.org/3/library/pickle.html
- NumPy `numpy.load` documentation: `allow_pickle` defaults to `False`, changed specifically
  because loading pickled object arrays "is not secure against erroneous or maliciously
  constructed data." https://numpy.org/doc/stable/reference/generated/numpy.load.html
  This matters MORE, not less, as Python becomes universal: a ubiquitous format is one
  readers load reflexively without inspecting.

- Wilkinson, M.D. et al. (2016). "The FAIR Guiding Principles for scientific data management
  and stewardship." *Scientific Data* 3, 160018. doi:10.1038/sdata.2016.18 -- principle I1
  requires "a formal, accessible, shared, and broadly applicable language for knowledge
  representation"; R1.3 requires meeting "domain-relevant community standards." HDF5 satisfies
  both (H5MD for molecular simulation, NeXus for neutron/X-ray are built on it); pickle
  satisfies neither.

NOT a factor in this decision: MATLAB's native `h5read` support. It is true, but willll
notes the group's MATLAB code is legacy and the field has moved to Python (2026-07-26), so
it carries no weight here. The decision rests on reference class and on pickle's security
model.

## DNA base and base-pair oxidation potentials (EVIDENCED, retrieved 2026-07-30)

Supplied by willll. Both verified against the Crossref REST API -- authors, title, journal,
volume, page range and year all match with no discrepancy. Verification was done against
structured metadata rather than a search-engine summary, because during this check a
WebSearch result confidently attributed the 2005 paper to an entirely different author list.

- Caruso, T.; Carotenuto, M.; Vasca, E.; Peluso, A. "Direct Experimental Observation of the
  Effect of the Base Pairing on the Oxidation Potential of Guanine." *J. Am. Chem. Soc.*
  **2005**, 127, 15040-15041. doi:10.1021/ja055130s
- Caruso, T.; Capobianco, A.; Peluso, A. "The Oxidation Potential of Adenosine and
  Adenosine-Thymidine Base Pair in Chloroform Solution." *J. Am. Chem. Soc.* **2007**, 129,
  15347-15353. doi:10.1021/ja076181n

Both report ELECTROCHEMICAL OXIDATION POTENTIALS measured by voltammetry in chloroform, and
specifically the effect of Watson-Crick pairing on them (2005: a 0.34 V lowering for a
guanosine derivative; 2007: adenosine versus the A-T pair, with no thymidine oxidation signal
detected). These are NOT gas-phase vertical ionization potentials and are not directly
comparable to them -- see `docs/model-results.md` section 4b for why, and for the retraction
of an earlier entry that wrongly implied the citation could not be traced.

## Statistical methodology for the training-configuration factorial (EVIDENCED, retrieved 2026-08-09)

Cited from `docs/doe-methods.md`. Each entry below was verified by web retrieval on
2026-08-09 (title, authors, venue, volume and pages checked against the publisher's or an
indexing service's record), not quoted from memory.

- Benjamini, Y.; Hochberg, Y. (1995). "Controlling the false discovery rate: a practical
  and powerful approach to multiple testing." *J. R. Statist. Soc. B* 57(1), 289-300.
  doi:10.1111/j.2517-6161.1995.tb02031.x -- the FDR procedure used for multiplicity
  control over the factorial's test family.
- Box, G.E.P.; Meyer, R.D. (1986). "Dispersion effects from fractional designs."
  *Technometrics* 28(1), 19-27. -- precedent for analysing dispersion (spread) responses
  from factorial designs on a log scale.
- Nelder, J.A. (1977). "A reformulation of linear models." *J. R. Statist. Soc. A* 140(1),
  48-63. doi:10.2307/2344517 -- the marginality principle: interactions are meaningful
  only in models containing their parent terms; the basis for hierarchical backward
  elimination.
- Peixoto, J.L. (1990). "A property of well-formulated polynomial regression models."
  *The American Statistician* 44(1), 26-30. doi:10.1080/00031305.1990.10475687 -- a
  polynomial model is invariant under coding transformations (e.g. centring) iff it
  respects effect hierarchy; the basis for centring b and restricting elimination to
  well-formulated models.
- Shapiro, S.S.; Wilk, M.B. (1965). "An analysis of variance test for normality (complete
  samples)." *Biometrika* 52(3/4), 591-611. -- the normality check applied to raw
  cell-level spreads before trusting the raw-scale dispersion analysis.

---

# INTRODUCTION AND RELATED WORK (EVIDENCED, retrieved 2026-08-11)

Assembled by two independent literature passes, both instructed to verify every DOI against
a registration-agency record rather than a search-engine summary. **Seven entries were then
re-verified independently against the Crossref REST API before this section was written**
(marked `[SPOT-CHECKED]`); all seven matched the reported metadata exactly, including one
the search results actively misattribute. Diacritics are transliterated per the repo's
ASCII rule -- restore them in the final BibTeX (Artes = Artes with acute e, Schonenberger,
Rosch, Nogues, Kohler, Ordejon).

## PRIOR ART -- machine learning applied to DNA charge transport

**Read these before writing any novelty claim.** Both literature passes surfaced them
independently, from different search framings.

- **[SPOT-CHECKED]** Korol, R., Segal, D. "Machine Learning Prediction of DNA Charge
  Transport." *J. Phys. Chem. B* **123**, 2801-2811 (2019).
  doi:[10.1021/acs.jpcb.8b12557](https://doi.org/10.1021/acs.jpcb.8b12557)
  ML trained on n = 3-7 bp junctions (conductance from quantum scattering) to predict
  conductance of long duplexes. Predicts a SCALAR OBSERVABLE, not a Hamiltonian.

- **[SPOT-CHECKED]** Aggarwal, A., Vinayak, V., Bag, S., Bhattacharyya, C., Waghmare, U. V.,
  Maiti, P. K. "Predicting the DNA Conductance Using a Deep Feedforward Neural Network
  Model." *J. Chem. Inf. Model.* **61**, 106-114 (2021).
  doi:[10.1021/acs.jcim.0c01072](https://doi.org/10.1021/acs.jcim.0c01072)
  Feedforward network for DNA conductance. Again a scalar readout, no intermediate H.
  Cite as 2021 (issue); online 2020-12-15.

The honest positioning: ML for DNA conductance is NOT new. What is not present in either is
a learned intermediate **Hamiltonian** supervised only on transport spectra. That is the
distinction the paper must draw, and it should be drawn explicitly rather than by omission.

## PRIOR ART -- tight-binding models of DNA

- **[SPOT-CHECKED]** Lambropoulos, K., Simserides, C. "Tight-Binding Modeling of Nucleic
  Acid Sequences: Interplay between Various Types of Order or Disorder and Charge
  Transport." *Symmetry* **11**(8), 968 (2019).
  doi:[10.3390/sym11080968](https://doi.org/10.3390/sym11080968) [REVIEW]
  The most direct prior-art review for a tight-binding-Hamiltonian paper: base-pair and
  wire/ladder TB models, ordered vs disordered sequences.

- Voityuk, A. A., Rosch, N., Bixon, M., Jortner, J. "Electronic Coupling for Charge Transfer
  and Transport in DNA." *J. Phys. Chem. B* **104**, 9740-9745 (2000).
  doi:[10.1021/jp001109w](https://doi.org/10.1021/jp001109w)
  Sequence- and stacking-dependent inter-base couplings -- the off-diagonal elements a TB H
  must reproduce. Companion to the Voityuk 2001 JCP entry already used in `physics.py`.

## Direct measurements: DNA conducts, contradictorily

- Fink, H.-W., Schonenberger, C. "Electrical conduction through DNA molecules." *Nature*
  **398**, 407-410 (1999). doi:[10.1038/18855](https://doi.org/10.1038/18855) [PRIMARY]
- de Pablo, P. J., Moreno-Herrero, F., Colchero, J., et al. "Absence of dc-Conductivity in
  lambda-DNA." *Phys. Rev. Lett.* **85**, 4992-4995 (2000).
  doi:[10.1103/PhysRevLett.85.4992](https://doi.org/10.1103/PhysRevLett.85.4992) [PRIMARY]
  The opposite result. Cite WITH Fink to support "early measurements were contradictory".
- **[SPOT-CHECKED via companion]** Porath, D., Bezryadin, A., de Vries, S., Dekker, C.
  "Direct measurement of electrical transport through DNA molecules." *Nature* **403**,
  635-638 (2000). doi:[10.1038/35001029](https://doi.org/10.1038/35001029) [PRIMARY]
  10.4 nm poly(G)-poly(C); large-bandgap semiconducting behaviour.
- Xu, B., Zhang, P., Li, X., Tao, N. "Direct Conductance Measurement of Single DNA Molecules
  in Aqueous Solution." *Nano Lett.* **4**, 1105-1108 (2004).
  doi:[10.1021/nl0494295](https://doi.org/10.1021/nl0494295) [PRIMARY]
  STM break junction. NOTE: ACS deposits surnames only to Crossref; given names (Bingqian,
  Peiming, Xiulan, Nongjian) come from Semantic Scholar, agreeing with a published reference
  list. Confirm the byline from the PDF before submission.
- Cohen, H., Nogues, C., Naaman, R., Porath, D. "Direct measurement of electrical transport
  through single DNA molecules of complex sequence." *PNAS* **102**, 11589-11593 (2005).
  doi:[10.1073/pnas.0505272102](https://doi.org/10.1073/pnas.0505272102) [PRIMARY]
  Non-repeating 26-bp sequences, i.e. beyond homopolymers.
- Guo, X., Gorodetsky, A. A., Hone, J., Barton, J. K., Nuckolls, C. "Conductivity of a single
  DNA duplex bridging a carbon nanotube gap." *Nat. Nanotechnol.* **3**, 163-167 (2008).
  doi:[10.1038/nnano.2008.4](https://doi.org/10.1038/nnano.2008.4) [PRIMARY]
  A single GT or CA mismatch raises resistance ~300-fold.
- Livshits, G. I., Stern, A., Rotem, D., et al. (14 authors) "Long-range charge transport in
  single G-quadruplex DNA molecules." *Nat. Nanotechnol.* **9**, 1040-1046 (2014).
  doi:[10.1038/nnano.2014.246](https://doi.org/10.1038/nnano.2014.246) [PRIMARY]
- **[SPOT-CHECKED]** Artes, J. M., Li, Y., Qi, J., Anantram, M. P., Hihath, J.
  "Conformational gating of DNA conductance." *Nat. Commun.* **6**, 8870 (2015).
  doi:[10.1038/ncomms9870](https://doi.org/10.1038/ncomms9870) [PRIMARY]
  Conductance rises ~1 order of magnitude on the B-to-A transition, reversibly, AT FIXED
  SEQUENCE. **This is willll's own group.** Search engines misattribute it to
  Bruot/Palma/Xiang/Mujica/Ratner/Tao; Crossref confirms the byline above. It also bears
  directly on the `geom=off` default -- see the cautions below.
- **[SPOT-CHECKED]** Zhuravel, R., Huang, H., Polycarpou, G., et al. (14 authors) "Backbone
  charge transport in double-stranded DNA." *Nat. Nanotechnol.* **15**, 836-840 (2020).
  doi:[10.1038/s41565-020-0741-2](https://doi.org/10.1038/s41565-020-0741-2) [PRIMARY]
  Concludes the BACKBONE, not only the base stack, mediates long-distance transport in 30 nm
  duplexes. A live challenge to any base-stack-only TB model. Cite it as a limitation.
- Endres, R. G., Cox, D. L., Singh, R. R. P. "Colloquium: The quest for high-conductance
  DNA." *Rev. Mod. Phys.* **76**, 195-214 (2004).
  doi:[10.1103/RevModPhys.76.195](https://doi.org/10.1103/RevModPhys.76.195) [REVIEW]
  Documents why the early measurements disagreed: contacts, environment, sample geometry.

## Mechanism: tunnelling, hopping, and the regime between

- Jortner, J., Bixon, M., Langenbacher, T., Michel-Beyerle, M. E. "Charge transfer and
  transport in DNA." *PNAS* **95**, 12759-12765 (1998).
  doi:[10.1073/pnas.95.22.12759](https://doi.org/10.1073/pnas.95.22.12759) [THEORY]
- Giese, B. "Long-Distance Charge Transport in DNA: The Hopping Mechanism." *Acc. Chem. Res.*
  **33**, 631-636 (2000). doi:[10.1021/ar990040b](https://doi.org/10.1021/ar990040b) [REVIEW]
- Berlin, Y. A., Burin, A. L., Ratner, M. A. "Charge Hopping in DNA." *J. Am. Chem. Soc.*
  **123**, 260-268 (2001). doi:[10.1021/ja001496n](https://doi.org/10.1021/ja001496n) [THEORY]
  Cite as 2001 (issue); Crossref's bare `issued` says 2000-12-15, which is the ASAP posting.
- Giese, B., Amaudrut, J., Kohler, A.-K., Spormann, M., Wessely, S. "Direct observation of
  hole transfer through DNA by hopping between adenine bases and by tunnelling." *Nature*
  **412**, 318-320 (2001). doi:[10.1038/35085542](https://doi.org/10.1038/35085542) [PRIMARY]
  The distance dependence changes character past ~3 intervening A:T pairs. Best single
  citation for "tunnelling at short range, hopping at long range".
- Conwell, E. M. "Charge transport in DNA in solution: The role of polarons." *PNAS* **102**,
  8795-8799 (2005). doi:[10.1073/pnas.0501406102](https://doi.org/10.1073/pnas.0501406102)
  [THEORY] The competing polaron / self-trapping picture.
- **[SPOT-CHECKED]** Xiang, L., Palma, J. L., Bruot, C., Mujica, V., Ratner, M. A., Tao, N.
  "Intermediate tunnelling-hopping regime in DNA charge transport." *Nat. Chem.* **7**,
  221-226 (2015). doi:[10.1038/nchem.2183](https://doi.org/10.1038/nchem.2183) [PRIMARY]
  Coherent and incoherent transport COEXIST in stacked G:C sequences. The key modern
  crossover citation. **A corrigendum exists** -- *Nat. Chem.* **9**, 295 (2017),
  doi:[10.1038/nchem.2731](https://doi.org/10.1038/nchem.2731) -- check it before citing.
- Genereux, J. C., Barton, J. K. "Mechanisms for DNA Charge Transport." *Chem. Rev.* **110**,
  1642-1662 (2010). doi:[10.1021/cr900228f](https://doi.org/10.1021/cr900228f) [REVIEW]
  The standard comprehensive mechanism review. Cite as 2010 (issue); online 2009-11-23.
- Beratan, D. N. "Why Are DNA and Protein Electron Transfer So Different?" *Annu. Rev. Phys.
  Chem.* **70**, 71-97 (2019).
  doi:[10.1146/annurev-physchem-042018-052353](https://doi.org/10.1146/annurev-physchem-042018-052353)
  [REVIEW] Which mechanism operates is set by donor-bridge energetics and fluctuations, not
  fixed.

## Sequence dependence

- Sugiyama, H., Saito, I. "Theoretical Studies of GG-Specific Photocleavage of DNA via
  Electron Transfer..." *J. Am. Chem. Soc.* **118**, 7063-7068 (1996).
  doi:[10.1021/ja9609821](https://doi.org/10.1021/ja9609821) [THEORY]
  Guanine has the lowest IP, lowered further by GG stacking, HOMO on the 5'-G. **This is the
  properly-sourced citation for the guanine claim** that the UNRESOLVED section above could
  not source to "Caruso".
- Hihath, J., Xu, B., Zhang, P., Tao, N. "Study of single-nucleotide polymorphisms by means
  of electrical conductance measurements." *PNAS* **102**, 16979-16983 (2005).
  doi:[10.1073/pnas.0505175102](https://doi.org/10.1073/pnas.0505175102) [PRIMARY]
  A single base-pair mismatch changes conductance by up to an order of magnitude.
- **[SPOT-CHECKED]** Aminiranjbar, Z., Akin Gultakti, C., Zhang, A., Oren, E. E., Hihath, J.
  "Developing design guidelines for controlling charge transport in DNA." *Nat. Chem.* **18**,
  519-525 (2026). doi:[10.1038/s41557-025-01999-2](https://doi.org/10.1038/s41557-025-01999-2)
  [PRIMARY] Nearest-neighbour effects reshape conductance in G:C-rich duplexes WITHOUT
  changing composition; 20-bp designs above 1e-3 G0. The best "why sequence-resolved models
  matter" hook, and the most current statement of the problem this paper addresses.
  Cite as 2026 (issue); online 2025-11-18. A search engine returned a truncated 4-author
  list omitting Hihath -- Crossref and PubMed both give the 5 above.

## Length dependence and beta

- Lewis, F. D., Wu, T., Zhang, Y., et al. "Distance-Dependent Electron Transfer in DNA
  Hairpins." *Science* **277**, 673-676 (1997).
  doi:[10.1126/science.277.5326.673](https://doi.org/10.1126/science.277.5326.673) [PRIMARY]
- Kelley, S. O., Barton, J. K. "Electron Transfer Between Bases in Double Helical DNA."
  *Science* **283**, 375-381 (1999).
  doi:[10.1126/science.283.5400.375](https://doi.org/10.1126/science.283.5400.375) [PRIMARY]
  beta spans **0.1 to 1.0 per angstrom** depending on stacking. The citation for "beta is
  not a single number for DNA".
- Slinker, J. D., Muren, N. B., Renfrew, S. E., Barton, J. K. "DNA charge transport over
  34 nm." *Nat. Chem.* **3**, 228-233 (2011).
  doi:[10.1038/nchem.982](https://doi.org/10.1038/nchem.982) [PRIMARY]
  **Page caveat:** a widely-mirrored PMC-derived citation gives 230-235. That is wrong.
  Crossref gives 228-233.

## Applications: fabrication, sensing, damage detection

- Braun, E., Eichen, Y., Sivan, U., Ben-Yoseph, G. "DNA-templated assembly and electrode
  attachment of a conducting silver wire." *Nature* **391**, 775-778 (1998).
  doi:[10.1038/35826](https://doi.org/10.1038/35826) [PRIMARY]
  Exact title is "conducting", not the "conductive" form that circulates.
- Rothemund, P. W. K. "Folding DNA to create nanoscale shapes and patterns." *Nature* **440**,
  297-302 (2006). doi:[10.1038/nature04586](https://doi.org/10.1038/nature04586) [PRIMARY]
- Maune, H. T., Han, S.-p., Barish, R. D., et al. "Self-assembly of carbon nanotubes into
  two-dimensional geometries using DNA origami templates." *Nat. Nanotechnol.* **5**, 61-66
  (2010). doi:[10.1038/nnano.2009.311](https://doi.org/10.1038/nnano.2009.311) [PRIMARY]
  Cite as 2010 (issue); online 2009-11-08. Crossref renders Goddard as "III, W. A. G."; the
  correct form is William A. Goddard III.
- Dey, S., Fan, C., Gothelf, K. V., et al. "DNA origami." *Nat. Rev. Methods Primers* **1**,
  13 (2021). doi:[10.1038/s43586-020-00009-8](https://doi.org/10.1038/s43586-020-00009-8)
  [REVIEW] Article number from the publisher meta tag; Crossref carries no page.
- Zhan, P., Peil, A., Jiang, Q., et al. "Recent advances in DNA origami-engineered
  nanomaterials and applications." *Chem. Rev.* **123**, 3976-4050 (2023).
  doi:[10.1021/acs.chemrev.3c00028](https://doi.org/10.1021/acs.chemrev.3c00028) [REVIEW]
- Michelson, A., Shani, L., Kahn, J. S., et al. (13 authors) "Scalable fabrication of
  chip-integrated 3D-nanostructured electronic devices via DNA-programmable assembly."
  *Sci. Adv.* **11**(13), eadt5620 (2025).
  doi:[10.1126/sciadv.adt5620](https://doi.org/10.1126/sciadv.adt5620) [PRIMARY]
  DNA-programmed assembly reaching chip-integrated devices.
- Dunn, K. E., Elfick, A. "Harnessing DNA nanotechnology and chemistry for applications in
  photonics and electronics." *Bioconjugate Chem.* **34**, 97-104 (2023).
  doi:[10.1021/acs.bioconjchem.2c00286](https://doi.org/10.1021/acs.bioconjchem.2c00286)
  [REVIEW] Cite as 2023 (issue); online 2022-09-19.
- Drummond, T. G., Hill, M. G., Barton, J. K. "Electrochemical DNA sensors." *Nat.
  Biotechnol.* **21**, 1192-1199 (2003). doi:[10.1038/nbt873](https://doi.org/10.1038/nbt873)
  [REVIEW] Sensing whose transduction mechanism IS base-stack charge transport.
- Zwolak, M., Di Ventra, M. "Colloquium: Physical approaches to DNA sequencing and
  detection." *Rev. Mod. Phys.* **80**, 141-165 (2008).
  doi:[10.1103/RevModPhys.80.141](https://doi.org/10.1103/RevModPhys.80.141) [REVIEW]
- Tsutsui, M., Taniguchi, M., Yokota, K., Kawai, T. "Identifying single nucleotides by
  tunnelling current." *Nat. Nanotechnol.* **5**, 286-290 (2010).
  doi:[10.1038/nnano.2010.42](https://doi.org/10.1038/nnano.2010.42) [PRIMARY]
- Ivanov, A. P., Instuli, E., McGilvery, C. M., et al. "DNA tunneling detector embedded in a
  nanopore." *Nano Lett.* **11**, 279-285 (2011).
  doi:[10.1021/nl103873a](https://doi.org/10.1021/nl103873a) [PRIMARY]
  Cite as 2011 (issue); online 2010-12-06.
- Dorey, A., Howorka, S. "Nanopore DNA sequencing technologies and their applications towards
  single-molecule proteomics." *Nat. Chem.* **16**, 314-334 (2024).
  doi:[10.1038/s41557-023-01322-x](https://doi.org/10.1038/s41557-023-01322-x) [REVIEW]
- Chu, M., Zhang, Y., Ji, C., et al. "DNA nanomaterial-based electrochemical biosensors for
  clinical diagnosis." *ACS Nano* **18**, 31713-31736 (2024).
  doi:[10.1021/acsnano.4c11857](https://doi.org/10.1021/acsnano.4c11857) [REVIEW]
- Hall, D. B., Holmlin, R. E., Barton, J. K. "Oxidative DNA damage through long-range
  electron transfer." *Nature* **382**, 731-735 (1996).
  doi:[10.1038/382731a0](https://doi.org/10.1038/382731a0) [PRIMARY]
- Boon, E. M., Ceres, D. M., Drummond, T. G., Hill, M. G., Barton, J. K. "Mutation detection
  by electrocatalysis at DNA-modified electrodes." *Nat. Biotechnol.* **18**, 1096-1100
  (2000). doi:[10.1038/80301](https://doi.org/10.1038/80301) [PRIMARY]
  Single-base mismatch detection with charge transport as the transducer.
- Genereux, J. C., Boal, A. K., Barton, J. K. "DNA-mediated charge transport in redox sensing
  and signaling." *J. Am. Chem. Soc.* **132**, 891-905 (2010).
  doi:[10.1021/ja907669c](https://doi.org/10.1021/ja907669c) [REVIEW]
- Jang, J., Yoon, H. J. "Long-range charge transport in molecular wires." *J. Am. Chem. Soc.*
  **146**, 32206-32221 (2024). doi:[10.1021/jacs.4c11431](https://doi.org/10.1021/jacs.4c11431)
  [REVIEW] Places DNA in the wider molecular-wire context.
- Yao, C., Li, Y., Zhang, H., et al. "Molecular electronic devices based on atomic
  manufacturing methods." *Microsyst. Nanoeng.* **11**, 232 (2025).
  doi:[10.1038/s41378-025-01037-8](https://doi.org/10.1038/s41378-025-01037-8) [REVIEW]

### SCOPE WARNINGS -- verified real, but do not cite for what they look like

- Wang, K., Deng, P., Lin, H., Sun, W., Shen, J. "DNA-Based Conductors: From Materials Design
  to Ultra-Scaled Electronics." *Small Methods* **9**(6), 2400694 (2025).
  doi:[10.1002/smtd.202400694](https://doi.org/10.1002/smtd.202400694) [REVIEW]
  Verified real, but it is mostly about DNA-TEMPLATED / METALLIZED conductors and assembly,
  plus ionic gating -- **not** intrinsic transport through the base stack. Do not cite it as
  "a recent review of DNA charge transport".
- **No strong 2020-2026 review of intrinsic base-stack DNA charge transport in a top venue
  was found**, across four differently-phrased searches by one pass. If the intro needs one,
  that gap is real. Beratan 2019 and Lambropoulos 2019 are the honest substitutes.

### COULD NOT VERIFY -- do not cite

- "Engineering the electronic properties of DNA." *Nat. Chem.* **18**, 441-442 (2026).
  doi:10.1038/s41557-026-02066-0 -- the DOI resolves and the article is real (Crossref,
  OpenAlex, PubMed PMID 41577961 all agree on title, venue, volume, pages), but **no source
  checked lists any author**: Crossref has no author field, PubMed no AuthorList, OpenAlex
  no authorships. Contextually it is near-certainly a commentary on Aminiranjbar et al. in
  the same issue, but that is exactly the plausible reconstruction that manufactures a fake
  citation. Either read the byline off the PDF, or cite Aminiranjbar et al. instead.

## What the introduction may and may not claim from this set

**May claim.** That transport through DNA was measured directly and contradictorily -- from
efficient conduction (Fink 1999) to no measurable dc conductivity (de Pablo 2000) to
semiconducting behaviour (Porath 2000) -- with the discrepancies traced to contacts,
environment and geometry (Endres 2004); that single-molecule conductance of short duplexes
in solution is now routine (Xu 2004, Cohen 2005); that the mechanism is not one mechanism,
with superexchange tunnelling at short range and multistep hopping at long range (Jortner
1998, Berlin 2001), a crossover observed at roughly three intervening A:T pairs (Giese 2001)
and an intermediate regime where both coexist (Xiang 2015); that guanine is the hole carrier
by virtue of the lowest ionization potential, lowered further by GG stacking (Sugiyama 1996);
that inter-base couplings are sequence- and stacking-dependent (Voityuk 2000); that a single
base substitution changes conductance by up to an order of magnitude (Hihath 2005, Guo 2008);
that sequence rearrangement at FIXED composition substantially changes conductance
(Aminiranjbar 2026); and that beta is not a constant, spanning 0.1-1.0 per angstrom
(Kelley 1999), while transport is nonetheless detected over 34 nm (Slinker 2011).

**Must not claim.**
1. **A single canonical beta for duplex DNA.** Kelley 1999 is explicitly a range, and the
   tunnelling-to-hopping crossover means one exponential does not hold across lengths. Quote
   a number only with its sequence and measurement system named.
2. **That transport is exclusively through the base stack.** Zhuravel 2020 concludes the
   backbone mediates long-distance transport in 30 nm duplexes. This is a live challenge to
   a base-stack-only TB model and belongs in Limitations, stated, not omitted.
3. **That conductance is a function of sequence alone.** Artes 2015 -- willll's own group --
   shows a reversible order-of-magnitude change from a B-to-A conformational transition at
   fixed sequence. **This bears directly on the `geom=off` default.** Our own geometry result
   is null (MR 9), but that is a statement about our IDEALIZED FIBER geometry, which is
   near-constant across sequences by construction (see `docs/dataset.md`), not about geometry
   being unimportant to DNA conductance. State the distinction ourselves rather than let a
   referee draw it.
4. **That ML for DNA charge transport is new.** Korol and Segal 2019 and Aggarwal et al. 2021
   both predate this work. The novel element is the learned intermediate Hamiltonian
   supervised only on transport spectra, and the claim must be scoped to that.
5. **That commercial nanopore sequencing reads tunnelling current.** It reads ionic blockade
   current. Tunnelling-current sequencing is demonstrated (Tsutsui 2010, Ivanov 2011) but is
   not the deployed modality. Keep them separate.
