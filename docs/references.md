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
