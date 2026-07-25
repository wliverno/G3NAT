# LDOS loss: design notes (NOT a spec -- brainstorm paused mid-flight)

**Status: no implementation, no result.** This records an unfinished brainstorming session so
it can be resumed with `superpowers:brainstorming` rather than redone. The brainstorm reached
"propose approaches" and design section 1 of 4, then stopped when the work turned out to
depend on regenerating the dataset. Sections 2-4 (loss details, testing, run plan) were never
presented.

**Do not cite any of this as a finding.** Nothing here has been measured except the context
checks explicitly marked VERIFIED.

## The idea

Supervise the model on per-site local DOS, not just total DOS and transmission. DOS pins
eigenvalues; transmission pins one contact-to-contact matrix element of G; neither says WHERE
a state lives. Per-site LDOS pins eigenvectors site by site, which is the under-determination
this project has been circling.

This became the primary direction after 2026-07-24, when the "recover a universal per-base
onsite table" framing was retired (see `docs/dataset.md`). With absolute per-base values off
the table, "is H interpretable/physical" reduces largely to "does H put spectral weight in
the right places" -- which is what LDOS measures.

## VERIFIED context (measured, safe to build on)

- `DOSAtom [n_atoms, n_energy]` in `<seq>/run<k>/DOS_<seq>_gammaL_*.mat`. **Row order matches
  PDB atom order exactly** (253 == 253 for `aaac`).
- **`sum_atoms(DOSAtom) / DOS = 1.0000` at every energy.** The atom-resolved LDOS is an EXACT
  decomposition of the DOS we already train on, not an approximation. Consequence: magnitudes
  are directly comparable and there is NO free scale factor to fit -- an earlier plan to fit a
  global scale was unnecessary.
- The `.mat` Energy grid and the pickle `Egrid` are **bit-identical** (`max|diff| = 0`), so
  centring each by its own mean is safe.
- Residue count == model site count (8 == 8 for `aaac`), so residues map 1:1 onto TB sites.
- Coverage is complete: all 515 pickle sequences have both a DOS `.mat` and a PDB.
- The model already computes per-site LDOS and traces it away. `self.ldos` now exposes it
  (`hamiltonian.py`, additive; verified to sum to DOS to 2.6e-7 on trained checkpoints,
  non-negative, all 201 energy points unclamped). Committed in `1bd1890`.

## DECISIONS MADE

1. **It is a training signal, not a "discriminator".** willll rejected that framing: a
   quantity good enough to judge H by is good enough to train H on. The no-grad evaluation on
   an existing checkpoint is the same code, so "diagnostic first, training later" was a false
   dichotomy.
2. **Loss form** (willll's parameterization, better than the alternatives considered):
   `loss = a*T_loss + b*LDOS_loss + (1-b)*DOS_loss`
   - `b=0` reproduces today byte-identically; `b=1` drops explicit DOS (matching all sites
     implies matching the total). So one knob spans "current model" to "replace DOS entirely".
   - `a` already exists implicitly and equals 1 (`trainer.py:152,185`).
   - Unlike `alpha`, `b` is a REAL parameter: the model cannot rescale a loss term.
   - willll's framing for the paper: `b` is a **localization parameter**.
3. **Both DFT references get computed**: whole-residue and base-atoms-only. Only whole-residue
   preserves `sum_i LDOS_i = DOS`, so base-only would ask the model to reproduce less than the
   total it is already fit to. Plan: whole-residue for the loss, base-only reported alongside;
   the gap measures how much backbone DOS sits in the window.
4. **Contact vs interior sites reported separately.** The model is HANDED the contact coupling
   (`edge_attr[:,4]`) and the DFT used the same gamma, so LDOS at contact-attached sites is
   largely set by shared broadening rather than by the learned H. Weight the verdict on
   interior sites.
5. **Compare in log space.** LDOS spans ~4 orders of magnitude (dynamic range 3.3e4). A linear
   metric is decided entirely by the two contact sites and is blind to the interior.
6. The current loss is **HuberLoss, not MSE** (`trainer.py:47`).

## OPEN PROBLEMS (unresolved -- resume here)

- **The floor.** Per-site LDOS reaches ~1e-7 on empty interior sites, so `log10` residuals
  there would dominate and Huber's linear tail will not save it. Needs a floor analogous to
  the model's existing `log_floor`; the value is a physics choice, not a detail. No
  recommendation was reached.
- **Scale comparability of the two loss terms.** For `b` to be a meaningful localization
  parameter in a paper, `DOS_loss` and `LDOS_loss` must be on comparable scales, or `b=0.5`
  means nothing. Not yet analysed.
- **Residue -> node mapping is THE correctness risk.** Node order is contacts, then primary
  strand in order, then complementary strand in order (`construction.py:118-131`) -- NOT
  reversed. An off-by-one or strand reversal produces plausible numbers that are entirely
  wrong. This mapping MUST have a test that permutes it and asserts the loss rises. A scratch
  script during this session hit exactly this bug and it was only caught because the alpha=1.0
  model is a control whose shift must be exactly zero. Build that control in.
- **Sections 2-4 never presented**: loss details, testing strategy, run plan (`b` sweep).

## BLOCKED ON

`docs/dataset.md` -- the dataset must be regenerated to carry LDOS, and it is being published
with the paper, so it needs its own spec first. Section 1 of this design (the data pipeline)
gets simpler as a result: "load a field from the pickle" instead of a training-time `.mat`
reader doing residue mapping on the fly, which moves the quiet-failure-prone part into a
one-time generation step where it can be checked properly.

Also blocked on willll's call to establish a better baseline first (2026-07-24).

## When resuming

Invoke `superpowers:brainstorming`, start from "present design sections", and pick up at
section 2 (the loss). Sections 1 (data pipeline) and the approach choice are settled above.
Terminal state is a written spec, then `superpowers:writing-plans`.
