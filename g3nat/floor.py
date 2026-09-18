"""The one log10 floor for every transport quantity, everywhere.

Every log10 of a DOS, per-site LDOS or transmission -- in the reference
pipeline (DNADataset/), in the pickle loader, in both models and hence in
every loss and metric -- is log10(max(x, LOG_FLOOR)). One number, one rule.

Why 1e-25: on the unoccupied side of the window the DFT+NEGF reference
transmission of the 12- and 16-bp duplexes stops decaying and flattens to a
smooth, sequence- and coupling-dependent background between 1e-22 and 1e-28.
Transmission in a band gap decays exponentially, so that plateau is not a
property of the molecule. It is also not a precision artifact: rebuilding the
Hamiltonian from a 16-digit matrix export reproduces it to three decimals. It
is intrinsic to the reference model itself -- the Lowdin-orthogonalized
all-electron Hamiltonian carries long-range couplings (about 1e-8 eV eleven
base pairs apart) that a chain of nearest-neighbour sites has no counterpart
for. 1e-25 sits seven decades below the smallest genuine value in the
training set (T = 6.7e-19), so it never binds on training data, and removes
the bulk of that background from the held-out comparison (it still leaves the
part of the plateau that lies above 1e-25 on the 0.6 eV contact-coupling
records, and clips a few genuine points below 1e-25 on the mixed 16-mers).
"""

LOG_FLOOR = 1e-25
LOG10_FLOOR = -25.0
