#!/usr/bin/env python3
"""Python port of readMAT.m.

Parses the plaintext dump produced by the `readmat` Fortran tool (the ASCII
"Label ... / RArr= ..." format Gaussian's rwfdump-style output uses), rebuilds
the packed-triangular Overlap and (alpha) Fock matrices, computes the
generalized-eigenvalue spectrum EV = eig(Overlap^-1 * Fock) exactly as
readMAT.m does, and computes the orthogonalized Hamiltonian

    H0 = S^(-1/2) (Fock * au_to_eV) S^(-1/2)

matching readMAT.m line for line. Does NOT overwrite the seq.mat file with H0
by default (that MATLAB behavior of clobbering the raw Gaussian dump is why
the ported pipeline keeps H0 and the raw matrices in clearly separate files);
see `main()` for the file layout chosen here.

MATLAB-vs-Python semantics preserved deliberately (see inline comments):
  - `M1_size`/`M2_size` extraction: MATLAB's `textscan(line, '%*[^=]= %d')`
    walks every `key=value` token on the header line and keeps the LAST
    integer it can parse before hitting a non-numeric one (`ASym=F` stops
    it). Replicated by `_last_matlab_int`.
  - `textscan(fid, 'RArr= %f ...')` after a header line greedily consumes
    every immediately-following "RArr=" line (not just enough for one
    matrix) until a non-matching line is hit, THEN only the first
    n*(n+1)/2 values are used to build the matrix. Replicated by only
    reading contiguous "RArr=" lines and slicing to `vlength`.
  - `tri2mat`: values are packed column-major into the lower (Msize>0) or
    upper (Msize<0) triangle of an n x n matrix, then mirrored:
    M = M + M.' - diag(diag(M)). Replicated via `_fill_triangular_colmajor`.
  - `eig(Overlap^-1 * Fock)`: MATLAB computes an explicit inverse then a
    general (non-symmetric) eigendecomposition, NOT eigh on the pencil
    (Fock, Overlap). We replicate this literally with `scipy.linalg.eig` on
    `solve(Overlap, Fock)` (a linear solve rather than an explicit inverse,
    which is better conditioned and mathematically identical to
    `Overlap^-1 @ Fock`). Eigenvalue ORDER is not guaranteed to match
    MATLAB's LAPACK call across platforms; only the eigenvalue SET should be
    compared (see the validation report).
  - `Overlap^(-0.5)`: MATLAB's non-integer matrix power on a symmetric
    positive-definite matrix is computed via eigendecomposition; replicated
    with `scipy.linalg.eigh` (numerically identical for SPD input, and
    exact for the exact-arithmetic case).
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.linalg as sla

AU_TO_EV = 27.211396


def _last_matlab_int(line: str) -> int:
    """Replicate `textscan(line, '%*[^=]= %d'); take the last element`.

    MATLAB's `%*[^=]` skips to the next '=', then `%d` parses the integer
    right after it (leading whitespace allowed). Repeated application walks
    every 'key=value' token on the line and stops at the first one whose
    value is not an integer (e.g. `ASym=F`), keeping the last integer
    successfully parsed before that. Splitting on '=' and taking the
    leading integer of each subsequent segment reproduces this: for
    "...NTot= 9419970 LenBuf=    4000 N= -4340  4340 ... ASym=F" the walk
    is 0, 1, 1, 9419970, 4000, -4340, then stops at "F" -- landing on
    -4340, the value MATLAB actually uses for M1_size/M2_size here.
    """
    last = None
    for seg in line.split("=")[1:]:
        m = re.match(r"\s*([+-]?\d+)", seg)
        if not m:
            break
        last = int(m.group(1))
    if last is None:
        raise ValueError(f"No integer found after '=' in header line: {line!r}")
    return last


def _fill_triangular_colmajor(vals: np.ndarray, n: int, lower: bool) -> np.ndarray:
    """Place `vals` (length n*(n+1)/2) into the lower/upper triangle of an
    n x n matrix in MATLAB column-major order, then mirror to full matrix.

    MATLAB fills `M(M ~= 0) = 1:vlength` where the mask is `tril(ones(n))`
    (Msize > 0, "IsLow=0" despite the name -- see readMAT.m) or
    `triu(ones(n))` (Msize < 0, "IsLow=1"), and MATLAB linear indexing over
    a boolean mask always proceeds column-major. The trick used here: the
    row-major traversal of `np.triu_indices` and the desired column-major
    traversal of the lower triangle are the same index pairs with rows and
    columns swapped (and vice versa for the upper triangle) -- verified by
    hand for n=3 during development.
    """
    if lower:
        r0, c0 = np.triu_indices(n)
        rows, cols = c0, r0
    else:
        r0, c0 = np.tril_indices(n)
        rows, cols = c0, r0
    M = np.zeros((n, n), dtype=np.float64)
    M[rows, cols] = vals
    M = M + M.T - np.diag(np.diag(M))
    return M


def _read_block(lines_iter, first_line):
    """Given the header line just read (already matched), consume every
    contiguous following 'RArr=' line and return the flat float64 array of
    all values found, in file order, plus the parsed Msize (n, lower flag).
    """
    Msize = _last_matlab_int(first_line)
    n = abs(Msize)
    lower = Msize > 0  # readMAT.m: Msize>0 -> IsLow=0 -> tril() used (lower)
    vlength = n * (n + 1) // 2

    # MATLAB's textscan greedily consumes every contiguous following
    # "RArr=" line (needed so the outer header scan resumes in the right
    # place), but only the first `vlength` values are ever used to build
    # the matrix. For large dumps (millions of RArr lines belonging to
    # later blocks such as MO coefficients) accumulating every value would
    # blow up memory, so once `vlength` values are collected we keep
    # scanning lines (cheap prefix check only) without parsing further
    # numbers.
    vals = []
    have_enough = False
    for line in lines_iter:
        if not line.lstrip().startswith("RArr="):
            return np.asarray(vals[:vlength], dtype=np.float64), n, lower, line
        if not have_enough:
            nums = line.split("RArr=", 1)[1].split()
            vals.extend(float(x) for x in nums)
            if len(vals) >= vlength:
                have_enough = True
    return np.asarray(vals[:vlength], dtype=np.float64), n, lower, None


def parse_gaussian_dump(txt_path: Path):
    """Parse the readmat plaintext dump; return (Overlap, Fock) as dense
    float64 n x n numpy arrays, matching readMAT.m's `tri2mat` output.
    """
    txt_path = Path(txt_path)
    overlap = None
    fock = None

    with open(txt_path, "r", errors="replace") as fh:
        line_iter = iter(fh)
        pending = None
        while overlap is None or fock is None:
            if pending is not None:
                line = pending
                pending = None
            else:
                try:
                    line = next(line_iter)
                except StopIteration:
                    break
            if "Label OVERLAP" in line and overlap is None:
                vals, n, lower, leftover = _read_block(line_iter, line)
                overlap = _fill_triangular_colmajor(vals, n, lower)
                pending = leftover
            elif "Label ALPHA FOCK MATRIX" in line and fock is None:
                vals, n, lower, leftover = _read_block(line_iter, line)
                fock = _fill_triangular_colmajor(vals, n, lower)
                pending = leftover

    if overlap is None:
        raise ValueError(f"'Label OVERLAP' block not found in {txt_path}")
    if fock is None:
        raise ValueError(f"'Label ALPHA FOCK MATRIX' block not found in {txt_path}")
    return overlap, fock


def compute_eigenvalues(overlap: np.ndarray, fock: np.ndarray) -> np.ndarray:
    """EV = eig(Overlap^-1 * Fock), matching readMAT.m literally (general,
    non-symmetric eigendecomposition of the explicit matrix product -- NOT
    a symmetric generalized eigh on the (Fock, Overlap) pencil). Returns a
    REAL array. Overlap^-1 Fock is similar to the symmetric S^-1/2 F S^-1/2, so
    its eigenvalues are real and any imaginary part is numerical dust -- measured
    at exactly 0.0 for the L=12 record (8748 eigenvalues, max|imag| = 0.000e+00).

    Returning complex here caused a real failure: the downstream HOMO/LUMO search
    in TransportSetup.py propagated complex values into the energy range, and
    `np.arange` with complex bounds silently returns an EMPTY array. The whole
    conversion then completed with exit code 0 and produced Energy of shape
    (1, 0), DOS (1, 0) and DOSAtom (760, 0) -- correct atom count, no spectrum.
    Nothing errored. Cast here so that cannot recur.
    """
    M = sla.solve(overlap, fock)  # == overlap^-1 @ fock, better conditioned
    ev = sla.eig(M, right=False)
    imag_max = float(np.abs(np.imag(ev)).max())
    if imag_max > 1e-6 * max(1.0, float(np.abs(np.real(ev)).max())):
        raise ValueError(
            "eigenvalues of Overlap^-1 Fock are not real (max|imag| = %.3e); "
            "the Fock/Overlap pair is not symmetric-definite as assumed" % imag_max
        )
    return np.real(ev)


def ascii_roundtrip(M: np.ndarray, sig_digits: int = 8) -> np.ndarray:
    """Emulate MATLAB `save(f,'H0','-ASCII')` followed by `load -ASCII f`.

    THIS IS DELIBERATE PRECISION LOSS AND IT IS REQUIRED. Do not "fix" it.

    readMAT.m:75-77 does:
        save(Fname, 'H0', '-ASCII');     % writes H0 as TEXT
        eval(['load -ASCII ', Fname]);   % reads it straight back
        save(Fname, strand_name);        % stores THAT as the binary .mat
    MATLAB's `-ASCII` (without `-double`) writes 8 significant digits, so every
    element of the Hamiltonian used by the transport stage carries only ~1e-8
    relative precision. All 2077 published records were produced this way.

    Skipping this step makes the Python disagree with MATLAB by 1e-4 to 1e-2 on
    T / DOS / DOSAtom -- small on H, but amplified through 201 resolvent solves,
    and worse for larger systems. Measured on aaac (2869 orbitals), max relative
    deviation vs the stored MATLAB output:

        H0 at full float64      T 9.40e-05   DOS 8.32e-05   DOSAtom 1.52e-04
        H0 via this round-trip  T 2.88e-09   DOS 1.73e-09   DOSAtom 3.78e-09

    Five orders of magnitude, and the second line is round-off for a quantity
    that has been through 201 complex solves. Reproducing the truncation is what
    makes new records commensurable with the existing dataset.
    """
    fmt = "%." + str(sig_digits) + "g"
    flat = M.ravel()
    out = np.fromiter((float(fmt % v) for v in flat), dtype=np.float64,
                      count=flat.size)
    return out.reshape(M.shape)


def compute_H0(overlap: np.ndarray, fock: np.ndarray, au_to_eV: float = AU_TO_EV,
               match_matlab_ascii: bool = True) -> np.ndarray:
    """H0 = S^(-1/2) (Fock * au_to_eV) S^(-1/2), matching readMAT.m exactly
    (after `Fock_Mod = Fock_Mod * au_to_eV`), including its ASCII round-trip.

    match_matlab_ascii=True (the default, and what you want) reproduces the
    8-significant-digit truncation readMAT.m applies before the transport stage.
    Pass False ONLY to study the effect of that truncation; results will not be
    comparable to the existing 2077 records.
    """
    fock_ev = fock * au_to_eV
    w, v = sla.eigh(overlap)  # overlap is real symmetric positive definite
    if np.any(w <= 0):
        raise np.linalg.LinAlgError("Overlap matrix is not positive definite")
    s_inv_sqrt = (v * (w ** -0.5)) @ v.T
    H0 = s_inv_sqrt @ fock_ev @ s_inv_sqrt
    if match_matlab_ascii:
        H0 = ascii_roundtrip(H0, sig_digits=8)
    return H0


def process_strand(strand_name: str, work_dir: Path, txt_path: Path = None):
    """Full readMAT.m equivalent for one strand.

    Writes <strand>_Fock.mat, <strand>_Overlap.mat, <strand>_eigen.mat, and
    <strand>_H0.mat (the orthogonalized Hamiltonian) into work_dir.

    NOTE: unlike readMAT.m, this does NOT overwrite <strand>.mat -- it
    writes H0 to <strand>_H0.mat instead. readMAT.m's overwrite-in-place
    behavior is a MATLAB-workflow artifact (the raw Gaussian dump file
    <strand>.mat never existed as a persistent artifact separate from the
    text dump in this pipeline); this port keeps outputs unambiguous. If a
    consumer specifically needs <strand>.mat containing H0 to match the old
    layout, copy/rename <strand>_H0.mat.
    """
    work_dir = Path(work_dir)
    if txt_path is None:
        txt_path = work_dir / f"{strand_name}.txt"

    overlap, fock = parse_gaussian_dump(txt_path)

    sio.savemat(work_dir / f"{strand_name}_Fock.mat", {f"{strand_name}_Fock": fock})
    sio.savemat(work_dir / f"{strand_name}_Overlap.mat", {f"{strand_name}_Overlap": overlap})

    ev = compute_eigenvalues(overlap, fock)
    sio.savemat(work_dir / f"{strand_name}_eigen.mat", {"EV": ev.reshape(-1, 1)})

    H0 = compute_H0(overlap, fock)
    sio.savemat(work_dir / f"{strand_name}_H0.mat", {strand_name: H0})

    print(f"Finished getHamiltonian for {strand_name}!")
    return overlap, fock, ev, H0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: readmat_parse.py <strand_name> [work_dir]")
        sys.exit(1)
    strand = sys.argv[1]
    wd = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    process_strand(strand, wd)
