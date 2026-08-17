"""R4: the numpy reference NEGF must not plateau where the model does not.

`calculate_NEGF` used to `np.clip(..., 1e-16, None)`, so any model-vs-reference
tail figure compared a real tail against a flat line at -16. It now uses the
same smooth floor as the model's floor_mode='smooth': clamp at zero, add a
1e-38 log10(0) guard.
"""
import numpy as np

from g3nat.utils.physics import calculate_NEGF


def _deep_tail_case():
    # Weakly coupled 6-site chain, evaluated far off resonance: transmission
    # runs far below 1e-16 without being zero.
    n = 6
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = 1e-3
    GammaL = np.zeros(n)
    GammaR = np.zeros(n)
    GammaL[0] = 0.1
    GammaR[-1] = 0.1
    grid = np.array([3.0, 4.0], dtype=np.float64)
    return H, GammaL, GammaR, grid


def test_reference_transmission_is_not_plateaued_at_1e_minus_16():
    H, GammaL, GammaR, grid = _deep_tail_case()
    T, DOS = calculate_NEGF(H, GammaL, GammaR, grid)
    assert T.max() < 1e-16, (
        f"test case is not in the deep tail: max T = {T.max()}")
    # The old hard clip returned exactly 1e-16 at every one of these points.
    assert np.all(T > 0)
    assert np.all(T != 1e-16)
    # And the two energies must still be distinguishable -- a clip destroys the
    # ordering the tail figure is about.
    assert T[0] != T[1]
    assert np.all(DOS > 0)


def test_reference_floor_guards_log10_of_zero():
    # A structurally decoupled site pair gives an exactly-zero transmission;
    # the eps must keep log10 finite rather than the old clip pinning it to -16.
    n = 4
    H = np.zeros((n, n))
    H[0, 1] = H[1, 0] = 0.2  # site 2,3 block never reaches the right contact
    GammaL = np.array([0.1, 0.0, 0.0, 0.0])
    GammaR = np.array([0.0, 0.0, 0.0, 0.1])
    grid = np.array([0.0], dtype=np.float64)
    T, DOS = calculate_NEGF(H, GammaL, GammaR, grid)
    assert np.isfinite(np.log10(T)).all()
    assert np.isfinite(np.log10(DOS)).all()
    assert np.log10(T)[0] < -30.0, (
        "a decoupled chain must read far below the old -16 plateau, "
        f"got {np.log10(T)[0]}")
