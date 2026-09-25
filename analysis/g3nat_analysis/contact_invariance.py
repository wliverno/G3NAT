"""Contact-invariance metric: drift of the learned H across the four contact setups
of one strand. Headline number is INTERIOR-SITE d_mode (the attachment-mode axis;
coupling drift is reported beside it, never merged)."""
import numpy as np

SETUPS = (('0.1', 'same'), ('0.1', 'cross'), ('0.6', 'same'), ('0.6', 'cross'))
SETUP_RUN_KEYS = ('run1', 'run2', 'run3', 'run4')   # convert_to_pickle.py RUN_MAP


def axis_decompose(y):
    """2x2 factorial decomposition of the four setup values along the last axis:
    coupling (0.1 vs 0.6 eV), mode (same vs cross attachment), interaction."""
    y = np.asarray(y, float)
    y00, y01, y10, y11 = (y[..., 0], y[..., 1], y[..., 2], y[..., 3])
    mean = y.mean(axis=-1)
    a = (y10 + y11 - y00 - y01) / 2.0
    b = (y01 + y11 - y00 - y10) / 2.0
    ab = (y00 + y11 - y01 - y10) / 2.0
    return {'d_total': np.sqrt(((y - mean[..., None]) ** 2).mean(axis=-1)),
            'd_coupling': np.abs(a) / 2.0,
            'd_mode': np.abs(b) / 2.0,
            'd_interaction': np.abs(ab) / 2.0}


def strand_onsite_drift(levels_by_setup, L, n_orb):
    """levels_by_setup: [4, 2L, n_orb] onsite levels in SETUPS order. RMS drift per
    site class (all / attached / interior) and axis. Attached = the union over setups
    of contact-attached H rows {0, L-1, L}."""
    lv = np.asarray(levels_by_setup, float)
    if lv.shape != (4, 2 * L, n_orb):
        raise ValueError(f'expected (4, {2*L}, {n_orb}), got {lv.shape}')
    d = axis_decompose(np.moveaxis(lv, 0, -1))       # each [2L, n_orb]
    attached = np.zeros(2 * L, bool)
    attached[[0, L - 1, L]] = True
    classes = {'all': np.ones(2 * L, bool), 'attached': attached,
               'interior': ~attached}
    out = {'per_site': {k: v.copy() for k, v in d.items()}}
    for cls, mask in classes.items():
        for axis in ('total', 'coupling', 'mode', 'interaction'):
            vals = d[f'd_{axis}'][mask]
            out[f'D_{cls}_{axis}'] = float(np.sqrt((vals ** 2).mean()))
    out['D_onsite'] = out['D_all_total']
    return out


def strand_coupling_drift(Hs, edge_pairs, L, n_orb):
    """Coupling-block drift. Hs is [4, 2L*n_orb, 2L*n_orb] in SETUPS order; edge_pairs
    the (m, m') DNA-site pairs carrying a backbone/hbond edge. Per edge and setup the
    singular values of the off-diagonal block (basis-rotation invariant at both sites),
    then the same RMS-over-setups / RMS-over-edges reduction as the onsite drift. The
    'signed' companion repeats the construction on the raw block entries."""
    Hs = np.asarray(Hs, float)
    n = 2 * L * n_orb
    if Hs.shape != (4, n, n):
        raise ValueError(f'expected (4, {n}, {n}), got {Hs.shape}')
    n_edges = len(edge_pairs)
    sv = np.zeros((4, n_edges, n_orb))
    signed = np.zeros((4, n_edges, n_orb * n_orb))
    for k in range(4):
        for e, (m, mp) in enumerate(edge_pairs):
            block = Hs[k, m * n_orb:(m + 1) * n_orb, mp * n_orb:(mp + 1) * n_orb]
            sv[k, e, :] = np.linalg.svd(block, compute_uv=False)  # descending
            signed[k, e, :] = block.reshape(-1)

    d = axis_decompose(np.moveaxis(sv, 0, -1))              # each [n_edges, n_orb]
    d_signed = axis_decompose(np.moveaxis(signed, 0, -1))   # each [n_edges, n_orb**2]

    out = {'per_edge': {k: v.copy() for k, v in d.items()},
           'per_edge_signed': {k: v.copy() for k, v in d_signed.items()}}
    axis_label = {'total': 'edges', 'coupling': 'coupling',
                  'mode': 'mode', 'interaction': 'interaction'}
    for axis, label in axis_label.items():
        vals = d[f'd_{axis}']
        out[f'D_coupling_{label}'] = float(np.sqrt((vals ** 2).mean())) if vals.size else 0.0
        vals_s = d_signed[f'd_{axis}']
        out[f'D_coupling_signed_{label}'] = float(np.sqrt((vals_s ** 2).mean())) if vals_s.size else 0.0
    return out


def frozen_rows(L, num_layers):
    """H rows whose site lies more than num_layers hops from EVERY contact node under
    BOTH attachment modes (coupling magnitude does not change the graph topology).

    Ladder convention: primary site i -> H row i; complementary site j -> H row L+j.
    Relabelling j' = L-1-j aligns the complementary rail with the primary one, rung
    partners p_i -- c'_i. Left electrode attaches to p_0 always; right electrode to
    p_{L-1} in 'same' mode and to c'_{L-1} in 'cross' mode. Hop distances:
      left        = 1 + i + [rail == c']
      right-same  = 1 + (L-1-i) + [rail == c']
      right-cross = 1 + (L-1-i) + [rail == p]
    """
    frozen = set()
    for i in range(L):
        d_left_p = 1 + i
        d_right_same_p = 1 + (L - 1 - i)
        d_right_cross_p = 1 + (L - 1 - i) + 1
        if (min(d_left_p, d_right_same_p) > num_layers
                and min(d_left_p, d_right_cross_p) > num_layers):
            frozen.add(i)
        # rail c' (relabelled complementary), original H row = 2L-1-i
        d_left_c = 1 + i + 1
        d_right_same_c = 1 + (L - 1 - i) + 1
        d_right_cross_c = 1 + (L - 1 - i)
        if (min(d_left_c, d_right_same_c) > num_layers
                and min(d_left_c, d_right_cross_c) > num_layers):
            frozen.add(2 * L - 1 - i)
    return frozen
