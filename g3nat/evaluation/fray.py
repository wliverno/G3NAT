"""Read-only probe: how the predicted Hamiltonian responds to terminal destacking.

Perturb the geometry of the terminal primary stacking edge and observe the whole
`model.H`. Pure helpers (edge location, region masks) + sweep + metrics; no model
or pipeline changes. Assumes n_orb=1 and the frozen sequence_to_graph node layout
(node 0/1 = contacts, primary bases at nodes 2..2+Np-1).
"""
import numpy as np
import torch


def terminal_backbone_rows(graph, n_primary):
    """edge_geom rows of the terminal primary backbone edge (both directed copies).

    Primary position k is node 2+k; the last two primary bases are nodes
    n_primary and n_primary+1. The terminal stacking edge is the backbone edge
    (edge_attr[:, 0] == 1) between them.
    """
    ei = graph.edge_index
    ea = graph.edge_attr
    tgt = {n_primary, n_primary + 1}
    rows = []
    for r in range(ei.shape[1]):
        if ea[r, 0] == 1 and {int(ei[0, r]), int(ei[1, r])} == tgt:
            rows.append(r)
    return rows


def terminal_h_indices(n_primary):
    """Hamiltonian indices (terminal base, stacked neighbor) for n_orb=1."""
    return (n_primary - 1, n_primary - 2)


def region_masks(n_dna, n_primary):
    """Boolean n_dna x n_dna masks decomposing the Hamiltonian into regions."""
    M = n_dna
    idx = np.arange(M)
    diag = np.eye(M, dtype=bool)
    offdiag = ~diag
    term = {n_primary - 1, n_primary - 2}
    ti = np.array([i in term for i in idx])
    terminal_local = ti[:, None] | ti[None, :]
    distal = offdiag & ~terminal_local
    is_primary = idx < n_primary
    primary = is_primary[:, None] & is_primary[None, :]
    comp = (~is_primary)[:, None] & (~is_primary)[None, :]
    cross = ~(primary | comp)
    return {"diag": diag, "offdiag": offdiag, "terminal_local": terminal_local,
            "distal": distal, "primary": primary, "comp": comp, "cross": cross}


def run_fray_sweep(model, graph, rows, deltas):
    """Stack model.H over the destack sweep (numpy [n_delta, M, M]).

    At each delta sets edge_geom[rows, 0] = d0 + delta and edge_geom[rows, 3] =
    r0 + delta (d0/r0 read from the unmorphed graph), forwards a 1-graph batch,
    and restores the caller's graph afterward. deltas[0] must be 0.
    """
    from torch_geometric.data import Batch
    assert float(deltas[0]) == 0.0, "deltas[0] must be 0 (unmorphed reference)"
    base = graph.edge_geom.clone()
    d0 = base[rows[0], 0].item()
    r0 = base[rows[0], 3].item()
    out = []
    model.eval()
    try:
        for delta in deltas:
            eg = base.clone()
            for r in rows:
                eg[r, 0] = d0 + float(delta)
                eg[r, 3] = r0 + float(delta)
            graph.edge_geom = eg
            with torch.no_grad():
                model(Batch.from_data_list([graph]))
            out.append(model.H[0].detach().cpu().numpy().copy())
    finally:
        graph.edge_geom = base
    return np.stack(out)


def sweep_metrics(Hstack, deltas, n_dna, n_primary):
    """Per-delta whole-Hamiltonian response metrics vs the unmorphed H (Hstack[0])."""
    masks = region_masks(n_dna, n_primary)
    t0, t1 = terminal_h_indices(n_primary)
    H0 = Hstack[0]
    n = Hstack.shape[0]
    term = np.empty(n)
    fro = np.empty(n)
    amax = np.empty((n, 2), int)
    region = {k: np.empty(n) for k in masks}
    for k in range(n):
        D = np.abs(Hstack[k] - H0)
        term[k] = abs(Hstack[k][t0, t1])
        fro[k] = float(np.sqrt((D ** 2).sum()))
        amax[k] = np.unravel_index(int(np.argmax(D)), D.shape)
        for name, msk in masks.items():
            region[name][k] = float(D[msk].sum())
    return {"term_coupling": term, "argmax_ij": amax, "fro": fro, "region": region}
