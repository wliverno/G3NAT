#!/usr/bin/env python3
"""Collect the structured-onsite alpha sweep into the discriminator table.

For each cell checkpoint (outputs_<prefix>_a<alpha>_s<seed>/hamiltonian_pickle_model.pth):
  - final-epoch val loss (val_losses[-1]) + a convergence flag (tail slope), so we compare
    CONVERGED fits, not lucky dips. final-epoch is the honest object here (willll's call):
    the converged H is what we judge, and a per-alpha tail slope near 0 confirms convergence.
  - onsite + eigenvalue in-window fractions (physicality) and coupling bandwidth, measured on
    model.H over a FIXED sample of real DFT sequences (identical across cells -> comparable).
  - per-base baseline distinctness + eta^2 (how strongly onsite is tied to base identity).

!! THE DISCRIMINATOR LOGIC BELOW IS RETRACTED (2026-07-24). Kept for provenance only. !!

The original reading was: "val_loss flat as alpha->1 => onsite needs little context, a
per-base baseline suffices, interpretable H at ~no fit cost, structured head is a real win;
rising sharply => the data wants context-dependent onsite."

It cannot discriminate either of those. For every alpha < 1 the mixing is a vacuous
reparametrization -- collapse the baseline table to a constant and rescale onsite_proj's last
layer by 1/(1-alpha) and you recover the free model exactly -- so the hypothesis class is
IDENTICAL across [0,1) and only alpha=1.0 changes it. The sweep measures how expensive the
free solution is to reach at each reparametrization, not how much context the data wants.
See docs/model-results.md, "CORRECTION: the alpha sweep does not measure what it was designed
to measure".

Separately, the per-base table it extracts is HOMO-referenced: G is pinned near 0 by the
energy convention for 495 of 515 sequences, so "G on top" is not a fit result.

The columns below are still correct as measurements. Only the interpretation was wrong.

Physicality uses a fixed DFT sequence sample (not each cell's val split) on purpose: it is a
property of the learned H, and holding the sample constant across alphas keeps the columns
comparable. The val LOSS is already the honest grouped-split number from training.

Usage:
  conda run -n g3nat python scripts/collect_onsite_sweep.py \
    --prefix onsite_sweep --alphas 0 0.25 0.5 0.75 0.9 1.0 --seed 42 --n 400
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch_geometric.data import Batch
import g3nat
from g3nat.graph import sequence_to_graph
from g3nat.data.pickle import load_single_pickle
from g3nat.evaluation.physicality import (onsite_metrics, eig_metrics,
                                          coupling_bandwidth, baseline_distinctness)

BASES = 'ATGC'
WINDOW = (-1.0, 1.0)


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    eg = np.asarray(ck['energy_grid'], dtype=float)
    m = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=a['hidden_dim'], num_layers=a['num_layers'], num_heads=a['num_heads'],
        energy_grid=eg, n_orb=a['n_orb'], conv_type=a.get('conv_type', 'gat'),
        structured_onsite=a.get('structured_onsite', False),
        alpha_granularity=a.get('alpha_granularity', 'global'),
        alpha_mode=a.get('alpha_mode', 'fixed'),
        alpha_value=a.get('alpha_value', 0.0), alpha_init=a.get('alpha_init', 0.9))
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, a, ck


def build_graphs(n, seed=0):
    """Fixed sample of unique real DFT sequences (dedup), same for every cell."""
    files = sorted(glob.glob('pickle_files/*.pkl'))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    graphs, seen = [], set()
    for f in files:
        if len(graphs) >= n:
            break
        d = load_single_pickle(f)
        if d is None:
            continue
        seq = d['sequence']
        if seq in seen:
            continue
        seen.add(seq)
        data = sequence_to_graph(
            primary_sequence=seq, complementary_sequence=d['complementary_sequence'],
            left_contact_positions=d['left_contact_pos'], right_contact_positions=d['right_contact_pos'],
            left_contact_coupling=float(d['coupling']), right_contact_coupling=float(d['coupling']))
        graphs.append((seq, data))
    return graphs


def convergence(val_losses, tail=50):
    """(final, tail_mean, tail_slope_per_epoch). |slope| ~ 0 => converged."""
    vl = np.asarray(val_losses, dtype=float)
    t = vl[-tail:] if len(vl) >= tail else vl
    slope = float(np.polyfit(np.arange(len(t)), t, 1)[0]) if len(t) > 1 else 0.0
    return float(vl[-1]), float(t.mean()), slope


def physicality_over(model, n_orb, graphs):
    onsite_diag, eig_fracs, bw = [], [], []
    per_base = {b: [] for b in BASES}
    for seq, data in graphs:
        with torch.no_grad():
            model(Batch.from_data_list([data]))
        H = model.H[0].cpu().numpy()
        onsite_diag.append(np.diag(H))
        eig_fracs.append(eig_metrics(H, WINDOW)['frac_eig_in_window'])
        bw.append(coupling_bandwidth(H))
        for i in range(len(seq)):
            blk = H[i * n_orb:(i + 1) * n_orb, i * n_orb:(i + 1) * n_orb]
            per_base[seq[i]].append(float(np.mean(np.diag(blk))))
    om = onsite_metrics(np.concatenate(onsite_diag), WINDOW)
    return {'ons_in_win': om['frac_in_window'], 'ons_min': om['min'], 'ons_max': om['max'],
            'eig_in_win': float(np.mean(eig_fracs)), 'coup_bw': float(np.mean(bw))}, per_base


def eta2(per_base):
    vals = {b: np.array(per_base[b]) for b in BASES if len(per_base[b])}
    allv = np.concatenate(list(vals.values()))
    grand = allv.mean()
    ss_tot = float(np.sum((allv - grand) ** 2))
    ss_bet = float(sum(len(v) * (v.mean() - grand) ** 2 for v in vals.values()))
    return ss_bet / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='onsite_sweep')
    ap.add_argument('--alphas', nargs='+', default=['0', '0.25', '0.5', '0.75', '0.9', '1.0'])
    ap.add_argument('--seed', default='42')
    ap.add_argument('--n', type=int, default=400, help='unique DFT sequences for physicality')
    args = ap.parse_args()

    graphs = build_graphs(args.n, seed=0)
    print(f"physicality over {len(graphs)} unique DFT sequences (fixed across cells)\n")
    cols = f"{'alpha':>6} {'val_loss':>9} {'tail_mean':>9} {'slope/ep':>10} {'ons_in_win':>10} {'eig_in_win':>10} {'coup_bw':>8} {'eta2':>6} {'distinct':>8}"
    print(cols)
    print('-' * len(cols))
    for a in args.alphas:
        path = f"outputs_{args.prefix}_a{a}_s{args.seed}/hamiltonian_pickle_model.pth"
        if not os.path.exists(path):
            print(f"{a:>6}   MISSING: {path}")
            continue
        m, margs, ck = load_model(path)
        fin, tmean, slope = convergence(ck['val_losses'])
        phys, per_base = physicality_over(m, margs['n_orb'], graphs)
        e2 = eta2(per_base)
        bl = m.onsite_baseline.detach().numpy() if getattr(m, 'structured_onsite', False) else np.zeros((4, 1))
        dist = baseline_distinctness(bl)['min_pairwise']
        print(f"{a:>6} {fin:>9.4f} {tmean:>9.4f} {slope:>+10.2e} {phys['ons_in_win']:>10.3f} "
              f"{phys['eig_in_win']:>10.3f} {phys['coup_bw']:>8.2f} {e2:>6.3f} {dist:>8.3f}")
    print("\nval_loss flat as alpha->1  => onsite needs little context (per-base baseline OK; H interpretable).")
    print("val_loss rises sharply     => data wants context-dependent onsite; per-base reduction inadequate.")
    print("(ons_in_win / eig_in_win rising WITH alpha at ~flat val_loss = the physical win.)")


if __name__ == '__main__':
    main()
