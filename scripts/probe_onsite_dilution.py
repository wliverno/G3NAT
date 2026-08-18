#!/usr/bin/env python3
"""Probe: is base identity DILUTED by message passing before the onsite head?

Hypothesis (structured-onsite-head motivation): the base one-hot is the input, but
the GNN conv layers mix in neighbor context, so by the onsite head the embedding is
context-dominated. Same-base sites are therefore UNTIED and free to diverge (some run
to -33 eV on the DFT model).

Observable signature of dilution:
  - LOW eta^2  (fraction of onsite variance explained by base identity)
  - LARGE within-base spread of onsite energies (same base -> very different onsite)
  - extreme min/max occurring WITHIN a single base group

Method:
  - Probe on the REAL DFT sequences (in-distribution for the DFT model), so we are not
    confounding dilution with the known OOD-extrapolation blowup (fray finding).
  - Report per-base stats + one-way variance decomposition (ANOVA eta^2).
  - CONTROL: run the identical probe, on the identical graphs, through the synthetic-TB
    GAT model. If base identity anchors onsite there (high eta^2, tight, means ~ Roche),
    the probe is sound and the DFT result is a training-data effect, not an artifact.

Usage:
  python scripts/probe_onsite_dilution.py \
      --dft trained_models/hamiltonian_DFT_gat_baseaware.pth \
      --tb  outputs_regen_gat/hamiltonian_tb_model.pth \
      --n 400
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch_geometric.data import Batch
import g3nat
from g3nat.graph import sequence_to_graph
from g3nat.data.pickle import load_single_pickle
from g3nat.evaluation.inference import (per_base_onsite_from_args,
                                        drop_legacy_alpha_state)

BASES = 'ATGC'
# Roche 2003 onsite energies. These ARE the ground truth for the synthetic TB control
# (g3nat/utils/physics.py generates that data from them), so the comparison is legitimate
# there. They are NOT ground truth for a DFT-trained model: DFT onsite here is HOMO-referenced
# per sequence, so absolute magnitudes are not comparable to a literature table, and G is
# pinned near 0 by the centring convention rather than by the fit (docs/dataset.md). When this
# column appears in a DFT model's report, read ordering only -- never agreement in magnitude.
ONSITE_ROCHE_SYNTHETIC_TRUTH = {'A': -0.49, 'T': -1.39, 'G': 0.00, 'C': -1.12}
ONSITE_TRUE = ONSITE_ROCHE_SYNTHETIC_TRUTH  # back-compat alias; prefer the explicit name
WINDOW = (-1.0, 1.0)  # transmission window used throughout the project


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    eg = np.asarray(ck['energy_grid'], dtype=float)
    m = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=a['hidden_dim'], num_layers=a['num_layers'], num_heads=a['num_heads'],
        energy_grid=eg, n_orb=a['n_orb'], conv_type=a.get('conv_type', 'gat'),
        per_base_onsite=per_base_onsite_from_args(a, path, ck['model_state_dict']))
    # Pre-boolean checkpoints carry the removed alpha-mix state.
    ck['model_state_dict'] = drop_legacy_alpha_state(
        ck['model_state_dict'], m.per_base_onsite)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, a


def build_graphs(n, seed=0):
    """Build graphs from real DFT pickles (dedup by sequence). Returns list of (seq, Data)."""
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
            primary_sequence=seq,
            complementary_sequence=d['complementary_sequence'],
            left_contact_positions=d['left_contact_pos'],
            right_contact_positions=d['right_contact_pos'],
            left_contact_coupling=float(d['coupling']),
            right_contact_coupling=float(d['coupling']),
        )
        graphs.append((seq, data))
    return graphs


def extract_onsite(model, n_orb, graphs):
    """Return dict base -> list of onsite energies (primary strand only, unambiguous)."""
    onsite = {b: [] for b in BASES}
    for seq, data in graphs:
        with torch.no_grad():
            model(Batch.from_data_list([data]))
        H = model.H[0].cpu().numpy()
        L = len(seq)
        for i in range(L):
            blk = H[i * n_orb:(i + 1) * n_orb, i * n_orb:(i + 1) * n_orb]
            onsite[seq[i]].append(float(np.mean(np.diag(blk))))
    return onsite


def variance_decomposition(onsite):
    """One-way ANOVA of onsite by base. Returns eta^2 and pooled within/between std."""
    all_vals = np.concatenate([np.array(onsite[b]) for b in BASES if len(onsite[b])])
    grand = all_vals.mean()
    ss_total = float(np.sum((all_vals - grand) ** 2))
    ss_between = 0.0
    within_var_pool_num = 0.0
    ntot = 0
    for b in BASES:
        v = np.array(onsite[b])
        if len(v) == 0:
            continue
        ss_between += len(v) * (v.mean() - grand) ** 2
        within_var_pool_num += np.sum((v - v.mean()) ** 2)
        ntot += len(v)
    ss_within = ss_total - ss_between
    eta2 = ss_between / ss_total if ss_total > 0 else float('nan')
    within_std = np.sqrt(within_var_pool_num / ntot)   # pooled within-base std
    between_std = np.sqrt(ss_between / ntot)            # between-base std (base "signal")
    return eta2, within_std, between_std, grand


def report(name, model, n_orb, graphs):
    onsite = extract_onsite(model, n_orb, graphs)
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"{'base':>4} {'n':>5} {'mean':>9} {'std':>8} {'min':>9} {'max':>9} "
          f"{'%in[-1,1]':>10} {'Roche':>7}")
    for b in BASES:
        v = np.array(onsite[b])
        if len(v) == 0:
            print(f"{b:>4} {'0':>5}  (no sites)")
            continue
        pin = 100.0 * np.mean((v >= WINDOW[0]) & (v <= WINDOW[1]))
        print(f"{b:>4} {len(v):>5} {v.mean():>+9.3f} {v.std():>8.3f} "
              f"{v.min():>+9.3f} {v.max():>+9.3f} {pin:>9.1f}% {ONSITE_TRUE[b]:>+7.2f}")
    eta2, wstd, bstd, grand = variance_decomposition(onsite)
    allv = np.concatenate([np.array(onsite[b]) for b in BASES if len(onsite[b])])
    pin_all = 100.0 * np.mean((allv >= WINDOW[0]) & (allv <= WINDOW[1]))
    print(f"\n  eta^2 (variance explained by base) = {eta2:.3f}   "
          f"[1.0 = fully anchored, 0.0 = fully diluted]")
    print(f"  between-base std (base signal)     = {bstd:.3f}")
    print(f"  within-base  std (dilution noise)  = {wstd:.3f}")
    print(f"  within/between ratio               = {wstd/bstd:.2f}   "
          f"[>1 means noise dominates signal]")
    print(f"  overall onsite range               = [{allv.min():+.2f}, {allv.max():+.2f}] eV")
    print(f"  overall %% in [-1,1] window          = {pin_all:.1f}%")
    return onsite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dft', default='trained_models/hamiltonian_DFT_gat_baseaware.pth')
    ap.add_argument('--tb', default='outputs_regen_gat/hamiltonian_tb_model.pth')
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()

    print(f"Building graphs from {args.n} real DFT sequences (dedup by sequence)...")
    graphs = build_graphs(args.n)
    lens = [len(s) for s, _ in graphs]
    print(f"Got {len(graphs)} unique sequences, lengths {min(lens)}-{max(lens)}, "
          f"{sum(lens)} total primary sites.")

    dft_model, da = load_model(args.dft)
    print(f"\nDFT model: conv={da.get('conv_type')}, hidden={da['hidden_dim']}, n_orb={da['n_orb']}")
    report("DFT-trained GAT  (the model under investigation)", dft_model, da['n_orb'], graphs)

    if os.path.exists(args.tb):
        tb_model, ta = load_model(args.tb)
        print(f"\nTB control model: conv={ta.get('conv_type')}, hidden={ta['hidden_dim']}, n_orb={ta['n_orb']}")
        report("Synthetic-TB GAT  (CONTROL: same arch, same graphs, physical training data)",
               tb_model, ta['n_orb'], graphs)
    else:
        print(f"\n(skipping TB control: {args.tb} not found)")

    print(f"\n{'='*72}")
    print("READ: dilution is confirmed if the DFT model has LOW eta^2 + large within-base")
    print("std + extreme min/max inside single base groups, WHILE the TB control (identical")
    print("graphs) stays tightly anchored (high eta^2, means ~ Roche, in-window).")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
