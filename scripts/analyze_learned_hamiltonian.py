#!/usr/bin/env python3
"""Compare a trained hamiltonian model's LEARNED H to the known synthetic tight-binding
parameters (Roche onsites, Voityuk stacking + H-bond couplings).

Extracts, from the model's internal H over many random duplexes:
  - onsite(base)        from H[i,i] on the primary strand
  - backbone |coupling| from |H[i,i+1]| (dinucleotide step), sign is a gauge -> compare magnitude
  - H-bond |coupling|   from |H[i, partner]| (Watson-Crick pair)
and compares each to the ground-truth values baked into g3nat.utils.create_hamiltonian.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch_geometric.data import Batch
import g3nat
from g3nat.graph import sequence_to_graph
from g3nat.evaluation.inference import (per_base_onsite_from_args,
                                        drop_legacy_alpha_state)

# ---- ground truth (verbatim from g3nat/utils/physics.py::create_hamiltonian) ----
ONSITE_TRUE = {'A': -0.49, 'T': -1.39, 'G': 0.00, 'C': -1.12}
HBOND_TRUE = {'AT': 0.034, 'TA': 0.034, 'CG': 0.050, 'GC': 0.050}  # others 0
NN_TRUE = {'AA':0.030,'CC':0.041,'GG':0.084,'TT':0.158,'AT':0.105,'AG':0.049,
           'AC':0.061,'CT':0.100,'CG':0.042,'CA':0.029,'GT':0.137,'GC':0.110,
           'GA':0.089,'TA':0.086,'TC':0.076,'TG':0.085}
COMP = {'A':'T','T':'A','G':'C','C':'G'}
BASES = 'ATGC'

def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']; eg = np.asarray(ck['energy_grid'], dtype=float)
    n_orb = int(a.get('n_orb', 1))
    assert n_orb == 1, (
        f"this script reads diag(H) assuming n_orb=1 but the checkpoint has n_orb="
        f"{n_orb}; use g3nat.evaluation.physicality.onsite_block_eigs instead")
    m = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=a['hidden_dim'], num_layers=a['num_layers'], num_heads=a['num_heads'],
        energy_grid=eg, n_orb=a['n_orb'], conv_type=a.get('conv_type', 'gat'),
        per_base_onsite=per_base_onsite_from_args(a, path))
    # Pre-boolean checkpoints carry the removed alpha-mix state.
    ck['model_state_dict'] = drop_legacy_alpha_state(
        ck['model_state_dict'], m.per_base_onsite)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, a

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'outputs_regen_transformer/hamiltonian_tb_model.pth'
    model, a = load_model(path)
    print(f"loaded {path}  (conv={a.get('conv_type')}, hidden={a['hidden_dim']}, n_orb={a['n_orb']})")

    rng = np.random.RandomState(7)
    onsite = {b: [] for b in BASES}
    nn = {}; hb = {}
    N = 300
    for _ in range(N):
        L = int(rng.randint(6, 11))
        seq = ''.join(rng.choice(list(BASES), L))
        cseq = ''.join(COMP[b] for b in seq)[::-1]   # WC complement, 5'->3', full duplex
        data = sequence_to_graph(seq, cseq, left_contact_positions=0, right_contact_positions=L-1)
        with torch.no_grad():
            model(Batch.from_data_list([data]))
        H = model.H[0].cpu().numpy()   # [2L, 2L], primary strand = indices 0..L-1
        for i in range(L):
            onsite[seq[i]].append(H[i, i])
        for i in range(L-1):
            # physics.py puts nn_energies[seq[i+1]+seq[i]] on the (i, i+1) bond
            # (higher-index base first), so key the learned coupling the same way.
            nn.setdefault(seq[i+1]+seq[i], []).append(abs(H[i, i+1]))
        for i in range(L):                       # primary i pairs with comp node at L + (L-1-i)
            j = L + (L-1-i)
            hb.setdefault(seq[i]+cseq[L-1-i], []).append(abs(H[i, j]))

    # ---- report ----
    print("\n== ONSITE  (signed; H[i,i] on primary strand) ==")
    print(f"{'base':>4} {'model mean':>11} {'std':>7} {'true':>7}")
    allm, allt = [], []
    for b in BASES:
        v = np.array(onsite[b]); print(f"{b:>4} {v.mean():>+11.3f} {v.std():>7.3f} {ONSITE_TRUE[b]:>+7.3f}")
        allm += list(v); allt += [ONSITE_TRUE[b]]*len(v)
    allm, allt = np.array(allm), np.array(allt)
    print(f"global offset (mean model - mean true) = {allm.mean()-allt.mean():+.3f}")
    print(f"onsite Pearson r (per-sample vs true)  = {np.corrcoef(allm, allt)[0,1]:.3f}")

    print("\n== BACKBONE COUPLING  |H[i,i+1]|  vs Voityuk nn_energies ==")
    print(f"{'step':>4} {'model mean':>11} {'std':>7} {'true':>7}")
    ms, ts = [], []
    for s in sorted(NN_TRUE):
        v = np.array(nn.get(s, [np.nan]))
        print(f"{s:>4} {np.nanmean(v):>11.3f} {np.nanstd(v):>7.3f} {NN_TRUE[s]:>7.3f}")
        ms.append(np.nanmean(v)); ts.append(NN_TRUE[s])
    ms, ts = np.array(ms), np.array(ts)
    print(f"backbone Pearson r (16 steps, model vs true) = {np.corrcoef(ms, ts)[0,1]:.3f}")
    print(f"backbone best-fit scale (model = k*true)     = {np.sum(ms*ts)/np.sum(ts*ts):.3f}")

    print("\n== H-BOND COUPLING  |H[i,partner]|  vs Voityuk (WC pairs) ==")
    print(f"{'pair':>5} {'model mean':>11} {'std':>7} {'true':>7}")
    for p in ['AT', 'TA', 'CG', 'GC']:
        v = np.array(hb.get(p, [np.nan]))
        print(f"{p:>5} {np.nanmean(v):>11.3f} {np.nanstd(v):>7.3f} {HBOND_TRUE[p]:>7.3f}")

if __name__ == '__main__':
    main()
