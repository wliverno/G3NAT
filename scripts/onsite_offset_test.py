"""Is the free model's 'context' term mostly absorbing the HOMO reference offset?

CLAIM UNDER TEST. Egrid is centered per-sequence on the HOMO, and E_HOMO is ~0.81 eV higher
for G-containing sequences than for AT-only ones. The onsite a model must produce is
    onsite_HOMOref(base, seq) = E_absolute(base) - E_HOMO(seq)
so the SAME base should sit ~0.7-0.8 eV LOWER in a G-containing duplex than in an AT-only one,
purely because the reference moved.

PREDICTIONS
  free model (alpha=0): its context head CAN represent a sequence-level offset, so onsite for
    a fixed base (A or T, which occur in both classes) should be systematically LOWER in
    G-containing sequences. Expected shift ~ -0.7 eV.
  per-base model (alpha=1.0): onsite IS a per-base constant, so the shift is EXACTLY 0 by
    construction. It cannot represent the offset -- which would explain its 24% fit penalty
    as a reference-frame artifact rather than a statement about DNA.

If the free model shows the predicted shift, the structured-onsite investigation was fighting
the energy convention, and the fix is an explicit per-sequence offset term.
"""
import sys, os, glob, re, pickle
sys.path.insert(0, '/mmfs1/gscratch/anantram/willll/G3NAT')
os.chdir('/mmfs1/gscratch/anantram/willll/G3NAT')
import numpy as np
import torch
from torch_geometric.data import Batch
import g3nat
from g3nat.graph import sequence_to_graph
from g3nat.data.pickle import load_single_pickle

MODELS = {
    'free (alpha=0)':     'outputs_onsite_sweep_a0_s42/hamiltonian_pickle_model.pth',
    'per-base (a=1.0)':   'outputs_onsite_sweep_a1.0_s42/hamiltonian_pickle_model.pth',
}


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    n_orb = int(a.get('n_orb', 1))
    assert n_orb == 1, (
        f"this script reads diag(H) assuming n_orb=1 but the checkpoint has n_orb="
        f"{n_orb}; use g3nat.evaluation.physicality.onsite_block_eigs instead")
    m = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=a['hidden_dim'], num_layers=a['num_layers'], num_heads=a['num_heads'],
        energy_grid=np.asarray(ck['energy_grid'], dtype=float), n_orb=a['n_orb'],
        conv_type=a.get('conv_type', 'gat'),
        structured_onsite=a.get('structured_onsite', False),
        alpha_granularity=a.get('alpha_granularity', 'global'),
        alpha_mode=a.get('alpha_mode', 'fixed'),
        alpha_value=a.get('alpha_value', 0.0), alpha_init=a.get('alpha_init', 0.9))
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m


# --- pick sequences: all AT-only, plus a matched set of G-containing ---------
files = sorted(glob.glob('pickle_files/*_run1.pkl'))
at_only, g_cont = [], []
for f in files:
    s = re.sub(r'_run\d+\.pkl$', '', os.path.basename(f))
    (at_only if not set(s.lower()) & set('gc') else g_cont).append(f)
# match on length distribution so length is not a confound
at_lens = [len(re.sub(r'_run\d+\.pkl$', '', os.path.basename(f))) for f in at_only]
g_sel = []
for L in at_lens:
    for f in g_cont:
        s = re.sub(r'_run\d+\.pkl$', '', os.path.basename(f))
        if len(s) == L and f not in g_sel:
            g_sel.append(f); break
print(f'AT-only: {len(at_only)}   G-containing (length-matched): {len(g_sel)}')


def onsite_by_base(model, path):
    d = load_single_pickle(path)
    if d is None:
        return None, None
    data = sequence_to_graph(
        primary_sequence=d['sequence'], complementary_sequence=d['complementary_sequence'],
        left_contact_positions=d['left_contact_pos'], right_contact_positions=d['right_contact_pos'],
        left_contact_coupling=float(d['coupling']), right_contact_coupling=float(d['coupling']))
    with torch.no_grad():
        model(Batch.from_data_list([data]))
    H = model.H[0].cpu().numpy()
    # Node order (construction.py:118-131): contacts, then primary IN ORDER, then
    # complementary IN ORDER. NOT reversed. The alpha=1.0 model is the control that
    # catches this: its onsite is a per-base constant, so any label scramble shows up
    # as a nonzero shift where zero is guaranteed.
    full = d['sequence'] + d['complementary_sequence']
    out = {}
    for i, b in enumerate(full[:H.shape[0]]):
        out.setdefault(b.upper(), []).append(float(H[i, i]))
    # raw Egrid (E_HOMO) needs the unprocessed pickle; load_single_pickle centers it
    raw = pickle.load(open(path, 'rb'))
    return out, float(np.mean(np.asarray(raw['Egrid']).ravel()))


for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f'{name}: MISSING'); continue
    m = load_model(path)
    agg = {'AT-only': {}, 'G-containing': {}}
    ehomo = {'AT-only': [], 'G-containing': []}
    for label, flist in [('AT-only', at_only), ('G-containing', g_sel)]:
        for f in flist:
            ob, eh = onsite_by_base(m, f)
            if ob is None:
                continue
            ehomo[label].append(eh)
            for b, v in ob.items():
                agg[label].setdefault(b, []).extend(v)

    print(f'\n=== {name} ===')
    print(f"  mean E_HOMO: AT-only {np.mean(ehomo['AT-only']):.4f}   "
          f"G-containing {np.mean(ehomo['G-containing']):.4f}   "
          f"delta {np.mean(ehomo['G-containing'])-np.mean(ehomo['AT-only']):+.4f} eV")
    print(f"  {'base':>5} {'AT-only':>18} {'G-containing':>18} {'shift':>9}")
    for b in 'ATGC':
        a_v = np.array(agg['AT-only'].get(b, []))
        g_v = np.array(agg['G-containing'].get(b, []))
        if a_v.size == 0 or g_v.size == 0:
            print(f'  {b:>5} {"n/a" if a_v.size==0 else f"{a_v.mean():8.4f} (n={a_v.size})":>18}'
                  f' {"n/a" if g_v.size==0 else f"{g_v.mean():8.4f} (n={g_v.size})":>18}'
                  f' {"--":>9}')
            continue
        print(f'  {b:>5} {a_v.mean():9.4f} (n={a_v.size:4d}) {g_v.mean():9.4f} (n={g_v.size:4d})'
              f' {g_v.mean()-a_v.mean():+9.4f}')
    print('  PREDICTION: free model shows ~-0.7 eV shift for A and T; per-base shows exactly 0.')
