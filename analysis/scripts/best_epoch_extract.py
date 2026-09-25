"""Extract the selected (best-checkpoint) epoch of all 120 campaign-v3 runs.

Read-only over outputs_v3/. Writes the JSON cache doe_v3.py's training-time section
reads, to the path given as argv[1] (conventionally outputs/best_epoch_cache.json).

    G3NAT_ROOT=/path/to/G3NAT python scripts/best_epoch_extract.py outputs/best_epoch_cache.json
"""
import json, os, sys
import torch

REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(REPO, 'outputs_v3')
out_path = sys.argv[1]

rows = {}
names = sorted(os.listdir(ROOT))
for name in names:
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        continue
    if name.startswith('ham_'):
        ckpt = os.path.join(d, 'hamiltonian_pickle_model_best.pth')
        parts = name.split('_')
        assert len(parts) == 6, name
        _, sup, norb, lay, geom, seed = parts
        assert norb.startswith('n') and lay.startswith('L') and seed.startswith('s')
        row = dict(family='ham', supervision=sup, n_orb=int(norb[1:]),
                   num_layers=int(lay[1:]),
                   geometry={'geom': 1, 'nogeom': 0}[geom], seed=int(seed[1:]))
    elif name.startswith('blind_'):
        ckpt = os.path.join(d, 'standard_pickle_model_best.pth')
        parts = name.split('_')
        assert len(parts) == 5, name
        _, sup, lay, geom, seed = parts
        row = dict(family='blind', supervision=sup, num_layers=int(lay[1:]),
                   geometry={'geom': 1, 'nogeom': 0}[geom], seed=int(seed[1:]))
    else:
        raise SystemExit('FATAL: unexpected directory %r in outputs_v3' % name)

    if not os.path.exists(ckpt):
        raise SystemExit('FATAL: missing published best checkpoint %s' % ckpt)
    d_ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    ep = d_ck.get('saved_at_epoch', d_ck.get('epoch'))
    if ep is None:
        raise SystemExit('FATAL: no saved_at_epoch/epoch in %s' % ckpt)
    row['best_epoch'] = int(ep)
    row['n_val_records'] = len(d_ck['val_losses']) if d_ck.get('val_losses') is not None else None
    row['selection_metric'] = d_ck.get('selection_metric')
    rows[name] = row
    print(name, ep, flush=True)

n_ham = sum(1 for r in rows.values() if r['family'] == 'ham')
n_blind = sum(1 for r in rows.values() if r['family'] == 'blind')
assert n_ham == 96, n_ham
assert n_blind == 24, n_blind
assert len(rows) == 120, len(rows)
assert all(r['best_epoch'] is not None for r in rows.values())
with open(out_path, 'w') as f:
    json.dump(rows, f, indent=1)
print('OK 96 ham + 24 blind ->', out_path)
