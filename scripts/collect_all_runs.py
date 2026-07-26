#!/usr/bin/env python3
"""Cross-seed collection over every finished run: alpha sweep, layers sweep,
learned-alpha, transformer check.

Deliberately does NO forward passes. Everything here comes straight out of the
checkpoint (val_losses, train_losses, args, onsite_baseline), so it is fast and
cannot be confounded by which sequences a physicality sample happened to draw.
Forward-pass metrics (eta^2) are a separate, slower pass.

The headline question: the single-seed run said "only G is resolved; A/T/C sit
within 0.14 eV". That was n=1. With three seeds we can finally ask whether each
pairwise gap exceeds the cross-seed scatter, which is the only thing that makes
"resolved" mean anything.

Usage (from a compute node):
  srun -A anantram-ckpt -p ckpt-all -c 2 --mem=8G -t 20 \
    conda run -n g3nat python scripts/collect_all_runs.py
"""
import sys, os, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

BASES = ['A', 'T', 'G', 'C']


def load_ck(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def curve_stats(ck, tail=50):
    """(final, tail_mean, tail_std, slope_per_epoch) from val_losses."""
    vl = np.asarray(ck.get('val_losses', []), dtype=float)
    if vl.size == 0:
        return None
    t = vl[-tail:] if vl.size >= tail else vl
    slope = float(np.polyfit(np.arange(len(t)), t, 1)[0]) if len(t) > 1 else 0.0
    return float(vl[-1]), float(t.mean()), float(t.std()), slope


def train_val_gap(ck, tail=50):
    """Overfit signature: (train_tail_mean, val_tail_mean, val_min, epochs_since_val_min)."""
    vl = np.asarray(ck.get('val_losses', []), dtype=float)
    tl = np.asarray(ck.get('train_losses', []), dtype=float)
    if vl.size == 0:
        return None
    vmin_i = int(np.argmin(vl))
    tmean = float(tl[-tail:].mean()) if tl.size else float('nan')
    return tmean, float(vl[-tail:].mean()), float(vl[vmin_i]), int(vl.size - 1 - vmin_i)


def baselines_of(ck):
    b = ck.get('model_state_dict', {}).get('onsite_baseline')
    return None if b is None else b.numpy().ravel()


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------- alpha sweep
section('1. ALPHA SWEEP  val_loss across seeds  (grouped split, 5000 epochs)')
ALPHAS = ['0', '0.25', '0.5', '0.75', '0.9', '1.0']
SEEDS = ['42', '43', '44']
print(f"{'alpha':>6} " + ' '.join(f'{"s"+s:>9}' for s in SEEDS) + f" {'mean':>9} {'std':>8}  n")
print('-' * 62)
alpha_means = {}
for a in ALPHAS:
    vals, cells = [], []
    for s in SEEDS:
        p = f'outputs_onsite_sweep_a{a}_s{s}/hamiltonian_pickle_model.pth'
        if not os.path.exists(p):
            cells.append('    --   ')
            continue
        st = curve_stats(load_ck(p))
        vals.append(st[0])
        cells.append(f'{st[0]:>9.4f}')
    if vals:
        alpha_means[a] = (float(np.mean(vals)), float(np.std(vals)))
        print(f'{a:>6} ' + ' '.join(cells) +
              f' {np.mean(vals):>9.4f} {np.std(vals):>8.4f}  {len(vals)}')
    else:
        print(f'{a:>6} ' + ' '.join(cells) + '   (no cells)')

if alpha_means:
    spread = max(m for m, _ in alpha_means.values()) - min(m for m, _ in alpha_means.values())
    typ_std = float(np.mean([s for _, s in alpha_means.values()]))
    print(f'\n  range of alpha-means = {spread:.4f}   typical across-seed std = {typ_std:.4f}')
    print('  If the range is not clearly larger than the seed scatter, the alpha curve is noise.')

# ------------------------------------------------------- baselines across seeds
section('2. PER-BASE BASELINES across seeds  -- does "only G is resolved" replicate?')
for a in ['0.9', '1.0']:
    rows = {}
    for s in SEEDS:
        p = f'outputs_onsite_sweep_a{a}_s{s}/hamiltonian_pickle_model.pth'
        if not os.path.exists(p):
            continue
        b = baselines_of(load_ck(p))
        if b is not None and b.size == 4:
            rows[s] = b
    if not rows:
        print(f'  alpha={a}: no cells')
        continue
    print(f'\n  alpha={a}')
    print('    seed  ' + '  '.join(f'{x:>9}' for x in BASES) + '   range')
    for s, b in rows.items():
        print(f'    {s:>4}  ' + '  '.join(f'{v:>9.4f}' for v in b) +
              f'   {b.max()-b.min():.4f}')
    M = np.stack(list(rows.values()))                     # [nseed, 4]
    # Referenced to G so a global energy offset per run cannot masquerade as scatter.
    Mg = M - M[:, [2]]
    print('    --- referenced to G (removes per-run global offset) ---')
    for s, row in zip(rows.keys(), Mg):
        print(f'    {s:>4}  ' + '  '.join(f'{v:>9.4f}' for v in row))
    mean_g, std_g = Mg.mean(0), Mg.std(0)
    print('    mean  ' + '  '.join(f'{v:>9.4f}' for v in mean_g))
    print('    std   ' + '  '.join(f'{v:>9.4f}' for v in std_g))
    if len(rows) > 1:
        print('\n    pairwise gap vs cross-seed scatter (RESOLVED needs gap >> scatter):')
        for i in range(4):
            for j in range(i + 1, 4):
                gap = abs(mean_g[i] - mean_g[j])
                sc = float(np.sqrt(std_g[i] ** 2 + std_g[j] ** 2))
                verdict = 'RESOLVED' if sc > 0 and gap > 2 * sc else 'not resolved'
                print(f'      {BASES[i]}-{BASES[j]}: gap {gap:6.3f}  scatter {sc:6.3f}  -> {verdict}')

# --------------------------------------------------------------- layers sweep
section('3. LAYERS SWEEP  -- does a smaller receptive field cost fit?')
found = sorted(glob.glob('outputs_layers_sweep_L*_s*/hamiltonian_pickle_model.pth'))
if not found:
    print('  no layers-sweep cells found')
else:
    by_L = {}
    for p in found:
        m = re.search(r'_L(\d+)_s(\d+)', p)
        L, s = m.group(1), m.group(2)
        st = curve_stats(load_ck(p))
        by_L.setdefault(L, {})[s] = st[0]
    print(f"{'layers':>7} {'cells':>28} {'mean':>9} {'std':>8}")
    print('-' * 56)
    for L in sorted(by_L, key=int):
        d = by_L[L]
        vals = list(d.values())
        cells = ' '.join(f's{s}={v:.4f}' for s, v in sorted(d.items()))
        print(f'{L:>7} {cells:>28} {np.mean(vals):>9.4f} {np.std(vals):>8.4f}')
    print('\n  Compare L=4 against the alpha=0 cells above -- same config, so any gap')
    print('  between them is pure run-to-run noise and is the error bar for this table.')

# ---------------------------------------------------------------- learned alpha
section('4. LEARNED ALPHA  -- where does it settle when free?')
for p in sorted(glob.glob('outputs_onsite_learned*/hamiltonian_pickle_model.pth')):
    ck = load_ck(p)
    th = ck.get('model_state_dict', {}).get('onsite_alpha_theta')
    st = curve_stats(ck)
    a_val = torch.sigmoid(th).numpy().ravel() if th is not None else None
    print(f'  {os.path.dirname(p)}')
    print(f'    final val {st[0]:.4f}   tail {st[1]:.4f}')
    print(f'    learned alpha = {np.round(a_val,4) if a_val is not None else "ABSENT"}')
    b = baselines_of(ck)
    if b is not None:
        print(f'    baselines A/T/G/C = {np.round(b,4)}  range {b.max()-b.min():.4f}')
    print('    (prediction on record: a naive learned alpha drifts LOW, toward context,')
    print('     because alpha<1 does not constrain the model class.)')

# ------------------------------------------------------------ transformer check
section('5. TRANSFORMER on the clean split -- overfitting?')
cands = [d for d in glob.glob('outputs_*tf*/hamiltonian_pickle_model.pth')
         + glob.glob('outputs_*transformer*/hamiltonian_pickle_model.pth')]
if not cands:
    print('  no transformer checkpoint found; searched outputs_*tf* / outputs_*transformer*')
else:
    for p in sorted(set(cands)):
        ck = load_ck(p)
        g = train_val_gap(ck)
        st = curve_stats(ck)
        conv = ck.get('args', {}).get('conv_type', '?')
        print(f'  {os.path.dirname(p)}  (conv={conv})')
        if g:
            tr, va, vmin, since = g
            print(f'    train tail {tr:.4f}   val tail {va:.4f}   gap {va-tr:+.4f}')
            print(f'    best val {vmin:.4f}, reached {since} epochs before the end')
            print(f'    tail slope {st[3]:+.2e}/epoch')
            print('    OVERFIT signature = val rising after its minimum (positive slope)')
            print('    AND a large train-val gap. Both must hold; a gap alone is not overfit.')

print('\ndone.')
