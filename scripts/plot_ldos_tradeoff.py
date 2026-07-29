"""Plot held-out DOS+T and LDOS agreement against epoch, for one or more loss_b groups.

The point of the figure is the relationship between two curves that are measured the
same way at every loss_b:

  DOS + T (unweighted)  -- the transport objective, always trained
  LDOS agreement        -- measured at EVERY loss_b, trained only when loss_b > 0

At loss_b = 0 the LDOS curve is a passive observer, and it rises after ~epoch 400 while
DOS+T keeps falling (docs/model-results.md section 5c). Overlaying a loss_b > 0 group
answers the question Phase B exists for: does supervision stop that climb, and at what
cost in DOS+T?

BINNING. Curves are medians within bins uniform in log(epoch), NOT a fixed-epoch moving
average. A fixed window is wrong on a log axis -- it spans two decades at epoch 10 and a
sliver at epoch 10000, over-smoothing exactly where the structure is. An earlier version
of this figure used a 301-epoch mean and manufactured a smooth LDOS valley at epoch 201
that does not exist in the data.

Usage:
    python scripts/plot_ldos_tradeoff.py --out docs/figures/ldos-tradeoff.png \
        --group "outputs_ldos_A_b0.0_s*:b = 0 (baseline)"

    python scripts/plot_ldos_tradeoff.py --out docs/figures/ldos-b-comparison.png \
        --group "outputs_ldos_A_b0.0_s*:b = 0 (baseline)" \
        --group "outputs_ldos_B_b0.5_s*:b = 0.5"

Each --group is "GLOB:LABEL". Every directory matched must contain
hamiltonian_pickle_model_best.pth with a metric_history written by scripts/train.py.
"""
import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOS_T_KEY = 'val_dos_t_unweighted'
LDOS_KEYS = ('val_ldos_residue', 'val_ldos_base_only')


def load_group(pattern):
    """Return (ldos [nseed, nepoch], dost [nseed, nepoch], best_epochs, ldos_key)."""
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no directories match {pattern!r}")
    ldos, dost, best, key_used = [], [], [], None
    for d in dirs:
        path = os.path.join(d, 'hamiltonian_pickle_model_best.pth')
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing -- did that cell finish?")
        ck = torch.load(path, map_location='cpu', weights_only=False)
        mh = ck.get('metric_history')
        if not mh:
            raise SystemExit(f"{path} has no metric_history")
        # Whichever aggregation was trained is the finite one; the other is nan by design.
        key = next((k for k in LDOS_KEYS if np.isfinite(mh[0].get(k, np.nan))), None)
        if key is None:
            raise SystemExit(f"{path}: no finite LDOS metric in metric_history")
        if key_used is None:
            key_used = key
        elif key != key_used:
            raise SystemExit(f"{pattern}: mixes {key_used} and {key} across seeds")
        ldos.append([e[key] for e in mh])
        dost.append([e[DOS_T_KEY] for e in mh])
        best.append(int(np.asarray(ck['val_losses']).argmin()))
    n = min(len(x) for x in ldos)
    return (np.array([x[:n] for x in ldos]), np.array([x[:n] for x in dost]),
            best, key_used, len(dirs))


def log_bin(epochs, values, n_bins=140):
    """Median per seed within bins uniform in log(epoch); return centre, mean, min, max."""
    edges = np.unique(np.round(np.logspace(0, np.log10(epochs[-1]), n_bins + 1)).astype(int))
    centre, mean, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (epochs >= a) & (epochs < b)
        if not m.any():
            continue
        per_seed = np.median(values[:, m], axis=1)
        centre.append(np.exp(np.log(epochs[m]).mean()))
        mean.append(per_seed.mean())
        lo.append(per_seed.min())
        hi.append(per_seed.max())
    return (np.array(centre), np.array(mean), np.array(lo), np.array(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', action='append', required=True,
                    help='GLOB:LABEL, repeatable. First group is drawn as the reference.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bins', type=int, default=140)
    ap.add_argument('--title', default='Fitting DOS and transmission alone degrades '
                                       'where the model puts spectral weight')
    args = ap.parse_args()

    C_DOST, C_LDOS = '#0b6fa4', '#c2410c'
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for gi, spec in enumerate(args.group):
        pattern, _, label = spec.partition(':')
        ldos, dost, best, key, n_seed = load_group(pattern)
        epochs = np.arange(ldos.shape[1]) + 1
        # First group solid; later groups dashed, so a b>0 overlay is distinguishable
        # in greyscale as well as colour.
        style = '-' if gi == 0 else '--'
        alpha_band = 0.20 if gi == 0 else 0.13
        for values, colour, what in ((dost, C_DOST, 'DOS + T'), (ldos, C_LDOS, 'LDOS')):
            c, m, lo, hi = log_bin(epochs, values, args.bins)
            ax.fill_between(c, lo, hi, color=colour, alpha=alpha_band, lw=0)
            ax.plot(c, m, color=colour, lw=1.9, ls=style,
                    label=f'{what}  --  {label}')
        print(f"{label}: {n_seed} seeds, LDOS key={key}, "
              f"best-val epochs {sorted(e + 1 for e in best)}")
        c, m, _, _ = log_bin(epochs, ldos, args.bins)
        bm = int(np.mean(best))
        print(f"    binned LDOS: min {m.min():.4f} at epoch {c[m.argmin()]:.0f}; "
              f"at best-val {m[np.argmin(np.abs(c - bm))]:.4f}; final {m[-1]:.4f}")
        c, m, _, _ = log_bin(epochs, dost, args.bins)
        print(f"    binned DOS+T: at best-val {m[np.argmin(np.abs(c - bm))]:.4f}; "
              f"final {m[-1]:.4f}")

    ax.set_xscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('held-out Huber loss')
    ax.set_title(args.title, fontsize=10.5, loc='left', pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left', bbox_to_anchor=(.015, 1.0))
    ax.text(.30, .035, f'mean of seeds, band = seed range; '
                       f'median in {args.bins} log-spaced epoch bins',
            transform=ax.transAxes, fontsize=7.5, color='0.45')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18, lw=.6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
