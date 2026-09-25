"""Mean/SD of every edge geometry feature over the training and held-out sets.

Backs the paper's Geometry paragraph (rise SD, propeller SD, COM/centroid
distance SD, twist SD). Uses g3nat's own assemble_graph_geometry so the edge
definition (backbone vs hydrogen-bond pair edges, one row per edge) is never
re-derived here.

    python scripts/geometry_spread.py
    G3NAT_ROOT=/path/to/G3NAT python scripts/geometry_spread.py
"""
import os
import pickle
import sys

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)

import numpy as np  # noqa: E402
from g3nat.graph.geometry import assemble_graph_geometry  # noqa: E402

BACKBONE_NAMES = ['centroid_distance', 'shift', 'slide', 'rise',
                  'tilt', 'roll', 'twist']
PAIR_NAMES = ['centroid_distance', 'shear', 'stretch', 'stagger',
              'buckle', 'propeller', 'opening']

TRAIN_CACHE = os.path.join(REPO, 'geom_cache', 'geometry_v2.pkl')
HELDOUT_CACHE = os.path.join(REPO, 'geom_cache', 'geometry_heldout_L12_L16.pkl')
OUT = os.path.join(_ANALYSIS, 'outputs', 'geometry_spread.out')


def collect_rows(cache):
    """Assemble every backbone and pair (hbond) 7-vector across all sequences
    in `cache`, using assemble_graph_geometry's own edge definition.

    A sequence's own key doubles as both primary and complementary sequence
    argument: assemble_graph_geometry only uses the cached entry (centroids,
    bp_pars, step_pars) to build edges, not the sequence strings, so this is
    equivalent to calling it the way graph construction does.
    """
    back, pair = [], []
    for seq, entry in cache.items():
        geo = assemble_graph_geometry(seq, seq, entry)
        for edge_id, vec in geo.items():
            if edge_id[0] == 'backbone':
                back.append(vec)
            elif edge_id[0] == 'hbond':
                pair.append(vec)
    back_arr = np.array(back, dtype=float) if back else np.zeros((0, 7))
    pair_arr = np.array(pair, dtype=float) if pair else np.zeros((0, 7))
    return back_arr, pair_arr


def stats_table(rows, names):
    """{feature_name: (mean, population_sd)}, population SD (ddof=0)."""
    out = {}
    for i, name in enumerate(names):
        col = rows[:, i]
        out[name] = (float(col.mean()), float(col.std()))
    return out


def format_table(title, n_edges, table, names):
    lines = [f"{title} ({n_edges} edges)",
             f"{'feature':20s} {'mean':>12s} {'sd':>12s}"]
    for name in names:
        mean, sd = table[name]
        lines.append(f"{name:20s} {mean:12.4f} {sd:12.4f}")
    return lines


def main(argv=None):
    lines = []
    with open(TRAIN_CACHE, 'rb') as f:
        train_cache = pickle.load(f)
    tback, tpair = collect_rows(train_cache)
    lines.append(f"TRAINING SET: {len(train_cache)} duplexes "
                f"(pickle_files_v2), cache = "
                f"{os.path.relpath(TRAIN_CACHE, REPO)}")
    lines += format_table('backbone', tback.shape[0],
                          stats_table(tback, BACKBONE_NAMES), BACKBONE_NAMES)
    lines.append('')
    lines += format_table('pair (hydrogen bond)', tpair.shape[0],
                          stats_table(tpair, PAIR_NAMES), PAIR_NAMES)
    lines.append('')

    if os.path.exists(HELDOUT_CACHE):
        with open(HELDOUT_CACHE, 'rb') as f:
            heldout_cache = pickle.load(f)
        hback, hpair = collect_rows(heldout_cache)
        lines.append(f"HELD-OUT SET: {len(heldout_cache)} duplexes "
                    f"(L=12, L=16), cache = "
                    f"{os.path.relpath(HELDOUT_CACHE, REPO)}")
        lines += format_table('backbone', hback.shape[0],
                              stats_table(hback, BACKBONE_NAMES), BACKBONE_NAMES)
        lines.append('')
        lines += format_table('pair (hydrogen bond)', hpair.shape[0],
                              stats_table(hpair, PAIR_NAMES), PAIR_NAMES)
    else:
        lines.append("HELD-OUT SET: cache not found at "
                    f"{os.path.relpath(HELDOUT_CACHE, REPO)}, skipped")

    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(text)
    print(text)
    return text


if __name__ == '__main__':
    main(sys.argv[1:])
