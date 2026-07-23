"""Stage 1 fray probe: destack the terminal primary stacking edge and record how the
whole predicted Hamiltonian responds. Read-only inference; writes outputs_fray/."""
import os
import sys
import csv
import json
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from g3nat.evaluation import load_trained_model
from g3nat.graph.construction import sequence_to_graph
from g3nat.graph.geometry import compute_norm_stats
from g3nat.evaluation.fray import (
    terminal_backbone_rows, terminal_h_indices, run_fray_sweep, sweep_metrics,
)

MODEL = "outputs_pickle_gat_geom/hamiltonian_pickle_model.pth"
CACHE = "geom_cache/geometry.pkl"
OUT = "outputs_fray"
# training sequences spanning terminal steps (validated present in the cache at runtime)
CANDIDATES = ["aaac", "ggtc", "gtcc", "cagc", "tttc", "caaa"]
DELTAS = np.linspace(0.0, 5.0, 35)


def complement(seq):
    c = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(c[b] for b in seq.upper())[::-1]


def main():
    os.makedirs(OUT, exist_ok=True)
    model, egrid, dev = load_trained_model(MODEL, device='cpu')
    assert getattr(model, 'n_orb', 1) == 1, "probe assumes n_orb=1"
    cache = pickle.load(open(CACHE, "rb"))

    stats = compute_norm_stats(cache)
    band = {"d": {"mean": stats["backbone"]["mean"][0], "std": stats["backbone"]["std"][0]},
            "rise": {"mean": stats["backbone"]["mean"][3], "std": stats["backbone"]["std"][3]}}
    json.dump(band, open(os.path.join(OUT, "norm_band.json"), "w"), indent=2)
    print(f"in-distribution backbone d: mean={band['d']['mean']:.3f} std={band['d']['std']:.4f}; "
          f"rise: mean={band['rise']['mean']:.3f} std={band['rise']['std']:.4f}")

    seqs = [s for s in CANDIDATES if s in cache][:4]
    rows_out = []
    Hsave = {}
    for seq in seqs:
        N = len(seq)
        g = sequence_to_graph(seq.upper(), complement(seq), geometry=cache[seq])
        rows = terminal_backbone_rows(g, N)
        if not rows:
            print(f"WARN {seq}: no terminal backbone geometry, skipping")
            continue
        d0 = g.edge_geom[rows[0], 0].item()
        H = run_fray_sweep(model, g, rows, DELTAS)
        m = sweep_metrics(H, DELTAS, 2 * N, N)
        Hsave[f"{seq}_H"] = H
        for k, delta in enumerate(DELTAS):
            rows_out.append([seq, float(delta), d0 + float(delta), m["term_coupling"][k],
                             int(m["argmax_ij"][k, 0]), int(m["argmax_ij"][k, 1]),
                             m["fro"][k], m["region"]["terminal_local"][k],
                             m["region"]["distal"][k], m["region"]["diag"][k],
                             m["region"]["cross"][k]])
        ti = terminal_h_indices(N)
        print(f"{seq} (N={N}, terminal H idx {ti}): fro[0]={m['fro'][0]:.3g} "
              f"max||D||_F={m['fro'].max():.4g} final argmax={tuple(m['argmax_ij'][-1])}")

    with open(os.path.join(OUT, "sweep_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "delta", "d", "term_coupling", "argmax_i", "argmax_j", "fro",
                    "terminal_local", "distal", "diag", "cross"])
        w.writerows(rows_out)
    np.savez_compressed(os.path.join(OUT, "Hmats.npz"), **Hsave)
    print(f"wrote {OUT}/sweep_metrics.csv, Hmats.npz, norm_band.json for {len(seqs)} sequences")


if __name__ == "__main__":
    main()
