"""Fray probe plots (Stage 1). Reads outputs_fray/ and writes 3 PNGs.

Palette: Okabe-Ito (colorblind-safe) for categorical series; perceptually-uniform
'magma' for the |dH| heatmap. Thin marks, recessive grid, finding-stating titles.
"""
import os
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs_fray"
# Okabe-Ito
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
INK = "#222222"
MUTED = "#888888"
BAND = "#DddddD"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#EEEEEE", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    rows = list(csv.DictReader(open(os.path.join(OUT, "sweep_metrics.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[r["seq"]].append({k: (v if k == "seq" else float(v)) for k, v in r.items()})
    for s in by:
        by[s].sort(key=lambda x: x["delta"])
    band = json.load(open(os.path.join(OUT, "norm_band.json")))
    H = np.load(os.path.join(OUT, "Hmats.npz"))
    return by, band, H


def dband(band):
    m, s = band["d"]["mean"], band["d"]["std"]
    return m - 3 * s, m + 3 * s


def plot_terminal_coupling(by, band):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    lo, hi = dband(band)
    ax.axvspan(lo, hi, color=BAND, alpha=0.6, lw=0, zorder=0)
    seq_colors = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"]]
    for i, (seq, rr) in enumerate(sorted(by.items())):
        d = [r["d"] for r in rr]
        t = [r["term_coupling"] for r in rr]
        ax.plot(d, t, color=seq_colors[i % 4], lw=1.8, label=seq.upper(), zorder=3)
    # physical expectation: coupling should DECAY toward 0
    d0 = by[sorted(by)[0]][0]["d"]
    t0 = by[sorted(by)[0]][0]["term_coupling"]
    xx = np.linspace(d0, max(r["d"] for r in by[sorted(by)[0]]), 50)
    ax.plot(xx, t0 * np.exp(-(xx - d0) / 0.6), "--", color=MUTED, lw=1.6,
            label="physical expectation (decay)", zorder=2)
    ax.scatter([d0], [t0], color=OI["black"], s=28, zorder=4)
    ax.annotate(f"real geometry\n= {t0:.2f} eV", (d0, t0), textcoords="offset points",
                xytext=(10, 14), fontsize=9, color=INK)
    ax.text(sum(dband(band)) / 2, ax.get_ylim()[1] * 0.42, "in-distribution\n(distance)",
            ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlabel("terminal stacking distance  d  (Angstrom)")
    ax.set_ylabel("|terminal stacking coupling|  (eV)")
    ax.set_title("Model predicts runaway GROWTH, not physical decay",
                 fontsize=12.5, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.99))
    fig.text(0.01, 0.005, "note: rise sigma in training ~ 0.005 A, so the model is already "
             "far out-of-distribution just past the real point (~30 sigma at +0.15 A).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "terminal_coupling.png"), bbox_inches="tight")
    plt.close(fig)


def plot_region_curves(by, band):
    seq = sorted(by)[0]
    rr = by[seq]
    d = np.array([r["d"] for r in rr])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    lo, hi = dband(band)
    ax.axvspan(lo, hi, color=BAND, alpha=0.6, lw=0, zorder=0)
    series = [("terminal_local", OI["vermillion"], "terminal-local (perturbed edge)"),
              ("diag", OI["green"], "onsite (diagonal)"),
              ("distal", OI["sky"], "distal (far bases)"),
              ("cross", OI["purple"], "cross-strand")]
    for key, col, lab in series:
        y = np.array([max(r[key], 1e-6) for r in rr])
        ax.plot(d, y, color=col, lw=1.9, label=lab, zorder=3)
    ax.plot(d, np.array([max(r["fro"], 1e-6) for r in rr]), color=OI["black"], lw=1.3,
            ls=":", label="total ||dH||_F", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("terminal stacking distance  d  (Angstrom)")
    ax.set_ylabel("summed |dH| over region  (eV, log)")
    ax.set_title("The blow-up is LOCALIZED to the perturbed edge, not spread",
                 fontsize=12.5, weight="bold", loc="left")
    ax.text(sum((lo, hi)) / 2, ax.get_ylim()[1] * 0.5, "in-distribution\n(distance)",
            ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "region_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(by, H):
    seq = sorted(by)[0]
    Hs = H[f"{seq}_H"]                       # [n_delta, M, M]
    deltas = [r["delta"] for r in by[seq]]
    ds = [r["d"] for r in by[seq]]
    N = int(Hs.shape[1] // 2)
    picks = [0, 1, len(deltas) - 1]          # real, tiny destack, deep OOD
    vmax = np.abs(Hs[picks[-1]] - Hs[0]).max()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, k in zip(axes, picks):
        D = np.abs(Hs[k] - Hs[0])
        im = ax.imshow(D, cmap="magma", vmin=0, vmax=max(vmax, 1e-9))
        ax.set_title(f"d = {ds[k]:.2f} A  (destack +{deltas[k]:.2f})", fontsize=10.5)
        ax.axhline(N - 0.5, color="white", lw=0.8, alpha=0.6)
        ax.axvline(N - 0.5, color="white", lw=0.8, alpha=0.6)
        ax.add_patch(plt.Rectangle((N - 2.5, N - 1.5), 1, 1, fill=False,
                                   edgecolor=OI["sky"], lw=1.6))  # terminal (3,2)
        ax.set_xticks(range(2 * N)); ax.set_yticks(range(2 * N))
        ax.set_xticklabels(list(range(N)) + [f"c{j}" for j in range(N)], fontsize=8)
        ax.set_yticklabels(list(range(N)) + [f"c{j}" for j in range(N)], fontsize=8)
        ax.set_xlabel("primary 0..N-1 | complementary", fontsize=8.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="|dH| (eV)")
    fig.suptitle(f"Where the Hamiltonian changes ({seq.upper()}): hot spot stays at the "
                 f"terminal stacking element (boxed), rest stays cold",
                 fontsize=12.5, weight="bold", x=0.02, ha="left")
    fig.savefig(os.path.join(OUT, "response_heatmaps.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    by, band, H = load()
    plot_terminal_coupling(by, band)
    plot_region_curves(by, band)
    plot_heatmaps(by, H)
    print("wrote", [f for f in os.listdir(OUT) if f.endswith(".png")])


if __name__ == "__main__":
    main()
