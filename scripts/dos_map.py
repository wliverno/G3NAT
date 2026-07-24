"""2D DOS map: DFT atom-resolved DOS segmented per base pair (contour) with the
learned tight-binding onsite energies overlaid. v1 for a single sequence.

Assumptions (flagged for review):
- Energy centered as Energy - mean(Energy), matching the pickle loader, so the DFT DOS
  and the model's onsite energies share an energy reference.
- DOSAtom row order == PDB atom order == DFT/gjf atom order (same source). Asserted.
- Per-base-pair segmentation: pair p = primary residue p + its Watson-Crick partner.
- Model = the DFT model of record (outputs_pickle_gat), onsite = diag(model.H).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SEQ = "aaac"
DEFAULT_MODEL = "outputs_pickle_gat/hamiltonian_pickle_model.pth"
DATASET_ROOT = "/mmfs1/gscratch/anantram/asyed4/DNADataSet"
OUT = "outputs_dosmap"


def load_mat(path):
    try:
        from scipy.io import loadmat
        m = loadmat(path)
        return np.asarray(m["Energy"]).ravel(), np.asarray(m["DOSAtom"])
    except Exception as e:
        import h5py
        with h5py.File(path, "r") as f:
            return np.asarray(f["Energy"]).ravel(), np.asarray(f["DOSAtom"]).T


def parse_pdb_atoms(pdb):
    atoms = []          # (chain_index, resseq) per atom, in file order
    chain = 0
    for ln in open(pdb):
        if ln.startswith("TER"):
            chain += 1
            continue
        if ln.startswith(("ATOM", "HETATM")):
            atoms.append((chain, int(ln[22:26])))
    return atoms


def complement(seq):
    c = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(c[b] for b in seq.upper())[::-1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=DEFAULT_SEQ)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="", help="suffix on output filename to distinguish runs")
    args = ap.parse_args()
    SEQ, MODEL, TAG = args.seq, args.model, args.tag
    RUNDIR = f"{DATASET_ROOT}/{SEQ}/run1"
    MAT = f"{RUNDIR}/DOS_{SEQ}_gammaL_0.1_gammaR_0.1.mat"
    PDB = f"{DATASET_ROOT}/{SEQ}/{SEQ}.pdb"

    os.makedirs(OUT, exist_ok=True)
    E, DOSAtom = load_mat(MAT)                       # E [NE], DOSAtom [NAtoms, NE]
    # DOSAtom may come as [NAtoms, NE] or [NE, NAtoms]; orient so rows=atoms
    if DOSAtom.shape[0] == E.shape[0] and DOSAtom.shape[1] != E.shape[0]:
        DOSAtom = DOSAtom.T
    atoms = parse_pdb_atoms(PDB)
    print(f"NE={E.shape[0]} DOSAtom={DOSAtom.shape} PDB_atoms={len(atoms)}")
    assert DOSAtom.shape[0] == len(atoms), "atom count mismatch DOSAtom vs PDB"

    # group atoms -> residue key (chain, resseq); sum DOSAtom per residue
    from collections import OrderedDict
    groups = OrderedDict()
    for i, key in enumerate(atoms):
        groups.setdefault(key, []).append(i)
    res_keys = list(groups.keys())
    localDOS = np.array([DOSAtom[idx, :].sum(0) for idx in groups.values()])  # [Nres, NE]
    print(f"residues={len(res_keys)} keys(head)={res_keys[:6]}")

    N = len(SEQ)                                     # base pairs
    # primary chain 0 resseq 1..N ; comp chain 1 resseq N+1..2N ; pair p: primary p + comp (2N+1-p)
    def res_row(chain, resseq):
        return res_keys.index((chain, resseq))
    bp_dos = np.array([localDOS[res_row(0, p)] + localDOS[res_row(1, 2 * N + 1 - p)]
                       for p in range(1, N + 1)])    # [N, NE]

    # energy reference: center like the pickle loader
    Ec = E - E.mean()

    # model onsite energies = diag(model.H)
    from g3nat.evaluation import load_trained_model
    from g3nat.graph.construction import sequence_to_graph
    from torch_geometric.data import Batch
    import torch
    model, _, _ = load_trained_model(MODEL, device='cpu')
    g = sequence_to_graph(SEQ.upper(), complement(SEQ))
    with torch.no_grad():
        model(Batch.from_data_list([g]))
    H = model.H[0].detach().cpu().numpy()
    onsite = np.diag(H)                              # [2N], primary 0..N-1, comp N..2N-1
    # pair p (1..N): primary index p-1 ; comp index that pairs it = 2N - p
    prim_on = [onsite[p - 1] for p in range(1, N + 1)]
    comp_on = [onsite[2 * N - p] for p in range(1, N + 1)]
    print(f"onsite(diag H) range [{onsite.min():.3f}, {onsite.max():.3f}] eV")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    Xp = np.arange(1, N + 1)
    # DOS spans orders of magnitude -> log10 color scale (like transmission)
    logdos = np.log10(np.clip(bp_dos, 1e-4, None))
    cs = ax.contourf(Xp, Ec, logdos.T, levels=30, cmap="magma")
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("log10  DFT local DOS (per base pair)  [states/eV]")
    ax.scatter(Xp, prim_on, marker="o", s=70, facecolor="none",
               edgecolor="#56B4E9", lw=2, label="learned onsite (primary base)")
    ax.scatter(Xp, comp_on, marker="s", s=70, facecolor="none",
               edgecolor="#009E73", lw=2, label="learned onsite (complementary base)")
    ax.set_xlabel(f"base-pair position along {SEQ.upper()} (5' -> 3')")
    ax.set_ylabel("energy - E_mean  (eV)")
    ax.set_title(f"DFT local DOS along {SEQ.upper()} vs learned onsite energies  [{TAG or 'model'}]",
                 fontsize=12, weight="bold", loc="left")
    ax.set_xticks(Xp)
    ax.legend(frameon=True, fontsize=9, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    out = os.path.join(OUT, f"dos_map_{SEQ}{('_' + TAG) if TAG else ''}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
