import os
import numpy as np
import pytest
from g3nat.graph import geometry

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
DATASET = "/mmfs1/gscratch/anantram/asyed4/DNADataSet"


def test_parse_dssr_out_aaac():
    out = open(os.path.join(FIX, "aaac.out")).read()
    r = geometry.parse_dssr_out(out)
    bp, step = r["bp_pars"], r["step_pars"]
    assert bp.shape == (4, 6), bp.shape          # aaac = 4 base pairs
    assert step.shape == (3, 6), step.shape      # 3 steps, deduped from the doubled block
    # first bp row: Shear,Stretch,Stagger,Buckle,Propeller,Opening
    np.testing.assert_allclose(bp[0], [0.00, -0.09, -0.00, 0.01, -1.23, -2.68], atol=1e-6)
    # first step row: Shift,Slide,Rise,Tilt,Roll,Twist
    np.testing.assert_allclose(step[0], [0.00, -0.20, 3.37, 0.01, -2.81, 35.90], atol=1e-6)


def test_base_centroids_and_distances():
    pdb = os.path.join(FIX, "aaac.pdb")
    cent = geometry.base_centroids(pdb)
    # aaac: strand 0 = DA1 DA2 DA3 DC4 (resseq 1..4);
    #       strand 1 = DG5 DT6 DT7 DT8 -- resseq CONTINUES 5..8 (one TER between strands).
    assert (0, 1) in cent and (1, 8) in cent
    # Watson-Crick: primary resseq k pairs with comp resseq (9-k); DA1 : DT8
    d_pair = geometry.centroid_distance(cent[(0, 1)], cent[(1, 8)])
    assert 4.5 < d_pair < 7.5, d_pair          # ~6.0 A, NOT the ~0.09 A frame-origin degeneracy
    # stacking (neighbor) distance on strand 0
    d_stack = geometry.centroid_distance(cent[(0, 1)], cent[(0, 2)])  # DA1 : DA2
    assert 3.0 < d_stack < 4.5, d_stack        # ~3.7 A


def _toy_entry():
    # 4 base pairs, distinct param values so a mis-mapping is detectable.
    bp = np.array([[1, 2, 3, 4, 5, 6],
                   [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18],
                   [19, 20, 21, 22, 23, 24]], float)
    step = np.array([[101, 102, 103, 104, 105, 106],
                     [111, 112, 113, 114, 115, 116],
                     [121, 122, 123, 124, 125, 126]], float)  # Nstep = 3
    # primary stacked along +z (3.4 apart); comp ordered 5'->3' antiparallel so
    # comp[n-1-i] sits at the same helical level as primary[i] (WC pair, 6 A across).
    pc = np.array([[0, 0, 0], [0, 0, 3.4], [0, 0, 6.8], [0, 0, 10.2]], float)
    cc = np.array([[6, 0, 10.2], [6, 0, 6.8], [6, 0, 3.4], [6, 0, 0]], float)
    return {"bp_pars": bp, "step_pars": step,
            "primary_centroids": pc, "comp_centroids": cc}


def test_assemble_backbone_primary_and_hbond():
    g = geometry.assemble_graph_geometry("acgt", "acgt", _toy_entry())
    # primary backbone step i: [dist(p_i,p_{i+1})=3.4, *step[i]]
    np.testing.assert_allclose(g[("backbone", "primary", 0)], [3.4, 101, 102, 103, 104, 105, 106])
    np.testing.assert_allclose(g[("backbone", "primary", 2)], [3.4, 121, 122, 123, 124, 125, 126])
    # hbond pair i: [dist(p_i, c_{n-1-i})=6.0, *bp[i]]
    np.testing.assert_allclose(g[("hbond", 0)], [6.0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(g[("hbond", 3)], [6.0, 19, 20, 21, 22, 23, 24])


def test_assemble_backbone_complementary_shares_step():
    g = geometry.assemble_graph_geometry("acgt", "acgt", _toy_entry())
    # comp step j shares primary step index n-2-j (n=4): j=0->2, j=1->1, j=2->0
    np.testing.assert_allclose(g[("backbone", "complementary", 0)], [3.4, 121, 122, 123, 124, 125, 126])
    np.testing.assert_allclose(g[("backbone", "complementary", 1)], [3.4, 111, 112, 113, 114, 115, 116])
    np.testing.assert_allclose(g[("backbone", "complementary", 2)], [3.4, 101, 102, 103, 104, 105, 106])


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="DSSR dataset not present")
def test_build_cache_and_norm_stats(tmp_path):
    out = str(tmp_path / "geom.pkl")
    cache = geometry.build_geometry_cache(DATASET, out, sequences=["aaac", "aaat"])
    assert "aaac" in cache
    e = cache["aaac"]
    assert e["bp_pars"].shape == (4, 6) and e["step_pars"].shape == (3, 6)
    assert e["primary_centroids"].shape == (4, 3) and e["comp_centroids"].shape == (4, 3)
    assert os.path.exists(out)
    stats = geometry.compute_norm_stats(cache)
    for t in ("backbone", "hbond"):
        assert np.asarray(stats[t]["mean"]).shape == (7,)
        assert np.all(np.asarray(stats[t]["std"]) >= 1e-6)
    # hbond distance channel (slot 0) is the atom distance ~6, not the degenerate ~0.09
    assert stats["hbond"]["mean"][0] > 4.0


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="DSSR dataset not present")
def test_geometry_is_se3_invariant(tmp_path):
    src = os.path.join(DATASET, "aaac", "aaac.pdb")
    lines = open(src).read().splitlines()
    rng = np.random.RandomState(3)
    A = rng.randn(3, 3)
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.array([11.0, -22.0, 33.0])
    rot = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            v = Q @ xyz + t
            ln = ln[:30] + f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}" + ln[54:]
        rot.append(ln)
    rp = str(tmp_path / "aaac_rot.pdb")
    open(rp, "w").write("\n".join(rot) + "\n")

    p0 = geometry.parse_dssr_out(geometry.run_dssr(src, workdir=str(tmp_path)))
    p1 = geometry.parse_dssr_out(geometry.run_dssr(rp, workdir=str(tmp_path)))
    np.testing.assert_allclose(p0["step_pars"], p1["step_pars"], atol=0.05)
    np.testing.assert_allclose(p0["bp_pars"], p1["bp_pars"], atol=0.05)
    c0, c1 = geometry.base_centroids(src), geometry.base_centroids(rp)
    d0 = geometry.centroid_distance(c0[(0, 1)], c0[(0, 2)])
    d1 = geometry.centroid_distance(c1[(0, 1)], c1[(0, 2)])
    assert abs(d0 - d1) < 1e-3
