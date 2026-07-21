import os
import numpy as np
from g3nat.graph import geometry

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


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
