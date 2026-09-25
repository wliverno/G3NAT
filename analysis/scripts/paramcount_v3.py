"""Parameter counts of the paper's model cells, split into encoder and readout.

Encoder = node_proj + edge_proj + convs + norms (+ geom_encoder when the geometry
channel is on); readout = onsite_proj + coupling_proj (Hamiltonian model) or
dos_proj + transmission_proj (direct model). Source of the settings-table numbers:
encoder 268,032 / 533,248 at 2 / 4 layers, +67,840 with geometry; readout
263,170 / 264,712 at n_orb = 1 / 2 (Hamiltonian), 117,650 (direct).

    G3NAT_ROOT=/path/to/G3NAT python scripts/paramcount_v3.py
"""
import os
import sys

REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from g3nat.evaluation import load_trained_model  # noqa: E402

CELLS = {
    'ham  ldosonly n2 L4 geom':   'outputs_v3/ham_ldosonly_n2_L4_geom_s1179027592/hamiltonian_pickle_model_best.pth',
    'ham  ldosonly n2 L4 nogeom': 'outputs_v3/ham_ldosonly_n2_L4_nogeom_s1179027592/hamiltonian_pickle_model_best.pth',
    'ham  dos n1 L2 nogeom':      'outputs_v3/ham_dos_n1_L2_nogeom_s1179027592/hamiltonian_pickle_model_best.pth',
    'direct dos L2 geom':         'outputs_v3/blind_dos_L2_geom_s1179027592/standard_pickle_model_best.pth',
    'direct dos L2 nogeom':       'outputs_v3/blind_dos_L2_nogeom_s1179027592/standard_pickle_model_best.pth',
    'direct dos L4 geom':         'outputs_v3/blind_dos_L4_geom_s1179027592/standard_pickle_model_best.pth',
}
ENCODER = ('node_proj', 'edge_proj', 'convs', 'norms', 'geom_encoder')

for name, rel in CELLS.items():
    model, _, _ = load_trained_model(os.path.join(REPO, rel), device='cpu')
    groups = {}
    for pname, t in model.named_parameters():
        groups[pname.split('.')[0]] = groups.get(pname.split('.')[0], 0) + t.numel()
    total = sum(groups.values())
    enc = sum(v for k, v in groups.items() if k in ENCODER)
    print('%-30s total=%9d  encoder=%9d  readout=%9d' % (name, total, enc, total - enc))
    print('     modules:', groups)
