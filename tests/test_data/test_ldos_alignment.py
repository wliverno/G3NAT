from pathlib import Path

import torch

from g3nat.graph import sequence_to_graph

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'dataset' / 'aaac' / 'aaac.pdb'

RESNAME_TO_BASE = {'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C'}
# BASE_FEATURES in g3nat/graph/construction.py:14-19
ONEHOT_TO_BASE = {
    (1, 0, 0, 0): 'A',
    (0, 1, 0, 0): 'T',
    (0, 0, 1, 0): 'G',
    (0, 0, 0, 1): 'C',
}


def _pdb_residue_bases(path):
    """Base letter of each residue, in ascending resseq order, read from the PDB."""
    first_seen = {}
    for line in path.read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        resseq = int(line[22:26])
        resname = line[17:20].strip()
        first_seen.setdefault(resseq, RESNAME_TO_BASE[resname])
    return [first_seen[k] for k in sorted(first_seen)]


def test_graph_node_order_matches_pdb_residue_order():
    # LDOS row i is residue i+1 (aggregation orders by ascending resseq).
    # A DNA node's Hamiltonian index is (node index - 2), because nodes 0 and 1
    # are the contacts (hamiltonian.py:385). So H index i must be residue i+1.
    # If this ever stops holding, the LDOS loss silently trains against a
    # permuted target -- the shapes match either way.
    graph = sequence_to_graph(primary_sequence='AAAC', complementary_sequence='GTTT')

    rows = graph.x.to(torch.int64).tolist()
    assert rows[0] == [0, 0, 0, 0], 'node 0 must be the left contact'
    assert rows[1] == [0, 0, 0, 0], 'node 1 must be the right contact'

    graph_bases = [ONEHOT_TO_BASE[tuple(r)] for r in rows[2:]]
    pdb_bases = _pdb_residue_bases(FIXTURE)

    assert graph_bases == pdb_bases
    assert graph_bases == ['A', 'A', 'A', 'C', 'G', 'T', 'T', 'T']


def test_complementary_half_is_the_reverse_complement_not_the_complement():
    # aaac -> gttt, NOT tttg. Both strands are written 5'->3', so residue 5 (the
    # first complementary residue) pairs with residue 4, not residue 1.
    pdb_bases = _pdb_residue_bases(FIXTURE)

    assert ''.join(pdb_bases[:4]) == 'AAAC'
    assert ''.join(pdb_bases[4:]) == 'GTTT'


def test_graph_node_order_follows_sequence_position_not_base_identity():
    # The aaac fixture concatenates to A,A,A,C,G,T,T,T -- already in
    # alphabetical order, with three indistinguishable A's and three T's.
    # A bug that sorted or grouped nodes by base identity would pass that
    # test unchanged. TAGC is not alphabetically ordered and all four bases
    # are distinct, so this case fails if node order is ever sorted or
    # permuted. GCTA is the reverse complement of TAGC.
    graph = sequence_to_graph(primary_sequence='TAGC', complementary_sequence='GCTA')

    rows = graph.x.to(torch.int64).tolist()
    assert rows[0] == [0, 0, 0, 0], 'node 0 must be the left contact'
    assert rows[1] == [0, 0, 0, 0], 'node 1 must be the right contact'

    bases = [ONEHOT_TO_BASE[tuple(r)] for r in rows[2:]]
    assert bases == ['T', 'A', 'G', 'C', 'G', 'C', 'T', 'A']
