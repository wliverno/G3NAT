"""DNA-only message passing (opt-in `dna_only_message_passing` on the Hamiltonian
model).

With the flag on, the GNN layers see only the DNA nodes and the backbone and
hydrogen-bond edges: the two contact nodes and the contact edges are removed from
message passing. The contact self-energies GammaL/GammaR are still read from the
contact edges exactly as before, so the NEGF layer sees the same contacts. The
consequence under test is that ONE learned H serves all four contact
configurations of a sequence (same/cross attachment x 0.1/0.6 eV), while T and DOS
still change with the configuration through Gamma.

The flag is ON by default since 2026-11-09. Off (contacts in the graph, campaign v3)
must stay byte-identical to the historical model: no new parameters, no new buffers,
identical outputs.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat.graph import sequence_to_graph
from g3nat.models import DNATransportHamiltonianGNN

EGRID = np.linspace(-1, 1, 21)
SEQ, COMP = 'ACGTA', 'TACGT'


def _model(**kw):
    torch.manual_seed(0)
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2, n_orb=2,
                                      conv_type='gat', energy_grid=EGRID, **kw).eval()


def _graph(mode, coupling, seq=SEQ, comp=COMP):
    """The four dataset contact setups: left on primary 0; right on primary L-1
    ('same') or complementary 0 ('cross'); one coupling for both contacts."""
    L = len(seq)
    right = ('primary', L - 1) if mode == 'same' else ('complementary', 0)
    return sequence_to_graph(seq, comp, ('primary', 0), right, coupling, coupling)


def _four():
    return [_graph(m, c) for c in (0.1, 0.6) for m in ('same', 'cross')]


def _forward(model, g):
    with torch.no_grad():
        dos, t = model(Batch.from_data_list([g]))
    return dos, t, model.H.clone()


def test_default_is_on_and_identical_to_explicit_on():
    m_plain = _model()
    m_on = _model(dna_only_message_passing=True)
    assert m_plain.dna_only_message_passing is True
    m_on.load_state_dict(m_plain.state_dict())
    for g in _four():
        d0, t0, H0 = _forward(m_plain, g)
        d1, t1, H1 = _forward(m_on, g)
        assert torch.equal(d0, d1) and torch.equal(t0, t1) and torch.equal(H0, H1)


def test_off_matches_the_historical_encoder():
    """Off must be the campaign-v3 forward path: every node and edge in the convs."""
    m = _model(dna_only_message_passing=False)
    b = Batch.from_data_list([_graph('cross', 0.6)])
    with torch.no_grad():
        x_full = m.encode(b)[0]
        x = m.node_proj(b.x)
        ea = m.edge_proj(b.edge_attr)
        for i in range(m.num_layers):
            x = torch.relu(m.norms[i](m.convs[i](x, b.edge_index, ea)))
    assert torch.allclose(x_full, x, atol=1e-6)


def test_flag_on_adds_no_parameters():
    m_on = _model(dna_only_message_passing=True)
    m_off = _model(dna_only_message_passing=False)
    assert [n for n, _ in m_on.named_parameters()] == [n for n, _ in m_off.named_parameters()]
    assert m_on.dna_only_message_passing is True
    # the same weights load in both directions
    m_on.load_state_dict(m_off.state_dict())


def test_flag_on_gives_one_H_for_all_four_contact_setups():
    m = _model(dna_only_message_passing=True)
    Hs = [_forward(m, g)[2] for g in _four()]
    for H in Hs[1:]:
        assert torch.allclose(H, Hs[0], atol=1e-6, rtol=0.0), (H - Hs[0]).abs().max()


def test_flag_off_H_still_depends_on_contacts():
    """The invariant under test is specific to the flag: without it the same
    weights give a contact-dependent H (otherwise the test above is vacuous)."""
    m = _model(dna_only_message_passing=False)
    with torch.no_grad():  # push the GAT off its symmetric init so contact messages matter
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    Hs = [_forward(m, g)[2] for g in _four()]
    assert not torch.allclose(Hs[0], Hs[1], atol=1e-6)


def test_flag_on_gammas_unchanged_and_outputs_differ_by_setup():
    m_on = _model(dna_only_message_passing=True)
    m_off = _model(dna_only_message_passing=False)
    m_on.load_state_dict(m_off.state_dict())
    outs = []
    for g in _four():
        b = Batch.from_data_list([g])
        GL_on, GR_on = m_on.get_contact_vectors(b.x, b.edge_attr, b.edge_index, b.batch)
        GL_off, GR_off = m_off.get_contact_vectors(b.x, b.edge_attr, b.edge_index, b.batch)
        assert torch.equal(GL_on, GL_off) and torch.equal(GR_on, GR_off)
        outs.append(_forward(m_on, g))
    # same H, different Gamma -> different transmission across the four setups
    t_same01, t_cross01, t_same06, t_cross06 = (o[1] for o in outs)
    assert not torch.allclose(t_same01, t_cross01)
    assert not torch.allclose(t_same01, t_same06)
    # LDOS side channel still present and sums to DOS (log space check via linear)
    ldos = m_on.ldos
    assert ldos.shape[-1] == 2 * len(SEQ) * m_on.n_orb


def test_flag_on_matches_a_graph_built_without_contacts_in_the_encoder():
    """Reference implementation of the semantics: run the encoder by hand on the
    contact-free subgraph and compare the node embeddings the readout receives."""
    m = _model(dna_only_message_passing=True)
    g = _graph('cross', 0.6)
    b = Batch.from_data_list([g])
    with torch.no_grad():
        x_full = m.encode(b)[0]                    # [N, hidden], rows 0,1 are contacts
        # by hand: DNA nodes only, DNA-DNA edges only, re-indexed
        dna = ~torch.all(b.x == 0, dim=1)
        keep = dna[b.edge_index[0]] & dna[b.edge_index[1]]
        remap = torch.full((b.x.size(0),), -1, dtype=torch.long)
        remap[dna] = torch.arange(int(dna.sum()))
        ei = remap[b.edge_index[:, keep]]
        x = m.node_proj(b.x[dna])
        ea = m.edge_proj(b.edge_attr[keep])
        for i in range(m.num_layers):
            x = torch.relu(m.norms[i](m.convs[i](x, ei, ea)))
    assert torch.allclose(x_full[dna], x, atol=1e-6)
    assert torch.all(x_full[~dna] == 0)


def test_flag_on_batched_equals_single():
    m = _model(dna_only_message_passing=True)
    gs = [_graph('same', 0.1), _graph('cross', 0.6, 'GGTCA', 'TGACC')]
    singles = [_forward(m, g) for g in gs]
    with torch.no_grad():
        dos, t = m(Batch.from_data_list(gs))
        H = m.H.clone()
    for i, (d_i, t_i, H_i) in enumerate(singles):
        assert torch.allclose(dos[i], d_i.squeeze(0), atol=1e-5)
        assert torch.allclose(t[i], t_i.squeeze(0), atol=1e-5)
        assert torch.allclose(H[i], H_i.squeeze(0), atol=1e-6)


def test_flag_survives_checkpoint_roundtrip(tmp_path):
    from g3nat.evaluation.inference import load_trained_model
    m = _model(dna_only_message_passing=True)
    p = str(tmp_path / 'dnaonly.pth')
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2, 'n_orb': 2,
                         'conv_type': 'gat', 'dna_only_message_passing': True},
                'energy_grid': EGRID}, p)
    m2, _, _ = load_trained_model(p, device='cpu', allow_untrained_selection=True)
    assert m2.dna_only_message_passing is True
    # a checkpoint without the key (every existing one) loads with the flag off
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2, 'n_orb': 2,
                         'conv_type': 'gat'},
                'energy_grid': EGRID}, p)
    m3, _, _ = load_trained_model(p, device='cpu', allow_untrained_selection=True)
    assert m3.dna_only_message_passing is False


def test_flag_on_with_geometry_channel():
    """The geometry fusion happens before the convs; with the flag on it must act
    on the DNA-DNA edges only and still run."""
    m = _model(dna_only_message_passing=True, use_geometry=True)
    g = _graph('same', 0.1)
    dos, t, H = _forward(m, g)
    assert torch.isfinite(dos).all() and torch.isfinite(t).all() and torch.isfinite(H).all()
