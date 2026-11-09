"""Readout pooling of the direct model (opt-in `readout_pooling='sum'`).

Default ('mean'): the GNN node embeddings are mean-pooled per graph and the two heads
map the pooled vector to log10 DOS and log10 T. A mean does not change with the number
of nodes, so the readout cannot tell a long duplex from a short one of similar
composition. 'sum' sums the same node embeddings instead (global_add_pool over the same
nodes, contacts included as in the mean); both heads read the summed vector.

Default must be byte-identical to the existing model; the flag adds no parameters.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat.graph import sequence_to_graph
from g3nat.models import DNATransportGNN

EGRID = np.linspace(-1, 1, 21)
SEQ, COMP = 'ACGTA', 'TACGT'


def _model(**kw):
    torch.manual_seed(0)
    return DNATransportGNN(hidden_dim=32, num_layers=2, num_heads=2, output_dim=len(EGRID),
                           conv_type='gat', **kw).eval()


def _graph(seq=SEQ, comp=COMP, coupling=0.1):
    return sequence_to_graph(seq, comp, ('primary', 0), ('complementary', 0), coupling, coupling)


def _forward(model, gs):
    with torch.no_grad():
        return model(Batch.from_data_list(gs))


def _node_embeddings(m, b):
    """The encoder by hand: projections and conv stack, as DNATransportGNN.forward."""
    x = m.node_proj(b.x)
    ea = m.edge_proj(b.edge_attr)
    for i in range(m.num_layers):
        x = torch.relu(m.norms[i](m.convs[i](x, b.edge_index, ea)))
    return x


def test_default_is_mean_with_identical_state_and_output():
    m_plain = _model()
    m_mean = _model(readout_pooling='mean')
    assert m_plain.readout_pooling == 'mean'
    assert [n for n, _ in m_plain.named_parameters()] == [n for n, _ in m_mean.named_parameters()]
    assert [n for n, _ in m_plain.named_buffers()] == [n for n, _ in m_mean.named_buffers()]
    m_mean.load_state_dict(m_plain.state_dict())
    d0, t0 = _forward(m_plain, [_graph()])
    d1, t1 = _forward(m_mean, [_graph()])
    assert torch.equal(d0, d1) and torch.equal(t0, t1)


@pytest.mark.parametrize('pooling,reduce', [('mean', torch.mean), ('sum', torch.sum)])
def test_readout_reads_the_pooled_node_embeddings(pooling, reduce):
    m = _model(readout_pooling=pooling)
    b = Batch.from_data_list([_graph()])
    with torch.no_grad():
        pooled = reduce(_node_embeddings(m, b), dim=0, keepdim=True)   # every node, contacts included
        dos, t = m(b)
    assert torch.allclose(dos, m.dos_proj(pooled).clamp_min(-25.0), atol=1e-5)
    assert torch.allclose(t, m.transmission_proj(pooled).clamp_min(-25.0), atol=1e-5)


def test_sum_adds_no_parameters_and_loads_the_same_weights():
    m_sum = _model(readout_pooling='sum')
    m_mean = _model()
    assert [n for n, _ in m_sum.named_parameters()] == [n for n, _ in m_mean.named_parameters()]
    m_sum.load_state_dict(m_mean.state_dict())


def test_sum_sees_length_where_mean_does_not():
    """A homopolymer and its doubled copy: interior embeddings are near-identical,
    so the mean barely moves while the sum grows with the node count."""
    m_sum, m_mean = _model(readout_pooling='sum'), _model()
    m_sum.load_state_dict(m_mean.state_dict())
    short, long_ = _graph('GGGG', 'CCCC'), _graph('GGGGGGGG', 'CCCCCCCC')
    with torch.no_grad():
        e_s = _node_embeddings(m_mean, Batch.from_data_list([short]))
        e_l = _node_embeddings(m_mean, Batch.from_data_list([long_]))
    mean_gap = (e_l.mean(0) - e_s.mean(0)).norm()
    sum_gap = (e_l.sum(0) - e_s.sum(0)).norm()
    assert sum_gap > 5 * mean_gap


def test_sum_batched_equals_single_across_lengths():
    m = _model(readout_pooling='sum')
    gs = [_graph(), _graph('GGTCAGG', 'CCTGACC'), _graph('AT', 'AT', 0.6)]
    singles = [_forward(m, [g]) for g in gs]
    dos, t = _forward(m, gs)
    for i, (d_i, t_i) in enumerate(singles):
        assert torch.allclose(dos[i], d_i.squeeze(0), atol=1e-5)
        assert torch.allclose(t[i], t_i.squeeze(0), atol=1e-5)


def test_readout_pooling_survives_checkpoint_roundtrip(tmp_path):
    from g3nat.evaluation.inference import load_trained_model
    m = _model(readout_pooling='sum')
    base = {'model_type': 'standard', 'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2,
            'conv_type': 'gat', 'dropout': 0.0}
    p = str(tmp_path / 'direct.pth')
    torch.save({'model_state_dict': m.state_dict(),
                'args': dict(base, readout_pooling='sum'),
                'energy_grid': EGRID}, p)
    m2, _, _ = load_trained_model(p, device='cpu', allow_untrained_selection=True)
    assert m2.readout_pooling == 'sum'
    # every existing checkpoint has no such key and loads with the historical mean
    torch.save({'model_state_dict': m.state_dict(), 'args': base, 'energy_grid': EGRID}, p)
    m3, _, _ = load_trained_model(p, device='cpu', allow_untrained_selection=True)
    assert m3.readout_pooling == 'mean'


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError):
        _model(readout_pooling='max')
