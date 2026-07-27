import numpy as np
import torch
from torch_geometric.data import Batch

from g3nat.data.datasets import create_dna_dataset


def _build(ldos_list=None, ldos_target='residue'):
    # UPPERCASE is required: BASE_FEATURES is keyed 'A'/'T'/'G'/'C' and
    # g3nat/graph/construction.py does no case normalization. The real pipeline
    # only works because load_single_pickle calls .upper() (pickle.py:37).
    seqs = ['AAAC', 'AAAG']
    dos = np.zeros((2, 5), dtype=np.float64)
    trans = np.zeros((2, 5), dtype=np.float64)
    egrid = np.linspace(-1, 1, 5)
    ldos_data = None
    if ldos_list is not None:
        ldos_data = {'residue': ldos_list, 'base_only': ldos_list}
    return create_dna_dataset(
        sequences=seqs,
        dos_data=dos,
        transmission_data=trans,
        energy_grid=egrid,
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data=ldos_data,
        ldos_target=ldos_target,
    )


def test_no_ldos_attribute_when_dataset_has_none():
    # PyG DROPS an attribute assigned None -- it never becomes a key. So the
    # dataset must not assign it at all, and consumers must use hasattr.
    dataset = _build(ldos_list=None)

    data = dataset[0]

    assert not hasattr(data, 'ldos') or 'ldos' not in data.keys()


def test_ldos_attached_with_expected_shape():
    ldos = [np.arange(8 * 5, dtype=np.float64).reshape(8, 5) for _ in range(2)]
    dataset = _build(ldos_list=ldos)

    data = dataset[0]

    assert data.ldos.shape == (8, 5)
    assert data.ldos.dtype == torch.float


def test_batching_concatenates_on_dim_zero_and_reshapes():
    # Verified against torch_geometric 2.6.1: a 2-D custom attribute defaults
    # to __cat_dim__ == 0, so a batch of B graphs gives [B*2L, n_energy].
    ldos = [
        np.full((8, 5), 1.0),
        np.full((8, 5), 2.0),
    ]
    dataset = _build(ldos_list=ldos)

    batch = Batch.from_data_list([dataset[0], dataset[1]])

    assert batch.ldos.shape == (16, 5)
    reshaped = batch.ldos.view(2, 8, 5)
    assert torch.allclose(reshaped[0], torch.ones(8, 5))
    assert torch.allclose(reshaped[1], torch.full((8, 5), 2.0))


def test_ldos_target_selects_the_named_aggregation():
    residue = [np.full((8, 5), 1.0), np.full((8, 5), 1.0)]
    base_only = [np.full((8, 5), 9.0), np.full((8, 5), 9.0)]
    # UPPERCASE is required: BASE_FEATURES is keyed 'A'/'T'/'G'/'C' and
    # g3nat/graph/construction.py does no case normalization. The real pipeline
    # only works because load_single_pickle calls .upper() (pickle.py:37).
    seqs = ['AAAC', 'AAAG']
    dataset = create_dna_dataset(
        sequences=seqs,
        dos_data=np.zeros((2, 5)),
        transmission_data=np.zeros((2, 5)),
        energy_grid=np.linspace(-1, 1, 5),
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data={'residue': residue, 'base_only': base_only},
        ldos_target='base_only',
    )

    assert torch.allclose(dataset[0].ldos, torch.full((8, 5), 9.0))
