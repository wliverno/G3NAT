import numpy as np
import pytest
from g3nat.data import create_dna_dataset


def test_cache_miss_is_a_hard_error():
    grid = np.linspace(-1, 1, 8)
    dos = [np.zeros(8)]
    trans = [np.zeros(8)]
    with pytest.raises(KeyError, match='gatt'):
        create_dna_dataset(sequences=['GATT'], dos_data=dos, transmission_data=trans,
                           energy_grid=grid, geometry_cache={'aaaa': object()})


def test_no_cache_still_fine():
    grid = np.linspace(-1, 1, 8)
    create_dna_dataset(sequences=['GATT'], dos_data=[np.zeros(8)],
                       transmission_data=[np.zeros(8)], energy_grid=grid,
                       geometry_cache=None)
