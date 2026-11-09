"""--dna_only_message_passing is on by default (2026-11-09) and can be switched off.

The model-class default and the command-line default must agree, and the direct
model records False (it keeps the contacts in its graph).
"""
import importlib.util
import inspect
import pathlib
import sys

import pytest

from g3nat.models import DNATransportHamiltonianGNN

TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'
BASE = ['train.py', '--data_source', 'pickle', '--init_seed', '1']


@pytest.fixture(scope='module')
def train():
    spec = importlib.util.spec_from_file_location('train_under_test', TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(train, monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', BASE + extra)
    return train.parse_args()


def test_cli_default_is_on(train, monkeypatch):
    assert _parse(train, monkeypatch, []).dna_only_message_passing is True


def test_cli_can_switch_it_off(train, monkeypatch):
    assert _parse(train, monkeypatch, ['--no-dna_only_message_passing']).dna_only_message_passing is False


def test_explicit_on_still_accepted(train, monkeypatch):
    assert _parse(train, monkeypatch, ['--dna_only_message_passing']).dna_only_message_passing is True


def test_class_default_matches_cli_default():
    default = inspect.signature(DNATransportHamiltonianGNN.__init__).parameters['dna_only_message_passing'].default
    assert default is True


def test_direct_model_records_false():
    src = TRAIN_PY.read_text()
    assert ("if args.model_type in ('standard', 'standard_deep'):\n"
            "        # The direct model has no Hamiltonian GNN to restrict") in src
    i = src.index("# The direct model has no Hamiltonian GNN to restrict")
    j = src.index('write_run_metadata(args.output_dir', i)
    assert 'args.dna_only_message_passing = False' in src[i:j], (
        'the direct model must record False before the run metadata is written')
