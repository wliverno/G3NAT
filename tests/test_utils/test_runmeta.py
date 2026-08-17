import json
import os
from g3nat.utils.runmeta import write_run_metadata


def test_writes_complete_metadata(tmp_path):
    path = write_run_metadata(str(tmp_path), {'n_orb': 2, 'learning_rate': 1e-3})
    assert os.path.basename(path) == 'resolved_config.json'
    with open(path) as f:
        meta = json.load(f)
    assert meta['args']['n_orb'] == 2
    for key in ('git_sha', 'git_dirty', 'hostname', 'timestamp', 'g3nat_version'):
        assert key in meta
