"""Per-run provenance: the resolved config and code identity, written at start.

The campaign's legibility requirement: a run's exact parameters must be readable
from its artifacts, not reconstructed from runner scripts and defaults."""
import json
import os
import socket
import subprocess
import time


def _git(cmd, cwd):
    try:
        return subprocess.run(['git'] + cmd, cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def write_run_metadata(output_dir: str, args_dict: dict) -> str:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sha = _git(['rev-parse', 'HEAD'], repo) or 'unknown'
    dirty = bool(_git(['status', '--porcelain'], repo))
    try:
        import g3nat
        version = getattr(g3nat, '__version__', 'unknown')
    except Exception:
        version = 'unknown'
    meta = {
        'args': args_dict,
        'git_sha': sha,
        'git_dirty': dirty,
        'hostname': socket.gethostname(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'g3nat_version': version,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'resolved_config.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path
