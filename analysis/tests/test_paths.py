import os
import re
import subprocess
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ANALYSIS / "scripts").glob("*.py"))
BAD = re.compile(r"/mmfs1|/gscratch|/home/|willll|G3NAT-internal|analysis/docs/|scratch/ci")


def test_no_absolute_paths_or_names():
    hits = []
    for p in list(SCRIPTS) + sorted((ANALYSIS / "g3nat_analysis").glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if BAD.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()[:80]}")
    assert hits == []


def test_paths_resolve_without_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "G3NAT_ROOT"}
    code = ("import runpy,sys; sys.argv=['x','--help']; "
            f"runpy.run_path({str(ANALYSIS / 'scripts' / 'posthoc_v3.py')!r}, run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
