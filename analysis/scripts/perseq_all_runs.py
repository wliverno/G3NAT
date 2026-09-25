"""Per-record Huber loss on log10 T for EVERY v3 run (96 Hamiltonian + 24 direct)
against all 32 held-out records. Dumps one CSV row per (run, record). Reuses the
scoring path of scripts/persequence_v3.py verbatim."""
import csv, glob, os, sys
import numpy as np, torch
_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)
sys.path.insert(0, os.path.join(_ANALYSIS, 'scripts'))
from g3nat.evaluation.inference import load_trained_model
from g3nat_analysis import evaluator as EV
from posthoc_v3 import load_geometry, _check_grid, HELDOUT_DIRS, GEOM_CACHES
from persequence_v3 import load_heldout_indexed

OUT = os.path.join(_ANALYSIS, 'outputs', 'perseq_all_runs.csv')
runs = sorted(d for d in glob.glob(os.path.join(REPO, 'outputs_v3', '*'))
              if os.path.isdir(d))
geometry_cache = load_geometry(list(GEOM_CACHES))
heldout = {L: load_heldout_indexed(L) for L in sorted(HELDOUT_DIRS)}
loss_fn = EV.LOSS_FUNCS['huber']
n_runs = 0
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['run_name', 'family', 'supervision', 'n_orb', 'num_layers', 'geometry',
                'seed', 'length', 'sequence', 'contact_run', 'huber_t'])
    for run_dir in runs:
        name = os.path.basename(run_dir)
        p = EV.parse_run_name(name)
        if p is None:
            raise SystemExit(f'FATAL: unparsed run dir {name}')
        weights = ('hamiltonian_pickle_model_best.pth' if p['family'] == 'ham'
                   else 'standard_pickle_model_best.pth')
        model, grid, _ = load_trained_model(os.path.join(run_dir, weights), device='cpu')
        use_geom = bool(getattr(model, 'use_geometry', False))
        if use_geom != bool(p['geometry']):
            raise SystemExit(f'FATAL: geometry mismatch {name}')
        for L, recs in heldout.items():
            _check_grid(grid, list(recs.values()), name)
            for (seq, rk), rec in recs.items():
                batch = EV.gated_batch(rec, grid, geometry_cache, use_geom)
                with torch.no_grad():
                    _, t_p = model(batch)
                t_p = t_p.squeeze(0).cpu().numpy()
                t_t = np.asarray(rec['transmission'], float).ravel()
                assert len(t_p) == len(t_t) == len(grid)
                w.writerow([name, p['family'], p['supervision'], p['n_orb'], p['num_layers'],
                            p['geometry'], p['seed'], L, seq, rk, f'{loss_fn(t_p, t_t):.6f}'])
        n_runs += 1
        print(f'{n_runs:3d} {name}', flush=True)
print(f'wrote {OUT} ({n_runs} runs)')
