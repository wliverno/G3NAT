"""Merge the retrained-campaign per-run JSONs into the two inputs doe_v4.py reads:
  outputs/posthoc_v4_report.json      {'provenance': ..., 'runs': {name: row}}
  outputs/best_epoch_cache_v4.json    {name: {factors..., 'best_epoch'}}

Sources (all report-only, written earlier):
  outputs/heldout_v4/<run>.json       held-out 12/16-bp T and DOS + val T (score_heldout_v4.py), 144 runs
  outputs/select_v4_val/<run>.json    val DOS, val per-residue LDOS, onsite near/far over all 2077
                                      records (select_v4_val.py), 96 Hamiltonian runs
Row keys follow posthoc_v3_report.json; the direct family gains 'pooling' (mean, sum) and its
sum-pooled run names keep their _sumpool_ infix. Hamiltonian rows gain val_ldos_residue.
"""
import glob
import json
import os
import re

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ANALYSIS, 'outputs')
HELD = os.path.join(OUT, 'heldout_v4')
SEL = os.path.join(OUT, 'select_v4_val')

HAM = re.compile(r'^ham_(?P<sup>dos|ldos|ldosonly|tonly)_n(?P<orb>[12])_L(?P<lay>[24])_(?P<geom>geom|nogeom)_s(?P<seed>\d+)$')
BLIND = re.compile(r'^blind_(?P<sup>dos|tonly)_L(?P<lay>[24])_(?P<geom>geom|nogeom)(?P<pool>_sumpool)?_s(?P<seed>\d+)$')


def main():
    held = {json.load(open(f))['run']: json.load(open(f)) for f in glob.glob(os.path.join(HELD, '*.json'))}
    sel = {json.load(open(f))['run']: json.load(open(f)) for f in glob.glob(os.path.join(SEL, '*.json'))}
    if len(held) != 144 or len(sel) != 96:
        raise SystemExit(f'FATAL: {len(held)} held-out rows (want 144), {len(sel)} validation rows (want 96)')
    runs, epochs = {}, {}
    for name, h in sorted(held.items()):
        m = HAM.match(name)
        if m:
            s = sel[name]
            row = {'family': 'ham', 'supervision': m['sup'], 'n_orb': int(m['orb']),
                   'num_layers': int(m['lay']), 'geometry': int(m['geom'] == 'geom'), 'seed': int(m['seed'])}
            for k in ('supervision', 'n_orb', 'num_layers', 'geometry', 'seed'):
                if s[k] != row[k]:
                    raise SystemExit(f'FATAL: {name}: {k} {s[k]!r} vs {row[k]!r}')
            if abs(s['val_transmission'] - h['val_transmission']) > 1e-12 or s['saved_at_epoch'] != h['saved_at_epoch']:
                raise SystemExit(f'FATAL: {name}: the two sources read different checkpoints')
            row.update(onsite_near=s['onsite_near'], onsite_far=s['onsite_far'],
                       onsite_n_graphs=s['onsite_n_graphs'], val_ldos_residue=s['val_ldos_residue'])
        else:
            m = BLIND.match(name)
            if not m:
                raise SystemExit(f'FATAL: unparsed run name {name}')
            pool = 'sum' if m['pool'] else 'mean'
            if h['pooling'] != pool:
                raise SystemExit(f'FATAL: {name}: name says {pool}, checkpoint says {h["pooling"]}')
            row = {'family': 'blind', 'supervision': m['sup'], 'n_orb': None, 'num_layers': int(m['lay']),
                   'geometry': int(m['geom'] == 'geom'), 'pooling': pool, 'seed': int(m['seed'])}
        row.update(val_transmission_at_selection=h['val_transmission'], saved_at_epoch=h['saved_at_epoch'],
                   l12_transmission=h['l12_transmission'], l12_dos=h['l12_dos'],
                   l16_transmission=h['l16_transmission'], l16_dos=h['l16_dos'])
        runs[name] = row
        epochs[name] = {k: row[k] for k in ('family', 'supervision', 'num_layers', 'geometry', 'seed')}
        if row['family'] == 'ham':
            epochs[name]['n_orb'] = row['n_orb']
        else:
            epochs[name]['pooling'] = row['pooling']
        epochs[name]['best_epoch'] = h['saved_at_epoch']
    report = {'provenance': {
        'sources': ['analysis/outputs/heldout_v4/*.json (score_heldout_v4.py)',
                    'analysis/outputs/select_v4_val/*.json (select_v4_val.py)'],
        'weights': 'best (own-objective argmin): outputs_v4/ham_*, outputs_v4/blind_*_sumpool_*, outputs_v3/blind_*',
        'onsite_records': 'all pickle_files_v2 records (2077)'}, 'runs': runs}
    json.dump(report, open(os.path.join(OUT, 'posthoc_v4_report.json'), 'w'), indent=1)
    json.dump(epochs, open(os.path.join(OUT, 'best_epoch_cache_v4.json'), 'w'), indent=1)
    print(f"wrote posthoc_v4_report.json ({len(runs)} runs) and best_epoch_cache_v4.json")


if __name__ == '__main__':
    main()
