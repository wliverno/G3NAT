"""Results 3.4 worst-cell detail: the three ham_dos_n1_L4_nogeom seeds (CI median, companions, posthoc row, worst strands)."""
import os
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
import json, numpy as np
rep = json.load(open(os.path.join(_OUT, 'posthoc_v3_report.json')))['runs']
ci = json.load(open(os.path.join(_OUT, 'contact_invariance_v3.json')))['runs']
for name in sorted(k for k in ci if k.startswith('ham_dos_n1_L4_nogeom_')):
    r, c = rep[name], ci[name]
    v = np.asarray(c['vectors']['D_interior_mode'])
    print(name)
    print('   ci median=%.4f  p25=%.4f p75=%.4f  max=%.3f' % (np.median(v), *np.percentile(v, [25, 75]), v.max()))
    print('   companion:', {k: round(x, 4) for k, x in c['companion'].items()})
    print('   posthoc: epoch=%s val_T=%.4f l16_T=%.4f l16_dos=%.4f onsite_near=%.4f' % (
        r.get('saved_at_epoch', r.get('best_epoch')), r['val_transmission_at_selection'],
        r['l16_transmission'], r['l16_dos'], r['onsite_near']))
    print('   worst-3 strands:', [(w['sequence'], round(w['D_interior_mode'], 2)) for w in c['worst10_D_interior_mode'][:3]])
