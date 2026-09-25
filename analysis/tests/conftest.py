"""Shared fixtures for the merged doe_v3 test suite.

The module under test is loaded from ``../scripts`` RELATIVE to this file, so
the suite travels with the staged tree and has no absolute path in it. Set
G3NAT_DOE_SCRIPTS to point the suite at a different checkout of the script
(used to re-run the suite against the pre-fix original: the tests that then
fail are exactly the ones documenting the defects that were fixed).

Every fixture here is built from the script's OWN declared constants
(HAM_FACTORS / HAM_LEVELS / SEEDS), never copied from a real output file, so a
fixture can never silently encode the answer the test is checking.
"""
import inspect
import itertools
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get(
    'G3NAT_DOE_SCRIPTS',
    os.path.join(os.path.dirname(_HERE), 'scripts'))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import doe_v3 as D                                    # noqa: E402


@pytest.fixture(scope='session')
def mod():
    return D


# ---------------------------------------------------------------------------
# Signature shim: the pruned script takes (report, ci_report, best_epoch_cache,
# out=...); the pre-fix original took (report, out=..., ci_report=...,
# best_epoch_cache=..., best_epoch_cache_path=...). Every test in this suite
# supplies both side inputs, so the only difference is argument order.
# ---------------------------------------------------------------------------

def run_analysis(report, ci_report, best_epoch_cache, out):
    params = list(inspect.signature(D.run_analysis).parameters)
    if len(params) > 1 and params[1] == 'ci_report':
        return D.run_analysis(report, ci_report, best_epoch_cache, out=out)
    kw = {}
    if 'best_epoch_cache_path' in params:
        kw['best_epoch_cache_path'] = '(supplied directly)'
    return D.run_analysis(report, out=out, ci_report=ci_report,
                          best_epoch_cache=best_epoch_cache, **kw)


# ---------------------------------------------------------------------------
# Synthetic, schema-exact inputs
# ---------------------------------------------------------------------------

def cells_of(factors, levels):
    return [dict(zip(factors, combo))
            for combo in itertools.product(*(levels[f] for f in factors))]


def run_name(family, cell, seed):
    return (family + '_' + '_'.join(f"{k}{cell[k]}" for k in sorted(cell))
            + f"_s{seed}")


def make_runs(seed=0, resp_fn=None, sub_n=33):
    """A schema-exact 120-run report body: 96 ham + 24 blind.

    resp_fn(response, cell, seed, family) -> float overrides the default
    seeded-noise response, so a test can plant an exactly known effect.
    """
    rng = np.random.default_rng(seed)

    def value(r, cell, s, family):
        if resp_fn is not None:
            v = resp_fn(r, cell, s, family)
            if v is not None:
                return float(v)
        return float(rng.normal(5.0, 1.0))

    runs = {}
    for family, factors, levels in (
            ('ham', D.HAM_FACTORS, D.HAM_LEVELS),
            ('blind', D.BLIND_FACTORS, D.BLIND_LEVELS)):
        for cell in cells_of(factors, levels):
            for s in D.SEEDS:
                row = {'family': family, 'seed': s,
                       'sub_n_comparisons': sub_n}
                row.update(cell)
                for r in D.RESPONSES_COMMON:
                    row[r] = value(r, cell, s, family)
                if family == 'ham':
                    row[D.HAM_EXTRA_RESPONSE] = value(
                        D.HAM_EXTRA_RESPONSE, cell, s, family)
                    row['onsite_far'] = value('onsite_far', cell, s, family)
                    row['onsite_n_graphs'] = 80
                runs[run_name(family, cell, s)] = row
    return runs


def make_ci(runs, seed=1, n_strands=6):
    """A contact_invariance_v3.json-shaped report covering every ham run."""
    rng = np.random.default_rng(seed)
    out = {'runs': {}}
    for name, row in runs.items():
        if row['family'] != 'ham':
            continue
        out['runs'][name] = {
            'family': 'ham',
            'vectors': {
                'D_interior_mode':
                    (1.0 + 0.03 * rng.standard_normal(n_strands)).tolist(),
                'D_interior_total':
                    (1.5 + 0.03 * rng.standard_normal(n_strands)).tolist(),
            },
            'companion': {f: float(1.0 + rng.standard_normal() * 0.1)
                          for f in D.CI_COMPANION_FIELDS},
        }
    return out


def make_epochs(runs, seed=2, epoch_fn=None):
    """A best_epoch_extract.py-shaped cache covering every run."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, row in runs.items():
        e = {'family': row['family'], 'supervision': row['supervision'],
             'num_layers': row['num_layers'], 'geometry': row['geometry'],
             'seed': row['seed']}
        if row['family'] == 'ham':
            e['n_orb'] = row['n_orb']
        e['best_epoch'] = (int(epoch_fn(row)) if epoch_fn else
                           int(round(400.0 * np.exp(0.15 * rng.standard_normal()))))
        out[name] = e
    return out


@pytest.fixture
def runs():
    return make_runs()


@pytest.fixture
def ci_report(runs):
    return make_ci(runs)


@pytest.fixture
def epochs(runs):
    return make_epochs(runs)


def ham_cells():
    return cells_of(D.HAM_FACTORS, D.HAM_LEVELS)


def in_cell(row, cell, factors):
    return all(row[f] == cell[f] for f in factors)
