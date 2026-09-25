"""Whole-pipeline behaviour: family sizes, the sections and caveats that must
appear, the printed labels, and the CLI's contract with its output file.

Oracles: family sizes recomputed from the design (terms x fitted responses),
planted effects with a known unique answer, and the printed lines parsed back
and compared against the objects run_analysis returns.
"""
import json

import numpy as np
import pytest

import doe_v3 as D
from conftest import make_runs, make_ci, make_epochs, run_analysis


def _full(seed=2026):
    runs = make_runs(seed=seed)
    return runs, make_ci(runs), make_epochs(runs)


def test_the_full_pipeline_prints_every_section_with_the_documented_sizes():
    """Oracle: ham 15 terms x 7 responses = 105, blind 7 x 5 = 35, ham
    dispersion 10 x 8, blind dispersion 3 x 6, best-epoch 15 x 2 and 7 x 2 --
    each recomputed from the design rather than read off the output."""
    runs, ci, cache = _full()
    lines = []
    res = run_analysis({'runs': runs}, ci, cache, lines.append)
    txt = '\n'.join(lines)

    assert res['ham'].bh_m == 15 * 7 == 105
    assert res['blind'].bh_m == 7 * 5 == 35
    assert res['ham_dispersion'].bh_m == 10 * 8 == 80
    assert res['blind_dispersion'].bh_m == 3 * 6 == 18
    assert res['best_epoch']['ham'].bh_m == 15 * 2 == 30
    assert res['best_epoch']['blind'].bh_m == 7 * 2 == 14

    for marker in ('CAMPAIGN-V3 DOE', 'ALWAYS CHECK THIS LINE',
                   'FAMILY: HAM', 'FAMILY: BLIND',
                   'SEED-TO-SEED DISPERSION ANALYSIS',
                   'TRAINING TIME: BEST-CHECKPOINT EPOCH', 'HEAD-TO-HEAD',
                   'MOST STABLE', 'LEAST STABLE',
                   'MEAN SPREAD BY SUPERVISION ARM',
                   'mean over the 8 cells of each arm',
                   'mean over the 4 cells of each arm',
                   'RESPONSE: ci_interior_mode', 'COMPANION @ BEST',
                   'DESCRIPTIVE onsite_far', 'DESCRIPTIVE ci_interior_total',
                   'NO RATE, SLOPE, OR THRESHOLD METRIC'):
        assert marker in txt, marker

    # the sub_t removal ruling is stated, and sub_t is never a response block
    assert 'sub_t' in txt and 'REMOVED' in txt
    assert 'RESPONSE: sub_t' not in txt
    # ci_interior_total is descriptive only: no RESPONSE block of its own
    assert 'RESPONSE: ci_interior_total' not in txt
    # the dispersion caveat appears exactly once, and the channel is raw-scale
    assert txt.count(
        'runs are seeded but GPU reductions are nondeterministic') == 1
    assert 'Shapiro' not in txt
    assert 'log10' not in txt.lower().replace('log10_best_epoch', '')
    # every companion field is printed beside its cell (spec section 3.3)
    comp = [l for l in lines if 'COMPANION @' in l]
    for field in D.CI_COMPANION_FIELDS:
        assert any(field in l for l in comp), field


def test_every_location_response_gets_its_own_dispersion_block():
    runs, ci, cache = _full(seed=33)
    lines = []
    run_analysis({'runs': runs}, ci, cache, lines.append)
    txt = '\n'.join(lines)
    for r in D.DISPERSION_RESPONSES + ('best_epoch',):
        assert f"DISPERSION RESPONSE: {r}" in txt, r
    for r in D.CI_DESCRIPTIVE_ONLY + D.ONSITE_DESCRIPTIVE_ONLY:
        assert f"DISPERSION RESPONSE: {r}" not in txt
    assert ('saturated dispersion model on 8 cells leaves 0 residual df; '
            'main effects only') in txt
    assert 'df_res=4' in txt and 'n=8' in txt


def test_one_planted_main_effect_is_the_only_bh_survivor():
    """Oracle: a 2.0 geometry shift against a 0.01 seed noise floor on a single
    response. BH over m = 15 x 1 must return exactly {geometry}, with the
    threshold at 0.05/15."""
    rng = np.random.default_rng(1234)
    runs = make_runs(resp_fn=lambda r, c, s, f: (
        5.0 + (2.0 if c['geometry'] else 0.0) + rng.normal(0, 0.01)
        if r == 'l12_transmission' else None))
    D.validate_structure(runs)
    fa = D.analyze_family(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          ['l12_transmission'])
    assert fa.bh_m == 15 and fa.bh_k_star == 1
    assert {(r, t) for r, t, _p in fa.bh_survivors} == {
        ('l12_transmission', 'geometry')}
    assert fa.bh_threshold == pytest.approx(0.05 / 15)


def test_a_planted_effect_in_a_multi_response_family_brings_no_companions():
    """Oracle: a +40 shift on one response's geometry contrast in the full
    m=90 ham family. That term must survive and no OTHER term of the same
    response may."""
    rng = np.random.default_rng(20260916)
    runs = make_runs(resp_fn=lambda r, c, s, f: (
        rng.normal(5.0, 1.0) + (40.0 if (r == 'l16_transmission' and
                                         c['geometry']) else 0.0)))
    D.validate_structure(runs)
    fa = D.analyze_family(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          list(D.RESPONSES_COMMON) + [D.HAM_EXTRA_RESPONSE])
    assert fa.bh_m == 15 * 6 == 90
    surv = {(r, t) for r, t, _p in fa.bh_survivors}
    assert ('l16_transmission', 'geometry') in surv
    assert {t for r, t in surv
            if r == 'l16_transmission' and t != 'geometry'} == set()


def test_pure_noise_has_no_survivors_but_still_reports_best_and_worst_cells():
    """n=3 is the design: a null is a statement about power, not a failure. The
    best/worst cell contrast must still be reported for every response."""
    runs = make_runs(seed=4)
    D.validate_structure(runs)
    for family, factors, levels, resp in (
            ('ham', D.HAM_FACTORS, D.HAM_LEVELS,
             list(D.RESPONSES_COMMON) + [D.HAM_EXTRA_RESPONSE]),
            ('blind', D.BLIND_FACTORS, D.BLIND_LEVELS,
             list(D.RESPONSES_COMMON))):
        fa = D.analyze_family(runs, family, factors, levels, resp)
        assert fa.bh_k_star == 0, f"{family}: {fa.bh_k_star} of {fa.bh_m}"
        for rr in fa.responses.values():
            assert rr.best_cell_name and rr.worst_cell_name
            assert rr.best_mean <= rr.worst_mean


def test_the_blind_family_has_seven_terms_carrying_one_df_each():
    """Oracle: 2x2x2 saturated -> 7 terms, 7 df, 8 cells - 1."""
    fa = D.analyze_family(make_runs(seed=9), 'blind', D.BLIND_FACTORS,
                          D.BLIND_LEVELS, list(D.RESPONSES_COMMON))
    assert fa.bh_m == 7 * 5 == 35
    for rr in fa.responses.values():
        assert len(rr.terms) == 7
        assert sum(t.df for t in rr.terms) == 7


def test_the_announced_ham_family_size_is_the_m_actually_used():
    """FIX (label defect 2). The banner used to be computed from
    2**n_factors - 1 times len(ham_responses) BEFORE fitting, so a response
    that dropped out left two different, both-authoritative family sizes in the
    same report. Oracle: the two printed numbers must be the same number, with
    one response deliberately made untestable (seeds identical within every
    cell) so 15 tests leave the family."""
    runs = make_runs(seed=13, resp_fn=lambda r, c, s, f: (
        10.0 * c.get('n_orb', 1) if r == 'l16_dos' else None))
    ci, cache = make_ci(runs), make_epochs(runs)
    lines = []
    run_analysis({'runs': runs}, ci, cache, lines.append)

    announced = next(l for l in lines if 'ham BH family size is now' in l)
    a = int(announced.split('responses =')[1].split('(')[0].strip())
    used = int(next(l for l in lines
                    if l.startswith('BENJAMINI-HOCHBERG')).split('m=')[1])
    assert a == used == 15 * 6 == 90, f"announced m={a}, used m={used}"


def test_a_dropped_response_is_labelled_zero_replicate_variance_not_constant():
    """FIX (label defect 1, printed face). The response below varies from 10 to
    20 across cells while the three seeds agree exactly inside each cell. The
    old output called that 'CONSTANT across all runs at 10', on the same block
    whose own WORST line reads mean=20.

    Oracle: the response takes exactly two distinct values, so any 'constant'
    claim is false; the block must instead report the cell-mean range."""
    runs = make_runs(seed=5, resp_fn=lambda r, c, s, f: (
        10.0 * (c['geometry'] + 1) if r == 'l16_dos' else None))
    vals = {row['l16_dos'] for row in runs.values() if row['family'] == 'ham'}
    assert vals == {10.0, 20.0}

    fa = D.analyze_family(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          list(D.RESPONSES_COMMON))
    lines = []
    D.print_family(lines.append, fa)
    txt = '\n'.join(lines)
    assert 'CONSTANT across all runs' not in txt
    assert 'ZERO REPLICATE VARIANCE' in txt
    assert 'cell means range 10 to 20' in txt
    assert 'EXCLUDED from the BH family' in txt
    # and the dropped response really did leave the family
    assert fa.bh_m == 15 * (len(D.RESPONSES_COMMON) - 1)


def test_the_bh_block_counts_tests_not_terms():
    """FIX (label defect 5). The line counts (response x term) PAIRS, so a term
    tested on seven responses contributes seven. Oracle: m - k*, and the word
    on the line must be 'tests'."""
    runs, ci, cache = _full(seed=2027)
    lines = []
    res = run_analysis({'runs': runs}, ci, cache, lines.append)
    line = next(l for l in lines if 'clearing nothing' in l)
    assert line.strip().startswith('tests clearing nothing')
    assert int(line.split(':')[1]) == res['ham'].bh_m - res['ham'].bh_k_star


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

def _write_inputs(tmp_path, runs, ci, cache):
    p = {}
    for name, obj in (('report', {'runs': runs}), ('ci', ci),
                      ('cache', cache)):
        p[name] = tmp_path / f"{name}.json"
        p[name].write_text(json.dumps(obj))
    return p


def test_main_writes_the_report_on_success(tmp_path, capsys):
    runs, ci, cache = _full(seed=77)
    p = _write_inputs(tmp_path, runs, ci, cache)
    out = tmp_path / 'sub' / 'doe.out'
    D.main(['--report', str(p['report']), '--ci', str(p['ci']),
            '--best-epoch-cache', str(p['cache']), '--out', str(out)])
    capsys.readouterr()
    txt = out.read_text()
    assert txt.startswith('=' * 100)
    assert 'CAMPAIGN-V3 DOE' in txt and 'HEAD-TO-HEAD' in txt
    assert not txt.startswith('FAILED')


def test_main_never_leaves_a_stale_report_behind_after_an_abort(tmp_path,
                                                               capsys):
    """FIX (defect 6). The --out file is documented as 'OVERWRITTEN every run'.
    It used to be written only after the analysis returned, so an abort left
    the PREVIOUS run's report in place, undated and indistinguishable from a
    fresh one -- while every structure check in this script is designed to
    abort. Oracle: after a run that aborts, the file must not still be the old
    report."""
    runs, ci, cache = _full(seed=78)
    p = _write_inputs(tmp_path, runs, ci, cache)
    out = tmp_path / 'doe.out'
    D.main(['--report', str(p['report']), '--ci', str(p['ci']),
            '--best-epoch-cache', str(p['cache']), '--out', str(out)])
    capsys.readouterr()
    assert 'CAMPAIGN-V3 DOE' in out.read_text()

    # now break the report: 119 rows instead of 120
    runs.pop(next(k for k, v in runs.items() if v['family'] == 'ham'))
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'runs': runs}))
    with pytest.raises(BaseException):
        D.main(['--report', str(bad), '--ci', str(p['ci']),
                '--best-epoch-cache', str(p['cache']), '--out', str(out)])
    capsys.readouterr()
    left = out.read_text()
    assert 'CAMPAIGN-V3 DOE' not in left, (
        "the previous run's report survived an aborted run")
    assert left.startswith('FAILED:')


def test_main_records_the_reason_a_run_aborted(tmp_path, capsys):
    """Oracle: the placeholder must name the exception that stopped the run, so
    the file cannot be mistaken for a report."""
    runs, ci, cache = _full(seed=79)
    p = _write_inputs(tmp_path, runs, ci, cache)
    ci['runs'].pop(next(iter(ci['runs'])))
    broken = tmp_path / 'broken_ci.json'
    broken.write_text(json.dumps(ci))
    out = tmp_path / 'doe.out'
    with pytest.raises(BaseException):
        D.main(['--report', str(p['report']), '--ci', str(broken),
                '--best-epoch-cache', str(p['cache']), '--out', str(out)])
    capsys.readouterr()
    text = out.read_text()
    assert text.startswith('FAILED:')
    assert 'StructureError' in text or 'CI report' in text
