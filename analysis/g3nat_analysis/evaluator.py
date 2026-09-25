"""Post-hoc evaluator units for campaign v3 (scoring of the published checkpoints).

Every batch scored here is built by `gated_batch`, which runs the geometry gate on the
batch it returns: passing geometry_cache=None to create_dna_dataset zeroes the geometry
edge features and their mask silently, with no error and no nan (the campaign-v2 defect).
"""
import os
import re
import sys

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

HAM_SUPERVISION = ('dos', 'ldos', 'ldosonly', 'tonly')
BLIND_SUPERVISION = ('dos', 'tonly')

_HAM = re.compile(r'^ham_(?P<sup>[a-z]+)_n(?P<orb>\d)_L(?P<lay>\d)'
                  r'_(?P<geom>geom|nogeom)_s(?P<seed>\d+)$')
_BLIND = re.compile(r'^blind_(?P<sup>[a-z]+)_L(?P<lay>\d)'
                    r'_(?P<geom>geom|nogeom)_s(?P<seed>\d+)$')


def parse_run_name(name):
    m = _HAM.match(name)
    if m and m['sup'] in HAM_SUPERVISION:
        return {'family': 'ham', 'supervision': m['sup'],
                'n_orb': int(m['orb']), 'num_layers': int(m['lay']),
                'geometry': 1 if m['geom'] == 'geom' else 0,
                'seed': int(m['seed'])}
    m = _BLIND.match(name)
    if m and m['sup'] in BLIND_SUPERVISION:
        return {'family': 'blind', 'supervision': m['sup'], 'n_orb': None,
                'num_layers': int(m['lay']),
                'geometry': 1 if m['geom'] == 'geom' else 0,
                'seed': int(m['seed'])}
    return None


def transmission_at_selection(ck):
    """val_transmission at the run's own selected epoch. Matched on the entry's
    absolute 'epoch' key, never the list index: requeued runs have a metric_history
    shorter than the epoch count."""
    mh = ck.get('metric_history') or []
    epoch = ck.get('saved_at_epoch', ck.get('epoch'))
    for entry in mh:
        if entry.get('epoch') == epoch:
            return float(entry['val_transmission'])
    raise ValueError(f'no metric_history entry for saved_at_epoch={epoch} '
                     f'({len(mh)} entries)')


class GateError(RuntimeError):
    """A batch failed its geometry gate."""


def gated_batch(rec, energy_grid, geometry_cache, use_geometry):
    """The ONLY batch constructor for scoring. Gate-checked before return."""
    from torch_geometric.loader import DataLoader
    from g3nat.data import create_dna_dataset
    from verify_gates import gate_geometry_reaches_cells

    ds = create_dna_dataset(
        sequences=[rec['sequence']],
        dos_data=np.asarray([rec['dos']]),
        transmission_data=np.asarray([rec['transmission']]),
        energy_grid=energy_grid,
        complementary_sequences=[rec['complementary_sequence']],
        left_contact_positions_list=[rec['left_contact_pos']],
        right_contact_positions_list=[rec['right_contact_pos']],
        left_contact_coupling_list=[rec['coupling']],
        right_contact_coupling_list=[rec['coupling']],
        geometry_cache=geometry_cache if use_geometry else None,
        ldos_data=None, ldos_target='residue')
    batch = next(iter(DataLoader(ds, batch_size=1)))
    result = gate_geometry_reaches_cells(batch, use_geometry)
    if not result.passed:
        raise GateError(f"{rec['sequence']}: {result.detail}")
    return batch


def huber_mean(pred, target, delta=1.0):
    r = np.abs(np.asarray(pred, float) - np.asarray(target, float))
    return float(np.mean(np.where(r <= delta, 0.5 * r ** 2,
                                  delta * (r - 0.5 * delta))))


def mse_mean(pred, target):
    """Mean squared error on the same log10 quantities (the loss-probe scoring unit)."""
    r = np.asarray(pred, float) - np.asarray(target, float)
    return float(np.mean(r * r))


LOSS_FUNCS = {'huber': huber_mean, 'mse': mse_mean}


def eval_heldout(model, recs, energy_grid, prefix, geometry_cache,
                 use_geometry):
    """Full-curve fit error at one held-out length. All 201 points, no subsets."""
    import torch
    t_all, d_all = [], []
    for rec in recs:
        batch = gated_batch(rec, energy_grid, geometry_cache, use_geometry)
        with torch.no_grad():
            dos_p, t_p = model(batch)
        t_p = t_p.squeeze(0).cpu().numpy()
        d_p = dos_p.squeeze(0).cpu().numpy()
        t_t = np.asarray(rec['transmission'], float).ravel()
        d_t = np.asarray(rec['dos'], float).ravel()
        if len(t_p) != len(t_t) or len(d_p) != len(d_t):
            raise ValueError(
                f"eval_heldout: length mismatch for record {rec.get('sequence', '?')!r}: "
                f"transmission pred={len(t_p)} vs target={len(t_t)}, "
                f"dos pred={len(d_p)} vs target={len(d_t)} (no silent truncation)")
        t_all.append(huber_mean(t_p, t_t))
        d_all.append(huber_mean(d_p, d_t))
    return {f'{prefix}_transmission': float(np.mean(t_all)) if t_all else float('nan'),
            f'{prefix}_dos': float(np.mean(d_all)) if d_all else float('nan'),
            f'{prefix}_n_records': len(t_all)}


def sample_onsite_records(records, per_length=4, seed=20260825):
    """Seeded sample: per_length sequences per training length, all contact variants."""
    by_len = {}
    for seq in sorted({s for s, _k in records}):
        by_len.setdefault(len(seq), []).append(seq)
    rng = np.random.default_rng(seed)
    out = []
    for L in sorted(by_len):
        group = sorted(by_len[L])
        if len(group) > per_length:
            idx = rng.choice(len(group), size=per_length, replace=False)
            group = [group[i] for i in sorted(idx)]
        for seq in group:
            for s, k in sorted(records):
                if s == seq:
                    out.append((s, k))
    return out


def onsite_over_sample(model, records, sample, energy_grid, geometry_cache,
                       use_geometry, n_orb):
    """Onsite distance from the HOMO, averaged over the seeded sample. n_bases comes
    from the SEQUENCE (2L), never from H, so the one-strand guard in onsite_distance
    is not vacuous."""
    import torch
    from g3nat_analysis.metrics import onsite_distance
    near, far = [], []
    for seq, key in sample:
        rec = records[(seq, key)]
        batch = gated_batch(rec, energy_grid, geometry_cache, use_geometry)
        with torch.no_grad():
            model(batch)
        H = model.H[0].cpu().numpy()
        d = onsite_distance(H, n_orb, n_bases=2 * len(rec['sequence']))
        near.append(d['near'])
        far.append(d['far'])
    return {'onsite_near': float(np.mean(near)) if near else float('nan'),
            'onsite_far': float(np.mean(far)) if far else float('nan'),
            'onsite_n_graphs': len(near)}


def substitution_response(model, records, pairs, energy_grid, geometry_cache,
                          use_geometry):
    """Huber on the predicted vs DFT CHANGE in log10 T, contact-matched. Not a paper
    response (removed by ruling 2026-08-26) but still computed so the report carries
    sub_n_comparisons, which doe_v3.validate_structure asserts is identical across runs."""
    import torch
    from g3nat_analysis.metrics import matched_records, substitution_loss

    cache = {}

    def predict(seq, key):
        if (seq, key) not in cache:
            batch = gated_batch(records[(seq, key)], energy_grid,
                                geometry_cache, use_geometry)
            with torch.no_grad():
                _dos, t = model(batch)
            cache[(seq, key)] = t.squeeze(0).cpu().numpy()
        return cache[(seq, key)]

    losses_by_len = {}
    n = 0
    for pair in pairs:
        a, b, _pos = pair
        for key, rec_a, rec_b in matched_records(records, pair):
            loss = substitution_loss(
                predict(a.lower(), key), predict(b.lower(), key),
                np.asarray(rec_a['transmission'], float).ravel(),
                np.asarray(rec_b['transmission'], float).ravel())
            losses_by_len.setdefault(str(len(a)), []).append(loss)
            n += 1
    all_losses = [v for vs in losses_by_len.values() for v in vs]
    return {'sub_t': float(np.mean(all_losses)) if all_losses else float('nan'),
            'sub_t_by_length': {L: float(np.mean(vs))
                                for L, vs in sorted(losses_by_len.items())},
            'sub_n_comparisons': n}
