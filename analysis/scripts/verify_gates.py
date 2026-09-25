"""The geometry gate every scored batch passes through (g3nat_analysis.evaluator.gated_batch).

Passing geometry_cache=None to create_dna_dataset zeroes edge_geom and edge_geom_mask,
and the Hamiltonian head multiplies by that mask, so the geometry channel contributes
nothing, silently. This gate is what makes that impossible to miss on a geometry-on run.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def gate_geometry_reaches_cells(batch, use_geometry: bool) -> GateResult:
    """For a geometry-on run, the geometry features and mask must be non-zero."""
    if not use_geometry:
        # A geometry-off run having a zeroed channel is correct behaviour.
        return GateResult('geometry_reaches_cells', True,
                          'geometry off for this run; nothing to check')
    geom = getattr(batch, 'edge_geom', None)
    mask = getattr(batch, 'edge_geom_mask', None)
    if geom is None or mask is None:
        return GateResult('geometry_reaches_cells', False,
                          'batch carries no edge_geom / edge_geom_mask')
    gmax = float(np.abs(np.asarray(geom)).max())
    msum = float(np.asarray(mask).sum())
    ok = gmax > 0.0 and msum > 0.0
    return GateResult('geometry_reaches_cells', ok,
                      f'edge_geom absmax {gmax:.6g}, mask sum {msum:.6g}')
