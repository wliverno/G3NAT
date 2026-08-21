"""Inference utilities for loading and using trained models."""
import torch
import numpy as np
from typing import Tuple, Union, List
from torch_geometric.data import Batch

from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph


_LEGACY_ALPHA_STATE_KEYS = ('onsite_alpha_fixed', 'onsite_alpha_theta')


def _uniform_alpha(buf) -> float:
    """The single value in a legacy alpha buffer, or None if it is not uniform.

    Legacy alpha could be global (one entry) or per-base (four). Only a buffer that
    holds the SAME value everywhere corresponds to one of the two endpoints the
    current model can express.
    """
    t = torch.as_tensor(buf).detach().reshape(-1).to(torch.float64)
    if t.numel() == 0:
        return None
    first = t[0]
    if not bool(torch.all(t == first)):
        return None
    return float(first)


def _cross_check_state_dict(resolved: bool, state_dict: dict, args: dict,
                            source: str) -> None:
    """Verify the args-derived onsite head against the state dict, which is the
    stronger authority (it is what the weights actually are).

    Reading args alone is how a per-base-trained checkpoint whose args were never
    recorded -- including the `args = {}` fallback in `load_trained_model` -- used to
    resolve to False and then have its `onsite_baseline` silently deleted by
    `drop_legacy_alpha_state`. It loaded as a pure-context model, with no error.
    """
    args_says = ('args per_base_onsite' if 'per_base_onsite' in args else
                 'args (structured_onsite/alpha_mode/alpha_value)')

    if 'onsite_alpha_theta' in state_dict:
        raise ValueError(
            f"{source}: state dict carries 'onsite_alpha_theta', i.e. the REMOVED "
            "LEARNED continuous onsite alpha. That model is unrepresentable here "
            "whatever the args say (they resolve to per_base_onsite="
            f"{resolved}). Check out a commit from before the alpha-booleanization "
            "change to evaluate it.")

    if 'onsite_alpha_fixed' in state_dict:
        alpha = _uniform_alpha(state_dict['onsite_alpha_fixed'])
        if alpha is None or alpha not in (0.0, 1.0):
            raise ValueError(
                f"{source}: state dict 'onsite_alpha_fixed' is "
                f"{state_dict['onsite_alpha_fixed'].reshape(-1).tolist()!r}, which "
                "is not uniformly 0.0 or 1.0. Only the two endpoints (0 == context "
                "head, 1 == per-base table) exist in the current model, so this "
                f"checkpoint cannot be loaded ({args_says} resolves to "
                f"per_base_onsite={resolved}). Use pre-boolean code.")
        if (alpha == 1.0) != resolved:
            raise ValueError(
                f"{source}: the checkpoint disagrees with itself. State dict "
                f"'onsite_alpha_fixed' is {alpha} (=> per_base_onsite="
                f"{alpha == 1.0}), but {args_says} resolves to per_base_onsite="
                f"{resolved}. The state dict is what the weights actually are; "
                "loading either answer would score a model that was never trained. "
                "Fix the recorded args, or load with pre-boolean code.")
        return

    if 'onsite_baseline' in state_dict and not resolved:
        raise ValueError(
            f"{source}: state dict carries 'onsite_baseline' (the per-base onsite "
            f"table) but {args_says} resolves to per_base_onsite={resolved}, and "
            "there is no 'onsite_alpha_fixed' buffer to prove the table was "
            "multiplied by zero. Dropping it would silently load a per-base "
            "checkpoint as a pure-context model. Record the flag in args, or load "
            "with pre-boolean code.")


def per_base_onsite_from_args(args: dict, source: str, state_dict: dict) -> bool:
    """Resolve the boolean `per_base_onsite` model flag from a checkpoint's args.

    Checkpoints written on or after the alpha-booleanization commit record
    `per_base_onsite` directly. Older ones record the REMOVED continuous mix
    (`structured_onsite` + `alpha_mode`/`alpha_value`), whose two endpoints are the
    only ones the current model can express:
      alpha=0 (or structured_onsite off) -> per_base_onsite=False (context head)
      alpha=1                            -> per_base_onsite=True  (per-base table)
    A fractional or learned alpha is a genuinely different model. Rather than load
    it as one of the endpoints and report the numbers as if nothing changed, this
    raises: those checkpoints are historical alpha-sweep artifacts and must be read
    with pre-boolean code.

    `state_dict` is REQUIRED, and is CROSS-CHECKED against the args-derived answer as
    the stronger authority: any disagreement raises rather than resolving to one of
    the two. It has no default on purpose. It used to default to None, which skipped
    the cross-check entirely and restored the args-only behaviour that silently loads
    a per-base checkpoint with unrecorded args as a pure-context model -- i.e. one
    forgotten argument re-opened the defect this cross-check exists to close
    (independent review). Every caller has the state dict in hand already.
    """
    resolved = _resolve_from_args(args, source)
    _cross_check_state_dict(resolved, state_dict, args, source)
    return resolved


def _resolve_from_args(args: dict, source: str) -> bool:
    if 'per_base_onsite' in args:
        return bool(args['per_base_onsite'])
    if not bool(args.get('structured_onsite', False)):
        return False
    mode = str(args.get('alpha_mode', 'fixed'))
    alpha = float(args.get('alpha_value', 0.0))
    if mode == 'learned' or alpha not in (0.0, 1.0):
        raise ValueError(
            f"{source}: this checkpoint was trained with the REMOVED continuous "
            f"onsite alpha mix (alpha_mode={mode!r}, alpha_value={alpha!r}, "
            f"alpha_granularity={args.get('alpha_granularity', 'global')!r}) and "
            "cannot be represented by the current model, which offers only the two "
            "endpoints (per_base_onsite=False == alpha 0, True == alpha 1). Check "
            "out a commit from before the alpha-booleanization change to evaluate "
            "it; loading it here would silently score a different model.")
    return alpha == 1.0


def drop_legacy_alpha_state(state_dict: dict, per_base_onsite: bool) -> dict:
    """Strip alpha-mix state that the current model has no home for.

    The alpha buffers/parameters are gone entirely. `onsite_baseline` survives only
    when per_base_onsite is True; at alpha=0 the table was multiplied by zero and
    never entered H, so dropping it is exact, not an approximation.
    """
    drop = set(_LEGACY_ALPHA_STATE_KEYS)
    if not per_base_onsite:
        drop.add('onsite_baseline')
    if not any(k in state_dict for k in drop):
        return state_dict
    return {k: v for k, v in state_dict.items() if k not in drop}


#: Which loss weight trains each term a selection metric is built from.
#:
#: This is an EXPLICIT per-metric map, not substring matching on the metric
#: name. Substring matching is wrong in both directions here and both errors
#: were caught by the tests for this function: `'dos' in 'val_ldos_residue'` is
#: True (LDOS contains DOS, so the guard fired on a valid arm), while
#: `'transmission' in 'val_dos_t_unweighted'` is False (the metric spells it
#: `_t_`, so the transmission check never fired at all).
#:
#: Metric definitions: docs/metrics.md sec. 1 and the metric_history key table.
#: Loss weights: `loss_a` transmission, `loss_b` LDOS/DOS mix, `loss_c` the DOS
#: family switch (trainer.py:288-332).
_METRIC_TERM_WEIGHTS = {
    'val_dos_t_unweighted': {'DOS': 'loss_c', 'transmission': 'loss_a'},
    'val_dos_t_shape_unweighted': {'DOS': 'loss_c', 'transmission': 'loss_a'},
    'val_dos': {'DOS': 'loss_c'},
    'val_dos_shape': {'DOS': 'loss_c'},
    'val_transmission': {'transmission': 'loss_a'},
    'val_ldos_residue': {'LDOS': 'loss_b'},
    'val_ldos_base_only': {'LDOS': 'loss_b'},
    'val_ldos_shape_residue': {'LDOS': 'loss_b'},
    'val_ldos_shape_base_only': {'LDOS': 'loss_b'},
}


def check_selection_metric_trained(args: dict, selection_metric, source: str = '') -> None:
    """Raise if the checkpoint was selected on a metric containing an UNTRAINED term.

    docs/metrics.md sec. 1b, private notes sec. 18d. Measured on the v2
    campaign: an arm whose selection metric includes a term it does not train
    selects on that term's own trajectory. On the transmission-only arm
    (`loss_c=0`, DOS never trained) the untrained `val_dos` term degrades
    monotonically from an early minimum, so `val_dos + val_transmission` is
    minimized within the first handful of epochs -- 11 of the 12 v2 cells that
    published before epoch 100 are that arm, some at epoch 5 of 15000. Those
    weights are not usable; the per-epoch curves in `metric_history` are.

    This exists because sec. 18d's decision ("report tonly from curves, not from
    published weights") otherwise rests on a human remembering to read a note.
    Sec. 16e made checkpoint provenance a programmatic check rather than a
    memory exercise; this is the same standard for selection validity.

    `selection_metric` records WHICH metric was used, never whether it suited the
    arm, so the check has to come from the loss weights in `args`.

    Raises:
        ValueError: the metric contains a term whose loss weight is 0.
    """
    if not selection_metric or not isinstance(args, dict):
        return
    terms = _METRIC_TERM_WEIGHTS.get(str(selection_metric))
    if terms is None:
        # Unknown metric: unknown is not known-bad. Adding a metric to
        # metric_history without adding it here silently disables the check,
        # which is why the map lives next to the docstring that explains it.
        return
    offending = []
    for term, weight_key in sorted(terms.items()):
        w = args.get(weight_key)
        if w is not None and float(w) == 0.0:
            offending.append((term, weight_key))
    if offending:
        terms = ', '.join(f"'{t}' (weight {k}=0)" for t, k in offending)
        raise ValueError(
            f"{source or 'checkpoint'}: selected on '{selection_metric}', which contains "
            f"{terms} -- never trained by this run. The published weights were chosen by "
            f"an untrained term's trajectory and do not sit at this arm's optimum "
            f"(docs/metrics.md sec. 1b). Use the per-epoch `metric_history` instead, or "
            f"pass allow_untrained_selection=True if you specifically want these weights."
        )


def load_trained_model(model_path: str, device: str = 'auto',
                       allow_untrained_selection: bool = False) -> Tuple[Union[DNATransportGNN, DNATransportHamiltonianGNN], np.ndarray, torch.device]:
    """
    Load a trained DNA Transport GNN model.

    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load model on ('auto', 'cpu', 'cuda')
        allow_untrained_selection: bypass the selection-validity check of
            `check_selection_metric_trained`. Only set this when the weights
            themselves are the object of study (e.g. characterizing the defect).

    Returns:
        Tuple of (model, energy_grid, device)
    """
    if device == 'auto':
        device_tensor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_tensor = torch.device(device)

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device_tensor}")

    # Load the saved model (allow numpy arrays for energy grid)
    checkpoint = torch.load(model_path, map_location=device_tensor, weights_only=False)

    # Extract model arguments
    args = checkpoint.get('args', {})
    energy_grid = checkpoint.get('energy_grid', np.linspace(-1, 1, 201))

    if not allow_untrained_selection:
        check_selection_metric_trained(
            args, checkpoint.get('selection_metric'), source=model_path)

    # Detect model type from state dict keys
    state_dict = checkpoint['model_state_dict']
    model_type = None

    # Check for Hamiltonian model (new DNATransportHamiltonianGNN has onsite_proj/coupling_proj)
    if any('onsite_proj' in key for key in state_dict.keys()) and any('coupling_proj' in key for key in state_dict.keys()):
        model_type = 'hamiltonian'
        print("Detected DNATransportHamiltonianGNN model")
    # Check for standard model (has dos_proj and transmission_proj layers)
    elif any('dos_proj' in key for key in state_dict.keys()) and any('transmission_proj' in key for key in state_dict.keys()):
        model_type = 'standard'
        print("Detected DNATransportGNN model")
    # Check for simple Hamiltonian model (has H_proj but no NEGF components)
    elif any('H_proj' in key for key in state_dict.keys()) and not any('NEGF' in key for key in state_dict.keys()):
        model_type = 'simple_hamiltonian'
        print("Detected DNAHamiltonianGNN model (legacy)")
    else:
        # Default to standard model
        model_type = 'standard'
        print("Could not detect model type, defaulting to DNATransportGNN")

    # Initialize model with same architecture
    if model_type == 'hamiltonian':
        # Resolve the onsite head first: an unrepresentable legacy alpha must abort
        # before anything is built or loaded.
        # The state dict is cross-checked against args here: an unrecorded flag must
        # not silently drop a per-base table (independent review, finding I3).
        per_base_onsite = per_base_onsite_from_args(args, model_path, state_dict)
        if 'solver_type' not in args:
            print("WARNING: this checkpoint predates solver_type being recorded in args "
                  "(scripts/train.py). Falling back to the constructor default 'complex', "
                  "which is what training actually used. NOTE: evaluations of this "
                  "checkpoint made before 2026-08-09 ran the 'frobenius' path instead. "
                  "The two solvers TYPICALLY agree to ~3e-5 in log10 T (measured median "
                  "3.2e-5 over 1296 L=12 model-record pairs), but the disagreement is "
                  "heavy-tailed at isolated near-resonance energies: 1.4% of pairs "
                  "differ by >0.1 decade and the worst measured case is 3.8 decades "
                  "(DOS is unaffected, max 2e-3; see "
                  "private notes section 15b). Transmission "
                  "numbers from pre-2026-08-09 evaluations of this checkpoint are "
                  "typically fine but not guaranteed. Pass solver_type explicitly if "
                  "you need to reproduce an older result.")
        if 'log_floor' not in args or 'floor_mode' not in args:
            print("WARNING: this checkpoint's args carry no explicit "
                  f"{'log_floor' if 'log_floor' not in args else 'floor_mode'} "
                  "(and possibly neither). Falling back to the LEGACY semantics: "
                  "log_floor=1e-16 with floor_mode='clamp', i.e. "
                  "log10(max(x, 1e-16)). That reproduces what this checkpoint was "
                  "trained and evaluated with. BUT: the deep-tail transmission it "
                  "produces is NOT comparable to numbers recorded on or after "
                  "2026-08-15, which use floor_mode='smooth' (log10(max(x,0)+eps)) "
                  "with eps=1e-38. Under 'smooth' the floor never binds and the tail "
                  "extends below -16; under 'clamp' every point below 1e-16 reads "
                  "exactly -16. Do not pool tail metrics across the two without "
                  "re-evaluating both under the same floor_mode.")
        model = DNATransportHamiltonianGNN(
            hidden_dim=args.get('hidden_dim', 128),
            num_layers=args.get('num_layers', 4),
            num_heads=args.get('num_heads', 4),
            energy_grid=energy_grid,
            n_orb=args.get('n_orb', 1),
            enforce_hermiticity=args.get('enforce_hermiticity', True),
            # solver_type MUST default to the CONSTRUCTOR default, which is what training
            # used. It previously defaulted to 'frobenius' here while the constructor
            # default is 'complex' and train.py passed neither -- so every model on record
            # was evaluated through a solver it was not trained with. Since 'frobenius'
            # clamped at a hardcoded 1e-16 and 'complex' honours self.log_floor, that
            # mismatch is also what made log_floor a dead knob at evaluation time and
            # silently invalidated the length curves behind private notes 12a.
            solver_type=args.get('solver_type', 'complex'),
            use_log_outputs=args.get('use_log_outputs', True),
            log_floor=args.get('log_floor', 1e-16),
            # 'clamp' is the legacy semantics and the constructor default; see
            # the warning above for what its absence from args implies.
            floor_mode=args.get('floor_mode', 'clamp'),
            complex_eta=args.get('complex_eta', 1e-12),
            conv_type=args.get('conv_type', 'gat'),
            use_geometry=args.get('use_geometry', False),
            per_base_onsite=per_base_onsite,
        )
        # Old alpha-mix checkpoints carry state this model no longer has.
        state_dict = drop_legacy_alpha_state(state_dict, per_base_onsite)
        checkpoint['model_state_dict'] = state_dict
        print("DNATransportHamiltonianGNN initialized successfully")
    else:  # standard
        model = DNATransportGNN(
            hidden_dim=args.get('hidden_dim', 128),
            num_layers=args.get('num_layers', 4),
            num_heads=args.get('num_heads', 4),
            # The stored energy grid is authoritative -- it is what the weights were
            # trained against. args['num_energy_points'] only applies to the synthetic
            # tight-binding data source; for pickle data the grid comes from the data
            # and that arg keeps its default of 100, so reading it here made every
            # direct-model checkpoint trained on the 201-point DFT grid fail to load
            # with a dos_proj/transmission_proj size mismatch.
            output_dim=len(energy_grid),
            dropout=args.get('dropout', 0.1),
            conv_type=args.get('conv_type', 'transformer')
        )
        print("DNATransportGNN initialized successfully")

    # Handle legacy Hamiltonian checkpoints that had Dropout layers and hidden_dim//2 intermediate
    if model_type == 'hamiltonian':
        has_legacy_proj = any(k.endswith('.3.weight') and ('onsite_proj' in k or 'coupling_proj' in k)
                              for k in state_dict.keys())
        if has_legacy_proj:
            import torch.nn as nn
            # Old architecture: Linear(hidden_dim, hidden_dim//2) → ReLU → Dropout → Linear(hidden_dim//2, n_orb²)
            hidden_dim = args.get('hidden_dim', 128)
            n_orb = args.get('n_orb', 1)
            model.onsite_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim // 2, n_orb * n_orb)
            )
            model.coupling_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim // 2, n_orb * n_orb)
            )
            print("Detected legacy checkpoint format — rebuilt projection layers to match")

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_tensor)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")

    return model, energy_grid, device_tensor


def predict_sequence(
    model: Union[DNATransportGNN, DNATransportHamiltonianGNN],
    sequence: str,
    complementary_sequence: str,
    left_contact_positions: Union[int, List[int]] = None,
    right_contact_positions: Union[int, List[int]] = None,
    left_contact_coupling: float = 0.1,
    right_contact_coupling: float = 0.1,
    geometry_cache: dict = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict DOS and transmission for a DNA sequence.

    Args:
        model: Trained DNATransportGNN or DNATransportHamiltonianGNN model
        sequence: DNA sequence string (e.g., "ACGTACGT")
        complementary_sequence: Complementary DNA sequence string (e.g., "__GCATGCAT__")
        left_contact_positions: Position(s) for left contact
        right_contact_positions: Position(s) for right contact (default: last position)
        left_contact_coupling: Coupling strength for left contact
        right_contact_coupling: Coupling strength for right contact
        geometry_cache: dict mapping lowercase sequence -> geometry entry (as loaded
            from geom_cache/geometry_v2.pkl). Required (and must contain `sequence`)
            when `model.use_geometry` is True -- a geometry-trained checkpoint run
            without its geometry previously ran silently with an all-zero geometry
            mask (spec B8). Ignored for models with use_geometry=False.

    Returns:
        Tuple of (dos_pred, transmission_pred) arrays
    """
    if right_contact_positions is None:
        right_contact_positions = len(sequence) - 1

    print(f"Predicting for sequence: {sequence}")
    print(f"                         {complementary_sequence[::-1]}")
    print(f"Left contact at position {left_contact_positions}, coupling: {left_contact_coupling}")
    print(f"Right contact at position {right_contact_positions}, coupling: {right_contact_coupling}")

    geometry = None
    if bool(getattr(model, 'use_geometry', False)):
        if geometry_cache is None:
            raise ValueError(
                "this checkpoint was trained with use_geometry=True but no "
                "geometry_cache was supplied -- scoring it with the geometry "
                "channel silently deleted (all-zero mask) biases geometry effects "
                "toward null. Pass the cache (geom_cache/geometry_v2.pkl).")
        key = sequence.lower()
        if key not in geometry_cache:
            raise ValueError(
                f"geometry cache has no entry for sequence {sequence!r} -- a silent "
                "miss would score this sequence with geometry deleted (mask 0).")
        geometry = geometry_cache[key]

    # Convert sequence to graph
    graph = sequence_to_graph(
        primary_sequence=sequence,
        complementary_sequence=complementary_sequence,
        left_contact_positions=left_contact_positions,
        right_contact_positions=right_contact_positions,
        left_contact_coupling=left_contact_coupling,
        right_contact_coupling=right_contact_coupling,
        geometry=geometry
    )

    if graph is None:
        raise ValueError(f"Failed to create graph for sequence: {sequence}")

    # Create batch (single graph)
    batch_data = Batch.from_data_list([graph])
    batch_data = batch_data.to(next(model.parameters()).device)

    # Make prediction
    with torch.no_grad():
        dos_pred, transmission_pred = model(batch_data)

        # Convert to numpy arrays
        dos_pred = dos_pred.cpu().numpy()[0]  # Remove batch dimension
        transmission_pred = transmission_pred.cpu().numpy()[0]

    print(f"Prediction completed!")
    print(f"DOS range: [{dos_pred.min():.4f}, {dos_pred.max():.4f}]")
    print(f"Transmission range: [{transmission_pred.min():.4f}, {transmission_pred.max():.4f}]")

    return dos_pred, transmission_pred
