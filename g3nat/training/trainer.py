"""Trainer class for DNA transport models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from torch_geometric.loader import DataLoader

from .config import TrainingConfig
from g3nat.models.hamiltonian import site_ldos_log10


def _center(x, dims):
    """Subtract the mean over `dims`, keeping dims, so only SHAPE is compared.

    Used only by Trainer._ldos_agreement's held-out 'shape' metric (dims=(1, 2),
    site AND energy jointly -- one offset per sequence, never per site), which
    is independent of config.shape_loss and kept for continuity with older
    runs. Trainer._compute_losses's TRAINED shape terms no longer call this: as
    of 2026-07-30 they share one offset (the median DOS residual) between DOS
    and LDOS instead of centering each independently -- see that method's
    docstring. Never applied to transmission.
    """
    return x - x.mean(dim=dims, keepdim=True)


class Trainer:
    """Trainer for DNA transport GNN models."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        **kwargs
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: TrainingConfig object (optional)
            **kwargs: Additional config parameters if config not provided
        """
        self.model = model
        self.config = config or TrainingConfig.from_kwargs(**kwargs)

        # Set device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Initialize optimizer. Default 'adam' + 1e-5 reproduces the historical hardcoded
        # behaviour exactly, so existing comparisons are unaffected.
        opt_name = getattr(self.config, 'optimizer', 'adam').lower()
        wd = getattr(self.config, 'weight_decay', 1e-5)
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=wd)
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=wd)
        else:
            raise ValueError(f"unknown optimizer {opt_name!r}; expected 'adam' or 'adamw'")

        # Loss function
        self.criterion = nn.HuberLoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.metric_history: List[Dict[str, float]] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        start_epoch: int = 0
    ) -> tuple[List[float], List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_callback: Optional callback for saving checkpoints. Called as
                checkpoint_callback(model, optimizer, epoch, train_losses, val_losses,
                metric_history=self.metric_history) -- the metric_history keyword lets
                a callback that forwards it to save_checkpoint() survive a preemption
                without losing per-epoch LDOS/DOS/transmission history.
            progress_callback: Optional callback for tracking progress
            start_epoch: Starting epoch for resumption (default: 0)

        Returns:
            Tuple of (train_losses, val_losses) lists
        """
        # Verify all model parameters are on the same device
        param_devices = set(p.device for p in self.model.parameters())
        if len(param_devices) > 1:
            print(f"WARNING: Model parameters on different devices: {param_devices}")
            self.model = self.model.to(self.device)

        # Verify optimizer state is on the correct device
        optimizer_devices = set()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_devices.add(v.device)

        if len(optimizer_devices) > 1:
            print(f"WARNING: Optimizer state on different devices: {optimizer_devices}")
            # Move all optimizer state tensors to the model's device
            model_device = next(self.model.parameters()).device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(model_device)

        for epoch in range(start_epoch, self.config.num_epochs):
            # Learning rate warmup: ramp from lr/100 to lr over warmup_epochs epochs
            if self.config.warmup_epochs > 0 and epoch < self.config.warmup_epochs:
                warmup_scale = (epoch + 1) / self.config.warmup_epochs
                lr = self.config.learning_rate * max(0.01, warmup_scale)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(epoch, train_loss, val_loss)

            # Save checkpoint periodically
            if checkpoint_callback is not None:
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    checkpoint_callback(self.model, self.optimizer, epoch, self.train_losses, self.val_losses,
                                       metric_history=self.metric_history)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save final checkpoint
        if checkpoint_callback is not None:
            checkpoint_callback(self.model, self.optimizer, self.config.num_epochs - 1, self.train_losses, self.val_losses,
                               metric_history=self.metric_history)

        return self.train_losses, self.val_losses

    def _compute_losses(self, batch, dos_pred, transmission_pred):
        """Compose the loss for one batch.

        DOS/LDOS are compared BY ABSOLUTE MAGNITUDE by default (config.shape_loss
        =False): the DOS offset is large, systematic and composition-dependent, so
        it is a real measurement of what the one-orbital-per-base ansatz is
        missing, not an artifact to be centered away. The level-counting
        explanation this docstring used to give for the offset's SIZE is retracted
        (see TrainingConfig.shape_loss's docstring and docs/model-results.md sec.
        7a); the decision does not rest on it.

        Both the raw (magnitude) and shape (offset-corrected) versions of
        DOS/LDOS are always computed here, regardless of config.shape_loss, so
        metric_history never loses information; config.shape_loss selects only
        which pair feeds 'total', the optimized scalar.

        The shape variant uses ONE offset for both DOS and LDOS, derived from
        DOS alone (the MEDIAN, not mean, of the per-sequence DOS residual --
        the LDOS residual std is ~1.49 against Huber delta=1.0, so the loss
        operates mostly in its linear regime, where the median is optimal, not
        the mean). Sharing a single offset matters because sum_i LDOS_i = DOS
        holds exactly on both sides, so
            mean_i log10 LDOS_i = log10(DOS) - log10(n_sites) - J
        where J is the Jensen/AM-GM gap, a log-space localization measure.
        Subtracting INDEPENDENT offsets from DOS and LDOS (the pre-2026-07-30
        shape_loss implementation) removes <J_pred - J_target> entirely, i.e.
        deletes exactly the quantity the LDOS term exists to measure. Sharing
        the DOS-derived offset instead leaves that signal in the LDOS
        residual -- see val_ldos_localization_gap in _validate_epoch, which
        reports it directly. Transmission is a dimensionless, basis-size-
        independent observable and is NEVER centered, under either setting.

        Returns a dict with 'total' (the optimized scalar), the individual
        'dos'/'dos_shape'/'transmission'/'ldos'/'ldos_shape' terms, and
        'dos_t_unweighted'/'dos_t_shape_unweighted' -- the quantities
        comparable across different loss_b values and back to the v1
        record, because 'total' is loss_b-weighted and so differently
        scaled at each b.
        """
        batch_size = dos_pred.size(0)
        num_energy_points = dos_pred.size(1)

        dos_target = batch.dos.view(batch_size, num_energy_points)
        transmission_target = batch.transmission.view(batch_size, num_energy_points)

        dos_loss = self.criterion(dos_pred, dos_target)
        # Shared offset for the shape variant: MEDIAN (not mean) of the
        # per-sequence DOS residual over the energy axis (dim=1) -- see the
        # docstring above for why median and why shared with LDOS below.
        off = (dos_pred - dos_target).median(dim=1, keepdim=True).values  # [B, 1]
        dos_shape_loss = self.criterion(dos_pred - off, dos_target)
        transmission_loss = self.criterion(transmission_pred, transmission_target)

        a = self.config.loss_a
        b = self.config.loss_b
        c = getattr(self.config, 'loss_c', 1.0)
        shape = self.config.shape_loss
        dos_term = dos_shape_loss if shape else dos_loss

        ldos_loss = None
        ldos_shape_loss = None
        if c == 0.0:
            # Transmission-only training (loss_c=0): the previously unreachable
            # arm. The whole DOS family is skipped by branch -- no LDOS target is
            # required, no backward graph is built for DOS/LDOS -- while the
            # diagnostics above (dos_loss, dos_shape_loss) remain measured and
            # logged, so held-out DOS is still tracked on a model never trained
            # on it.
            total = a * transmission_loss
        elif b != 0.0:
            # PyG drops an attribute assigned None, so presence is hasattr,
            # never `is not None`.
            if not hasattr(batch, 'ldos'):
                raise ValueError(
                    "loss_b > 0 but the batch carries no LDOS target. The data "
                    "directory has no DOSAtom (a v1 dataset); use pickle_files_v2 "
                    "or set loss_b = 0."
                )
            if not hasattr(self.model, 'ldos'):
                raise ValueError(
                    f"loss_b > 0 but {type(self.model).__name__} exposes no "
                    "'ldos' attribute after a forward pass. Only the Hamiltonian "
                    "model supports the LDOS term."
                )

            ldos_pred, ldos_target = self._ldos_pred_and_target(batch, batch_size)

            ldos_loss = self.criterion(ldos_pred, ldos_target)
            # SAME offset as dos_shape_loss (derived from DOS alone), broadcast
            # over both site and energy dims, so DOS and LDOS shift TOGETHER
            # rather than independently -- see the docstring above.
            ldos_shape_loss = self.criterion(ldos_pred - off.unsqueeze(-1), ldos_target)
            ldos_term = ldos_shape_loss if shape else ldos_loss
            total = a * transmission_loss + c * (b * ldos_term + (1.0 - b) * dos_term)
        else:
            # Skipped by branch, never multiplied by zero, so a dataset with no
            # LDOS target still trains and no backward pass is built for it.
            total = a * transmission_loss + c * dos_term

        return {
            'total': total,
            'dos': dos_loss,
            'dos_shape': dos_shape_loss,
            'transmission': transmission_loss,
            'ldos': ldos_loss,
            'ldos_shape': ldos_shape_loss,
            'dos_t_unweighted': dos_loss + transmission_loss,
            'dos_t_shape_unweighted': dos_shape_loss + transmission_loss,
        }

    def _ldos_pred_and_target(self, batch, batch_size):
        """Reshape/predict the LDOS pair shared by the loss and the metric.

        Used by both `_compute_losses`'s b != 0 branch and `_ldos_agreement`,
        so a fix to one path cannot silently diverge from the other -- the
        whole point of the held-out metric is that it measures the same
        quantity the loss trains. Callers must already have established
        hasattr(batch, 'ldos') and hasattr(self.model, 'ldos').

        Returns:
            (pred, target), both [batch, n_sites, n_energy].
        """
        num_energy_points = self.model.ldos.size(1)

        # Target [batch * 2L, n_energy] -> [batch, 2L, n_energy]
        n_sites = batch.ldos.size(0) // batch_size
        target = batch.ldos.view(batch_size, n_sites, num_energy_points)

        # Prediction [batch, n_energy, n_sites*n_orb] -> [batch, n_energy, n_sites]
        pred = site_ldos_log10(self.model.ldos, n_sites, self.model.log_floor)
        # ... then transpose to match the target's [batch, n_sites, n_energy].
        pred = pred.transpose(1, 2)

        if pred.shape != target.shape:
            raise ValueError(
                f"LDOS shape mismatch: prediction {tuple(pred.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        return pred, target

    def _ldos_agreement(self, batch, dos_pred):
        """Held-out LDOS Huber + localization gap, measured whether or not
        the LDOS term is being trained.

        Returns (raw, shape, localization_gap) -- all nan when the batch
        carries no target. Presence of the target governs this metric;
        loss_b/shape_loss govern only what feeds the trained loss.

        `shape` centers pred and target INDEPENDENTLY, jointly over (site,
        energy) per sequence, dims=(1, 2). This is NOT the same computation as
        `_compute_losses`'s trained shape term any more: as of 2026-07-30 that
        term shares one DOS-derived offset with DOS instead of centering LDOS
        on its own (see `_compute_losses`'s docstring). This metric keeps its
        own independent centering for continuity with older runs; it is a
        held-out diagnostic, not the trained quantity.

        `localization_gap` is
            <log10 dos_pred - log10 dos_target> - <log10 ldos_pred - log10 ldos_target>
        Both dos_pred/dos_target and ldos_pred/ldos_target are already log10
        (see docstrings in g3nat/data/datasets.py and
        g3nat/models/hamiltonian.py::site_ldos_log10), so this is directly a
        difference of log-residual means -- no extra log10 call needed. Since
        sum_i LDOS_i = DOS holds exactly on both sides,
            mean_i log10 LDOS_i = log10(DOS) - log10(n_sites) - J
        (J the Jensen/AM-GM gap, a log-space localization measure), and the
        shared log10(n_sites) term cancels in the pred-minus-target
        differences above, so localization_gap == J_pred - J_target exactly.
        POSITIVE means the model concentrates spectral weight onto fewer
        sites than DFT (more log-localized); negative means it spreads weight
        more than DFT.
        """
        if not hasattr(batch, 'ldos') or not hasattr(self.model, 'ldos'):
            return float('nan'), float('nan'), float('nan')
        batch_size = int(batch.batch.max().item() + 1)
        pred, target = self._ldos_pred_and_target(batch, batch_size)
        raw = self.criterion(pred, target).item()
        shape = self.criterion(_center(pred, (1, 2)), _center(target, (1, 2))).item()

        num_energy_points = dos_pred.size(1)
        dos_target = batch.dos.view(batch_size, num_energy_points)
        localization_gap = (dos_pred - dos_target).mean().item() - (pred - target).mean().item()
        return raw, shape, localization_gap

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            dos_pred, transmission_pred = self.model(batch)

            losses = self._compute_losses(batch, dos_pred, transmission_pred)
            total_loss = losses['total']

            total_loss.backward()

            # Gradient clipping to prevent gradient explosion in physics-informed models
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        return train_loss

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch.

        Args:
            epoch: ABSOLUTE epoch number (already accounts for start_epoch on
                a resumed run -- it is the loop variable from fit(), not a
                loop-local index). Stored on each metric_history entry so a
                consumer can align on epoch number rather than list position,
                which matters once a resume makes metric_history shorter than
                val_losses (see train_model's metric_history/metric_history_out
                params).
        """
        self.model.eval()
        val_loss = 0.0
        agg_dos = 0.0
        agg_dos_shape = 0.0
        agg_trans = 0.0
        agg_unweighted = 0.0
        agg_shape_unweighted = 0.0
        agg_ldos = 0.0
        agg_ldos_shape = 0.0
        agg_localization_gap = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                dos_pred, transmission_pred = self.model(batch)

                losses = self._compute_losses(batch, dos_pred, transmission_pred)
                total_loss = losses['total']

                val_loss += total_loss.item()
                agg_dos += losses['dos'].item()
                agg_dos_shape += losses['dos_shape'].item()
                agg_trans += losses['transmission'].item()
                agg_unweighted += losses['dos_t_unweighted'].item()
                agg_shape_unweighted += losses['dos_t_shape_unweighted'].item()
                ldos_raw, ldos_shape, localization_gap = self._ldos_agreement(batch, dos_pred)
                agg_ldos += ldos_raw
                agg_ldos_shape += ldos_shape
                agg_localization_gap += localization_gap
                n_batches += 1

        val_loss /= len(val_loader)

        # Key the measured LDOS agreement under whichever aggregation this run
        # is actually configured against (self.config.ldos_target); the other
        # key is always nan. Both keys are always present so the schema is
        # stable across runs regardless of ldos_target -- a consumer never has
        # to check which key exists before reading it. Reporting the
        # non-trained aggregation would require carrying BOTH the residue and
        # base_only LDOS arrays on every Data object simultaneously, which is
        # deliberately out of scope here. The experiment covers the other
        # aggregation by running it directly in a later phase (Phase C), not
        # through this trainer computing both at once. The '_shape' keys
        # mirror this exact same dynamic keying, one level up: shape_loss
        # only changes what 'total' trains on, never what is reported here --
        # every run measures and stores both raw and shape variants of DOS
        # and LDOS so no information is lost and old runs stay comparable.
        measured_key = f"val_ldos_{self.config.ldos_target}"
        other_key = 'val_ldos_base_only' if self.config.ldos_target == 'residue' else 'val_ldos_residue'
        shape_measured_key = f"val_ldos_shape_{self.config.ldos_target}"
        shape_other_key = (
            'val_ldos_shape_base_only' if self.config.ldos_target == 'residue'
            else 'val_ldos_shape_residue'
        )
        entry = {
            'epoch': epoch,
            'val_dos': agg_dos / n_batches,
            'val_dos_shape': agg_dos_shape / n_batches,
            'val_transmission': agg_trans / n_batches,
            'val_dos_t_unweighted': agg_unweighted / n_batches,
            'val_dos_t_shape_unweighted': agg_shape_unweighted / n_batches,
            measured_key: agg_ldos / n_batches,
            other_key: float('nan'),
            shape_measured_key: agg_ldos_shape / n_batches,
            shape_other_key: float('nan'),
            # Reported every run regardless of shape_loss -- this is the
            # quantity the shape_loss=True independent-offset bug (fixed
            # 2026-07-30) was silently deleting. See _ldos_agreement's
            # docstring for the derivation. Positive: model more
            # log-localized than DFT; negative: model spreads weight more
            # than DFT; nan when the batch carries no LDOS target.
            'val_ldos_localization_gap': agg_localization_gap / n_batches,
            # Last batch's value -- a per-epoch spot reading, cheap and sufficient
            # to see whether an arm is living at the log floor.
            'floored_frac_dos': float(getattr(self.model, 'last_floored_frac_dos', float('nan'))),
            'floored_frac_t': float(getattr(self.model, 'last_floored_frac_t', float('nan'))),
        }
        self.metric_history.append(entry)
        return val_loss

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set a custom optimizer (useful for loading from checkpoint)."""
        self.optimizer = optimizer

    def set_losses(self, train_losses: List[float], val_losses: List[float]):
        """Set training history (useful for loading from checkpoint)."""
        self.train_losses = train_losses
        self.val_losses = val_losses


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               num_epochs: int = 100, learning_rate: float = 1e-3, device: str = 'auto',
               checkpoint_dir: str = None, checkpoint_frequency: int = 10,
               start_epoch: int = 0, train_losses: List[float] = None,
               val_losses: List[float] = None, optimizer: torch.optim.Optimizer = None,
               checkpoint_callback=None, progress_callback=None, max_grad_norm: float = 1.0,
               warmup_epochs: int = 50, optimizer_name: str = 'adam',
               weight_decay: float = 1e-5, loss_a: float = 1.0, loss_b: float = 0.0,
               loss_c: float = 1.0,
               ldos_target: str = 'residue', shape_loss: bool = False,
               metric_history: Optional[List[Dict[str, float]]] = None,
               metric_history_out: Optional[List[Dict[str, float]]] = None):
    """
    Train the DNA Transport GNN model (backward-compatible function).

    Args:
        model: DNATransportGNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints (optional)
        checkpoint_frequency: Save checkpoint every N epochs (default: 10)
        start_epoch: Starting epoch for resumption (default: 0)
        train_losses: Existing training losses for resumption (optional)
        val_losses: Existing validation losses for resumption (optional)
        optimizer: Existing optimizer for resumption (optional)
        checkpoint_callback: Function to call for saving checkpoints (optional)
        progress_callback: Function to call for saving progress (optional)
        max_grad_norm: Maximum gradient norm for gradient clipping (default: 1.0)
        loss_a: weight on the transmission loss term (default 1.0 reproduces history)
        loss_b: convex mixing weight b*LDOS + (1-b)*DOS (default 0.0 reproduces history)
        ldos_target: which LDOS aggregation ('residue' or 'base_only') this run is
            trained/measured against -- selects which key in each metric_history
            entry holds the measured value; the other key is always nan.
        shape_loss: compare DOS/LDOS by a shared offset-corrected shape rather
            than absolute magnitude (default False -- absolute is correct; see
            TrainingConfig.shape_loss's docstring for why the basis-size
            justification for True was wrong and has been retracted). When
            True, DOS and LDOS share ONE offset (see Trainer._compute_losses)
            so the LDOS localization signal is not deleted. Transmission is
            never centered either way.
        metric_history: existing per-epoch metric_history for resumption
            (optional), analogous to train_losses/val_losses. Each entry
            already carries an absolute 'epoch' key, so seeding here and then
            letting fit() append more keeps the whole list self-describing
            even though the seed and the appended entries come from different
            process invocations.
        metric_history_out: if provided, filled in-place with the trainer's
            full (seeded + newly appended) per-epoch metric_history so a
            caller that only unpacks (train_losses, val_losses) can still
            recover it without changing this function's return arity.

    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    # Create config
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        max_grad_norm=max_grad_norm,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_dir=checkpoint_dir,
        warmup_epochs=warmup_epochs,
        # `optimizer_name`, not `optimizer` -- the latter is the resume OBJECT below.
        optimizer=optimizer_name,
        weight_decay=weight_decay,
        loss_a=loss_a,
        loss_b=loss_b,
        loss_c=loss_c,
        ldos_target=ldos_target,
        shape_loss=shape_loss
    )

    # Create trainer
    trainer = Trainer(model, config)

    # Set optimizer if provided (for resumption)
    if optimizer is not None:
        trainer.set_optimizer(optimizer)

    # Set losses if provided (for resumption)
    if train_losses is not None:
        trainer.train_losses = train_losses
    if val_losses is not None:
        trainer.val_losses = val_losses
    if metric_history is not None:
        # Copy rather than alias: the caller's list (e.g. scripts/train.py's
        # resume_metric_history, read from a checkpoint) should not be mutated
        # by this trainer's subsequent appends.
        trainer.metric_history = list(metric_history)

    # Train
    train_losses, val_losses = trainer.fit(
        train_loader,
        val_loader,
        checkpoint_callback=checkpoint_callback,
        progress_callback=progress_callback,
        start_epoch=start_epoch
    )

    if metric_history_out is not None:
        metric_history_out.extend(trainer.metric_history)

    return train_losses, val_losses
