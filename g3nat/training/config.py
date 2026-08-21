from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for training DNA transport models."""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = 'auto'
    max_grad_norm: float = 1.0
    checkpoint_frequency: int = 10
    checkpoint_dir: Optional[str] = None
    warmup_epochs: int = 50
    # Optimizer. Defaults reproduce the historical hardcoded Adam(weight_decay=1e-5) exactly.
    # 'adamw' decouples the decay: Loshchilov & Hutter, ICLR 2019 (arXiv:1711.05101) show that
    # Adam's `weight_decay` is NOT true weight decay -- it is folded into the gradient and then
    # rescaled by Adam's per-parameter adaptive rates, so the effective regularization is
    # weaker and parameter-dependent. See docs/references.md.
    optimizer: str = 'adam'          # 'adam' | 'adamw'
    weight_decay: float = 1e-5
    # Loss weights: total = loss_a * T + loss_c * (loss_b * LDOS + (1 - loss_b) * DOS).
    # loss_a=1.0, loss_b=0.0, loss_c=1.0 reproduces the historical dos + transmission
    # loss exactly. loss_b is a convex mixing weight between local and global DOS;
    # the model cannot rescale it, unlike alpha.
    #
    # loss_c (added 2026-08-10, willll) scales the WHOLE DOS-family term. Before it
    # existed, transmission-only training was structurally unreachable: b and (1-b)
    # sum to 1, so the DOS family always carried weight 1 and the knob "train on T
    # alone" had never been run in either model family. loss_c=0 is that arm. The
    # prediction on record (REASONED, not measured): T-only training worsens the
    # under-determination of H (the -33 eV era) -- running it tests whether the
    # DOS/LDOS supervision is what disciplines H, which is the honest version of
    # "DOS is a training tool for transmission accuracy".
    loss_a: float = 1.0
    loss_b: float = 0.0
    loss_c: float = 1.0
    # Which LDOS aggregation this run is trained/measured against. Controls
    # only which of val_ldos_residue / val_ldos_base_only in metric_history
    # holds the measured value (the other stays nan) -- it does not change
    # what any loss or aggregation computes.
    ldos_target: str = 'residue'  # 'residue' | 'base_only'
    # DOS/LDOS are compared by ABSOLUTE magnitude (Huber loss on the raw log10
    # values), not by shape. The basis-size justification previously used for
    # shape_loss=True (2026-07-30) is WRONG and has been retracted: the training
    # window is HOMO +/- 1 eV, which holds only frontier levels, not the whole
    # basis -- e.g. for `aaac` (2869 basis functions) the window contains 14
    # occupied MOs and ZERO virtual ones (HOMO-LUMO gap 4.43 eV puts the upper
    # half of the window inside the gap). The basis-size story predicts
    # log10(2869/8) = 2.55 decades, wrong by a factor of ~200 in linear terms.
    #
    # The LEVEL-COUNTING replacement for it is ALSO retracted (2026-07-30, and
    # again on other grounds 2026-08-09; private notes sec. 7a). It was
    # claimed to match the measurement to 0.02 decades; that was one seed against
    # a cross-seed range 0.62 decades wide, and counted properly it explains only
    # ~37% of the measured offset, leaving a -0.30 decade residual. Do not quote
    # the level count as the reason for anything.
    #
    # The DECISION stands on its own and does not depend on either story: the
    # offset is large, systematic, and varies with base composition, so it is a
    # MEASUREMENT of what the one-orbital-per-base ansatz is missing rather than
    # a normalisation artifact to be centered away. Absolute comparison is the
    # default so that signal is not deleted. Transmission is a dimensionless probability and is NEVER
    # centered, under either setting. shape_loss=True remains available
    # (Trainer now shares one offset between the DOS and LDOS shape terms, see
    # trainer.py, so it no longer deletes the LDOS localization signal), kept
    # reachable for older runs and for anyone who wants shape-only comparison.
    shape_loss: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Create config from keyword arguments."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)
