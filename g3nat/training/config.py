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
    # Loss weights: total = loss_a * T + loss_b * LDOS + (1 - loss_b) * DOS.
    # loss_a=1.0, loss_b=0.0 reproduces the historical dos + transmission loss
    # exactly. loss_b is a convex mixing weight between local and global DOS;
    # the model cannot rescale it, unlike alpha.
    loss_a: float = 1.0
    loss_b: float = 0.0

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Create config from keyword arguments."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)
