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

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Create config from keyword arguments."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)
