import torch.nn as nn

from g3nat.models.standard import DNATransportGNN


class DNATransportGNNDeep(DNATransportGNN):
    """Direct model with one extra hidden layer in each readout perceptron.

    Probe variant. Deliberately a subclass rather than a flag on the parent so
    the campaign code path is untouched; the parent's forward only calls
    self.dos_proj / self.transmission_proj, so replacing the modules is enough.
    """

    def __init__(self, hidden_dim: int = 128, output_dim: int = 100,
                 dropout: float = 0.0, **kwargs):
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim,
                         dropout=dropout, **kwargs)

        def head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        self.dos_proj = head()
        self.transmission_proj = head()
