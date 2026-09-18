import torch.nn as nn

from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


class DNATransportHamiltonianGNNDeep(DNATransportHamiltonianGNN):
    """Hamiltonian model with one extra hidden layer in each readout perceptron.

    Probe variant. Deliberately a subclass rather than a flag on the parent so
    the campaign code path is untouched; the parent's forward only calls
    self.onsite_proj / self.coupling_proj, so replacing the modules is enough.
    """

    def __init__(self, hidden_dim: int = 128, n_orb: int = 1, **kwargs):
        super().__init__(hidden_dim=hidden_dim, n_orb=n_orb, **kwargs)
        # Replacing (not appending) keeps the optional-module init order the
        # parent relies on; the shared core's RNG stream is consumed identically
        # up to this point.
        self.onsite_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_orb * n_orb),
        )
        self.coupling_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_orb * n_orb),
        )
        # Same near-zero output init as the parent: a small initial H keeps
        # (E*I - H) well-conditioned early in training.
        nn.init.normal_(self.onsite_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.onsite_proj[-1].bias)
        nn.init.normal_(self.coupling_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.coupling_proj[-1].bias)
