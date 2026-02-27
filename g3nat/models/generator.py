"""DNA sequence generator using Gumbel-Softmax for differentiable sequence generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNASequenceGenerator(nn.Module):
    """Generates soft DNA sequence representations via an MLP + Gumbel-Softmax."""

    def __init__(self, seq_length: int, latent_dim: int = 32,
                 hidden_dim: int = 128, num_hidden_layers: int = 2,
                 tau: float = 1.0):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.tau = tau

        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, seq_length * 4))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=next(self.parameters()).device)

        logits = self.mlp(z).view(-1, self.seq_length, 4)
        if self.training:
            soft_bases = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        else:
            # Deterministic: use one-hot of argmax for eval
            soft = F.softmax(logits / self.tau, dim=-1)
            idx = soft.argmax(dim=-1)
            soft_bases = F.one_hot(idx, num_classes=4).float()
        return soft_bases, logits
