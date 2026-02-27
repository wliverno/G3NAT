"""DNA sequence generator using Gumbel-Softmax for differentiable sequence generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNASequenceGenerator(nn.Module):
    """Generates soft DNA sequence representations via an MLP + Gumbel-Softmax."""

    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}

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

    def decode_sequences(self, soft_bases):
        """Convert soft base tensor to list of DNA sequence strings via argmax."""
        indices = soft_bases.argmax(dim=-1)  # [batch, seq_length]
        sequences = []
        for i in range(indices.shape[0]):
            seq = ''.join(self.IDX_TO_BASE[idx.item()] for idx in indices[i])
            sequences.append(seq)
        return sequences

    def get_complement(self, sequence):
        """Return Watson-Crick complement of a DNA sequence."""
        return ''.join(self.COMPLEMENT[b] for b in sequence)
