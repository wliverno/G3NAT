"""Direct sequence optimizer using Softmax Straight-Through with adaptive entropy.

Based on Fast SeqProp:
    Linder, J., & Seelig, G. (2021). Fast activation maximization for molecular
    sequence design. BMC Bioinformatics, 22, 510.
    https://doi.org/10.1186/s12859-021-04437-5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from g3nat.graph.construction import sequence_to_graph


class SequenceOptimizer(nn.Module):
    """Optimizes DNA sequences via Softmax ST with instance normalization.

    Maintains learnable per-position logits with per-channel instance
    normalization (learnable gamma/beta). Gamma adaptively controls
    sampling entropy — no manual temperature schedule needed.

    Based on Fast SeqProp (Linder & Seelig, 2021).
    """

    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.logits = nn.Parameter(torch.randn(seq_length, 4))
        # Per-channel instance-norm parameters (Linder & Seelig Eq. 5-7)
        self.gamma = nn.Parameter(torch.ones(4))
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self):
        """Produce one-hot DNA bases via Softmax Straight-Through estimator.

        Returns:
            (one_hot, scaled_logits): one_hot is [seq_length, 4],
                scaled_logits is [seq_length, 4].
        """
        # Instance normalization: per-channel across positions (paper Eq. 5)
        mu = self.logits.mean(dim=0, keepdim=True)
        var = self.logits.var(dim=0, keepdim=True, unbiased=False)
        normalized = (self.logits - mu) / torch.sqrt(var + 1e-5)

        # Learnable scale and offset (paper Eq. 6-7)
        # gamma controls sampling entropy adaptively
        scaled = normalized * self.gamma + self.beta

        if self.training:
            # Softmax ST: categorical sample in forward, softmax grad in backward
            probs = F.softmax(scaled, dim=-1)
            with torch.no_grad():
                indices = torch.multinomial(probs, 1).squeeze(-1)
                hard = F.one_hot(indices, num_classes=4).float()
            # Straight-through: forward uses hard, backward uses probs
            one_hot = hard - probs.detach() + probs
        else:
            # Deterministic: argmax one-hot
            idx = scaled.argmax(dim=-1)
            one_hot = F.one_hot(idx, num_classes=4).float()

        return one_hot, scaled

    def decode_sequence(self, one_hot):
        """Convert one-hot tensor to DNA string via argmax.

        Args:
            one_hot: Tensor [seq_length, 4].

        Returns:
            DNA sequence string.
        """
        indices = one_hot.argmax(dim=-1)
        return ''.join(self.IDX_TO_BASE[idx.item()] for idx in indices)

    def get_complement(self, sequence):
        """Return Watson-Crick reverse complement of a DNA sequence."""
        return ''.join(self.COMPLEMENT[b] for b in reversed(sequence))

    def build_graph_with_soft_features(self, soft_bases, complementary_sequence=None):
        """Build a graph using sequence_to_graph topology but with soft node features.

        Args:
            soft_bases: Tensor [seq_length, 4] - ST one-hot for primary strand.
            complementary_sequence: Optional complement string. None means single-stranded.
        """
        seq_length = soft_bases.shape[0]
        dummy_primary = 'A' * seq_length

        if complementary_sequence is None:
            data = sequence_to_graph(dummy_primary)
        else:
            data = sequence_to_graph(dummy_primary, complementary_sequence)

        # Replace primary strand node features with soft bases
        # Nodes: 0=left_contact, 1=right_contact, 2..2+N-1=primary strand
        new_x = data.x.clone()
        new_x[2:2 + seq_length, :4] = soft_bases
        data.x = new_x
        return data

    def compute_loss(self, transmission_single, transmission_double, energy_mask=None):
        """Compute loss as negative L1 norm of transmission difference.

        Returns negative because we maximize difference by minimizing loss.
        """
        diff = transmission_single - transmission_double
        if energy_mask is not None:
            diff = diff * energy_mask
        return -torch.norm(diff, p=1)

    def optimize(self, predictor, num_steps, lr=0.1, energy_mask=None,
                 log_every=100):
        """Fast SeqProp optimization loop.

        Optimizes logits, gamma, and beta jointly with Adam. Gamma
        adaptively controls sampling entropy (no manual temperature
        schedule). Requires a differentiable predictor.

        Args:
            predictor: Frozen differentiable predictor model (GNN).
            num_steps: Number of optimization steps.
            lr: Learning rate for Adam optimizer.
            energy_mask: Optional mask for energy sub-window optimization.
            log_every: Print loss every N steps.

        Returns:
            List of loss values per step.
        """
        # Freeze predictor
        predictor.requires_grad_(False)
        predictor.eval()

        # Optimizer for logits + gamma + beta
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass
            one_hot, scaled_logits = self.forward()

            # Decode discrete sequence for complement
            sequence = self.decode_sequence(one_hot)

            # Build single-stranded graph
            graph_single = self.build_graph_with_soft_features(one_hot)

            # Build double-stranded graph
            complement = self.get_complement(sequence)
            graph_double = self.build_graph_with_soft_features(one_hot, complement)

            # Forward through frozen predictor
            _, trans_single = predictor(graph_single)
            _, trans_double = predictor(graph_double)

            # Compute loss and backprop
            loss = self.compute_loss(trans_single, trans_double, energy_mask)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if (step + 1) % log_every == 0:
                gamma_str = ', '.join(f'{g:.2f}' for g in self.gamma.data)
                print(f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | "
                      f"Seq: {sequence} | gamma: [{gamma_str}]")

        return losses
