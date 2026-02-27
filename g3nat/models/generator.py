"""Direct sequence optimizer using straight-through Gumbel-Softmax.

Based on Fast SeqProp:
    Linder, J., & Seelig, G. (2020). Fast activation maximization for molecular
    sequence design. BMC Bioinformatics, 21, 510.
    https://doi.org/10.1186/s12859-020-03846-2

Gumbel-Softmax straight-through estimator:
    Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with
    Gumbel-Softmax. ICLR 2017. https://arxiv.org/abs/1611.01144
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from g3nat.graph.construction import sequence_to_graph


class SequenceOptimizer(nn.Module):
    """Optimizes DNA sequence logits directly via straight-through Gumbel-Softmax.

    Instead of an MLP mapping random z vectors to sequences, this class
    maintains learnable per-position logits and optimizes them directly
    against a frozen predictor model.
    """

    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.logits = nn.Parameter(torch.randn(seq_length, 4))

    def forward(self, tau: float = 1.0):
        """Produce one-hot DNA bases via straight-through Gumbel-Softmax.

        Args:
            tau: Temperature for Gumbel-Softmax. Lower = sharper.

        Returns:
            (one_hot, normalized_logits): one_hot is [seq_length, 4],
                normalized_logits is [seq_length, 4].
        """
        # Normalize logits by subtracting per-position mean
        normalized = self.logits - self.logits.mean(dim=-1, keepdim=True)

        if self.training:
            one_hot = F.gumbel_softmax(normalized, tau=tau, hard=True, dim=-1)
        else:
            # Deterministic: argmax one-hot
            idx = normalized.argmax(dim=-1)
            one_hot = F.one_hot(idx, num_classes=4).float()

        return one_hot, normalized

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
            soft_bases: Tensor [seq_length, 4] - soft Gumbel-Softmax output for primary strand.
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
        """Compute loss as negative L2 norm of transmission difference.

        Returns negative because we maximize difference by minimizing loss.
        """
        diff = transmission_single - transmission_double
        if energy_mask is not None:
            diff = diff * energy_mask
        return -torch.norm(diff, p=2)

    def optimize(self, predictor, num_steps, tau_start=2.0, tau_end=0.1,
                 lr=0.1, energy_mask=None, log_every=100):
        """Full optimization loop with tau annealing.

        Args:
            predictor: Frozen predictor model (GNN).
            num_steps: Number of optimization steps.
            tau_start: Initial Gumbel-Softmax temperature.
            tau_end: Final Gumbel-Softmax temperature.
            lr: Learning rate for Adam optimizer.
            energy_mask: Optional mask for energy sub-window optimization.
            log_every: Print loss every N steps.

        Returns:
            List of loss values per step.
        """
        # Freeze predictor
        predictor.requires_grad_(False)
        predictor.eval()

        # Optimizer for logits only
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()

            # Linear tau annealing
            if num_steps > 1:
                tau = tau_start + (tau_end - tau_start) * step / (num_steps - 1)
            else:
                tau = tau_start

            # Forward pass
            one_hot, normalized_logits = self.forward(tau)

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
                print(f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | "
                      f"Seq: {sequence} | tau: {tau:.3f}")

        return losses
