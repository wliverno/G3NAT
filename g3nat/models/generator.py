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
    BASE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    def __init__(self, seq_length: int, use_instance_norm: bool = False,
                 init_sequence: str = None):
        super().__init__()
        self.seq_length = seq_length
        self.use_instance_norm = use_instance_norm

        if init_sequence is not None:
            if len(init_sequence) != seq_length:
                raise ValueError(
                    f"init_sequence length {len(init_sequence)} != seq_length {seq_length}")
            # Initialize logits to strongly favor the given sequence.
            # Small random noise ensures instance norm has per-column variance.
            logits = torch.randn(seq_length, 4) * 0.1
            for i, base in enumerate(init_sequence.upper()):
                logits[i, self.BASE_TO_IDX[base]] += 5.0
            self.logits = nn.Parameter(logits)
        else:
            self.logits = nn.Parameter(torch.randn(seq_length, 4))

        if use_instance_norm:
            # Per-channel instance-norm parameters (Linder & Seelig Eq. 5-7)
            self.gamma = nn.Parameter(torch.ones(4))
            self.beta = nn.Parameter(torch.zeros(4))

    def forward(self):
        """Produce one-hot DNA bases via Softmax Straight-Through estimator.

        Returns:
            (one_hot, scaled_logits): one_hot is [seq_length, 4],
                scaled_logits is [seq_length, 4].
        """
        if self.use_instance_norm:
            # Instance normalization: per-channel across positions (paper Eq. 5)
            mu = self.logits.mean(dim=0, keepdim=True)
            var = self.logits.var(dim=0, keepdim=True, unbiased=False)
            normalized = (self.logits - mu) / torch.sqrt(var + 1e-5)

            # Learnable scale and offset (paper Eq. 6-7)
            # gamma controls sampling entropy adaptively
            scaled = normalized * self.gamma + self.beta
        else:
            scaled = self.logits

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

    def _soft_complement(self, soft_bases):
        """Differentiable Watson-Crick reverse complement of soft one-hot features.

        Args:
            soft_bases: Tensor [seq_length, 4] with columns [A, T, G, C].

        Returns:
            Tensor [seq_length, 4] — reverse complement with full gradient.
        """
        # Reverse positions (5'→3' becomes 3'→5') then swap A↔T and G↔C
        # Column order: [A=0, T=1, G=2, C=3]
        # Swap: A(0)↔T(1), G(2)↔C(3) → reindex as [1, 0, 3, 2]
        return soft_bases.flip(0)[:, [1, 0, 3, 2]]

    def _build_double_soft(self, soft_bases):
        """Build double-stranded graph with differentiable features on BOTH strands.

        Uses dummy sequences for topology, then replaces both primary and
        complement node features with soft tensors so gradients flow through
        both strands.

        Args:
            soft_bases: Tensor [seq_length, 4] — ST one-hot for primary strand.

        Returns:
            Data graph with soft features on both strands.
        """
        seq_length = soft_bases.shape[0]
        dummy = 'A' * seq_length
        dummy_complement = 'T' * seq_length  # complement of all-A

        # Build topology from dummy sequences
        data = sequence_to_graph(dummy, dummy_complement)

        # Replace BOTH strands with differentiable features
        # Node layout: [left_contact, right_contact, primary[0..N-1], comp[0..N-1]]
        new_x = data.x.clone()
        new_x[2:2 + seq_length, :4] = soft_bases
        soft_comp = self._soft_complement(soft_bases)
        new_x[2 + seq_length:2 + 2 * seq_length, :4] = soft_comp
        data.x = new_x
        return data

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

    def _eval_deterministic(self, predictor, energy_mask=None):
        """Evaluate deterministic loss (eval mode, no sampling).

        Returns:
            (loss_value, sequence_string)
        """
        self.eval()
        with torch.no_grad():
            one_hot, _ = self.forward()
            seq = self.decode_sequence(one_hot)
            graph_single = self.build_graph_with_soft_features(one_hot)
            graph_double = self._build_double_soft(one_hot)
            _, trans_single = predictor(graph_single)
            _, trans_double = predictor(graph_double)
            loss_val = self.compute_loss(trans_single, trans_double, energy_mask).item()
        self.train()
        return loss_val, seq

    def optimize(self, predictor, num_steps, lr=0.001, energy_mask=None,
                 log_every=100, patience=None):
        """Fast SeqProp optimization loop with best-seen tracking.

        Optimizes logits (and gamma/beta when instance norm is enabled)
        jointly with Adam. Every ``log_every`` steps, evaluates the
        deterministic (argmax) sequence and tracks the best-seen
        parameters. After training, restores the best-seen parameters.

        Args:
            predictor: Frozen differentiable predictor model (GNN).
            num_steps: Number of optimization steps.
            lr: Learning rate for Adam optimizer (paper default 0.001).
            energy_mask: Optional mask for energy sub-window optimization.
            log_every: Print loss every N steps and evaluate deterministic loss.
            patience: Stop if deterministic loss doesn't improve for this many
                evaluation rounds. None disables early stopping.

        Returns:
            Dict with keys:
                ``losses``: list of per-step training losses,
                ``best_loss``: best deterministic loss seen,
                ``best_step``: step at which best loss occurred,
                ``best_sequence``: DNA string at best step.
        """
        # Freeze predictor
        predictor.requires_grad_(False)
        predictor.eval()

        # Optimizer for learnable parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        losses = []

        # Best-seen tracking
        best_loss = float('inf')
        best_step = 0
        best_sequence = ''
        best_state = {k: v.clone() for k, v in self.state_dict().items()}
        rounds_without_improvement = 0

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass (training mode — stochastic sampling)
            one_hot, scaled_logits = self.forward()

            # Build single-stranded graph (soft primary features)
            graph_single = self.build_graph_with_soft_features(one_hot)

            # Build double-stranded graph (soft features on BOTH strands)
            graph_double = self._build_double_soft(one_hot)

            # Forward through frozen predictor
            _, trans_single = predictor(graph_single)
            _, trans_double = predictor(graph_double)

            # Compute loss and backprop
            loss = self.compute_loss(trans_single, trans_double, energy_mask)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            # Periodic deterministic evaluation + best-seen tracking
            if (step + 1) % log_every == 0:
                det_loss, det_seq = self._eval_deterministic(predictor, energy_mask)

                if det_loss < best_loss:
                    best_loss = det_loss
                    best_step = step + 1
                    best_sequence = det_seq
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if self.use_instance_norm:
                    gamma_str = ', '.join(f'{g:.2f}' for g in self.gamma.data)
                    extra = f" | gamma: [{gamma_str}]"
                else:
                    extra = ''
                print(f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | "
                      f"Det loss: {det_loss:.4f} | Seq: {det_seq}{extra}")

                # Early stopping
                if patience is not None and rounds_without_improvement >= patience:
                    print(f"Early stopping at step {step + 1} "
                          f"(no improvement for {patience} rounds)")
                    break

        # Restore best-seen parameters
        self.load_state_dict(best_state)

        return {
            'losses': losses,
            'best_loss': best_loss,
            'best_step': best_step,
            'best_sequence': best_sequence,
        }
