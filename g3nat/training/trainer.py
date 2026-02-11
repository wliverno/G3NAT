"""Trainer class for DNA transport models."""

import torch
import torch.nn as nn
from typing import List, Optional, Callable
from torch_geometric.loader import DataLoader

from .config import TrainingConfig


class Trainer:
    """Trainer for DNA transport GNN models."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        **kwargs
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: TrainingConfig object (optional)
            **kwargs: Additional config parameters if config not provided
        """
        self.model = model
        self.config = config or TrainingConfig.from_kwargs(**kwargs)

        # Set device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Loss function
        self.criterion = nn.HuberLoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        start_epoch: int = 0
    ) -> tuple[List[float], List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_callback: Optional callback for saving checkpoints
            progress_callback: Optional callback for tracking progress
            start_epoch: Starting epoch for resumption (default: 0)

        Returns:
            Tuple of (train_losses, val_losses) lists
        """
        # Verify all model parameters are on the same device
        param_devices = set(p.device for p in self.model.parameters())
        if len(param_devices) > 1:
            print(f"WARNING: Model parameters on different devices: {param_devices}")
            self.model = self.model.to(self.device)

        # Verify optimizer state is on the correct device
        optimizer_devices = set()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_devices.add(v.device)

        if len(optimizer_devices) > 1:
            print(f"WARNING: Optimizer state on different devices: {optimizer_devices}")
            # Move all optimizer state tensors to the model's device
            model_device = next(self.model.parameters()).device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(model_device)

        for epoch in range(start_epoch, self.config.num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(epoch, train_loss, val_loss)

            # Save checkpoint periodically if checkpointing is enabled
            if self.config.checkpoint_dir is not None and checkpoint_callback is not None:
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    checkpoint_callback(self.model, self.optimizer, epoch, self.train_losses, self.val_losses)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save final checkpoint if checkpointing is enabled
        if self.config.checkpoint_dir is not None and checkpoint_callback is not None:
            checkpoint_callback(self.model, self.optimizer, self.config.num_epochs - 1, self.train_losses, self.val_losses)

        return self.train_losses, self.val_losses

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            dos_pred, transmission_pred = self.model(batch)

            # Reshape batched targets to match predictions
            batch_size = dos_pred.size(0)
            num_energy_points = dos_pred.size(1)

            dos_target = batch.dos.view(batch_size, num_energy_points)
            transmission_target = batch.transmission.view(batch_size, num_energy_points)

            # Combined loss for DOS and transmission
            dos_loss = self.criterion(dos_pred, dos_target)
            transmission_loss = self.criterion(transmission_pred, transmission_target)
            total_loss = dos_loss + transmission_loss

            total_loss.backward()

            # Gradient clipping to prevent gradient explosion in physics-informed models
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        return train_loss

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                dos_pred, transmission_pred = self.model(batch)

                # Reshape batched targets to match predictions
                batch_size = dos_pred.size(0)
                num_energy_points = dos_pred.size(1)

                dos_target = batch.dos.view(batch_size, num_energy_points)
                transmission_target = batch.transmission.view(batch_size, num_energy_points)

                dos_loss = self.criterion(dos_pred, dos_target)
                transmission_loss = self.criterion(transmission_pred, transmission_target)
                total_loss = dos_loss + transmission_loss

                val_loss += total_loss.item()

        val_loss /= len(val_loader)
        return val_loss

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set a custom optimizer (useful for loading from checkpoint)."""
        self.optimizer = optimizer

    def set_losses(self, train_losses: List[float], val_losses: List[float]):
        """Set training history (useful for loading from checkpoint)."""
        self.train_losses = train_losses
        self.val_losses = val_losses


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               num_epochs: int = 100, learning_rate: float = 1e-3, device: str = 'auto',
               checkpoint_dir: str = None, checkpoint_frequency: int = 10,
               start_epoch: int = 0, train_losses: List[float] = None,
               val_losses: List[float] = None, optimizer: torch.optim.Optimizer = None,
               checkpoint_callback=None, progress_callback=None, max_grad_norm: float = 1.0):
    """
    Train the DNA Transport GNN model (backward-compatible function).

    Args:
        model: DNATransportGNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints (optional)
        checkpoint_frequency: Save checkpoint every N epochs (default: 10)
        start_epoch: Starting epoch for resumption (default: 0)
        train_losses: Existing training losses for resumption (optional)
        val_losses: Existing validation losses for resumption (optional)
        optimizer: Existing optimizer for resumption (optional)
        checkpoint_callback: Function to call for saving checkpoints (optional)
        progress_callback: Function to call for saving progress (optional)
        max_grad_norm: Maximum gradient norm for gradient clipping (default: 1.0)

    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    # Create config
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        max_grad_norm=max_grad_norm,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_dir=checkpoint_dir
    )

    # Create trainer
    trainer = Trainer(model, config)

    # Set optimizer if provided (for resumption)
    if optimizer is not None:
        trainer.set_optimizer(optimizer)

    # Set losses if provided (for resumption)
    if train_losses is not None:
        trainer.train_losses = train_losses
    if val_losses is not None:
        trainer.val_losses = val_losses

    # Train
    train_losses, val_losses = trainer.fit(
        train_loader,
        val_loader,
        checkpoint_callback=checkpoint_callback,
        progress_callback=progress_callback,
        start_epoch=start_epoch
    )

    return train_losses, val_losses
