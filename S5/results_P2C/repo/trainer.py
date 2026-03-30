"""
trainer.py

Training and optimization module for the physics-constrained gResNet surrogate model.
Implements the paper's training methodology (Section 2.4) including Adam optimization
with exponential learning rate decay, early stopping (500 epochs patience), windowed
training with true initialization, and validation from empty system.

This module provides the SurrogateTrainer class that manages the complete training
pipeline for the gResNet model, including:
1. Training loop with autoregressive windowed predictions
2. Validation loop with empty system initialization
3. Loss computation (MSE on scaled h, Q, Q_w with equal weights)
4. Learning rate scheduling (1e-3 → 1e-4 exponential decay)
5. Early stopping (500 epochs without improvement)
6. Checkpoint saving and loading
7. Training history tracking and visualization

Classes:
    SurrogateTrainer: Main training management class
    TrainingHistory: Data structure for tracking training metrics
    CheckpointManager: Handles model checkpointing and restoration

Functions:
    create_optimizer: Factory function for creating Adam optimizer
    create_scheduler: Factory function for exponential LR decay
    compute_window_loss: Compute loss for a single window prediction

Note:
    Strictly follows paper methodology: 2000 maximum epochs, 500 epoch early stopping
    patience, equal loss weights after scaling, and autoregressive windowed training.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Project imports
from utils import Timer, set_seed, save_pickle, load_pickle, get_device, format_time
from config import Config, TrainingConfig, SystemConfig, ModelConfig
from dataset import SWMMDataset
from model import gResNet, ConstraintLayer


@dataclass
class TrainingHistory:
    """
    Data structure for tracking training metrics and history.
    
    Attributes:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        learning_rates: List of learning rates per epoch
        epoch_times: List of epoch durations in seconds
        best_val_loss: Best validation loss achieved
        best_epoch: Epoch number of best validation loss
        early_stopping_counter: Number of epochs without improvement
        config: Training configuration for reproducibility
        metadata: Additional training metadata
        
    Note:
        Used for tracking training progress, generating plots, and
        analysis of training dynamics.
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = -1
    early_stopping_counter: int = 0
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'created_at': datetime.now().isoformat(),
        'paper_reference': 'Section 2.4: Training methodology'
    })
    
    def add_epoch(self, train_loss: float, val_loss: float, 
                  learning_rate: float, epoch_time: float) -> None:
        """
        Add epoch results to history.
        
        Args:
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            learning_rate: Learning rate used this epoch
            epoch_time: Time taken for this epoch in seconds
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
        
        # Update best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.train_losses) - 1
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert history to dictionary for serialization.
        
        Returns:
            Dictionary representation of training history
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'early_stopping_counter': self.early_stopping_counter,
            'metadata': self.metadata,
            'num_epochs': len(self.train_losses)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save training history to disk.
        
        Args:
            filepath: Path to save history file
        """
        save_pickle(self.to_dict(), filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingHistory':
        """
        Load training history from disk.
        
        Args:
            filepath: Path to history file
            
        Returns:
            Loaded TrainingHistory object
        """
        data = load_pickle(filepath)
        history = cls()
        history.train_losses = data['train_losses']
        history.val_losses = data['val_losses']
        history.learning_rates = data['learning_rates']
        history.epoch_times = data['epoch_times']
        history.best_val_loss = data['best_val_loss']
        history.best_epoch = data['best_epoch']
        history.early_stopping_counter = data['early_stopping_counter']
        history.metadata = data['metadata']
        return history
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Training and validation loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch + 1})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.learning_rates, 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epoch times
        axes[1, 0].plot(epochs, self.epoch_times, 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Epoch Duration')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation loss only (zoomed)
        axes[1, 1].plot(epochs, self.val_losses, 'r-')
        axes[1, 1].axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                          label=f'Best: {self.best_val_loss:.6f}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Validation Loss (Zoomed)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of training history.
        
        Returns:
            String summary of training results
        """
        if not self.train_losses:
            return "No training history available."
        
        total_time = sum(self.epoch_times)
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        summary_lines = [
            "=" * 80,
            "TRAINING HISTORY SUMMARY",
            "=" * 80,
            f"Total epochs: {len(self.train_losses)}",
            f"Best epoch: {self.best_epoch + 1}",
            f"Best validation loss: {self.best_val_loss:.6f}",
            f"Final training loss: {self.train_losses[-1]:.6f}",
            f"Final validation loss: {self.val_losses[-1]:.6f}",
            f"Total training time: {format_time(total_time)}",
            f"Average epoch time: {format_time(avg_epoch_time)}",
            f"Early stopping counter: {self.early_stopping_counter}",
            f"Initial learning rate: {self.learning_rates[0]:.4f}",
            f"Final learning rate: {self.learning_rates[-1]:.4f}",
            "=" * 80
        ]
        
        return "\n".join(summary_lines)


class CheckpointManager:
    """
    Manages model checkpointing and restoration.
    
    Handles saving and loading of model checkpoints during training,
    including model weights, optimizer state, scheduler state, and
    training history.
    
    Attributes:
        checkpoint_dir: Directory for storing checkpoints
        keep_checkpoints: Number of checkpoints to keep
        checkpoint_format: Format string for checkpoint filenames
        
    Note:
        Follows paper's requirement for reproducibility by saving
        complete training state.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 keep_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_format = "checkpoint_epoch_{:04d}.pt"
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, epoch: int, model: nn.Module, 
                       optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler],
                       training_history: TrainingHistory,
                       config: Dict[str, Any],
                       is_best: bool = False) -> str:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            model: gResNet model
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            training_history: Training history object
            config: Configuration dictionary
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint file
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': training_history.to_dict(),
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Determine filename
        if is_best:
            filename = "best_model.pt"
        else:
            filename = self.checkpoint_format.format(epoch)
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Clean up old checkpoints (keep only latest N)
        if not is_best:
            self._cleanup_old_checkpoints(epoch)
        
        print(f"Checkpoint saved: {filepath}")
        return str(filepath)
    
    def _cleanup_old_checkpoints(self, current_epoch: int) -> None:
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            current_epoch: Current epoch number
        """
        if self.keep_checkpoints <= 0:
            return
        
        # List all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoint_files) <= self.keep_checkpoints:
            return
        
        # Sort by epoch number
        def get_epoch(filename):
            try:
                return int(filename.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1
        
        sorted_files = sorted(checkpoint_files, key=get_epoch)
        
        # Remove oldest files
        files_to_remove = sorted_files[:-self.keep_checkpoints]
        for file in files_to_remove:
            try:
                file.unlink()
                print(f"Removed old checkpoint: {file.name}")
            except OSError as e:
                warnings.warn(f"Failed to remove checkpoint {file}: {e}")
    
    def load_checkpoint(self, filepath: str, model: nn.Module,
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       load_optimizer: bool = True) -> Tuple[int, TrainingHistory]:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: gResNet model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Tuple of (epoch, training_history)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        print(f"Loading checkpoint from: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if requested
            if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if requested
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            history_data = checkpoint['training_history']
            training_history = TrainingHistory()
            training_history.train_losses = history_data['train_losses']
            training_history.val_losses = history_data['val_losses']
            training_history.learning_rates = history_data['learning_rates']
            training_history.epoch_times = history_data['epoch_times']
            training_history.best_val_loss = history_data['best_val_loss']
            training_history.best_epoch = history_data['best_epoch']
            training_history.early_stopping_counter = history_data['early_stopping_counter']
            training_history.metadata = history_data['metadata']
            
            epoch = checkpoint['epoch']
            
            print(f"  Loaded epoch: {epoch}")
            print(f"  Best validation loss: {training_history.best_val_loss:.6f}")
            print(f"  Training history: {len(training_history.train_losses)} epochs")
            
            return epoch, training_history
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by epoch number (extracted from filename)
        def get_epoch(filename):
            try:
                return int(filename.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1
        
        latest_file = max(checkpoint_files, key=get_epoch)
        return str(latest_file)
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Find the best checkpoint file.
        
        Returns:
            Path to best checkpoint, or None if not found
        """
        best_file = self.checkpoint_dir / "best_model.pt"
        return str(best_file) if best_file.exists() else None


class SurrogateTrainer:
    """
    Main training class for physics-constrained gResNet surrogate model.
    
    Implements the complete training pipeline following the paper's methodology:
    - Adam optimizer with exponential learning rate decay (1e-3 → 1e-4)
    - Early stopping with 500 epoch patience
    - Windowed training with true initialization from HiFi data
    - Validation from empty system (all zeros)
    - MSE loss on scaled h, Q, Q_w with equal weights
    
    Attributes:
        model: gResNet model instance
        constraint_layer: ConstraintLayer instance (from model)
        config: Configuration dictionary
        training_config: Training configuration
        system_config: System configuration
        device: PyTorch device (CPU/GPU)
        optimizer: Adam optimizer
        scheduler: Exponential learning rate scheduler
        criterion: MSE loss function
        checkpoint_manager: CheckpointManager instance
        training_history: TrainingHistory instance
        n_nodes: Number of nodes in network
        n_links: Number of links in network
        state_dim: State dimension (n_nodes + n_links)
        
    Methods:
        train: Main training loop
        validate: Validation loop with empty system initialization
        compute_loss: Compute MSE loss on h, Q, Q_w predictions
        save_checkpoint: Save training checkpoint
        load_checkpoint: Load training checkpoint
        plot_training_curves: Generate training loss plots
        get_training_summary: Get training results summary
        
    Note:
        Follows paper methodology exactly: 2000 max epochs, 500 epoch early
        stopping patience, equal loss weights after scaling.
    """
    
    def __init__(self, model: gResNet, constraint_layer: ConstraintLayer, 
                 config: Optional[Dict] = None):
        """
        Initialize SurrogateTrainer with model and configuration.
        
        Args:
            model: gResNet model instance
            constraint_layer: ConstraintLayer instance (integrated in model)
            config: Configuration dictionary. If None, uses default Config.
            
        Raises:
            ValueError: If model or constraint_layer is invalid
            RuntimeError: If optimizer initialization fails
        """
        # Store model and constraint layer
        self.model = model
        self.constraint_layer = constraint_layer
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.training_config = self.config.training
            self.system_config = self.config.system
            self.model_config = self.config.model
        else:
            self.config = config
            self.training_config = TrainingConfig(**config.get('training', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
            self.model_config = ModelConfig(**config.get('model', {}))
        
        # Set random seed for reproducibility
        set_seed(self.system_config.random_seed)
        
        # Get device
        self.device = get_device(config)
        
        # Move model to device
        self.model.to(self.device)
        
        # Get network dimensions from model
        self.n_nodes = self.model.n_nodes
        self.n_links = self.model.n_links
        self.state_dim = self.n_nodes + self.n_links
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function (MSE with equal weights)
        self.criterion = nn.MSELoss(reduction='mean')
        
        # Initialize checkpoint manager
        checkpoint_dir = os.path.join("S5/results_P2C", self.system_config.checkpoint_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_checkpoints=self.training_config.keep_checkpoints
        )
        
        # Initialize training history
        self.training_history = TrainingHistory(config=self.config)
        
        # Training state
        self.current_epoch = 0
        self.best_model_state = None
        
        # Create results directory
        self.results_dir = Path("S5/results_P2C") / self.system_config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized SurrogateTrainer:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {self.model.get_total_parameters():,}")
        print(f"  Learning rate: {self.training_config.learning_rate['initial']} → "
              f"{self.training_config.learning_rate['final']}")
        print(f"  Early stopping: {self.training_config.early_stopping['patience']} epochs patience")
        print(f"  Checkpoint directory: {self.checkpoint_manager.checkpoint_dir}")
        print(f"  Results directory: {self.results_dir}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create Adam optimizer as per paper methodology.
        
        Returns:
            Adam optimizer configured per paper settings
            
        Note:
            Paper Section 2.4: Adam optimizer with initial learning rate 1e-3.
            Weight decay not mentioned, so not used.
        """
        optimizer_config = {
            'lr': self.training_config.learning_rate['initial'],
            'betas': (0.9, 0.999),  # Adam default
            'eps': 1e-8,  # Adam default
            'weight_decay': 0.0,  # Not mentioned in paper
            'amsgrad': False  # Not mentioned in paper
        }
        
        return optim.Adam(self.model.parameters(), **optimizer_config)
    
    def _create_scheduler(self) -> optim.lr_scheduler.ExponentialLR:
        """
        Create exponential learning rate scheduler.
        
        Returns:
            ExponentialLR scheduler configured per paper settings
            
        Note:
            Paper Section 2.4: Learning rate decays exponentially from 1e-3 to 1e-4
            over the course of training (2000 epochs maximum).
        """
        initial_lr = self.training_config.learning_rate['initial']
        final_lr = self.training_config.learning_rate['final']
        max_epochs = self.training_config.epochs
        
        # Compute gamma for exponential decay: final_lr = initial_lr * gamma^max_epochs
        gamma = (final_lr / initial_lr) ** (1.0 / max_epochs)
        
        return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    
    def compute_loss(self, predictions: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Compute MSE loss on h, Q, and Q_w predictions.
        
        Implements the loss function from paper Section 2.3:
        L = MSE(h_pred, h_true) + MSE(Q_pred, Q_true) + MSE(Q_w_pred, Q_w_true)
        
        Args:
            predictions: Tuple of (h_pred, Q_pred, Q_w_pred) tensors
            targets: Tuple of (h_true, Q_true, Q_w_true) tensors
            
        Returns:
            Total MSE loss (scalar tensor)
            
        Note:
            All variables are assumed to be scaled to [0,1] range.
            Q_w_pred may be None if physical constraints disabled.
            Paper uses equal weighting after scaling (all weights = 1.0).
        """
        h_pred, Q_pred, Q_w_pred = predictions
        h_true, Q_true, Q_w_true = targets
        
        # Compute losses for each component
        h_loss = self.criterion(h_pred, h_true)
        Q_loss = self.criterion(Q_pred, Q_true)
        
        # Q_w loss (0 if physical constraints disabled)
        if Q_w_pred is not None and Q_w_true is not None:
            Q_w_loss = self.criterion(Q_w_pred, Q_w_true)
        else:
            Q_w_loss = torch.tensor(0.0, device=self.device)
        
        # Apply weights from config (default: all 1.0 after scaling)
        weights = self.training_config.loss['weights']
        total_loss = (weights['h'] * h_loss + 
                     weights['Q'] * Q_loss + 
                     weights['Q_w'] * Q_w_loss)
        
        return total_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch using windowed training.
        
        Args:
            train_loader: DataLoader for training windows
            
        Returns:
            Average training loss for the epoch
            
        Note:
            Each window is trained independently with true initialization
            at window start (paper's methodology). The model runs
            autoregressively within each window.
        """
        self.model.train()
        total_loss = 0.0
        total_windows = 0

        for batch_idx, (x_t, R_sequence, target_sequence) in enumerate(train_loader):
            batch_size = x_t.size(0)
            window_size = R_sequence.size(1)

            # --- Vectorized Q_w_true: one constraint_layer call for whole window ---
            # Reshape (batch, window, dim) → (batch*window, dim), run once, reshape back.
            # This replaces window_size separate calls (e.g. 60 → 1).
            with torch.no_grad():
                BW = batch_size * window_size
                Q_w_true_all = self.constraint_layer(
                    target_sequence[:, :, :self.n_nodes].reshape(BW, -1),
                    target_sequence[:, :, self.n_nodes:].reshape(BW, -1),
                    R_sequence.reshape(BW, -1),
                ).reshape(batch_size, window_size, -1)

            current_state = x_t
            self.optimizer.zero_grad()

            # Autoregressive forward: collect predictions for all timesteps
            h_preds, Q_preds, Q_w_preds = [], [], []
            for step in range(window_size):
                h_pred, Q_pred, Q_w_pred = self.model(current_state, R_sequence[:, step, :])
                h_preds.append(h_pred)
                Q_preds.append(Q_pred)
                Q_w_preds.append(Q_w_pred)
                current_state = torch.cat([h_pred.detach(), Q_pred.detach()], dim=1)

            # Single loss call on stacked (batch, window, dim) tensors
            window_loss = self.compute_loss(
                (torch.stack(h_preds, dim=1),
                 torch.stack(Q_preds, dim=1),
                 torch.stack(Q_w_preds, dim=1)),
                (target_sequence[:, :, :self.n_nodes],
                 target_sequence[:, :, self.n_nodes:],
                 Q_w_true_all),
            )

            window_loss.backward()

            if self.model_config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model_config.gradient_clipping
                )

            self.optimizer.step()

            total_loss += window_loss.item() * batch_size
            total_windows += batch_size

        avg_loss = total_loss / total_windows if total_windows > 0 else 0.0
        return avg_loss
    
    def validate(self, val_dataset: Union[SWMMDataset, DataLoader]) -> float:
        """
        Validate model on validation dataset.
        
        Args:
            val_dataset: Validation dataset or DataLoader
            
        Returns:
            Average validation loss
            
        Note:
            Follows paper methodology: initialize from empty system (all zeros)
            and run autoregressively through the entire validation sequence.
            This tests model stability without teacher forcing.
        """
        self.model.eval()

        if isinstance(val_dataset, SWMMDataset):
            val_loader = val_dataset.get_dataloader(
                batch_size=self.training_config.batch_size, shuffle=False
            )
        else:
            val_loader = val_dataset

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for x_t, R_sequence, target_sequence in val_loader:
                batch_size = x_t.size(0)
                window_size = R_sequence.size(1)

                # Vectorized Q_w_true for entire window in one call
                BW = batch_size * window_size
                Q_w_true_all = self.constraint_layer(
                    target_sequence[:, :, :self.n_nodes].reshape(BW, -1),
                    target_sequence[:, :, self.n_nodes:].reshape(BW, -1),
                    R_sequence.reshape(BW, -1),
                ).reshape(batch_size, window_size, -1)

                # Empty-system initialisation (paper methodology)
                current_state = torch.zeros_like(x_t)

                h_preds, Q_preds, Q_w_preds = [], [], []
                for step in range(window_size):
                    h_pred, Q_pred, Q_w_pred = self.model(current_state, R_sequence[:, step, :])
                    h_preds.append(h_pred)
                    Q_preds.append(Q_pred)
                    Q_w_preds.append(Q_w_pred)
                    current_state = torch.cat([h_pred, Q_pred], dim=1)

                # Single loss call; .item() syncs GPU only once per batch
                batch_loss = self.compute_loss(
                    (torch.stack(h_preds, dim=1),
                     torch.stack(Q_preds, dim=1),
                     torch.stack(Q_w_preds, dim=1)),
                    (target_sequence[:, :, :self.n_nodes],
                     target_sequence[:, :, self.n_nodes:],
                     Q_w_true_all),
                )
                total_loss += batch_loss.item()
                total_batches += 1

        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def train(self, train_dataset: SWMMDataset, val_dataset: SWMMDataset,
              start_epoch: int = 0, max_epochs: Optional[int] = None,
              early_stopping: bool = True) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            start_epoch: Epoch to start from (for resuming training)
            max_epochs: Maximum number of epochs (overrides config)
            early_stopping: Whether to use early stopping
            
        Returns:
            Dictionary with training results:
            - 'history': TrainingHistory object
            - 'best_val_loss': Best validation loss achieved
            - 'best_epoch': Epoch of best validation loss
            - 'total_time': Total training time in seconds
            
        Raises:
            RuntimeError: If training fails
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Training windows: {len(train_dataset)}")
        print(f"Validation windows: {len(val_dataset)}")
        print(f"Window size: {train_dataset.window_size} timesteps")
        print(f"Maximum epochs: {max_epochs or self.training_config.epochs}")
        print(f"Early stopping: {early_stopping} "
              f"(patience: {self.training_config.early_stopping['patience']})")
        
        # Create DataLoaders
        train_loader = train_dataset.get_dataloader(
            batch_size=self.training_config.batch_size,
            shuffle=True  # Shuffle training data
        )
        
        val_loader = val_dataset.get_dataloader(
            batch_size=self.training_config.batch_size,
            shuffle=False  # Don't shuffle validation data
        )
        
        # Set maximum epochs
        if max_epochs is None:
            max_epochs = self.training_config.epochs
        
        # Training loop
        start_time = datetime.now()
        
        try:
            for epoch in range(start_epoch, max_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                with Timer() as train_timer:
                    train_loss = self.train_epoch(train_loader)

                # Validate
                with Timer() as val_timer:
                    val_loss = self.validate(val_loader)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update training history
                self.training_history.add_epoch(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    epoch_time=train_timer.elapsed + val_timer.elapsed
                )
                
                # Step learning rate scheduler
                self.scheduler.step()
                
                # Print every 10 epochs to avoid stdout flush overhead
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:4d} | train={train_loss:.6f} "
                          f"val={val_loss:.6f} lr={current_lr:.2e} "
                          f"| {format_time(train_timer.elapsed+val_timer.elapsed)}")
                
                # Check for improvement
                is_best = val_loss < self.training_history.best_val_loss
                
                if is_best:
                    print(f"  *** New best validation loss: {val_loss:.6f} "
                          f"(improvement: {self.training_history.best_val_loss - val_loss:.6f})")
                    # Save best model state
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'val_loss': val_loss
                    }
                
                # Save checkpoint
                if (epoch + 1) % self.training_config.checkpoint_frequency == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping check
                if early_stopping:
                    early_stopping_config = self.training_config.early_stopping
                    patience = early_stopping_config['patience']
                    min_delta = early_stopping_config['min_delta']
                    
                    if self.training_history.early_stopping_counter >= patience:
                        print(f"\nEarly stopping triggered after {patience} epochs "
                              f"without improvement > {min_delta}")
                        break
            
            # Training completed
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Save final checkpoint
            self.save_checkpoint(self.current_epoch, is_best=False)
            
            # Restore best model weights if early stopping is enabled
            if early_stopping and self.training_config.early_stopping['restore_best_weights']:
                if self.best_model_state is not None:
                    print(f"\nRestoring best model weights from epoch "
                          f"{self.best_model_state['epoch'] + 1}")
                    self.model.load_state_dict(self.best_model_state['model_state_dict'])
            
            # Generate training summary
            results = self._create_training_results(total_time)
            
            print("\n" + "="*80)
            print("TRAINING COMPLETED")
            print("="*80)
            print(results['summary'])
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Training failed at epoch {self.current_epoch + 1}: {str(e)}") from e
    
    def _create_training_results(self, total_time: float) -> Dict[str, Any]:
        """
        Create training results dictionary.
        
        Args:
            total_time: Total training time in seconds
            
        Returns:
            Dictionary with training results
        """
        return {
            'history': self.training_history,
            'best_val_loss': self.training_history.best_val_loss,
            'best_epoch': self.training_history.best_epoch,
            'total_time': total_time,
            'final_train_loss': self.training_history.train_losses[-1] if self.training_history.train_losses else 0.0,
            'final_val_loss': self.training_history.val_losses[-1] if self.training_history.val_losses else 0.0,
            'config': self.config,
            'model_parameters': self.model.get_total_parameters(),
            'summary': self.training_history.get_summary()
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint file
        """
        return self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_history=self.training_history,
            config=self.config,
            is_best=is_best
        )
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None,
                       load_optimizer: bool = True) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Epoch number from which to resume training
            
        Raises:
            FileNotFoundError: If no checkpoint found
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            # Try to find latest checkpoint
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
            if checkpoint_path is None:
                # Try to find best checkpoint
                checkpoint_path = self.checkpoint_manager.find_best_checkpoint()
            
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found to load")
        
        # Load checkpoint
        self.current_epoch, self.training_history = self.checkpoint_manager.load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            scheduler=self.scheduler if load_optimizer else None,
            load_optimizer=load_optimizer
        )
        
        # Update learning rate scheduler to correct epoch
        for _ in range(self.current_epoch):
            self.scheduler.step()
        
        print(f"Resuming training from epoch {self.current_epoch + 1}")
        return self.current_epoch
    
    def plot_training_curves(self, save: bool = True) -> str:
        """
        Plot training and validation loss curves.
        
        Args:
            save: Whether to save the plot to file
            
        Returns:
            Path to saved plot file (if saved), otherwise empty string
        """
        if not self.training_history.train_losses:
            warnings.warn("No training history to plot")
            return ""
        
        # Create plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"training_curves_{timestamp}.png"
        plot_path = self.results_dir / plot_filename
        
        self.training_history.plot_training_curves(
            save_path=str(plot_path) if save else None
        )
        
        return str(plot_path) if save else ""
    
    def save_training_results(self, filename: Optional[str] = None) -> str:
        """
        Save training results to disk.
        
        Args:
            filename: Optional filename. If None, generates timestamped name.
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{timestamp}.pkl"
        
        filepath = self.results_dir / filename
        
        # Create results dictionary
        results = {
            'training_history': self.training_history.to_dict(),
            'config': self.config,
            'model_config': self.model.get_config(),
            'best_val_loss': self.training_history.best_val_loss,
            'best_epoch': self.training_history.best_epoch,
            'total_parameters': self.model.get_total_parameters(),
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'paper_reference': 'Training results following paper methodology (Section 2.4)'
        }
        
        save_pickle(results, str(filepath))
        print(f"Training results saved to: {filepath}")
        
        return str(filepath)
    
    def get_training_summary(self) -> str:
        """
        Get human-readable training summary.
        
        Returns:
            String summary of training results
        """
        if not self.training_history.train_losses:
            return "Training not started yet."
        
        return self.training_history.get_summary()
    
    def evaluate_model(self, test_dataset: SWMMDataset) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary with evaluation metrics:
            - 'test_loss': Average test loss
            - Additional metrics could be added (RMSE, R², etc.)
            
        Note:
            This is a simple evaluation. More comprehensive evaluation
            should be done using the Evaluator class.
        """
        print("\nEvaluating model on test dataset...")
        
        test_loss = self.validate(test_dataset)
        
        print(f"  Test loss: {test_loss:.6f}")
        
        return {
            'test_loss': test_loss,
            'test_loss_relative': test_loss / self.training_history.best_val_loss if self.training_history.best_val_loss > 0 else float('inf')
        }


def create_trainer_from_config(model: gResNet, constraint_layer: ConstraintLayer,
                              config_path: str = "config.yaml") -> SurrogateTrainer:
    """
    Factory function to create SurrogateTrainer from configuration file.
    
    Args:
        model: gResNet model instance
        constraint_layer: ConstraintLayer instance
        config_path: Path to configuration file
        
    Returns:
        SurrogateTrainer instance
    """
    from config import load_config_from_yaml
    
    config = load_config_from_yaml(config_path)
    trainer = SurrogateTrainer(model, constraint_layer, config.__dict__)
    
    return trainer


def test_trainer():
    """
    Test function for SurrogateTrainer.
    
    Creates a simple test to verify trainer functionality.
    """
    print("Testing SurrogateTrainer...")
    
    # Test parameters
    n_nodes = 6
    n_links = 6
    state_dim = n_nodes + n_links
    window_size = 10
    n_windows = 50
    
    # Create dummy adjacency matrices
    from scipy.sparse import csr_matrix
    import numpy as np
    
    upstream_data = np.ones(n_links, dtype=np.float32)
    upstream_rows = list(range(n_links))
    upstream_cols = [(i + 1) % n_links for i in range(n_links)]
    
    downstream_data = np.ones(n_links, dtype=np.float32)
    downstream_rows = list(range(n_links))
    downstream_cols = list(range(n_links))
    
    upstream_matrix = csr_matrix((upstream_data, (upstream_rows, upstream_cols)), 
                                shape=(n_nodes, n_links))
    downstream_matrix = csr_matrix((downstream_data, (downstream_rows, downstream_cols)), 
                                  shape=(n_nodes, n_links))
    
    try:
        # Create dummy data
        np.random.seed(42)
        
        # States: scaled to [0,1]
        states = np.random.rand(n_windows, state_dim).astype(np.float32)
        
        # Inputs: window_size steps per window
        inputs = np.random.rand(n_windows, window_size, n_nodes).astype(np.float32)
        
        # Targets: same shape as states for each step
        targets = np.random.rand(n_windows, window_size, state_dim).astype(np.float32)
        
        # Create dataset
        from dataset import SWMMDataset
        dataset = SWMMDataset(
            states=states,
            inputs=inputs,
            targets=targets,
            window_size=window_size,
            dataset_type="train"
        )
        
        # Create validation dataset (different data)
        val_states = np.random.rand(n_windows // 2, state_dim).astype(np.float32)
        val_inputs = np.random.rand(n_windows // 2, window_size, n_nodes).astype(np.float32)
        val_targets = np.random.rand(n_windows // 2, window_size, state_dim).astype(np.float32)
        
        val_dataset = SWMMDataset(
            states=val_states,
            inputs=val_inputs,
            targets=val_targets,
            window_size=window_size,
            dataset_type="val"
        )
        
        # Create model
        from model import gResNet
        model = gResNet(
            n_nodes=n_nodes,
            n_links=n_links,
            upstream_matrix=upstream_matrix,
            downstream_matrix=downstream_matrix
        )
        
        # Create trainer
        trainer = SurrogateTrainer(model, model.constraint_layer)
        
        # Test single epoch training
        print("\n1. Testing single epoch training...")
        train_loader = dataset.get_dataloader(batch_size=4, shuffle=True)
        train_loss = trainer.train_epoch(train_loader)
        print(f"  Training loss: {train_loss:.6f}")
        assert train_loss >= 0.0
        
        # Test validation
        print("\n2. Testing validation...")
        val_loss = trainer.validate(val_dataset)
        print(f"  Validation loss: {val_loss:.6f}")
        assert val_loss >= 0.0
        
        # Test loss computation
        print("\n3. Testing loss computation...")
        # Create dummy predictions and targets
        batch_size = 2
        h_pred = torch.randn(batch_size, n_nodes)
        Q_pred = torch.randn(batch_size, n_links)
        Q_w_pred = torch.randn(batch_size, n_nodes)
        
        h_true = torch.randn(batch_size, n_nodes)
        Q_true = torch.randn(batch_size, n_links)
        Q_w_true = torch.randn(batch_size, n_nodes)
        
        loss = trainer.compute_loss(
            (h_pred, Q_pred, Q_w_pred),
            (h_true, Q_true, Q_w_true)
        )
        print(f"  Computed loss: {loss.item():.6f}")
        assert loss.item() >= 0.0
        
        # Test checkpoint saving/loading
        print("\n4. Testing checkpoint saving/loading...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary checkpoint manager
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                epoch=10,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                training_history=trainer.training_history,
                config=trainer.config,
                is_best=True
            )
            
            print(f"  Checkpoint saved: {checkpoint_path}")
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            loaded_epoch, loaded_history = checkpoint_manager.load_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler
            )
            
            print(f"  Loaded epoch: {loaded_epoch}")
            print(f"  Loaded history length: {len(loaded_history.train_losses)}")
            assert loaded_epoch == 10
        
        # Test training history
        print("\n5. Testing training history...")
        # Add some dummy data
        trainer.training_history.add_epoch(
            train_loss=0.1,
            val_loss=0.2,
            learning_rate=0.001,
            epoch_time=10.5
        )
        
        trainer.training_history.add_epoch(
            train_loss=0.08,
            val_loss=0.15,
            learning_rate=0.0009,
            epoch_time=10.2
        )
        
        summary = trainer.get_training_summary()
        print(f"  Training summary generated: {len(summary.splitlines())} lines")
        assert "TRAINING HISTORY SUMMARY" in summary
        
        # Test plot generation (not saved in test)
        print("\n6. Testing plot generation...")
        plot_path = trainer.plot_training_curves(save=False)
        print(f"  Plot generation test passed")
        
        print("\nAll SurrogateTrainer tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_trainer()
    if success:
        print("\nTrainer test completed successfully!")
    else:
        print("\nTrainer test failed!")
