"""
dataset.py

PyTorch Dataset implementation for windowed SWMM simulation data used in training
the physics-constrained gResNet surrogate model.

This module implements the SWMMDataset class that provides windowed access to
hydraulic state and input sequences for autoregressive training of the gResNet
model. It strictly follows the paper's methodology for windowed training
(Section 2.3: "Surrogate training") and supports both training (true state
initialization) and validation (empty system initialization) modes.

Classes:
    SWMMDataset: PyTorch Dataset for windowed hydraulic state sequences
    BatchCollator: Custom collate function for batching windowed sequences
    SWMMDataLoader: Wrapper around PyTorch DataLoader with dataset-specific settings

Functions:
    create_data_loaders: Factory function for creating training, validation, and test loaders
    split_dataset_by_indices: Split dataset into subsets by window indices

Note:
    The dataset implements the windowed training approach described in the paper:
    - Window size: 60 minutes (selected from hyperparameter study)
    - Initialization: True HiFi states for training, empty system for validation
    - Autoregressive: Each window requires running model W times sequentially
    - Parallel processing: Windows can be processed in parallel during training
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

# Project imports
from utils import Timer, set_seed, save_pickle, load_pickle, get_device
from config import Config, DataConfig, TrainingConfig, SystemConfig


@dataclass
class DatasetMetadata:
    """
    Metadata container for SWMMDataset instances.
    
    Stores information about dataset composition, windowing strategy,
    and alignment with paper methodology for reproducibility.
    
    Attributes:
        source_file: Path to source data file or identifier
        n_windows: Total number of windows in dataset
        window_size: Window size in timesteps
        state_dim: Dimension of state vector (n_nodes + n_links)
        input_dim: Dimension of input vector (n_nodes)
        n_timesteps_total: Total timesteps in original series
        n_windows_created: Number of windows actually created (may be less than possible)
        initialization_mode: How windows are initialized ('true_state' or 'empty_system')
        dataset_type: Type of dataset ('train', 'val', 'test')
        created_at: Timestamp when dataset was created
        paper_reference: Reference to paper methodology sections
        scaling_applied: Whether data has been scaled to [0,1]
        node_ids: Ordered list of node IDs for state vector interpretation
        link_ids: Ordered list of link IDs for state vector interpretation
    """
    
    # Dataset properties
    source_file: str = ""
    n_windows: int = 0
    window_size: int = 0
    state_dim: int = 0
    input_dim: int = 0
    n_timesteps_total: int = 0
    n_windows_created: int = 0
    
    # Configuration
    initialization_mode: str = "true_state"  # 'true_state' or 'empty_system'
    dataset_type: str = "train"  # 'train', 'val', or 'test'
    
    # Metadata
    created_at: str = ""
    paper_reference: str = "Section 2.3: Surrogate training with windows"
    scaling_applied: bool = False
    node_ids: List[str] = field(default_factory=list)
    link_ids: List[str] = field(default_factory=list)
    
    # Performance metrics
    memory_size_mb: float = 0.0
    creation_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def validate(self) -> bool:
        """
        Validate dataset metadata for consistency.
        
        Returns:
            True if metadata is valid, False otherwise
            
        Raises:
            ValueError: If metadata is inconsistent or invalid
        """
        # Check required fields
        required_fields = ['n_windows', 'window_size', 'state_dim', 'input_dim']
        for field in required_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"Field {field} must be positive, got {getattr(self, field)}")
        
        # Check initialization mode
        valid_modes = ['true_state', 'empty_system']
        if self.initialization_mode not in valid_modes:
            raise ValueError(f"initialization_mode must be one of {valid_modes}, "
                           f"got {self.initialization_mode}")
        
        # Check dataset type
        valid_types = ['train', 'val', 'test']
        if self.dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, "
                           f"got {self.dataset_type}")
        
        # Validate dimensions
        if self.state_dim <= self.input_dim:
            raise ValueError(f"state_dim ({self.state_dim}) must be greater than "
                           f"input_dim ({self.input_dim}) since state includes "
                           f"both water levels and pipe flows")
        
        # Check window size consistency
        if self.n_windows_created > self.n_windows:
            raise ValueError(f"n_windows_created ({self.n_windows_created}) cannot "
                           f"exceed n_windows ({self.n_windows})")
        
        # Estimate required timesteps
        min_timesteps = self.window_size + self.n_windows_created - 1
        if self.n_timesteps_total < min_timesteps:
            warnings.warn(f"n_timesteps_total ({self.n_timesteps_total}) may be insufficient "
                         f"for {self.n_windows_created} windows of size {self.window_size}. "
                         f"Minimum required: {min_timesteps}")
        
        return True


class SWMMDataset(Dataset):
    """
    PyTorch Dataset for windowed SWMM hydraulic state sequences.
    
    This dataset provides access to windowed sequences of hydraulic states
    and inputs for autoregressive training of the gResNet surrogate model.
    Each window consists of:
    - Initial state x_t at window start
    - Input sequence R_t, R_{t+1}, ..., R_{t+W-1} for the window
    - Target sequence x_{t+1}, x_{t+2}, ..., x_{t+W} for autoregressive training
    
    The dataset implements the paper's windowed training methodology:
    - Window size: Configurable, with 60 minutes selected as optimal (Fig. 3)
    - Initialization: True HiFi states for training, empty system for validation
    - Overlap: Windows can overlap (stride=1) to maximize training data
    
    Attributes:
        states: Array of initial states for each window (n_windows, state_dim)
        inputs: Array of input sequences for each window (n_windows, window_size, input_dim)
        targets: Array of target sequences for each window (n_windows, window_size, state_dim)
        window_size: Window size in timesteps (W)
        metadata: DatasetMetadata object with dataset information
        device: PyTorch device for tensor placement
        _preloaded_tensors: Dictionary of pre-loaded tensors for faster access
        
    Methods:
        __len__: Return number of windows in dataset
        __getitem__: Get window by index (x_t, R_sequence, targets_sequence)
        get_window_info: Get information about specific window
        split: Split dataset into training and validation subsets
        save: Save dataset to disk for reproducibility
        load: Load dataset from disk (class method)
        
    Note:
        The dataset assumes data has already been scaled to [0,1] range by
        DataProcessor. All tensors are returned as torch.float32.
    """
    
    def __init__(self, states: np.ndarray, inputs: np.ndarray, 
                 targets: np.ndarray, window_size: int,
                 initialization_mode: str = "true_state",
                 dataset_type: str = "train",
                 config: Optional[Dict] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize SWMMDataset with windowed state sequences.
        
        Args:
            states: Initial states for each window, shape (n_windows, state_dim)
            inputs: Input sequences for each window, shape (n_windows, window_size, input_dim)
            targets: Target sequences for each window, shape (n_windows, window_size, state_dim)
            window_size: Window size in timesteps (W)
            initialization_mode: How windows are initialized ('true_state' or 'empty_system')
            dataset_type: Type of dataset ('train', 'val', 'test')
            config: Optional configuration dictionary
            metadata: Optional metadata dictionary for dataset
            
        Raises:
            ValueError: If arrays have incompatible shapes or dimensions
            RuntimeError: If window size doesn't match input/target sequences
            
        Note:
            The dataset follows paper methodology: training windows initialized
            from true HiFi states, validation windows from empty system.
        """
        # Store configuration
        if config is None:
            self.config = Config()
            self.data_config = self.config.data
            self.training_config = self.config.training
            self.system_config = self.config.system
        else:
            self.config = config
            self.data_config = DataConfig(**config.get('data', {}))
            self.training_config = TrainingConfig(**config.get('training', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set random seed for reproducibility
        set_seed(self.system_config.random_seed)
        
        # Validate input arrays
        self._validate_inputs(states, inputs, targets, window_size)
        
        # Store data as numpy arrays (convert to tensors in __getitem__ for memory efficiency)
        self.states = states.astype(np.float32)
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window_size = window_size
        
        # Determine device
        self.device = get_device(config)
        
        # Extract dimensions
        self.n_windows = states.shape[0]
        self.state_dim = states.shape[1]
        self.input_dim = inputs.shape[2]
        
        # Calculate total timesteps (approximate)
        self.n_timesteps_total = self.n_windows + window_size - 1
        
        # Create metadata
        self.metadata = self._create_metadata(
            initialization_mode, dataset_type, metadata
        )
        
        # Preload tensors to GPU if dataset is small and GPU is available
        self._preloaded_tensors = None
        if self._should_preload_tensors():
            self._preload_tensors()
        
        # Print initialization summary
        self._print_summary()
    
    def _validate_inputs(self, states: np.ndarray, inputs: np.ndarray,
                        targets: np.ndarray, window_size: int) -> None:
        """
        Validate input arrays for consistency and correctness.
        
        Args:
            states: Initial states array
            inputs: Input sequences array
            targets: Target sequences array
            window_size: Window size
            
        Raises:
            ValueError: If arrays have incompatible shapes or dimensions
        """
        # Check array dimensions
        if states.ndim != 2:
            raise ValueError(f"states must be 2D array, got shape {states.shape}")
        
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be 3D array, got shape {inputs.shape}")
        
        if targets.ndim != 3:
            raise ValueError(f"targets must be 3D array, got shape {targets.shape}")
        
        # Check number of windows consistency
        n_windows = states.shape[0]
        if inputs.shape[0] != n_windows:
            raise ValueError(f"Number of windows mismatch: states has {n_windows}, "
                           f"inputs has {inputs.shape[0]}")
        
        if targets.shape[0] != n_windows:
            raise ValueError(f"Number of windows mismatch: states has {n_windows}, "
                           f"targets has {targets.shape[0]}")
        
        # Check window size consistency
        if inputs.shape[1] != window_size:
            raise ValueError(f"Window size mismatch: inputs has {inputs.shape[1]} "
                           f"timesteps, window_size is {window_size}")
        
        if targets.shape[1] != window_size:
            raise ValueError(f"Window size mismatch: targets has {targets.shape[1]} "
                           f"timesteps, window_size is {window_size}")
        
        # Check state dimension consistency
        state_dim = states.shape[1]
        if targets.shape[2] != state_dim:
            raise ValueError(f"State dimension mismatch: states has {state_dim} "
                           f"features, targets has {targets.shape[2]}")
        
        # Check input dimension (should be less than state_dim)
        input_dim = inputs.shape[2]
        if input_dim > state_dim:
            warnings.warn(f"Input dimension ({input_dim}) > state dimension ({state_dim}). "
                         f"This may indicate incorrect data processing.")
        
        # Check for NaN values
        if np.any(np.isnan(states)):
            warnings.warn("NaN values found in states array")
        
        if np.any(np.isnan(inputs)):
            warnings.warn("NaN values found in inputs array")
        
        if np.any(np.isnan(targets)):
            warnings.warn("NaN values found in targets array")
        
        # Check data range (should be [0,1] if scaled)
        if np.min(states) < -0.1 or np.max(states) > 1.1:
            warnings.warn(f"states values outside expected [0,1] range: "
                         f"[{np.min(states):.3f}, {np.max(states):.3f}]")
        
        if np.min(inputs) < -0.1 or np.max(inputs) > 1.1:
            warnings.warn(f"inputs values outside expected [0,1] range: "
                         f"[{np.min(inputs):.3f}, {np.max(inputs):.3f}]")
        
        if np.min(targets) < -0.1 or np.max(targets) > 1.1:
            warnings.warn(f"targets values outside expected [0,1] range: "
                         f"[{np.min(targets):.3f}, {np.max(targets):.3f}]")
        
        print(f"Input validation passed: {n_windows} windows, "
              f"window_size={window_size}, state_dim={state_dim}, input_dim={input_dim}")
    
    def _create_metadata(self, initialization_mode: str, dataset_type: str,
                        extra_metadata: Optional[Dict[str, Any]]) -> DatasetMetadata:
        """
        Create metadata object for dataset.
        
        Args:
            initialization_mode: How windows are initialized
            dataset_type: Type of dataset
            extra_metadata: Additional metadata to include
            
        Returns:
            DatasetMetadata object
        """
        metadata = DatasetMetadata(
            source_file="windowed_data_from_processor",
            n_windows=self.n_windows,
            window_size=self.window_size,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            n_timesteps_total=self.n_timesteps_total,
            n_windows_created=self.n_windows,
            initialization_mode=initialization_mode,
            dataset_type=dataset_type,
            scaling_applied=True,  # Assumes DataProcessor has scaled data
            paper_reference="Section 2.3: Windowed training with autoregressive prediction"
        )
        
        # Add extra metadata if provided
        if extra_metadata:
            for key, value in extra_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
                else:
                    warnings.warn(f"Ignoring unknown metadata key: {key}")
        
        # Compute memory usage
        states_size = self.states.nbytes / (1024 * 1024)
        inputs_size = self.inputs.nbytes / (1024 * 1024)
        targets_size = self.targets.nbytes / (1024 * 1024)
        metadata.memory_size_mb = states_size + inputs_size + targets_size
        
        # Validate metadata
        try:
            metadata.validate()
        except ValueError as e:
            warnings.warn(f"Dataset metadata validation warning: {e}")
        
        return metadata
    
    def _should_preload_tensors(self) -> bool:
        """
        Determine whether to preload tensors to GPU.
        
        Returns:
            True if tensors should be preloaded, False otherwise
            
        Note:
            Preloading is beneficial for small datasets that fit in GPU memory
            but can cause memory issues for large datasets. Preloading is disabled
            when using multiple workers to avoid multiprocessing issues.
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False
        
        # Don't preload if using multiple workers (causes multiprocessing issues)
        if self.system_config.num_workers > 0:
            print(f"Dataset preloading disabled because num_workers > 0")
            return False
        
        # Check dataset size (don't preload if > 1GB)
        total_memory_mb = self.metadata.memory_size_mb
        if total_memory_mb > 1024:  # 1GB threshold
            print(f"Dataset size ({total_memory_mb:.1f} MB) too large for GPU preloading")
            return False
        
        # Check available GPU memory (leave 1GB for model and operations)
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_mb = free_memory / (1024 * 1024)
            
            if free_memory_mb > total_memory_mb + 1024:  # Need 1GB headroom
                return True
            else:
                print(f"Insufficient GPU memory for preloading: {free_memory_mb:.1f} MB free, "
                      f"dataset requires {total_memory_mb:.1f} MB")
                return False
        except Exception as e:
            warnings.warn(f"Failed to check GPU memory: {e}")
            return False
    
    def _preload_tensors(self) -> None:
        """
        Preload dataset tensors to GPU for faster access.
        
        Note:
            This can significantly speed up training for small datasets
            but uses more GPU memory.
        """
        print(f"Preloading dataset tensors to {self.device}...")
        
        with Timer("tensor_preloading") as timer:
            self._preloaded_tensors = {
                'states': torch.from_numpy(self.states).to(self.device),
                'inputs': torch.from_numpy(self.inputs).to(self.device),
                'targets': torch.from_numpy(self.targets).to(self.device)
            }
        
        print(f"  Preloaded {self.n_windows} windows to GPU "
              f"({self.metadata.memory_size_mb:.1f} MB) in {timer.elapsed:.2f}s")
    
    def _print_summary(self) -> None:
        """Print dataset summary information."""
        print("\n" + "="*80)
        print("SWMMDataset INITIALIZATION SUMMARY")
        print("="*80)
        print(f"Dataset type: {self.metadata.dataset_type}")
        print(f"Initialization mode: {self.metadata.initialization_mode}")
        print(f"Number of windows: {self.n_windows:,}")
        print(f"Window size: {self.window_size} timesteps ({self.window_size} minutes)")
        print(f"State dimension: {self.state_dim} (nodes: ~{self.input_dim}, links: ~{self.state_dim - self.input_dim})")
        print(f"Input dimension: {self.input_dim}")
        print(f"Total timesteps represented: ~{self.n_timesteps_total:,}")
        print(f"Memory usage: {self.metadata.memory_size_mb:.1f} MB")
        print(f"Device: {self.device}")
        print(f"Tensor preloading: {'Enabled' if self._preloaded_tensors else 'Disabled'}")
        print("\nPaper Methodology Alignment:")
        print(f"  ✓ Windowed training with size W={self.window_size} (selected from hyperparameter study)")
        print(f"  ✓ {self.metadata.initialization_mode} initialization")
        print(f"  ✓ Autoregressive target sequences for {self.window_size} steps")
        print(f"  ✓ Scaled data in [0,1] range (min-max per variable)")
        print("="*80 + "\n")
    
    def __len__(self) -> int:
        """
        Return number of windows in dataset.
        
        Returns:
            Number of windows (n_windows)
            
        Note:
            Each window represents a potential starting point for
            autoregressive training of the surrogate model.
        """
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get window by index.
        
        Args:
            idx: Window index (0 <= idx < n_windows)
            
        Returns:
            Tuple of (x_t, R_sequence, targets_sequence) where:
            - x_t: Initial state at window start, shape (state_dim,)
            - R_sequence: Input sequence for the window, shape (window_size, input_dim)
            - targets_sequence: Target states for the window, shape (window_size, state_dim)
            
        Raises:
            IndexError: If idx is out of bounds
            
        Note:
            All tensors are returned as torch.float32. If preloaded tensors
            exist, they are sliced directly on GPU. Otherwise, numpy arrays
            are converted to tensors and moved to the appropriate device.
        """
        # Validate index
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(f"Window index {idx} out of range [0, {self.n_windows-1}]")
        
        # Get data from preloaded tensors if available
        if self._preloaded_tensors is not None:
            x_t = self._preloaded_tensors['states'][idx]
            R_sequence = self._preloaded_tensors['inputs'][idx]
            targets_sequence = self._preloaded_tensors['targets'][idx]
            
            # Return slices (no copy needed)
            return x_t, R_sequence, targets_sequence
        
        # Otherwise, convert numpy arrays to tensors
        x_t = torch.from_numpy(self.states[idx]).to(torch.float32)
        R_sequence = torch.from_numpy(self.inputs[idx]).to(torch.float32)
        targets_sequence = torch.from_numpy(self.targets[idx]).to(torch.float32)
        
        # Move to device if not CPU
        if self.device.type != 'cpu':
            x_t = x_t.to(self.device)
            R_sequence = R_sequence.to(self.device)
            targets_sequence = targets_sequence.to(self.device)
        
        return x_t, R_sequence, targets_sequence
    
    def get_window_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific window.
        
        Args:
            idx: Window index
            
        Returns:
            Dictionary with window information:
            - 'index': Window index
            - 'initial_state_stats': Statistics of initial state x_t
            - 'input_sequence_stats': Statistics of input sequence
            - 'target_sequence_stats': Statistics of target sequence
            - 'has_nan': Whether window contains NaN values
            - 'data_range': Value ranges for validation
            
        Note:
            Useful for debugging and dataset inspection.
        """
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(f"Window index {idx} out of range [0, {self.n_windows-1}]")
        
        # Extract window data
        x_t = self.states[idx]
        R_sequence = self.inputs[idx]
        targets_sequence = self.targets[idx]
        
        # Compute statistics
        info = {
            'index': idx,
            'initial_state_stats': {
                'min': float(np.min(x_t)),
                'max': float(np.max(x_t)),
                'mean': float(np.mean(x_t)),
                'std': float(np.std(x_t))
            },
            'input_sequence_stats': {
                'min': float(np.min(R_sequence)),
                'max': float(np.max(R_sequence)),
                'mean': float(np.mean(R_sequence)),
                'std': float(np.std(R_sequence))
            },
            'target_sequence_stats': {
                'min': float(np.min(targets_sequence)),
                'max': float(np.max(targets_sequence)),
                'mean': float(np.mean(targets_sequence)),
                'std': float(np.std(targets_sequence))
            },
            'has_nan': bool(np.any(np.isnan(x_t)) or 
                           np.any(np.isnan(R_sequence)) or 
                           np.any(np.isnan(targets_sequence))),
            'data_range_valid': (
                np.all(x_t >= -0.1) and np.all(x_t <= 1.1) and
                np.all(R_sequence >= -0.1) and np.all(R_sequence <= 1.1) and
                np.all(targets_sequence >= -0.1) and np.all(targets_sequence <= 1.1)
            )
        }
        
        return info
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, random_seed: Optional[int] = None) -> Tuple['SWMMDataset', ...]:
        """
        Split dataset into training, validation, and test subsets.
        
        Args:
            train_ratio: Proportion of windows for training
            val_ratio: Proportion of windows for validation
            test_ratio: Proportion of windows for testing
            random_seed: Random seed for reproducible splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) where each is a
            Subset wrapper around this dataset with appropriate indices.
            
        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid
            
        Note:
            This creates Subset objects that reference the original dataset.
            The paper uses event-based splitting, not random window splitting,
            but this method provides flexibility for different strategies.
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}")
        
        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError(f"All ratios must be positive, got train={train_ratio}, "
                           f"val={val_ratio}, test={test_ratio}")
        
        # Generate indices
        n_windows = self.n_windows
        indices = np.arange(n_windows)
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Calculate split points
        n_train = int(n_windows * train_ratio)
        n_val = int(n_windows * val_ratio)
        n_test = n_windows - n_train - n_val
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        print(f"Dataset split:")
        print(f"  Training windows: {n_train} ({train_ratio:.1%})")
        print(f"  Validation windows: {n_val} ({val_ratio:.1%})")
        print(f"  Test windows: {n_test} ({test_ratio:.1%})")
        print(f"  Random seed: {random_seed if random_seed else 'None'}")
        
        # Create subsets
        train_dataset = Subset(self, train_indices.tolist())
        val_dataset = Subset(self, val_indices.tolist())
        test_dataset = Subset(self, test_indices.tolist())
        
        # Update metadata for subsets
        self._update_subset_metadata(train_dataset, 'train')
        self._update_subset_metadata(val_dataset, 'val')
        self._update_subset_metadata(test_dataset, 'test')
        
        return train_dataset, val_dataset, test_dataset
    
    def _update_subset_metadata(self, subset: Subset, dataset_type: str) -> None:
        """
        Update metadata for a subset.
        
        Args:
            subset: Subset object
            dataset_type: Type of dataset ('train', 'val', 'test')
            
        Note:
            Adds metadata as an attribute to the subset for reference.
        """
        # Create copy of metadata with updated type and window count
        subset_metadata = DatasetMetadata(
            source_file=self.metadata.source_file,
            n_windows=len(subset),
            window_size=self.window_size,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            n_timesteps_total=self.n_timesteps_total,
            n_windows_created=len(subset),
            initialization_mode=self.metadata.initialization_mode,
            dataset_type=dataset_type,
            scaling_applied=self.metadata.scaling_applied,
            node_ids=self.metadata.node_ids.copy() if self.metadata.node_ids else [],
            link_ids=self.metadata.link_ids.copy() if self.metadata.link_ids else []
        )
        
        # Attach to subset
        subset.metadata = subset_metadata
    
    def save(self, filepath: Optional[str] = None, 
             include_metadata: bool = True) -> str:
        """
        Save dataset to disk for reproducibility.
        
        Args:
            filepath: Path to save dataset. If None, generates timestamped name.
            include_metadata: Whether to include metadata in saved file
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file cannot be written
            
        Note:
            Saves dataset as a pickle file. Large datasets may be better
            saved in a more efficient format (e.g., HDF5).
        """
        if filepath is None:
            # Generate timestamped filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_type = self.metadata.dataset_type
            filename = f"swmm_dataset_{dataset_type}_{timestamp}.pkl"
            
            # Create directory if it doesn't exist
            save_dir = Path("S5/results_P2C/datasets")
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(save_dir / filename)
        else:
            # Ensure directory exists
            save_dir = Path(filepath).parent
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'states': self.states,
            'inputs': self.inputs,
            'targets': self.targets,
            'window_size': self.window_size,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }
        
        if include_metadata:
            save_data['metadata'] = self.metadata.to_dict()
        
        # Save to disk
        try:
            save_pickle(save_data, filepath)
            print(f"Dataset saved to: {filepath}")
            print(f"  Size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
            print(f"  Windows: {self.n_windows}")
            print(f"  Window size: {self.window_size}")
        except Exception as e:
            raise IOError(f"Failed to save dataset: {str(e)}") from e
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None,
             config: Optional[Dict] = None) -> 'SWMMDataset':
        """
        Load dataset from disk.
        
        Args:
            filepath: Path to dataset file
            device: PyTorch device to use (if None, uses config or default)
            config: Configuration dictionary (overrides saved config)
            
        Returns:
            Loaded SWMMDataset instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read or parsed
            
        Note:
            The loaded dataset uses the same windowing strategy and
            initialization mode as when it was saved.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        print(f"Loading dataset from: {filepath}")
        
        try:
            # Load data
            loaded_data = load_pickle(filepath)
            
            # Extract data
            states = loaded_data['states']
            inputs = loaded_data['inputs']
            targets = loaded_data['targets']
            window_size = loaded_data['window_size']
            
            # Get config (use provided config if available, otherwise saved config)
            saved_config = loaded_data.get('config', {})
            if config is not None:
                # Merge saved config with provided config (provided takes precedence)
                merged_config = {**saved_config, **config}
            else:
                merged_config = saved_config
            
            # Extract metadata if available
            metadata = loaded_data.get('metadata', {})
            
            # Set device in config if provided
            if device is not None:
                merged_config.setdefault('system', {})
                merged_config['system']['device'] = device
            
            # Create dataset
            dataset = cls(
                states=states,
                inputs=inputs,
                targets=targets,
                window_size=window_size,
                config=merged_config,
                metadata=metadata
            )
            
            print(f"  Loaded {dataset.n_windows} windows, window_size={window_size}")
            print(f"  Dataset type: {dataset.metadata.dataset_type}")
            print(f"  Memory: {dataset.metadata.memory_size_mb:.1f} MB")
            
            return dataset
            
        except Exception as e:
            raise IOError(f"Failed to load dataset: {str(e)}") from e
    
    def get_dataloader(self, batch_size: Optional[int] = None,
                      shuffle: Optional[bool] = None,
                      num_workers: Optional[int] = None,
                      pin_memory: bool = True) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size. If None, uses config value or 1.
            shuffle: Whether to shuffle data. If None, shuffles for training only.
            num_workers: Number of worker processes. If None, uses config value.
            pin_memory: Whether to pin memory for faster GPU transfer.
            
        Returns:
            PyTorch DataLoader configured for this dataset.
            
        Note:
            Follows paper's training approach: batch_size=1 (window as batch),
            shuffle=True for training, shuffle=False for validation/testing.
        """
        # Use defaults from config if not specified
        if batch_size is None:
            batch_size = self.training_config.batch_size
        
        if shuffle is None:
            # Shuffle training data, not validation/test data
            shuffle = (self.metadata.dataset_type == 'train')
        
        if num_workers is None:
            num_workers = self.system_config.num_workers
        
        # Create collate function that handles our tuple structure
        def collate_fn(batch):
            """
            Collate function for SWMMDataset batches.
            
            Args:
                batch: List of (x_t, R_sequence, targets_sequence) tuples
                
            Returns:
                Tuple of batched tensors:
                - x_t_batch: (batch_size, state_dim)
                - R_sequence_batch: (batch_size, window_size, input_dim)
                - targets_sequence_batch: (batch_size, window_size, state_dim)
            """
            # Unzip the batch
            x_t_list, R_sequence_list, targets_sequence_list = zip(*batch)
            
            # Stack tensors
            x_t_batch = torch.stack(x_t_list, dim=0)
            R_sequence_batch = torch.stack(R_sequence_list, dim=0)
            targets_sequence_batch = torch.stack(targets_sequence_list, dim=0)
            
            return x_t_batch, R_sequence_batch, targets_sequence_batch
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # tensors are already on self.device in __getitem__
            collate_fn=collate_fn,
            drop_last=False  # Keep all windows even if batch doesn't divide evenly
        )
        
        print(f"Created DataLoader for {self.metadata.dataset_type} dataset:")
        print(f"  Batch size: {batch_size}")
        print(f"  Shuffle: {shuffle}")
        print(f"  Number of workers: {num_workers}")
        print(f"  Pin memory: False (tensors pre-loaded to {self.device})")
        print(f"  Total batches: {len(dataloader)}")
        
        return dataloader
    
    def iterate_windows(self, batch_size: int = 1, shuffle: bool = False) -> Iterator:
        """
        Iterate through windows with custom batching.
        
        Args:
            batch_size: Number of windows per batch
            shuffle: Whether to shuffle windows before iterating
            
        Yields:
            Batches of (x_t_batch, R_sequence_batch, targets_sequence_batch)
            
        Note:
            Alternative to DataLoader for more control over iteration.
        """
        # Create indices
        indices = np.arange(self.n_windows)
        if shuffle:
            np.random.shuffle(indices)
        
        # Iterate in batches
        for start_idx in range(0, self.n_windows, batch_size):
            end_idx = min(start_idx + batch_size, self.n_windows)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            x_t_batch = []
            R_sequence_batch = []
            targets_sequence_batch = []
            
            for idx in batch_indices:
                x_t, R_seq, targets_seq = self[idx]
                x_t_batch.append(x_t)
                R_sequence_batch.append(R_seq)
                targets_sequence_batch.append(targets_seq)
            
            # Stack into batch tensors
            x_t_batch = torch.stack(x_t_batch, dim=0)
            R_sequence_batch = torch.stack(R_sequence_batch, dim=0)
            targets_sequence_batch = torch.stack(targets_sequence_batch, dim=0)
            
            yield x_t_batch, R_sequence_batch, targets_sequence_batch
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze dataset statistics and characteristics.
        
        Returns:
            Dictionary with dataset analysis:
            - 'basic_stats': Basic statistics about the dataset
            - 'window_stats': Statistics about window contents
            - 'data_quality': Data quality metrics
            - 'paper_alignment': How well dataset aligns with paper methodology
            
        Note:
            Useful for understanding dataset characteristics and
            verifying alignment with paper methodology.
        """
        print(f"Analyzing {self.metadata.dataset_type} dataset...")
        
        with Timer("dataset_analysis") as timer:
            # Basic statistics
            basic_stats = {
                'n_windows': self.n_windows,
                'window_size': self.window_size,
                'state_dim': self.state_dim,
                'input_dim': self.input_dim,
                'total_timesteps': self.n_timesteps_total,
                'memory_mb': self.metadata.memory_size_mb,
                'initialization_mode': self.metadata.initialization_mode
            }
            
            # Window statistics (sample a subset for efficiency)
            sample_size = min(100, self.n_windows)
            sample_indices = np.random.choice(self.n_windows, sample_size, replace=False)
            
            window_stats = {
                'initial_state_mean': np.mean(self.states[sample_indices]),
                'initial_state_std': np.std(self.states[sample_indices]),
                'input_sequence_mean': np.mean(self.inputs[sample_indices]),
                'input_sequence_std': np.std(self.inputs[sample_indices]),
                'target_sequence_mean': np.mean(self.targets[sample_indices]),
                'target_sequence_std': np.std(self.targets[sample_indices]),
                'sample_windows_analyzed': sample_size
            }
            
            # Data quality metrics
            has_nan = (
                np.any(np.isnan(self.states)) or
                np.any(np.isnan(self.inputs)) or
                np.any(np.isnan(self.targets))
            )
            
            data_range_valid = (
                np.all(self.states >= -0.1) and np.all(self.states <= 1.1) and
                np.all(self.inputs >= -0.1) and np.all(self.inputs <= 1.1) and
                np.all(self.targets >= -0.1) and np.all(self.targets <= 1.1)
            )
            
            data_quality = {
                'has_nan': bool(has_nan),
                'data_range_valid': bool(data_range_valid),
                'states_min': float(np.min(self.states)),
                'states_max': float(np.max(self.states)),
                'inputs_min': float(np.min(self.inputs)),
                'inputs_max': float(np.max(self.inputs)),
                'targets_min': float(np.min(self.targets)),
                'targets_max': float(np.max(self.targets))
            }
            
            # Paper alignment metrics
            paper_alignment = {
                'window_size_matches_paper': (self.window_size == 60),
                'initialization_correct': (
                    (self.metadata.dataset_type == 'train' and 
                     self.metadata.initialization_mode == 'true_state') or
                    (self.metadata.dataset_type in ['val', 'test'] and 
                     self.metadata.initialization_mode == 'empty_system')
                ),
                'data_scaled': self.metadata.scaling_applied,
                'autoregressive_targets': True,  # By design
                'window_overlap': True  # Windows created with stride=1
            }
            
            analysis = {
                'basic_stats': basic_stats,
                'window_stats': window_stats,
                'data_quality': data_quality,
                'paper_alignment': paper_alignment,
                'analysis_time_seconds': timer.elapsed
            }
            
            # Print summary
            print(f"\nDataset Analysis Summary:")
            print(f"  Windows: {basic_stats['n_windows']:,}")
            print(f"  Window size: {basic_stats['window_size']} (paper: 60)")
            print(f"  State dimension: {basic_stats['state_dim']}")
            print(f"  Data quality: {'GOOD' if not data_quality['has_nan'] and data_quality['data_range_valid'] else 'ISSUES'}")
            print(f"  Paper alignment: {sum(paper_alignment.values())}/{len(paper_alignment)} criteria met")
            
            return analysis


def create_data_loaders(train_dataset: SWMMDataset, val_dataset: SWMMDataset,
                       test_dataset: Optional[SWMMDataset] = None,
                       config: Optional[Dict] = None) -> Dict[str, DataLoader]:
    """
    Factory function to create DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        config: Configuration dictionary
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
        
    Note:
        Configures DataLoaders according to paper methodology:
        - Training: Shuffled, with batch_size from config
        - Validation/Test: Not shuffled, for sequential evaluation
    """
    if config is None:
        config_obj = Config()
        training_config = config_obj.training
        system_config = config_obj.system
    else:
        training_config = TrainingConfig(**config.get('training', {}))
        system_config = SystemConfig(**config.get('system', {}))
    
    # Create DataLoaders
    dataloaders = {}
    
    # Training DataLoader
    train_loader = train_dataset.get_dataloader(
        batch_size=training_config.batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=system_config.num_workers,
        pin_memory=True
    )
    dataloaders['train'] = train_loader
    
    # Validation DataLoader
    val_loader = val_dataset.get_dataloader(
        batch_size=training_config.batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=system_config.num_workers,
        pin_memory=True
    )
    dataloaders['val'] = val_loader
    
    # Test DataLoader (if provided)
    if test_dataset is not None:
        test_loader = test_dataset.get_dataloader(
            batch_size=training_config.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=system_config.num_workers,
            pin_memory=True
        )
        dataloaders['test'] = test_loader
    
    print(f"\nCreated {len(dataloaders)} DataLoaders:")
    for name, loader in dataloaders.items():
        print(f"  {name}: {len(loader)} batches, "
              f"batch_size={loader.batch_size}, "
              f"shuffle={loader.shuffle}")
    
    return dataloaders


def split_dataset_by_indices(dataset: SWMMDataset, train_indices: List[int],
                            val_indices: List[int], 
                            test_indices: Optional[List[int]] = None) -> Tuple:
    """
    Split dataset by explicit indices.
    
    Args:
        dataset: Dataset to split
        train_indices: Indices for training subset
        val_indices: Indices for validation subset
        test_indices: Optional indices for test subset
        
    Returns:
        Tuple of Subset objects (train, val[, test])
        
    Note:
        Useful for custom splitting strategies or when using pre-defined
        splits from event-based partitioning.
    """
    # Validate indices
    all_indices = set(train_indices) | set(val_indices)
    if test_indices is not None:
        all_indices |= set(test_indices)
    
    if len(all_indices) > len(dataset):
        raise ValueError(f"Total indices ({len(all_indices)}) exceed dataset size ({len(dataset)})")
    
    if len(set(train_indices)) != len(train_indices):
        warnings.warn("Duplicate indices found in train_indices")
    
    if len(set(val_indices)) != len(val_indices):
        warnings.warn("Duplicate indices found in val_indices")
    
    if test_indices is not None and len(set(test_indices)) != len(test_indices):
        warnings.warn("Duplicate indices found in test_indices")
    
    # Check for overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    
    overlap = train_set & val_set
    if overlap:
        warnings.warn(f"Overlap between train and validation indices: {len(overlap)} indices")
    
    if test_indices is not None:
        test_set = set(test_indices)
        overlap = (train_set | val_set) & test_set
        if overlap:
            warnings.warn(f"Overlap with test indices: {len(overlap)} indices")
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Update metadata
    dataset._update_subset_metadata(train_subset, 'train')
    dataset._update_subset_metadata(val_subset, 'val')
    
    print(f"Dataset split by indices:")
    print(f"  Training windows: {len(train_subset)}")
    print(f"  Validation windows: {len(val_subset)}")
    
    if test_indices is not None:
        test_subset = Subset(dataset, test_indices)
        dataset._update_subset_metadata(test_subset, 'test')
        print(f"  Test windows: {len(test_subset)}")
        return train_subset, val_subset, test_subset
    
    return train_subset, val_subset


def test_swmm_dataset():
    """
    Test function for SWMMDataset.
    
    Creates a simple test to verify the dataset works correctly.
    """
    print("Testing SWMMDataset...")
    
    # Create dummy data for testing
    n_windows = 100
    window_size = 10
    state_dim = 20
    input_dim = 6
    
    # Generate random data (already scaled to [0,1])
    np.random.seed(42)
    states = np.random.rand(n_windows, state_dim).astype(np.float32)
    inputs = np.random.rand(n_windows, window_size, input_dim).astype(np.float32)
    targets = np.random.rand(n_windows, window_size, state_dim).astype(np.float32)
    
    try:
        # Create dataset
        dataset = SWMMDataset(
            states=states,
            inputs=inputs,
            targets=targets,
            window_size=window_size,
            initialization_mode="true_state",
            dataset_type="train"
        )
        
        # Test __len__
        print(f"Test __len__: {len(dataset)} windows, expected {n_windows}")
        assert len(dataset) == n_windows
        
        # Test __getitem__
        sample_idx = 5
        x_t, R_seq, targets_seq = dataset[sample_idx]
        print(f"Test __getitem__[{sample_idx}]:")
        print(f"  x_t shape: {x_t.shape}, expected ({state_dim},)")
        print(f"  R_seq shape: {R_seq.shape}, expected ({window_size}, {input_dim})")
        print(f"  targets_seq shape: {targets_seq.shape}, expected ({window_size}, {state_dim})")
        
        assert x_t.shape == (state_dim,)
        assert R_seq.shape == (window_size, input_dim)
        assert targets_seq.shape == (window_size, state_dim)
        
        # Test tensor types
        assert isinstance(x_t, torch.Tensor)
        assert isinstance(R_seq, torch.Tensor)
        assert isinstance(targets_seq, torch.Tensor)
        assert x_t.dtype == torch.float32
        
        # Test get_window_info
        window_info = dataset.get_window_info(sample_idx)
        print(f"Test get_window_info: keys={list(window_info.keys())}")
        assert 'initial_state_stats' in window_info
        
        # Test split
        train_ds, val_ds, test_ds = dataset.split(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        print(f"Test split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_windows
        
        # Test DataLoader creation
        dataloader = dataset.get_dataloader(batch_size=4, shuffle=True)
        print(f"Test DataLoader: {len(dataloader)} batches")
        
        # Test iteration
        for batch_idx, (x_batch, R_batch, targets_batch) in enumerate(dataloader):
            print(f"  Batch {batch_idx}: x_batch shape={x_batch.shape}, "
                  f"R_batch shape={R_batch.shape}, targets_batch shape={targets_batch.shape}")
            assert x_batch.shape == (4, state_dim)
            assert R_batch.shape == (4, window_size, input_dim)
            assert targets_batch.shape == (4, window_size, state_dim)
            break  # Just test first batch
        
        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_dataset.pkl")
            dataset.save(save_path)
            
            loaded_dataset = SWMMDataset.load(save_path)
            print(f"Test save/load: loaded {len(loaded_dataset)} windows")
            assert len(loaded_dataset) == len(dataset)
            assert loaded_dataset.window_size == dataset.window_size
            
            # Compare a sample
            x_t_orig, R_seq_orig, targets_seq_orig = dataset[0]
            x_t_loaded, R_seq_loaded, targets_seq_loaded = loaded_dataset[0]
            
            assert torch.allclose(x_t_orig, x_t_loaded, rtol=1e-5)
            assert torch.allclose(R_seq_orig, R_seq_loaded, rtol=1e-5)
            assert torch.allclose(targets_seq_orig, targets_seq_loaded, rtol=1e-5)
        
        # Test analysis
        analysis = dataset.analyze_dataset()
        print(f"Test analyze_dataset: {len(analysis)} analysis categories")
        assert 'basic_stats' in analysis
        
        print("\nAll SWMMDataset tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_swmm_dataset()
    if success:
        print("\nSWMMDataset test completed successfully!")
    else:
        print("\nSWMMDataset test failed!")
