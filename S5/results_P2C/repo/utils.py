"""
utils.py

Utility functions for reproducibility, configuration management, performance
measurement, and data serialization.

This module provides foundational utilities that ensure reproducibility,
configuration management, and data persistence. It has no dependencies on
other project files.

Functions:
    set_seed(seed): Set random seeds for reproducibility
    load_config(config_path): Load YAML configuration file
    Timer: Context manager for performance measurement
    save_pickle(obj, path): Serialize object to pickle file
    load_pickle(path): Deserialize object from pickle file

Classes:
    Timer: Context manager for timing code execution
"""

import os
import random
import pickle
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager
from pathlib import Path

import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This function sets seeds for Python's random module, NumPy, and PyTorch
    (both CPU and GPU). It also configures PyTorch for deterministic behavior
    when using CUDA.
    
    Args:
        seed: Random seed value. Defaults to 42 if not specified.
    
    Note:
        Must be called at the beginning of the main pipeline and before any
        random operations. Aligns with paper's requirement for reproducibility
        (Section 2.5.3 mentions "different (random) initializations").
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA reproducibility settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.enabled = True  # Enable CuDNN
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic behavior on CUDA >= 10.2
    
    # Print confirmation
    print(f"Random seed set to: {seed}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("Deterministic CuDNN settings enabled for reproducibility")
    
    # Note: PyTorch 2.0+ has additional reproducibility controls
    # We enable them if available
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions don't have this function


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    This function loads the configuration file specified in config_path,
    parses it as YAML, and returns a nested dictionary. It performs basic
    validation to ensure required sections are present.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Dictionary containing the parsed configuration.
    
    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    
    Note:
        The configuration must contain all sections specified in the paper's
        methodology. Critical parameters are validated against paper values.
    """
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    # Validate required sections are present
    required_sections = ['data', 'model', 'training', 'evaluation', 'performance', 'system']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Paper-specific validation
    # Window size must be 60 minutes (selected from hyperparameter study)
    if config['data'].get('training_window_size') != 60:
        print(f"WARNING: Window size is {config['data'].get('training_window_size')}, "
              f"but paper selected 60 minutes as optimal")
    
    # Residue network architecture must be S4 (6 layers × 100 neurons)
    residue_config = config['model'].get('residue_network', {})
    if (residue_config.get('architecture') != 'S4' or 
        residue_config.get('hidden_layers') != 6 or 
        residue_config.get('hidden_units_per_layer') != 100):
        print("WARNING: Model architecture does not match paper's S4 configuration")
    
    # Learning rate schedule (1e-3 → 1e-4 exponential decay)
    lr_config = config['training'].get('learning_rate', {})
    if (lr_config.get('initial') != 0.001 or 
        lr_config.get('final') != 0.0001 or 
        lr_config.get('decay') != 'exponential'):
        print("WARNING: Learning rate schedule does not match paper (1e-3 → 1e-4 exponential decay)")
    
    # Early stopping patience (500 epochs)
    early_stopping_config = config['training'].get('early_stopping', {})
    if early_stopping_config.get('patience') != 500:
        print(f"WARNING: Early stopping patience is {early_stopping_config.get('patience')}, "
              f"but paper used 500 epochs")
    
    # Create directories specified in system configuration
    system_config = config.get('system', {})
    checkpoint_dir = system_config.get('checkpoint_dir', 'checkpoints')
    results_dir = system_config.get('results_dir', 'results')
    
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set default values for unspecified parameters
    config.setdefault('system', {})
    config['system'].setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['system'].setdefault('random_seed', 42)
    config['system'].setdefault('num_workers', 0)
    
    # Prior network hidden units (paper doesn't specify)
    model_config = config.get('model', {})
    prior_config = model_config.get('prior_network', {})
    if prior_config.get('hidden_units') is None:
        # Default to state dimension (paper mentions "single hidden layer" without specifying size)
        config['model']['prior_network']['hidden_units'] = None  # Will be set dynamically
    
    print(f"Configuration loaded from: {config_path}")
    return config


class Timer:
    """
    Context manager for measuring execution time.
    
    This class provides precise timing for performance benchmarking as required
    by the paper's computational performance analysis (Section 3.5, Table 2).
    
    Attributes:
        start_time: Time when the timer started (seconds since epoch)
        end_time: Time when the timer ended (seconds since epoch)
        elapsed: Total elapsed time in seconds
    
    Example:
        with Timer() as timer:
            # Code to time
            time.sleep(1)
        print(f"Elapsed: {timer.elapsed:.2f} seconds")
    
    Note:
        The paper requires timing measurements for:
        - Training time (0.5-2 hours for System 1)
        - Simulation time (~100s for 8,200 rain events)
        - Speed-up calculations (one to two orders of magnitude)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize timer with optional name for identification.
        
        Args:
            name: Optional identifier for the timer (e.g., "training", "inference")
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        """Start timing when entering context."""
        # Use time.perf_counter() for highest resolution timing
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        # Auto-print if name provided
        if self.name:
            self._print_elapsed()
    
    def _print_elapsed(self) -> None:
        """Print elapsed time in appropriate units."""
        if self.elapsed < 1.0:
            time_str = f"{self.elapsed * 1000:.2f} ms"
        elif self.elapsed < 60.0:
            time_str = f"{self.elapsed:.2f} seconds"
        elif self.elapsed < 3600.0:
            minutes = self.elapsed / 60.0
            time_str = f"{minutes:.2f} minutes"
        else:
            hours = self.elapsed / 3600.0
            time_str = f"{hours:.2f} hours"
        
        name_str = f" ({self.name})" if self.name else ""
        print(f"Elapsed time{name_str}: {time_str}")
    
    def reset(self) -> None:
        """Reset the timer to zero."""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def get_elapsed(self, unit: str = 'seconds') -> float:
        """
        Get elapsed time in specified unit.
        
        Args:
            unit: Time unit ('ms', 'seconds', 'minutes', 'hours')
        
        Returns:
            Elapsed time in requested unit
        
        Raises:
            ValueError: If invalid unit specified
        """
        if unit == 'ms':
            return self.elapsed * 1000.0
        elif unit == 'seconds':
            return self.elapsed
        elif unit == 'minutes':
            return self.elapsed / 60.0
        elif unit == 'hours':
            return self.elapsed / 3600.0
        else:
            raise ValueError(f"Invalid time unit: {unit}. Must be 'ms', 'seconds', 'minutes', or 'hours'.")


def save_pickle(obj: Any, path: str, create_dir: bool = True) -> None:
    """
    Serialize Python object to pickle file.
    
    This function saves Python objects to disk using pickle serialization.
    It's used for storing intermediate results during the multi-stage training
    process (hyperparameter selection, multiple random initializations).
    
    Args:
        obj: Python object to serialize
        path: File path where object will be saved
        create_dir: If True, create parent directories if they don't exist
    
    Raises:
        IOError: If file cannot be written
        pickle.PickleError: If object cannot be serialized
    
    Note:
        Critical data that must be serialized for reproducibility:
        - Model weights and architecture
        - Data scaling parameters (min/max per variable)
        - Network adjacency matrices
        - Training history (loss curves, validation metrics)
    """
    if create_dir:
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to write pickle file {path}: {e}")
    except (pickle.PickleError, pickle.PicklingError) as e:
        raise pickle.PickleError(f"Failed to serialize object to {path}: {e}")
    
    file_size = os.path.getsize(path)
    size_mb = file_size / (1024 * 1024)
    print(f"Object saved to {path} ({size_mb:.2f} MB)")


def load_pickle(path: str) -> Any:
    """
    Deserialize Python object from pickle file.
    
    This function loads Python objects from disk that were previously saved
    with save_pickle. It handles common error cases and provides informative
    error messages.
    
    Args:
        path: File path to load object from
    
    Returns:
        Deserialized Python object
    
    Raises:
        FileNotFoundError: If pickle file does not exist
        IOError: If file cannot be read
        pickle.PickleError: If object cannot be deserialized
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to read pickle file {path}: {e}")
    except (pickle.PickleError, pickle.UnpicklingError) as e:
        raise pickle.PickleError(f"Failed to deserialize object from {path}: {e}")
    
    file_size = os.path.getsize(path)
    size_mb = file_size / (1024 * 1024)
    print(f"Object loaded from {path} ({size_mb:.2f} MB)")
    return obj


@contextmanager
def timed_section(name: str):
    """
    Context manager for timing code sections with automatic reporting.
    
    This is a convenience wrapper around Timer for common use cases.
    
    Args:
        name: Name of the code section being timed
    
    Example:
        with timed_section("data_preprocessing"):
            # Code to time
            process_data()
    """
    timer = Timer(name)
    with timer:
        yield


def get_device(config: Optional[Dict] = None) -> torch.device:
    """
    Get PyTorch device based on configuration or availability.
    
    This function determines the appropriate PyTorch device (CPU or GPU)
    based on system configuration and CUDA availability.
    
    Args:
        config: Optional configuration dictionary. If provided, uses
                config['system']['device'] if specified.
    
    Returns:
        torch.device object for CPU or CUDA
    
    Note:
        The paper mentions using single CPU (no GPU benefit mentioned),
        but we follow modern best practice of using GPU if available.
    """
    if config and 'system' in config and 'device' in config['system']:
        device_str = config['system']['device']
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        # Default behavior: use CUDA if available
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "2h 30m 15s", "45.2s", "125ms")
    """
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60.0:
        return f"{seconds:.1f}s"
    elif seconds < 3600.0:
        minutes = int(seconds // 60)
        seconds_rem = seconds % 60
        return f"{minutes}m {seconds_rem:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_rem = seconds % 60
        return f"{hours}h {minutes}m {seconds_rem:.1f}s"


if __name__ == "__main__":
    # Test the utility functions
    print("Testing utils.py functions...")
    
    # Test set_seed
    set_seed(123)
    
    # Test Timer
    with Timer("test_timer") as t:
        time.sleep(0.1)
    print(f"Timer elapsed: {t.elapsed:.4f} seconds")
    
    # Test timed_section
    with timed_section("test_section"):
        time.sleep(0.05)
    
    # Test format_time
    print(f"0.0005 seconds = {format_time(0.0005)}")
    print(f"0.5 seconds = {format_time(0.5)}")
    print(f"65 seconds = {format_time(65)}")
    print(f"3665 seconds = {format_time(3665)}")
    
    # Test get_device
    device = get_device()
    print(f"Using device: {device}")
    
    print("All tests completed!")
