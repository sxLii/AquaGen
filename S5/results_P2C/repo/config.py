"""
config.py

Configuration management for reproducing the paper "Accelerating hydrodynamic 
simulations of urban drainage systems with physics-guided machine learning".

This module defines structured dataclasses for all experimental parameters,
hyperparameters, and system settings based on the paper's methodology and
the provided config.yaml file. It ensures type safety and validation of
all configuration parameters.

Classes:
    DataConfig: Data loading, preprocessing, and windowing configuration
    ModelConfig: Neural network architecture and physical constraints
    TrainingConfig: Optimization, loss, and training procedure
    EvaluationConfig: Metrics and analysis methods
    SystemConfig: Computational environment and reproducibility
    Config: Main configuration container integrating all sub-configurations

Functions:
    load_config_from_yaml: Load configuration from YAML file
    validate_config: Validate configuration against paper requirements

Note:
    All default values are extracted from the paper's methodology or
    the provided config.yaml. Deviations are documented with rationale.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union, Any
import torch


@dataclass
class DataConfig:
    """
    Configuration for data processing and management.
    
    Defines parameters for SWMM network parsing, simulation settings,
    data preprocessing, windowing strategy, and dataset splits.
    
    Attributes:
        inp_file: Path to SWMM .inp network file
        time_step: Surrogate model time step Δt in seconds (1 minute = 60s)
        simulation_time_step: HiFi SWMM simulation time step in seconds (5s)
        training_window_size: Optimal window size for training in minutes (60 min)
        window_sizes_tested: Window sizes tested in hyperparameter study [1, 10, 60, 120, 360]
        scaling_method: Data normalization method ("minmax" to [0,1])
        state_vector_format: Order of state vector elements ["h_nodes", "Q_links"]
        validation_split: Fraction of data for validation (approx. 0.2)
        test_split: Fraction of data for testing (approx. 0.2)
        training_series_points: Target training data points (73,990 from paper)
        validation_series_points: Target validation data points (19,095 from paper)
        min_training_events: Minimum number of distinct inflow events for training
        data_dir: Directory for storing processed datasets
        cache_dir: Directory for caching simulation results
        
    Note:
        Time step alignment: simulation_time_step (5s) must divide time_step (60s).
        Window size is in minutes, but internally converted to timesteps.
    """
    
    # File paths
    inp_file: str = "Ji.inp"
    data_dir: str = "data"
    cache_dir: str = "cache"
    
    # Time settings
    time_step: int = 60  # Δt = 1 minute in seconds (paper: 1 minute)
    simulation_time_step: int = 5  # HiFi time step in seconds (paper: 5s fixed step)
    
    # Data partitioning
    validation_split: float = 0.2  # Approximate (paper uses event-based split)
    test_split: float = 0.2  # Approximate (paper uses independent series)
    
    # Target data volumes (from paper's System 1, Table 1)
    training_series_points: int = 73990  # Series A: 73,990 1-minute points
    validation_series_points: int = 19095  # Series A: 19,095 1-minute points
    min_training_events: int = 10  # Minimum distinct inflow events for diversity
    
    # Windowing and preprocessing
    training_window_size: int = 60  # 60 minutes (selected from hyperparameter study, Fig. 3)
    window_sizes_tested: List[int] = field(default_factory=lambda: [1, 10, 60, 120, 360])
    scaling_method: str = "minmax"  # Min-max scaling to [0,1] per variable (paper Section 2.3)
    state_vector_format: List[str] = field(default_factory=lambda: ["h_nodes", "Q_links"])
    
    # Derived properties (computed on initialization)
    _window_timesteps: Optional[int] = None
    _steps_per_window: Optional[int] = None
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        # Validate time step alignment
        if self.time_step % self.simulation_time_step != 0:
            raise ValueError(
                f"Time step {self.time_step}s must be multiple of simulation time step "
                f"{self.simulation_time_step}s for consistent resampling"
            )
        
        # Compute window size in timesteps
        self._window_timesteps = self.training_window_size  # 1 timestep = 1 minute
        self._steps_per_window = int(self.time_step / self.simulation_time_step)
        
        # Validate splits
        if not (0 < self.validation_split < 1):
            raise ValueError(f"validation_split must be between 0 and 1, got {self.validation_split}")
        if not (0 < self.test_split < 1):
            raise ValueError(f"test_split must be between 0 and 1, got {self.test_split}")
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be less than 1.0")
        
        # Validate window sizes
        if self.training_window_size not in self.window_sizes_tested:
            raise ValueError(f"training_window_size {self.training_window_size} not in window_sizes_tested")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @property
    def window_timesteps(self) -> int:
        """Get window size in timesteps."""
        if self._window_timesteps is None:
            raise ValueError("Window timesteps not initialized. Call __post_init__ first.")
        return self._window_timesteps
    
    @property
    def steps_per_window(self) -> int:
        """Get number of HiFi steps per surrogate timestep."""
        if self._steps_per_window is None:
            raise ValueError("Steps per window not initialized. Call __post_init__ first.")
        return self._steps_per_window


@dataclass
class ModelConfig:
    """
    Configuration for neural network architecture and physical constraints.
    
    Defines the gResNet architecture including prior network (L),
    residue network (N), physical constraint layer, and alternative
    architectures for hyperparameter studies.
    
    Attributes:
        state_vector_format: Order of state variables ["h_nodes", "Q_links"]
        prior_network: Configuration for prior network L (single hidden layer)
        residue_network: Configuration for residue network N (S4: 6×100)
        physical_constraints_enabled: Whether to apply mass balance constraints
        spilling_configuration: Whether water doesn't reenter system (True)
        activation: Activation function ("relu")
        architecture_options: Alternative architectures S1-S4 for testing
        
    Note:
        Prior network hidden units default to state dimension (paper doesn't specify).
        Physical constraints implement Eq. 8 from the paper.
    """
    
    # State vector configuration
    state_vector_format: List[str] = field(default_factory=lambda: ["h_nodes", "Q_links"])
    
    # Prior network L (single hidden layer, approximates dynamic mode decomposition)
    prior_network: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_units": None,  # Will be set to state dimension dynamically
        "activation": "relu",
        "use_bias": True,
        "dropout_rate": 0.0
    })
    
    # Residue network N (selected architecture S4 from Table 1)
    residue_network: Dict[str, Any] = field(default_factory=lambda: {
        "architecture": "S4",  # Selected architecture (6 layers × 100 neurons)
        "hidden_layers": 6,  # Number of hidden layers
        "hidden_units_per_layer": 100,  # Neurons per hidden layer
        "activation": "relu",  # ReLU activation (paper standard)
        "use_bias": True,
        "dropout_rate": 0.0,
        "batch_norm": False  # Not mentioned in paper
    })
    
    # Physical constraints
    physical_constraints_enabled: bool = True  # Mass balance for Q_w (Eq. 8)
    spilling_configuration: bool = True  # Water doesn't reenter system
    
    # Alternative architectures for hyperparameter study (Table 1)
    architecture_options: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "S1": (2, 10),   # 2 hidden layers × 10 neurons
        "S2": (4, 20),   # 4 hidden layers × 20 neurons
        "S3": (6, 50),   # 6 hidden layers × 50 neurons
        "S4": (6, 100),  # 6 hidden layers × 100 neurons (selected)
    })
    
    # Training-specific model settings
    gradient_clipping: Optional[float] = None  # Not mentioned in paper
    weight_initialization: str = "default"  # PyTorch default (paper doesn't specify)
    
    def __post_init__(self):
        """Validate model configuration."""
        # Validate residue network matches S4 configuration
        if (self.residue_network["hidden_layers"] != 6 or 
            self.residue_network["hidden_units_per_layer"] != 100):
            raise ValueError(
                f"Residue network must be S4 (6×100), got "
                f"{self.residue_network['hidden_layers']}×"
                f"{self.residue_network['hidden_units_per_layer']}"
            )
        
        # Validate activation function
        valid_activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
        if self.residue_network["activation"] not in valid_activations:
            raise ValueError(f"Invalid activation: {self.residue_network['activation']}")
        
        # Validate architecture options match Table 1
        expected_architectures = {"S1": (2, 10), "S2": (4, 20), "S3": (6, 50), "S4": (6, 100)}
        for name, config in expected_architectures.items():
            if name not in self.architecture_options:
                raise ValueError(f"Missing architecture {name} in options")
            if self.architecture_options[name] != config:
                raise ValueError(f"Architecture {name} configuration doesn't match Table 1")
        
        # Validate gradient clipping
        if self.gradient_clipping is not None and self.gradient_clipping <= 0:
            raise ValueError(f"gradient_clipping must be > 0, got {self.gradient_clipping}")


@dataclass
class TrainingConfig:
    """
    Configuration for model training and optimization.
    
    Defines optimizer settings, learning rate schedule, loss function,
    early stopping criteria, and training regimen details.
    
    Attributes:
        optimizer: Optimization algorithm ("adam")
        learning_rate: LR schedule (initial 1e-3, final 1e-4, exponential decay)
        epochs: Maximum training epochs (2000)
        early_stopping: Early stopping configuration (patience 500 epochs)
        loss_weights: Weighting for h, Q, Q_w in loss (all 1.0 after scaling)
        batch_size: Batch size (paper uses window as batch = 1)
        random_initializations: Number of models with different seeds (5)
        window_initialization: How to initialize windows ("true_state" for training)
        validation_initialization: How to initialize validation ("empty_system")
        use_parallel_processing: Whether to use parallel training (False)
        checkpoint_frequency: How often to save checkpoints (epochs)
        
    Note:
        Learning rate decays exponentially from 1e-3 to 1e-4 over training.
        Early stopping monitors validation loss with 500 epoch patience.
    """
    
    # Optimization
    optimizer: str = "adam"  # Adam optimizer (paper Section 2.4)
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "initial": 0.001,  # 1e-3 starting value
        "final": 0.0001,   # 1e-4 decayed to
        "decay": "exponential",  # Exponential decay (paper: from 1e-3 to 1e-4)
        "decay_steps": 2000,  # Decay over entire training
        "warmup_epochs": 0  # No warmup mentioned
    })
    
    # Training regimen
    epochs: int = 2000  # Maximum training epochs (paper Section 2.4)
    batch_size: int = 32  # Windows per batch; paper uses 1 but GPU benefits from larger batches
    random_initializations: int = 5  # Train 5 models with different random seeds (paper 2.5.3)
    
    # Window initialization strategies
    window_initialization: str = "true_state"  # Initialize from HiFi results at window start
    validation_initialization: str = "empty_system"  # Initialize from empty system
    
    # Early stopping
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "patience": 500,  # Stop if validation MSE doesn't decrease for 500 epochs
        "min_delta": 0.0001,  # Minimum improvement threshold
        "metric": "val_loss",  # Metric to monitor
        "restore_best_weights": True  # Restore best weights when stopping
    })
    
    # Loss function
    loss: Dict[str, Any] = field(default_factory=lambda: {
        "type": "mse",  # Mean Squared Error (paper Eq. 9)
        "weights": {  # All weights = 1 after scaling (paper: equal impact after scaling)
            "h": 1.0,  # Water levels
            "Q": 1.0,  # Pipe flows
            "Q_w": 1.0  # Excess flows (surcharge/overflow)
        },
        "reduction": "mean"  # Average over batch and time
    })
    
    # Parallel processing (paper mentions parallel window processing)
    use_parallel_processing: bool = False  # Set based on available hardware
    num_workers: int = 0  # Data loading workers (0 for main process)
    
    # Checkpointing
    checkpoint_frequency: int = 100  # Save checkpoint every 100 epochs
    keep_checkpoints: int = 5  # Keep only 5 most recent checkpoints
    
    def __post_init__(self):
        """Validate training configuration."""
        # Validate optimizer
        valid_optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        # Validate learning rate schedule
        if self.learning_rate["initial"] <= 0 or self.learning_rate["final"] <= 0:
            raise ValueError("Learning rates must be positive")
        if self.learning_rate["initial"] < self.learning_rate["final"]:
            raise ValueError("Initial learning rate must be >= final learning rate")
        
        # Validate loss weights sum to positive value
        loss_weights = self.loss["weights"]
        total_weight = sum(loss_weights.values())
        if total_weight <= 0:
            raise ValueError("Loss weights must sum to positive value")
        
        # Validate early stopping
        if self.early_stopping["enabled"] and self.early_stopping["patience"] <= 0:
            raise ValueError("Early stopping patience must be positive")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate number of random initializations
        if self.random_initializations <= 0:
            raise ValueError("Number of random initializations must be positive")


@dataclass
class EvaluationConfig:
    """
    Configuration for model evaluation and analysis.
    
    Defines metrics to compute, analysis methods, and visualization settings
    for comparing surrogate model performance against HiFi SWMM simulations.
    
    Attributes:
        metrics: List of metrics to compute ["rmse", "r2", "volume_error"]
        event_based_analysis: Whether to perform event-based analysis (True)
        event_separation: Method to separate events ("peak_flow")
        spatial_analysis: Whether to create spatial R² maps (True)
        plot_time_series: Whether to generate time series plots (True)
        plot_format: Format for saved plots ("png")
        results_dir: Directory for saving evaluation results
        
    Note:
        RMSE computed in original units (m for h, m³/s for Q).
        R² computed on unscaled data.
        Volume errors compare total volumes (runoff, surcharge, overflow, outflow).
    """
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: ["rmse", "r2", "volume_error"])
    
    # Analysis methods
    event_based_analysis: bool = True  # Analyze performance per event (paper Section 3.2)
    event_separation: str = "peak_flow"  # Separate events based on outlet peak flow
    spatial_analysis: bool = True  # Create spatial R² maps (paper Fig. 4)
    
    # Visualization
    plot_time_series: bool = True  # Generate time series plots (paper Fig. 5)
    plot_format: str = "png"  # Format for saved plots
    dpi: int = 300  # Resolution for figures
    
    # Output directories
    results_dir: str = "results"
    plots_dir: str = "plots"
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        # Validate metrics
        valid_metrics = ["rmse", "r2", "mae", "mse", "volume_error", "nse"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}")
        
        # Validate event separation method
        valid_separation = ["peak_flow", "dry_period", "manual"]
        if self.event_separation not in valid_separation:
            raise ValueError(f"Invalid event separation method: {self.event_separation}")
        
        # Validate plot format
        valid_formats = ["png", "pdf", "svg", "jpg"]
        if self.plot_format.lower() not in valid_formats:
            raise ValueError(f"Invalid plot format: {self.plot_format}")
        
        # Create directories
        results_path = Path(self.results_dir)
        plots_path = results_path / self.plots_dir
        results_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PerformanceConfig:
    """
    Configuration for computational performance benchmarking.
    
    Defines parameters for comparing surrogate model speed against
    HiFi SWMM simulations and measuring training/inference times.
    
    Attributes:
        compare_with_swmm: Whether to benchmark against SWMM (True)
        swmm_time_step: HiFi model time step (5 seconds)
        surrogate_time_step: Surrogate model time step (60 seconds)
        benchmark_repetitions: Number of repetitions for timing (3)
        warmup_runs: Number of warmup runs before timing (1)
        
    Note:
        Speed-up factor computed as: SWMM_time / surrogate_time.
        Paper reports one to two orders of magnitude speed-up (Table 2).
    """
    
    compare_with_swmm: bool = True  # Benchmark against SWMM simulations
    swmm_time_step: int = 5  # 5-second time step for HiFi model
    surrogate_time_step: int = 60  # 1-minute time step for surrogate
    benchmark_repetitions: int = 3  # Number of repetitions for stable timing
    warmup_runs: int = 1  # Warmup runs before timing measurements
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.swmm_time_step <= 0 or self.surrogate_time_step <= 0:
            raise ValueError("Time steps must be positive")
        if self.benchmark_repetitions <= 0:
            raise ValueError("Benchmark repetitions must be positive")
        if self.warmup_runs < 0:
            raise ValueError("Warmup runs cannot be negative")
        
        # Verify time step alignment
        if self.surrogate_time_step % self.swmm_time_step != 0:
            print(f"WARNING: Surrogate time step {self.surrogate_time_step}s is not "
                  f"multiple of SWMM time step {self.swmm_time_step}s")
        
        # Compute theoretical maximum speed-up
        theoretical_speedup = self.surrogate_time_step / self.swmm_time_step
        print(f"Theoretical maximum speed-up (time step only): {theoretical_speedup:.1f}x")


@dataclass
class SystemConfig:
    """
    Configuration for computational environment and reproducibility.
    
    Defines system-level settings including random seeds, device selection,
    directory paths, and logging configuration.
    
    Attributes:
        random_seed: Base random seed for reproducibility (42)
        device: PyTorch device ("cuda" if available, else "cpu")
        num_workers: DataLoader workers (0 for main process)
        checkpoint_dir: Directory for model checkpoints
        results_dir: Directory for results and logs
        log_level: Logging verbosity ("INFO")
        deterministic: Whether to use deterministic algorithms (True)
        
    Note:
        Paper used single CPU (no GPU benefit mentioned), but we use GPU
        if available for modern best practices.
    """
    
    # Reproducibility
    random_seed: int = 42  # For reproducibility (paper doesn't specify)
    deterministic: bool = True  # Use deterministic algorithms when possible
    
    # Hardware
    device: str = "cuda"  # Use GPU if available (paper used CPU)
    num_workers: int = 0  # DataLoader workers (0 = main process)
    
    # Directories
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    log_dir: str = "logs"
    
    # Logging
    log_level: str = "INFO"  # Logging verbosity
    log_file: str = "training.log"
    
    def __post_init__(self):
        """Validate system configuration and set up directories."""
        # Validate random seed
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError(f"Invalid random seed: {self.random_seed}")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device.lower() not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Adjust device based on availability
        if self.device.lower() == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        elif self.device.lower() == "mps" and not hasattr(torch.backends, "mps"):
            print("WARNING: MPS requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Validate num_workers
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative: {self.num_workers}")
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set full log file path
        self.log_file = os.path.join(self.log_dir, self.log_file)
    
    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.device)


@dataclass
class Config:
    """
    Main configuration container integrating all sub-configurations.
    
    Aggregates all configuration sections into a single object for
    easy access throughout the pipeline. Provides validation and
    convenience methods for configuration management.
    
    Attributes:
        data: Data processing and management configuration
        model: Neural network architecture configuration
        training: Training and optimization configuration
        evaluation: Evaluation and analysis configuration
        performance: Performance benchmarking configuration
        system: System and environment configuration
        
    Methods:
        validate: Validate entire configuration against paper requirements
        save: Save configuration to YAML file
        load: Load configuration from YAML file (classmethod)
        
    Note:
        All sub-configurations are validated during initialization.
    """
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Validate entire configuration and ensure consistency."""
        self.validate()
        
        # Override results_dir in evaluation config to match system config
        # Ensure all results go to the same directory
        self.evaluation.results_dir = self.system.results_dir
    
    def validate(self) -> None:
        """
        Validate configuration consistency against paper requirements.
        
        Checks cross-sectional consistency between different configuration
        sections and validates against known paper constraints.
        
        Raises:
            ValueError: If configuration violates paper methodology
        """
        # Check time step consistency
        if self.data.time_step != self.performance.surrogate_time_step:
            raise ValueError(
                f"Data time step ({self.data.time_step}s) must match "
                f"performance surrogate time step ({self.performance.surrogate_time_step}s)"
            )
        
        if self.data.simulation_time_step != self.performance.swmm_time_step:
            raise ValueError(
                f"Data simulation time step ({self.data.simulation_time_step}s) must match "
                f"performance SWMM time step ({self.performance.swmm_time_step}s)"
            )
        
        # Check window size is positive
        if self.data.training_window_size <= 0:
            raise ValueError(f"Training window size must be positive: {self.data.training_window_size}")
        
        # Check that S4 architecture is used
        if self.model.residue_network["architecture"] != "S4":
            print(f"WARNING: Using architecture {self.model.residue_network['architecture']} "
                  f"instead of S4 (6×100) as specified in paper")
        
        # Check learning rate schedule matches paper
        if (self.training.learning_rate["initial"] != 0.001 or 
            self.training.learning_rate["final"] != 0.0001):
            print("WARNING: Learning rate schedule differs from paper (1e-3 → 1e-4)")
        
        # Check early stopping patience
        if self.training.early_stopping["patience"] != 500:
            print(f"WARNING: Early stopping patience {self.training.early_stopping['patience']} "
                  f"differs from paper (500 epochs)")
        
        # Check training epochs
        if self.training.epochs < 2000:
            print(f"WARNING: Maximum epochs {self.training.epochs} < paper's 2000")
        
        # Check physical constraints are enabled
        if not self.model.physical_constraints_enabled:
            print("WARNING: Physical constraints disabled (paper uses constraints for final model)")
        
        # Check spilling configuration
        if not self.model.spilling_configuration:
            print("WARNING: Spilling configuration disabled (paper enables it)")
    
    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration file
        
        Raises:
            IOError: If file cannot be written
            yaml.YAMLError: If configuration cannot be serialized
        """
        # Convert dataclasses to dictionaries
        config_dict = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "performance": self.performance.__dict__,
            "system": self.system.__dict__
        }
        
        # Remove private attributes (starting with underscore)
        for section in config_dict:
            private_keys = [k for k in config_dict[section] if k.startswith("_")]
            for key in private_keys:
                del config_dict[section][key]
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {path}")
        except (IOError, OSError) as e:
            raise IOError(f"Failed to write configuration file {path}: {e}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to serialize configuration to YAML: {e}")
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            Config object with loaded settings
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML cannot be parsed
            ValueError: If configuration is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
        
        # Create Config object from dictionary
        config = cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            system=SystemConfig(**config_dict.get("system", {}))
        )
        
        # Re-validate the loaded configuration
        config.validate()
        
        print(f"Configuration loaded from: {path}")
        return config
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of configuration.
        
        Returns:
            String summarizing key configuration parameters
        """
        summary_lines = [
            "=" * 80,
            "CONFIGURATION SUMMARY",
            "=" * 80,
            "",
            "DATA:",
            f"  Network file: {self.data.inp_file}",
            f"  Time step: {self.data.time_step}s (surrogate), {self.data.simulation_time_step}s (SWMM)",
            f"  Window size: {self.data.training_window_size} minutes ({self.data.window_timesteps} steps)",
            f"  Scaling: {self.data.scaling_method} per variable to [0,1]",
            "",
            "MODEL:",
            f"  Architecture: gResNet with physical constraints",
            f"  Prior network L: Single hidden layer",
            f"  Residue network N: S4 ({self.model.residue_network['hidden_layers']}×{self.model.residue_network['hidden_units_per_layer']})",
            f"  Activation: {self.model.residue_network['activation']}",
            f"  Physical constraints: {self.model.physical_constraints_enabled}",
            "",
            "TRAINING:",
            f"  Optimizer: {self.training.optimizer}",
            f"  Learning rate: {self.training.learning_rate['initial']} → {self.training.learning_rate['final']}",
            f"  Epochs: {self.training.epochs} (max), early stopping: {self.training.early_stopping['patience']} patience",
            f"  Batch size: {self.training.batch_size} (window as batch)",
            f"  Random initializations: {self.training.random_initializations}",
            "",
            "SYSTEM:",
            f"  Device: {self.system.device}",
            f"  Random seed: {self.system.random_seed}",
            f"  Checkpoint dir: {self.system.checkpoint_dir}",
            f"  Results dir: {self.system.results_dir}",
            "=" * 80
        ]
        
        return "\n".join(summary_lines)


def load_config_from_yaml(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file with error handling.
    
    Convenience function for loading configuration that handles common
    errors and provides default configuration if file doesn't exist.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded settings
        
    Note:
        If config_path doesn't exist, creates default configuration
        and saves it to the specified path.
    """
    try:
        config = Config.load(config_path)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Creating default configuration.")
        config = Config()
        config.save(config_path)
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error loading configuration from {config_path}: {e}")
        print("Falling back to default configuration.")
        config = Config()
    
    return config


def validate_config(config: Config) -> bool:
    """
    Validate configuration against paper requirements.
    
    Performs comprehensive validation of configuration parameters
    against the paper's methodology and experimental design.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if configuration is valid, False otherwise
        
    Note:
        Prints warnings for deviations from paper methodology.
        Critical errors are raised as exceptions.
    """
    warnings = []
    
    # Check paper-specific requirements
    
    # Window size should be 60 minutes (optimal from hyperparameter study)
    if config.data.training_window_size != 60:
        warnings.append(
            f"Window size {config.data.training_window_size} != 60 minutes (paper's optimal)"
        )
    
    # Residue network should be S4 (6×100)
    residue = config.model.residue_network
    if not (residue["hidden_layers"] == 6 and residue["hidden_units_per_layer"] == 100):
        warnings.append(
            f"Residue network {residue['hidden_layers']}×{residue['hidden_units_per_layer']} "
            f"!= S4 (6×100) from paper"
        )
    
    # Learning rate should start at 1e-3 and decay to 1e-4
    lr = config.training.learning_rate
    if lr["initial"] != 0.001:
        warnings.append(f"Initial LR {lr['initial']} != 1e-3 (paper)")
    if lr["final"] != 0.0001:
        warnings.append(f"Final LR {lr['final']} != 1e-4 (paper)")
    
    # Early stopping patience should be 500
    if config.training.early_stopping["patience"] != 500:
        warnings.append(
            f"Early stopping patience {config.training.early_stopping['patience']} != 500 (paper)"
        )
    
    # Physical constraints should be enabled
    if not config.model.physical_constraints_enabled:
        warnings.append("Physical constraints disabled (paper uses them for final model)")
    
    # Print warnings if any
    if warnings:
        print("=" * 80)
        print("CONFIGURATION WARNINGS:")
        print("=" * 80)
        for i, warning in enumerate(warnings, 1):
            print(f"{i:2d}. {warning}")
        print("=" * 80)
        return False
    
    print("Configuration validated successfully against paper requirements.")
    return True


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Test the configuration module
    print("Testing config.py...")
    
    # Create default configuration
    config = Config()
    print(config.get_summary())
    
    # Test saving and loading
    test_config_path = "test_config.yaml"
    config.save(test_config_path)
    
    loaded_config = Config.load(test_config_path)
    print(f"Configuration saved and loaded successfully. Match: {config == loaded_config}")
    
    # Clean up test file
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
        print(f"Test file {test_config_path} removed.")
    
    # Test validation
    is_valid = validate_config(config)
    print(f"Configuration valid: {is_valid}")
    
    print("All tests completed!")
