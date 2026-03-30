"""
data_processor.py

Data preprocessing and transformation module for the physics-constrained gResNet
surrogate model. Implements the paper's methodology for converting raw SWMM
simulation outputs into structured training, validation, and test datasets.

This module provides the DataProcessor class that handles:
1. State vector construction: x_t = [h_1...h_N, Q_1...Q_M] (Eq. 7)
2. Input vector construction: R_t = [R_1...R_N] (runoff at each node)
3. Min-max scaling: Each variable independently scaled to [0,1] using training data
4. Windowed training: Sliding windows of size W for autoregressive training
5. Excess flow computation: Q_w via mass balance (Eq. 8) using network adjacency

Classes:
    DataProcessor: Main data preprocessing and transformation class
    VariableScaler: Custom scaler for per-variable min-max scaling

Functions:
    validate_state_input_alignment: Validate state and input arrays have same timesteps
    compute_mass_balance_excess_flows: Compute Q_w via Eq. 8 using sparse matrices

Note:
    Strictly follows paper's methodology: per-variable min-max scaling, windowed
    training with configurable window size, and mass balance constraints.
"""

import os
import warnings
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

# Project imports
from utils import Timer, set_seed, save_pickle, load_pickle
from config import Config, DataConfig, ModelConfig, SystemConfig


@dataclass
class ScalingParameters:
    """
    Data class for storing scaling parameters for reproducibility.
    
    Attributes:
        state_scaler: MinMaxScaler for state variables (h and Q)
        input_scaler: MinMaxScaler for input variables (R)
        state_min: Minimum values for state variables (original units)
        state_max: Maximum values for state variables (original units)
        input_min: Minimum values for input variables (original units)
        input_max: Maximum values for input variables (original units)
        node_ids: Ordered list of node IDs for consistent indexing
        link_ids: Ordered list of link IDs for consistent indexing
        metadata: Additional scaling metadata
    """
    state_scaler: MinMaxScaler
    input_scaler: MinMaxScaler
    state_min: np.ndarray
    state_max: np.ndarray
    input_min: np.ndarray
    input_max: np.ndarray
    node_ids: List[str]
    link_ids: List[str]
    metadata: Dict[str, Any]
    
    def save(self, path: str) -> None:
        """Save scaling parameters to disk."""
        save_pickle(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'ScalingParameters':
        """Load scaling parameters from disk."""
        return load_pickle(path)


class VariableScaler:
    """
    Custom scaler for per-variable min-max scaling to [0,1].
    
    This class implements the paper's scaling methodology: each variable
    (each node's h, each link's Q, each node's R) is scaled independently
    to the [0,1] range using training data statistics.
    
    Attributes:
        scalers: List of MinMaxScaler objects, one per variable
        n_features: Number of features being scaled
        feature_names: Names of features (for debugging/validation)
        fitted: Whether scalers have been fitted
        
    Note:
        Paper Section 2.3: "Scaling is performed using a min-max approach,
        and the scaling range is defined individually for each state variable,
        based on the minimum and maximum values observed in the HiFi simulations."
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize VariableScaler with optional feature names.
        
        Args:
            feature_names: List of feature names for identification
        """
        self.scalers = []
        self.n_features = 0
        self.feature_names = feature_names if feature_names is not None else []
        self.fitted = False
        
        # Statistics for zero-variance handling
        self.has_zero_variance = None
        self.zero_variance_mask = None
        self.constant_values = None
    
    def fit(self, X: np.ndarray) -> 'VariableScaler':
        """
        Fit scalers to data.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If X has incorrect dimensions
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        
        n_samples, self.n_features = X.shape
        
        # Initialize feature names if not provided
        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        # Check for zero-variance features
        self.has_zero_variance = np.std(X, axis=0) == 0
        self.zero_variance_mask = self.has_zero_variance
        self.constant_values = X[0, self.zero_variance_mask] if np.any(self.zero_variance_mask) else None
        
        # Initialize and fit scalers for each feature
        self.scalers = []
        for i in range(self.n_features):
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            
            if self.has_zero_variance[i]:
                # Handle zero-variance features: set to 0.5 (midpoint of [0,1])
                # Paper doesn't specify, but this preserves the constant nature
                X_col = np.array([[0.0], [1.0]])  # Dummy data for fitting
                scaler.fit(X_col)
                # Override data range to [value, value] -> maps to 0.5
                scaler.data_min_ = np.array([X[0, i]])
                scaler.data_max_ = np.array([X[0, i]])
                scaler.scale_ = np.array([1.0])  # Will be ignored due to zero range
                scaler.min_ = np.array([0.5])   # Always output 0.5
            else:
                # Fit on actual data
                scaler.fit(X[:, i:i+1])
            
            self.scalers.append(scaler)
        
        self.fitted = True
        
        # Log zero-variance features
        if np.any(self.has_zero_variance):
            zero_var_features = [self.feature_names[i] for i in range(self.n_features) 
                                if self.has_zero_variance[i]]
            warnings.warn(f"Found {len(zero_var_features)} zero-variance features: "
                         f"{zero_var_features}. These will be scaled to 0.5.")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scalers.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            
        Returns:
            Scaled data array of shape (n_samples, n_features)
            
        Raises:
            NotFittedError: If scalers haven't been fitted
            ValueError: If X has wrong number of features
        """
        if not self.fitted:
            raise NotFittedError("Scaler must be fitted before transforming data")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        X_scaled = np.zeros_like(X, dtype=np.float32)
        
        for i in range(self.n_features):
            if self.has_zero_variance[i]:
                # Set zero-variance features to 0.5
                X_scaled[:, i] = 0.5
            else:
                # Transform using fitted scaler
                X_scaled[:, i] = self.scalers[i].transform(X[:, i:i+1]).flatten()
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scalers and transform data.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            
        Returns:
            Scaled data array of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original units.
        
        Args:
            X_scaled: Scaled data array of shape (n_samples, n_features)
            
        Returns:
            Original data array of shape (n_samples, n_features)
            
        Raises:
            NotFittedError: If scalers haven't been fitted
            ValueError: If X_scaled has wrong number of features
        """
        if not self.fitted:
            raise NotFittedError("Scaler must be fitted before inverse transforming")
        
        if X_scaled.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X_scaled.shape[1]}")
        
        X_original = np.zeros_like(X_scaled, dtype=np.float32)
        
        for i in range(self.n_features):
            if self.has_zero_variance[i]:
                # Zero-variance features: return constant value
                X_original[:, i] = self.constant_values[list(self.zero_variance_mask).index(True)] \
                    if self.constant_values is not None else 0.0
            else:
                # Inverse transform using fitted scaler
                X_original[:, i] = self.scalers[i].inverse_transform(
                    X_scaled[:, i:i+1]
                ).flatten()
        
        return X_original
    
    def get_scaling_params(self) -> Dict[str, Any]:
        """
        Get scaling parameters for reproducibility.
        
        Returns:
            Dictionary with scaling parameters for each feature
        """
        if not self.fitted:
            raise NotFittedError("Scaler must be fitted to get parameters")
        
        params = {
            'feature_names': self.feature_names,
            'has_zero_variance': self.has_zero_variance.tolist() if self.has_zero_variance is not None else None,
            'constant_values': self.constant_values.tolist() if self.constant_values is not None else None,
            'scalers': []
        }
        
        for i, scaler in enumerate(self.scalers):
            if self.has_zero_variance[i]:
                params['scalers'].append({
                    'data_min': scaler.data_min_.tolist(),
                    'data_max': scaler.data_max_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'min': scaler.min_.tolist(),
                    'zero_variance': True
                })
            else:
                params['scalers'].append({
                    'data_min': scaler.data_min_.tolist(),
                    'data_max': scaler.data_max_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'min': scaler.min_.tolist(),
                    'zero_variance': False
                })
        
        return params


class DataProcessor:
    """
    Main data preprocessing class for the physics-constrained gResNet.
    
    This class implements the complete data processing pipeline from raw SWMM
    simulation outputs to structured datasets ready for model training. It
    strictly follows the paper's methodology for state vector construction,
    min-max scaling, windowed training, and mass balance computations.
    
    Attributes:
        network_parser: NetworkParser instance for network topology
        config: Configuration dictionary
        data_config: Data configuration sub-object
        model_config: Model configuration sub-object
        system_config: System configuration sub-object
        node_ids: Ordered list of node IDs (from network_parser)
        link_ids: Ordered list of link IDs (from network_parser)
        n_nodes: Number of nodes in network
        n_links: Number of links in network
        upstream_matrix: Sparse matrix (N×M) where [i,j]=1 if link j ends at node i
        downstream_matrix: Sparse matrix (N×M) where [i,j]=1 if link j starts at node i
        state_scaler: VariableScaler for state variables (h and Q)
        input_scaler: VariableScaler for input variables (R)
        scaling_params: ScalingParameters for reproducibility
        processed_data_dir: Directory for saving processed datasets
        
    Methods:
        create_state_vectors: Construct state vectors from raw SWMM data
        create_input_vectors: Construct input vectors from inflow data
        scale_data: Apply min-max scaling to states/inputs
        create_windows: Create sliding windows for autoregressive training
        compute_excess_flows: Compute excess flows via mass balance (Eq. 8)
        save_processed_data: Save processed data to disk
        load_processed_data: Load processed data from disk
        
    Note:
        All data processing steps are performed in-memory. For very large
        datasets, consider implementing batch processing or memory mapping.
    """
    
    def __init__(self, network_parser: Any, config: Optional[Dict] = None):
        """
        Initialize DataProcessor with network parser and configuration.
        
        Args:
            network_parser: NetworkParser instance (must have parse_network() called)
            config: Configuration dictionary. If None, uses default Config.
            
        Raises:
            ValueError: If network_parser hasn't parsed network yet
            RuntimeError: If network data is invalid or incomplete
            
        Note:
            The network_parser must have already parsed the network and
            have valid node and link DataFrames and adjacency matrices.
        """
        self.network_parser = network_parser
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.data_config = self.config.data
            self.model_config = self.config.model
            self.system_config = self.config.system
        else:
            self.config = config
            self.data_config = DataConfig(**config.get('data', {}))
            self.model_config = ModelConfig(**config.get('model', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set random seed for reproducibility
        set_seed(self.system_config.random_seed)
        
        # Extract network information
        self._extract_network_info()
        
        # Initialize scalers
        self.state_scaler = None
        self.input_scaler = None
        self.scaling_params = None
        
        # Set up directories
        self.processed_data_dir = Path("S5/results_P2C/processed_data")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized DataProcessor for network: {self.n_nodes} nodes, {self.n_links} links")
        print(f"  State vector dimension: {self.n_nodes + self.n_links}")
        print(f"  Input vector dimension: {self.n_nodes}")
        print(f"  Processed data directory: {self.processed_data_dir}")
    
    def _extract_network_info(self) -> None:
        """
        Extract network topology information from network_parser.
        
        Raises:
            RuntimeError: If network_parser hasn't parsed network yet
            ValueError: If network data is invalid
        """
        try:
            # Get node and link DataFrames
            nodes_df = self.network_parser.get_node_data()
            links_df = self.network_parser.get_link_data()
            
            # Get adjacency matrices
            upstream_matrix, downstream_matrix = self.network_parser.get_adjacency_matrices()
            
            # Validate network data
            if nodes_df is None or links_df is None:
                raise RuntimeError("Network parser hasn't parsed network yet. Call parse_network() first.")
            
            if upstream_matrix is None or downstream_matrix is None:
                raise ValueError("Adjacency matrices not available from network parser")
            
            # Store network information
            self.node_ids = nodes_df['id'].tolist()
            self.link_ids = links_df['id'].tolist()
            self.n_nodes = len(self.node_ids)
            self.n_links = len(self.link_ids)
            
            # Store adjacency matrices
            self.upstream_matrix = upstream_matrix
            self.downstream_matrix = downstream_matrix
            
            # Create node and link ID to index mappings
            self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
            self.link_id_to_idx = {link_id: idx for idx, link_id in enumerate(self.link_ids)}
            
            # Validate dimensions
            if self.n_nodes != upstream_matrix.shape[0]:
                raise ValueError(f"Node count mismatch: {self.n_nodes} nodes vs "
                               f"{upstream_matrix.shape[0]} rows in upstream matrix")
            if self.n_links != upstream_matrix.shape[1]:
                raise ValueError(f"Link count mismatch: {self.n_links} links vs "
                               f"{upstream_matrix.shape[1]} columns in upstream matrix")
            
            # Print network summary
            print(f"  Network summary: {self.n_nodes} nodes, {self.n_links} links")
            print(f"    Nodes: {self.node_ids}")
            print(f"    Links: {self.link_ids}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract network info: {str(e)}") from e
    
    def create_state_vectors(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Construct state vectors from raw SWMM simulation data.
        
        This method implements Eq. 7 from the paper:
        x_t = [h_1, ..., h_N, Q_1, ..., Q_M]
        
        Args:
            raw_data: DataFrame with SWMM simulation results. Expected columns:
                     - Water levels: MultiIndex columns ('water_level', node_id)
                     - Pipe flows: MultiIndex columns ('pipe_flow', link_id)
                     - Timestamp information in index or columns
                     
        Returns:
            NumPy array of shape (n_timesteps, n_nodes + n_links) containing
            water levels followed by pipe flows in consistent ordering.
            
        Raises:
            ValueError: If required columns are missing or data is misaligned
            KeyError: If expected node/link IDs not found in raw_data
            
        Note:
            The raw_data is expected to come from SimulationResult.to_dataframe()
            which creates MultiIndex columns. This method handles both MultiIndex
            and simple column names for robustness.
        """
        print(f"Creating state vectors from raw data with shape: {raw_data.shape}")
        
        with Timer("state_vector_creation") as timer:
            # Extract water level columns
            if isinstance(raw_data.columns, pd.MultiIndex):
                # MultiIndex columns: ('water_level', node_id) and ('pipe_flow', link_id)
                water_level_columns = [col for col in raw_data.columns 
                                      if col[0] == 'water_level']
                pipe_flow_columns = [col for col in raw_data.columns 
                                    if col[0] == 'pipe_flow']
                
                # Extract node and link IDs from column names
                water_level_nodes = [col[1] for col in water_level_columns]
                pipe_flow_links = [col[1] for col in pipe_flow_columns]
                
                # Create DataFrames for easier indexing
                water_levels_df = raw_data[water_level_columns].copy()
                pipe_flows_df = raw_data[pipe_flow_columns].copy()
                
                # Rename columns to simple node/link IDs
                water_levels_df.columns = water_level_nodes
                pipe_flows_df.columns = pipe_flow_links
                
            else:
                # Simple column names: assume format 'h_NodeID' and 'Q_LinkID'
                water_level_pattern = re.compile(r'^h_(\w+)$')
                pipe_flow_pattern = re.compile(r'^Q_(\w+)$')
                
                water_level_columns = [col for col in raw_data.columns 
                                      if water_level_pattern.match(col)]
                pipe_flow_columns = [col for col in raw_data.columns 
                                    if pipe_flow_pattern.match(col)]
                
                water_level_nodes = [water_level_pattern.match(col).group(1) 
                                   for col in water_level_columns]
                pipe_flow_links = [pipe_flow_pattern.match(col).group(1) 
                                 for col in pipe_flow_columns]
                
                water_levels_df = raw_data[water_level_columns].copy()
                pipe_flows_df = raw_data[pipe_flow_columns].copy()
                
                water_levels_df.columns = water_level_nodes
                pipe_flows_df.columns = pipe_flow_links
            
            # Validate that all nodes and links in network have data
            missing_nodes = set(self.node_ids) - set(water_level_nodes)
            if missing_nodes:
                raise ValueError(f"Missing water level data for nodes: {sorted(missing_nodes)}")
            
            missing_links = set(self.link_ids) - set(pipe_flow_links)
            if missing_links:
                raise ValueError(f"Missing pipe flow data for links: {sorted(missing_links)}")
            
            # Reorder columns to match network ordering
            water_levels_df = water_levels_df[self.node_ids]
            pipe_flows_df = pipe_flows_df[self.link_ids]
            
            # Handle missing values (should be none from SWMM, but check)
            if water_levels_df.isna().any().any() or pipe_flows_df.isna().any().any():
                warnings.warn("Missing values found in raw data. Filling with zeros.")
                water_levels_df = water_levels_df.fillna(0.0)
                pipe_flows_df = pipe_flows_df.fillna(0.0)
            
            # Convert to numpy arrays
            water_levels_array = water_levels_df.values.astype(np.float32)
            pipe_flows_array = pipe_flows_df.values.astype(np.float32)
            
            # Concatenate: water levels first, then pipe flows
            states_array = np.concatenate([water_levels_array, pipe_flows_array], axis=1)
            
            # Validate dimensions
            n_timesteps = raw_data.shape[0]
            expected_shape = (n_timesteps, self.n_nodes + self.n_links)
            
            if states_array.shape != expected_shape:
                raise ValueError(f"State array shape mismatch: {states_array.shape} != {expected_shape}")
            
            # Compute basic statistics for validation
            h_stats = {
                'min': np.min(water_levels_array),
                'max': np.max(water_levels_array),
                'mean': np.mean(water_levels_array),
                'std': np.std(water_levels_array)
            }
            
            Q_stats = {
                'min': np.min(pipe_flows_array),
                'max': np.max(pipe_flows_array),
                'mean': np.mean(pipe_flows_array),
                'std': np.std(pipe_flows_array)
            }
            
            print(f"  Created state vectors: {states_array.shape}")
            print(f"    Water levels: shape={water_levels_array.shape}, "
                  f"range=[{h_stats['min']:.3f}, {h_stats['max']:.3f}] m")
            print(f"    Pipe flows: shape={pipe_flows_array.shape}, "
                  f"range=[{Q_stats['min']:.3f}, {Q_stats['max']:.3f}] m³/s")
            print(f"    Total state dimension: {self.n_nodes + self.n_links}")
            
            return states_array
    
    def create_input_vectors(self, inflow_data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Construct input vectors (runoff R_i) for each node.
        
        This method creates input vectors R_t = [R_1, ..., R_N] where R_i is
        the runoff inflow at node i (zero for non-inflow nodes).
        
        Args:
            inflow_data: Input runoff data. Can be:
                        - DataFrame with columns for each inflow node
                        - Dictionary mapping node_id to runoff time series
                        
        Returns:
            NumPy array of shape (n_timesteps, n_nodes) containing runoff
            inflows for all nodes (zero for non-inflow nodes).
            
        Raises:
            ValueError: If inflow data has wrong dimensions or missing timesteps
            TypeError: If inflow_data is not DataFrame or dict
            
        Note:
            The paper uses direct nodal inflows as runoff inputs since Ji.inp
            has no rainfall-runoff model. Non-inflow nodes have R_i = 0.
        """
        print("Creating input vectors from inflow data...")
        
        with Timer("input_vector_creation") as timer:
            # Determine number of timesteps
            if isinstance(inflow_data, pd.DataFrame):
                n_timesteps = inflow_data.shape[0]
                inflow_df = inflow_data.copy()
            elif isinstance(inflow_data, dict):
                # Check that all arrays have same length
                lengths = [len(arr) for arr in inflow_data.values()]
                if not all(l == lengths[0] for l in lengths):
                    raise ValueError("All inflow time series must have same length")
                n_timesteps = lengths[0]
                
                # Convert dict to DataFrame
                inflow_df = pd.DataFrame(inflow_data)
            else:
                raise TypeError(f"inflow_data must be DataFrame or dict, got {type(inflow_data)}")
            
            # Initialize input array with zeros
            inputs_array = np.zeros((n_timesteps, self.n_nodes), dtype=np.float32)
            
            # Fill in inflows for nodes that have data
            for node_idx, node_id in enumerate(self.node_ids):
                if node_id in inflow_df.columns:
                    inflow_series = inflow_df[node_id].values
                    
                    # Ensure length matches
                    if len(inflow_series) != n_timesteps:
                        raise ValueError(f"Inflow series for node {node_id} has length "
                                       f"{len(inflow_series)} != {n_timesteps}")
                    
                    # Handle missing values
                    if np.any(np.isnan(inflow_series)):
                        warnings.warn(f"NaN values in inflow data for node {node_id}. Filling with zeros.")
                        inflow_series = np.nan_to_num(inflow_series, nan=0.0)
                    
                    inputs_array[:, node_idx] = inflow_series.astype(np.float32)
            
            # Compute statistics
            inflow_nodes = [node_id for node_id in self.node_ids 
                          if node_id in inflow_df.columns]
            non_inflow_nodes = [node_id for node_id in self.node_ids 
                              if node_id not in inflow_df.columns]
            
            total_inflow = np.sum(inputs_array)
            max_inflow = np.max(inputs_array)
            
            print(f"  Created input vectors: {inputs_array.shape}")
            print(f"    Inflow nodes: {inflow_nodes}")
            print(f"    Non-inflow nodes: {non_inflow_nodes}")
            print(f"    Total inflow volume: {total_inflow * 60:.2f} m³ "
                  f"(assuming 1-minute timesteps)")
            print(f"    Maximum inflow rate: {max_inflow:.4f} m³/s")
            
            return inputs_array
    
    def scale_data(self, data: np.ndarray, data_type: str = 'state', 
                  fit: bool = False, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[VariableScaler]]:
        """
        Apply min-max scaling to data.
        
        This implements the paper's scaling methodology: each variable is
        independently scaled to [0,1] using training data statistics.
        
        Args:
            data: Data array to scale, shape (n_samples, n_features)
            data_type: Type of data ('state' or 'input')
            fit: Whether to fit scaler (True for training, False for val/test)
            feature_names: Optional names for features (for debugging)
            
        Returns:
            Tuple of (scaled_data, scaler) where scaler is None if fit=False
            
        Raises:
            ValueError: If data_type not 'state' or 'input'
            NotFittedError: If fit=False but scaler hasn't been fitted
            ValueError: If data has wrong number of features
            
        Note:
            For state data: n_features = n_nodes + n_links
            For input data: n_features = n_nodes
        """
        print(f"Scaling {data_type} data (fit={fit})...")
        
        with Timer(f"{data_type}_scaling") as timer:
            # Validate data type
            if data_type not in ['state', 'input']:
                raise ValueError(f"data_type must be 'state' or 'input', got {data_type}")
            
            # Determine expected number of features
            if data_type == 'state':
                expected_features = self.n_nodes + self.n_links
                scaler_attr = 'state_scaler'
            else:  # input
                expected_features = self.n_nodes
                scaler_attr = 'input_scaler'
            
            # Validate dimensions
            if data.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} features for {data_type} data, "
                               f"got {data.shape[1]}")
            
            # Get or create scaler
            scaler = getattr(self, scaler_attr)
            
            if fit:
                # Fit new scaler
                if feature_names is None:
                    # Generate feature names
                    if data_type == 'state':
                        feature_names = (self.node_ids + 
                                       [f"Q_{lid}" for lid in self.link_ids])
                    else:  # input
                        feature_names = self.node_ids
                
                scaler = VariableScaler(feature_names=feature_names)
                scaled_data = scaler.fit_transform(data)
                
                # Store scaler
                setattr(self, scaler_attr, scaler)
                
                # Update scaling parameters if both scalers are fitted
                if self.state_scaler is not None and self.input_scaler is not None:
                    self._update_scaling_parameters()
                
                print(f"  Fitted scaler to {data.shape[0]} samples, {data.shape[1]} features")
                
            else:
                # Use existing scaler
                if scaler is None:
                    raise NotFittedError(f"{data_type} scaler not fitted. Call with fit=True first.")
                
                scaled_data = scaler.transform(data)
                print(f"  Transformed {data.shape[0]} samples using pre-fitted scaler")
            
            # Compute scaling statistics
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            scaled_min = np.min(scaled_data, axis=0)
            scaled_max = np.max(scaled_data, axis=0)
            
            print(f"    Original range: [{data_min.min():.3f}, {data_max.max():.3f}]")
            print(f"    Scaled range: [{scaled_min.min():.3f}, {scaled_max.max():.3f}]")
            
            return scaled_data, (scaler if fit else None)
    
    def _update_scaling_parameters(self) -> None:
        """
        Update scaling parameters object with current scalers.
        
        Creates a ScalingParameters object containing all information needed
        to reproduce the scaling transformation.
        """
        if self.state_scaler is None or self.input_scaler is None:
            warnings.warn("Cannot update scaling parameters: state or input scaler not fitted")
            return
        
        # Get scaling parameters from scalers
        state_params = self.state_scaler.get_scaling_params()
        input_params = self.input_scaler.get_scaling_params()
        
        # Extract min/max values for easy access
        state_min = np.array([s['data_min'][0] for s in state_params['scalers']])
        state_max = np.array([s['data_max'][0] for s in state_params['scalers']])
        input_min = np.array([s['data_min'][0] for s in input_params['scalers']])
        input_max = np.array([s['data_max'][0] for s in input_params['scalers']])
        
        # Create metadata
        metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'state_feature_names': state_params['feature_names'],
            'input_feature_names': input_params['feature_names'],
            'state_zero_variance': state_params['has_zero_variance'],
            'input_zero_variance': input_params['has_zero_variance'],
            'paper_reference': "Min-max scaling per variable to [0,1] (Section 2.3)"
        }
        
        # Create ScalingParameters object
        self.scaling_params = ScalingParameters(
            state_scaler=self.state_scaler,
            input_scaler=self.input_scaler,
            state_min=state_min,
            state_max=state_max,
            input_min=input_min,
            input_max=input_max,
            node_ids=self.node_ids,
            link_ids=self.link_ids,
            metadata=metadata
        )
        
        print("Updated scaling parameters with current scalers")
    
    def create_windows(self, states: np.ndarray, inputs: np.ndarray, 
                      window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows for autoregressive training.
        
        This method creates training windows where each window consists of:
        - Initial state: x_t
        - Input sequence: R_t, R_{t+1}, ..., R_{t+W-1}
        - Target sequence: x_{t+1}, x_{t+2}, ..., x_{t+W}
        
        Args:
            states: State array of shape (n_timesteps, n_nodes + n_links)
            inputs: Input array of shape (n_timesteps, n_nodes)
            window_size: Window size in timesteps. If None, uses config value.
            
        Returns:
            Tuple of (windowed_states, windowed_inputs, windowed_targets) where:
            - windowed_states: (n_windows, n_nodes + n_links) initial states
            - windowed_inputs: (n_windows, window_size, n_nodes) input sequences
            - windowed_targets: (n_windows, window_size, n_nodes + n_links) target states
            
        Raises:
            ValueError: If states and inputs have different number of timesteps
            ValueError: If window_size is too large for available data
            
        Note:
            Windows are created with stride=1 (each possible starting point).
            The paper uses window size W=60 minutes (60 timesteps at Δt=1 minute).
        """
        if window_size is None:
            window_size = self.data_config.training_window_size
        
        print(f"Creating sliding windows (window_size={window_size})...")
        
        with Timer("window_creation") as timer:
            # Validate inputs
            n_timesteps = states.shape[0]
            if inputs.shape[0] != n_timesteps:
                raise ValueError(f"States and inputs have different timesteps: "
                               f"{states.shape[0]} != {inputs.shape[0]}")
            
            # Check window size
            if window_size <= 0:
                raise ValueError(f"Window size must be positive, got {window_size}")
            
            if window_size >= n_timesteps:
                raise ValueError(f"Window size {window_size} >= timesteps {n_timesteps}")
            
            # Number of windows
            n_windows = n_timesteps - window_size
            
            if n_windows <= 0:
                raise ValueError(f"No windows can be created: window_size={window_size}, "
                               f"n_timesteps={n_timesteps}")
            
            # Initialize arrays
            windowed_states = np.zeros((n_windows, states.shape[1]), dtype=np.float32)
            windowed_inputs = np.zeros((n_windows, window_size, inputs.shape[1]), dtype=np.float32)
            windowed_targets = np.zeros((n_windows, window_size, states.shape[1]), dtype=np.float32)
            
            # Create windows
            for i in range(n_windows):
                # Initial state at window start
                windowed_states[i] = states[i]
                
                # Input sequence for the window
                windowed_inputs[i] = inputs[i:i+window_size]
                
                # Target states (next states)
                windowed_targets[i] = states[i+1:i+window_size+1]
            
            # Validate window integrity
            self._validate_windows(windowed_states, windowed_inputs, windowed_targets, 
                                 states, inputs, window_size)
            
            print(f"  Created {n_windows} windows")
            print(f"    windowed_states shape: {windowed_states.shape}")
            print(f"    windowed_inputs shape: {windowed_inputs.shape}")
            print(f"    windowed_targets shape: {windowed_targets.shape}")
            print(f"    Window overlap: {n_windows / n_timesteps * 100:.1f}% of timesteps used as window starts")
            
            return windowed_states, windowed_inputs, windowed_targets
    
    def _validate_windows(self, windowed_states: np.ndarray, windowed_inputs: np.ndarray,
                         windowed_targets: np.ndarray, original_states: np.ndarray,
                         original_inputs: np.ndarray, window_size: int) -> None:
        """
        Validate window integrity and temporal consistency.
        
        Args:
            windowed_states: Windowed initial states
            windowed_inputs: Windowed input sequences
            windowed_targets: Windowed target states
            original_states: Original state array
            original_inputs: Original input array
            window_size: Window size used
            
        Raises:
            ValueError: If windows violate temporal consistency
        """
        n_windows = windowed_states.shape[0]
        
        # Check sample window
        sample_idx = min(10, n_windows - 1)
        
        # Check that initial state matches original
        if not np.allclose(windowed_states[sample_idx], original_states[sample_idx]):
            raise ValueError(f"Window {sample_idx}: initial state doesn't match original")
        
        # Check that input sequence matches original
        expected_inputs = original_inputs[sample_idx:sample_idx+window_size]
        if not np.allclose(windowed_inputs[sample_idx], expected_inputs):
            raise ValueError(f"Window {sample_idx}: input sequence doesn't match original")
        
        # Check that target states match original (next states)
        expected_targets = original_states[sample_idx+1:sample_idx+window_size+1]
        if not np.allclose(windowed_targets[sample_idx], expected_targets):
            raise ValueError(f"Window {sample_idx}: target states don't match original")
        
        # Check that windows don't have NaN values
        if (np.any(np.isnan(windowed_states)) or 
            np.any(np.isnan(windowed_inputs)) or 
            np.any(np.isnan(windowed_targets))):
            warnings.warn("NaN values found in windowed data")
        
        # Check temporal ordering within windows
        for i in range(n_windows):
            # Check that target[0] should be the next state after initial
            expected_next = original_states[i+1]
            if not np.allclose(windowed_targets[i, 0], expected_next, rtol=1e-5, atol=1e-7):
                warnings.warn(f"Window {i}: first target doesn't match next state")
                break
        
        print(f"  Window validation passed for {n_windows} windows")
    
    def compute_excess_flows(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compute excess flows Q_w via mass balance (Eq. 8).
        
        Implements: Q_w,i = max(∑_j Q_US,i,j - ∑_k Q_DS,i,k + R_i, 0)
        
        Args:
            states: State array of shape (n_timesteps, n_nodes + n_links)
            inputs: Input array of shape (n_timesteps, n_nodes)
            
        Returns:
            Excess flow array of shape (n_timesteps, n_nodes) in m³/s
            
        Raises:
            ValueError: If states and inputs have different number of timesteps
            RuntimeError: If adjacency matrices are not available
            
        Note:
            This is a post-processing step that computes Q_w from predicted
            or actual states. The paper uses this in the constraint layer
            and includes Q_w in the loss function.
        """
        print("Computing excess flows via mass balance (Eq. 8)...")
        
        with Timer("excess_flow_computation") as timer:
            # Validate inputs
            n_timesteps = states.shape[0]
            if inputs.shape[0] != n_timesteps:
                raise ValueError(f"States and inputs have different timesteps: "
                               f"{states.shape[0]} != {inputs.shape[0]}")
            
            # Check adjacency matrices
            if self.upstream_matrix is None or self.downstream_matrix is None:
                raise RuntimeError("Adjacency matrices not available. Check network parser.")
            
            # Extract water levels and flows from states
            # states = [h_1...h_N, Q_1...Q_M]
            h_states = states[:, :self.n_nodes]  # Water levels (not used in mass balance)
            Q_flows = states[:, self.n_nodes:]   # Pipe flows
            
            # Initialize excess flow array
            Q_w = np.zeros((n_timesteps, self.n_nodes), dtype=np.float32)
            
            # Compute for each timestep
            for t in range(n_timesteps):
                # Current flows
                Q_t = Q_flows[t, :]
                
                # Compute upstream inflows: upstream_matrix * Q_t
                upstream_inflows = self.upstream_matrix.dot(Q_t)
                
                # Compute downstream outflows: downstream_matrix * Q_t
                downstream_outflows = self.downstream_matrix.dot(Q_t)
                
                # Current runoff inputs
                R_t = inputs[t, :]
                
                # Mass balance: net inflow at each node
                net_inflow = upstream_inflows - downstream_outflows + R_t
                
                # Excess flow = max(net_inflow, 0)
                Q_w[t, :] = np.maximum(net_inflow, 0.0)
            
            # Validate results
            max_Q_w = np.max(Q_w)
            total_Q_w = np.sum(Q_w)
            n_surcharge_timesteps = np.sum(Q_w > 0)
            
            print(f"  Computed excess flows: shape={Q_w.shape}")
            print(f"    Maximum excess flow: {max_Q_w:.4f} m³/s")
            print(f"    Total excess volume: {total_Q_w * 60:.2f} m³ (assuming 1-minute Δt)")
            print(f"    Surcharge timesteps: {n_surcharge_timesteps} ({n_surcharge_timesteps/n_timesteps*100:.1f}%)")
            
            # Check mass balance consistency (for debugging)
            self._validate_mass_balance(states, inputs, Q_w)
            
            return Q_w
    
    def _validate_mass_balance(self, states: np.ndarray, inputs: np.ndarray, 
                              Q_w: np.ndarray) -> None:
        """
        Validate mass balance consistency for debugging.
        
        For each node and timestep, check that:
        Inflow = Outflow + Storage change + Excess
        This is simplified since we don't have storage change readily available.
        
        Args:
            states: State array
            inputs: Input array
            Q_w: Excess flow array
            
        Note:
            This is a simplified validation. Full mass balance would require
            storage volumes which aren't in our state vector.
        """
        n_timesteps = states.shape[0]
        Q_flows = states[:, self.n_nodes:]
        
        # Compute total system imbalance at each timestep
        imbalances = []
        
        for t in range(n_timesteps):
            Q_t = Q_flows[t, :]
            R_t = inputs[t, :]
            Q_w_t = Q_w[t, :]
            
            # Total inflow to system
            total_inflow = np.sum(R_t)
            
            # Total outflow through links (net)
            # For each link, flow is positive in downstream direction
            # Sum of all flows at downstream ends minus upstream ends
            upstream_inflows = self.upstream_matrix.dot(Q_t)
            downstream_outflows = self.downstream_matrix.dot(Q_t)
            net_link_flow = np.sum(downstream_outflows - upstream_inflows)
            
            # Total excess flow
            total_excess = np.sum(Q_w_t)
            
            # Imbalance (should be near 0 for conservation)
            imbalance = total_inflow - net_link_flow - total_excess
            imbalances.append(imbalance)
        
        imbalances = np.array(imbalances)
        max_imbalance = np.max(np.abs(imbalances))
        avg_imbalance = np.mean(np.abs(imbalances))
        
        if max_imbalance > 1e-3:  # 1 L/s threshold
            warnings.warn(f"Significant mass balance imbalance detected: "
                         f"max={max_imbalance:.6f} m³/s, avg={avg_imbalance:.6f} m³/s")
        else:
            print(f"    Mass balance validation: max imbalance={max_imbalance:.6f} m³/s")
    
    def save_processed_data(self, data_dict: Dict[str, Any], 
                          filename: Optional[str] = None) -> str:
        """
        Save processed data to disk for reproducibility.
        
        Args:
            data_dict: Dictionary containing processed data arrays
            filename: Optional filename. If None, generates timestamped name.
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file cannot be written
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.pkl"
        
        filepath = self.processed_data_dir / filename
        
        try:
            save_pickle(data_dict, str(filepath))
            print(f"Processed data saved to: {filepath}")
            
            # Also save scaling parameters if available
            if self.scaling_params is not None:
                scaler_path = self.processed_data_dir / f"scaling_params_{timestamp}.pkl"
                self.scaling_params.save(str(scaler_path))
                print(f"Scaling parameters saved to: {scaler_path}")
            
            return str(filepath)
            
        except Exception as e:
            raise IOError(f"Failed to save processed data: {str(e)}") from e
    
    def load_processed_data(self, filename: str) -> Dict[str, Any]:
        """
        Load processed data from disk.
        
        Args:
            filename: Filename to load (relative to processed_data_dir)
            
        Returns:
            Dictionary containing processed data arrays
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        filepath = self.processed_data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        try:
            data_dict = load_pickle(str(filepath))
            print(f"Processed data loaded from: {filepath}")
            return data_dict
        except Exception as e:
            raise IOError(f"Failed to load processed data: {str(e)}") from e
    
    def inverse_scale_states(self, scaled_states: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled states back to physical units.
        
        Args:
            scaled_states: Scaled state array
            
        Returns:
            State array in original physical units (m for h, m³/s for Q)
            
        Raises:
            NotFittedError: If state scaler hasn't been fitted
        """
        if self.state_scaler is None:
            raise NotFittedError("State scaler not fitted. Call scale_data with fit=True first.")
        
        return self.state_scaler.inverse_transform(scaled_states)
    
    def inverse_scale_inputs(self, scaled_inputs: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled inputs back to physical units.
        
        Args:
            scaled_inputs: Scaled input array
            
        Returns:
            Input array in original physical units (m³/s for R)
            
        Raises:
            NotFittedError: If input scaler hasn't been fitted
        """
        if self.input_scaler is None:
            raise NotFittedError("Input scaler not fitted. Call scale_data with fit=True first.")
        
        return self.input_scaler.inverse_transform(scaled_inputs)
    
    def get_state_vector_info(self) -> Dict[str, Any]:
        """
        Get information about state vector composition.
        
        Returns:
            Dictionary with state vector information for debugging
        """
        return {
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'state_dimension': self.n_nodes + self.n_links,
            'node_ids': self.node_ids,
            'link_ids': self.link_ids,
            'state_order': self.node_ids + [f"Q_{lid}" for lid in self.link_ids],
            'input_order': self.node_ids,
            'paper_reference': "State vector: x_t = [h_1...h_N, Q_1...Q_M] (Eq. 7)"
        }


def validate_state_input_alignment(states: np.ndarray, inputs: np.ndarray) -> bool:
    """
    Validate that state and input arrays have same number of timesteps.
    
    Args:
        states: State array
        inputs: Input array
        
    Returns:
        True if aligned, False otherwise
        
    Note:
        Used for debugging and validation during data processing.
    """
    if states.shape[0] != inputs.shape[0]:
        print(f"ERROR: State and input arrays have different timesteps: "
              f"{states.shape[0]} != {inputs.shape[0]}")
        return False
    
    print(f"State and input arrays aligned: {states.shape[0]} timesteps")
    return True


def compute_mass_balance_excess_flows(Q_flows: np.ndarray, inputs: np.ndarray,
                                     upstream_matrix: csr_matrix,
                                     downstream_matrix: csr_matrix) -> np.ndarray:
    """
    Compute excess flows via mass balance (Eq. 8) for given flows and inputs.
    
    This is a standalone function for use outside the DataProcessor class.
    
    Args:
        Q_flows: Pipe flow array of shape (n_timesteps, n_links)
        inputs: Input runoff array of shape (n_timesteps, n_nodes)
        upstream_matrix: Sparse upstream adjacency matrix (N×M)
        downstream_matrix: Sparse downstream adjacency matrix (N×M)
        
    Returns:
        Excess flow array of shape (n_timesteps, n_nodes)
    """
    n_timesteps, n_links = Q_flows.shape
    n_nodes = inputs.shape[1]
    
    if upstream_matrix.shape != (n_nodes, n_links):
        raise ValueError(f"upstream_matrix shape {upstream_matrix.shape} "
                       f"doesn't match (n_nodes={n_nodes}, n_links={n_links})")
    
    if downstream_matrix.shape != (n_nodes, n_links):
        raise ValueError(f"downstream_matrix shape {downstream_matrix.shape} "
                       f"doesn't match (n_nodes={n_nodes}, n_links={n_links})")
    
    Q_w = np.zeros((n_timesteps, n_nodes), dtype=np.float32)
    
    for t in range(n_timesteps):
        Q_t = Q_flows[t, :]
        R_t = inputs[t, :]
        
        upstream_inflows = upstream_matrix.dot(Q_t)
        downstream_outflows = downstream_matrix.dot(Q_t)
        
        net_inflow = upstream_inflows - downstream_outflows + R_t
        Q_w[t, :] = np.maximum(net_inflow, 0.0)
    
    return Q_w


def test_data_processor():
    """
    Test function for DataProcessor.
    
    Creates a simple test to verify the processor works correctly.
    """
    print("Testing DataProcessor...")
    
    # Test with dummy network parser
    class DummyNetworkParser:
        def __init__(self):
            self.nodes_df = pd.DataFrame({
                'id': ['1', '2', '3', '4', '5', '6'],
                'type': ['junction'] * 6,
                'elevation': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                'max_depth': [2.0] * 6
            })
            self.links_df = pd.DataFrame({
                'id': ['1', '2', '3', '4', '5', '6'],
                'from_node': ['1', '2', '3', '4', '5', '6'],
                'to_node': ['2', '3', '4', '5', '6', '1'],  # Circular for testing
                'length': [100.0] * 6,
                'roughness': [0.013] * 6,
                'diameter': [0.5] * 6,
                'shape': ['CIRCULAR'] * 6,
                'area': [0.196] * 6
            })
            
            # Create dummy adjacency matrices
            n_nodes = len(self.nodes_df)
            n_links = len(self.links_df)
            
            from scipy.sparse import csr_matrix
            # Simple adjacency: each link connects node i to node i+1 (mod 6)
            upstream_data = np.ones(n_links)
            upstream_rows = list(range(n_links))
            upstream_cols = [(i + 1) % n_links for i in range(n_links)]
            
            downstream_data = np.ones(n_links)
            downstream_rows = list(range(n_links))
            downstream_cols = list(range(n_links))
            
            self.upstream_matrix = csr_matrix((upstream_data, (upstream_rows, upstream_cols)), 
                                            shape=(n_nodes, n_links))
            self.downstream_matrix = csr_matrix((downstream_data, (downstream_rows, downstream_cols)), 
                                              shape=(n_nodes, n_links))
        
        def get_node_data(self):
            return self.nodes_df
        
        def get_link_data(self):
            return self.links_df
        
        def get_adjacency_matrices(self):
            return self.upstream_matrix, self.downstream_matrix
    
    try:
        # Create dummy parser and processor
        dummy_parser = DummyNetworkParser()
        processor = DataProcessor(dummy_parser)
        
        # Test state vector creation with dummy data
        n_timesteps = 100
        n_nodes = 6
        n_links = 6
        
        # Create dummy raw data
        raw_data_dict = {}
        for i, node_id in enumerate(processor.node_ids):
            raw_data_dict[('water_level', node_id)] = np.random.uniform(100, 102, n_timesteps)
        
        for i, link_id in enumerate(processor.link_ids):
            raw_data_dict[('pipe_flow', link_id)] = np.random.uniform(-0.5, 0.5, n_timesteps)
        
        raw_data = pd.DataFrame(raw_data_dict)
        
        # Test create_state_vectors
        states = processor.create_state_vectors(raw_data)
        print(f"Test create_state_vectors: shape={states.shape}, expected=({n_timesteps}, {n_nodes+n_links})")
        assert states.shape == (n_timesteps, n_nodes + n_links)
        
        # Test create_input_vectors
        inflow_data = pd.DataFrame({
            '1': np.random.uniform(0, 0.1, n_timesteps),
            '3': np.random.uniform(0, 0.1, n_timesteps)
        })
        inputs = processor.create_input_vectors(inflow_data)
        print(f"Test create_input_vectors: shape={inputs.shape}, expected=({n_timesteps}, {n_nodes})")
        assert inputs.shape == (n_timesteps, n_nodes)
        
        # Test scaling
        scaled_states, _ = processor.scale_data(states, data_type='state', fit=True)
        scaled_inputs, _ = processor.scale_data(inputs, data_type='input', fit=True)
        print(f"Test scaling: scaled_states shape={scaled_states.shape}, range=[{scaled_states.min():.3f}, {scaled_states.max():.3f}]")
        assert np.all(scaled_states >= 0) and np.all(scaled_states <= 1)
        
        # Test window creation
        window_size = 10
        windowed_states, windowed_inputs, windowed_targets = processor.create_windows(
            scaled_states, scaled_inputs, window_size
        )
        n_windows = n_timesteps - window_size
        print(f"Test create_windows: {n_windows} windows created")
        assert windowed_states.shape == (n_windows, n_nodes + n_links)
        assert windowed_inputs.shape == (n_windows, window_size, n_nodes)
        assert windowed_targets.shape == (n_windows, window_size, n_nodes + n_links)
        
        # Test excess flow computation
        Q_w = processor.compute_excess_flows(states, inputs)
        print(f"Test compute_excess_flows: shape={Q_w.shape}, expected=({n_timesteps}, {n_nodes})")
        assert Q_w.shape == (n_timesteps, n_nodes)
        assert np.all(Q_w >= 0)
        
        # Test inverse scaling
        original_states = processor.inverse_scale_states(scaled_states)
        print(f"Test inverse_scale_states: shape={original_states.shape}, "
              f"reconstruction error={np.mean(np.abs(original_states - states)):.6f}")
        assert np.allclose(original_states, states, rtol=1e-5, atol=1e-7)
        
        # Test saving and loading
        test_data = {
            'states': states,
            'inputs': inputs,
            'scaled_states': scaled_states,
            'scaled_inputs': scaled_inputs,
            'windowed_states': windowed_states,
            'windowed_inputs': windowed_inputs,
            'windowed_targets': windowed_targets,
            'Q_w': Q_w
        }
        
        save_path = processor.save_processed_data(test_data, "test_data.pkl")
        loaded_data = processor.load_processed_data("test_data.pkl")
        
        print(f"Test save/load: keys={list(loaded_data.keys())}")
        assert set(test_data.keys()) == set(loaded_data.keys())
        
        # Clean up test file
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Cleaned up test file: {save_path}")
        
        print("\nAll DataProcessor tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_data_processor()
    if success:
        print("\nDataProcessor test completed successfully!")
    else:
        print("\nDataProcessor test failed!")
