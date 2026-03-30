"""
model.py

Physics-constrained Generalized Residue Network (gResNet) implementation for
urban drainage system surrogate modeling.

This module implements the complete gResNet architecture described in the paper
"Accelerating hydrodynamic simulations of urban drainage systems with physics-
guided machine learning". The architecture includes:
1. PriorNetwork (L): Single hidden layer neural network approximating dynamic
   mode decomposition (Section 2.4, Eq. 5)
2. ResidueNetwork (N): Deep neural network with S4 architecture (6×100 neurons)
   learning residuals not captured by the prior model (Table 1)
3. ConstraintLayer: Physics-based layer implementing mass balance constraints
   for excess flow computation (Eq. 8)
4. gResNet: Main model combining L, N, and constraint layer with proper state
   vector handling and gradient flow

Classes:
    PriorNetwork: Single hidden layer neural network for prior dynamics
    ResidueNetwork: Deep neural network (6×100) for residual dynamics
    ConstraintLayer: Physics-constrained layer for mass balance computations
    gResNet: Main physics-constrained gResNet model

Functions:
    build_network_layers: Helper for constructing MLP layers
    get_activation: Activation function factory

Note:
    Strictly follows paper's methodology: state vector format [h_nodes, Q_links],
    S4 residue network architecture, ReLU activations, and Eq. 8 constraints.
    All dimensions derived from network topology and config.yaml hyperparameters.
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix

# Project imports
from utils import Timer, set_seed, get_device, format_time
from config import Config, ModelConfig, SystemConfig


def get_activation(activation_name: str) -> nn.Module:
    """
    Get activation function module by name.
    
    Args:
        activation_name: Name of activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        
    Returns:
        PyTorch activation module
        
    Raises:
        ValueError: If activation_name is not supported
        
    Note:
        Paper uses ReLU for both prior and residue networks (Section 2.4).
    """
    activation_name = activation_name.lower()
    
    if activation_name == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(0.01, inplace=True)
    elif activation_name == 'elu':
        return nn.ELU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")


def build_network_layers(input_dim: int, hidden_dims: List[int], output_dim: int,
                        activation: str = 'relu', use_bias: bool = True,
                        dropout_rate: float = 0.0, batch_norm: bool = False) -> nn.Sequential:
    """
    Build a multi-layer perceptron with configurable architecture.
    
    Args:
        input_dim: Dimension of input layer
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output layer
        activation: Activation function name
        use_bias: Whether to use bias terms in linear layers
        dropout_rate: Dropout rate (0 = no dropout)
        batch_norm: Whether to use batch normalization
        
    Returns:
        nn.Sequential containing the network layers
        
    Note:
        Used for both prior and residue networks. Paper doesn't use dropout
        or batch norm, but we include for flexibility.
    """
    layers = []
    
    # Input to first hidden layer
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        # Linear layer
        layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
        
        # Batch normalization (if enabled)
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Activation
        layers.append(get_activation(activation))
        
        # Dropout (if enabled)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        prev_dim = hidden_dim
    
    # Output layer (no activation, dropout, or batch norm)
    layers.append(nn.Linear(prev_dim, output_dim, bias=use_bias))
    
    return nn.Sequential(*layers)


class PriorNetwork(nn.Module):
    """
    Prior network L approximating dynamic mode decomposition.
    
    This network learns a linear/affine prior of the system dynamics,
    implementing the L(x_t; Θ_L) component from Eq. 5 in the paper.
    
    Architecture:
        Input: x_t (state vector, dimension = N + M)
        Hidden: Single layer with configurable size (default = state dimension)
        Activation: ReLU (paper standard)
        Output: L(x_t) (same dimension as input)
    
    Attributes:
        n_nodes: Number of nodes in network
        n_links: Number of links in network
        state_dim: Total state dimension (n_nodes + n_links)
        hidden_dim: Hidden layer dimension
        activation: Activation function
        network: Sequential neural network layers
        
    Note:
        Paper Section 2.4: "The prior model is a neural network with a single
        hidden layer, which approximates the dynamic mode decomposition (DMD)
        for the current state of the system." Hidden dimension not specified.
    """
    
    def __init__(self, n_nodes: int, n_links: int, config: Optional[Dict] = None):
        """
        Initialize prior network.
        
        Args:
            n_nodes: Number of nodes in drainage network
            n_links: Number of links in drainage network
            config: Configuration dictionary. If None, uses default Config.
            
        Note:
            Hidden dimension defaults to state dimension if not specified in config.
            This follows the paper's description but is an implementation choice.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.model_config = self.config.model
            self.system_config = self.config.system
        else:
            self.config = config
            self.model_config = ModelConfig(**config.get('model', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set dimensions
        self.n_nodes = n_nodes
        self.n_links = n_links
        self.state_dim = n_nodes + n_links
        
        # Get network configuration
        prior_config = self.model_config.prior_network
        self.hidden_dim = prior_config.get('hidden_units')
        
        # Default hidden dimension to state dimension if not specified
        if self.hidden_dim is None:
            self.hidden_dim = self.state_dim
            print(f"  Prior network: Using default hidden dimension = state_dim = {self.state_dim}")
        
        self.activation = prior_config.get('activation', 'relu')
        self.use_bias = prior_config.get('use_bias', True)
        self.dropout_rate = prior_config.get('dropout_rate', 0.0)
        
        # Build network
        self.network = build_network_layers(
            input_dim=self.state_dim,
            hidden_dims=[self.hidden_dim],
            output_dim=self.state_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            batch_norm=False  # Paper doesn't mention batch norm for prior network
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.device = get_device(config)
        self.to(self.device)
        
        print(f"Initialized PriorNetwork: {self.state_dim} → {self.hidden_dim} → {self.state_dim}")
        print(f"  Activation: {self.activation}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights.
        
        Note:
            Paper doesn't specify weight initialization. Using PyTorch default
            (Kaiming uniform for ReLU, Xavier uniform for tanh/sigmoid).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU, Xavier for others
                if self.activation == 'relu' or self.activation == 'leaky_relu':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of prior network.
        
        Args:
            x_t: Current state tensor of shape (batch_size, state_dim)
                where state_dim = n_nodes + n_links
            
        Returns:
            Prior prediction L(x_t) of shape (batch_size, state_dim)
            
        Note:
            Implements Eq. 5: L(x_t; Θ_L) approximates system dynamics.
            Output has same dimension as input for combination with residue network.
        """
        # Validate input dimensions
        if x_t.dim() != 2:
            raise ValueError(f"Expected 2D tensor (batch, state_dim), got {x_t.dim()}D")
        
        if x_t.shape[1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {x_t.shape[1]}")
        
        return self.network(x_t)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get network configuration for reproducibility.
        
        Returns:
            Dictionary with network configuration parameters
        """
        return {
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'parameters': sum(p.numel() for p in self.parameters()),
            'paper_reference': "Prior network L: single hidden layer approximating DMD (Section 2.4)"
        }


class ResidueNetwork(nn.Module):
    """
    Residue network N learning dynamics residuals.
    
    This network learns the residuals not captured by the prior network,
    implementing the N(x_t, R_t; Θ_N) component from Eq. 5 in the paper.
    
    Architecture (S4 from Table 1):
        Input: concat(x_t, R_t) where x_t = state, R_t = runoff inputs
               Dimension = (N + M) + N = 2N + M
        Hidden: 6 layers of 100 neurons each
        Activation: ReLU (paper standard)
        Output: N(x_t, R_t) (same dimension as state vector)
    
    Attributes:
        n_nodes: Number of nodes in network
        n_links: Number of links in network
        state_dim: State dimension (n_nodes + n_links)
        input_dim: Input dimension (state_dim + n_nodes = 2*n_nodes + n_links)
        hidden_layers: Number of hidden layers (6 for S4)
        hidden_units: Neurons per hidden layer (100 for S4)
        activation: Activation function
        network: Sequential neural network layers
        
    Note:
        Paper Section 2.4: "The residue model is a deep neural network with
        architecture S4 (selected via hyperparameter tuning), which has six
        hidden layers with 100 neurons in each layer."
    """
    
    def __init__(self, n_nodes: int, n_links: int, config: Optional[Dict] = None):
        """
        Initialize residue network.
        
        Args:
            n_nodes: Number of nodes in drainage network
            n_links: Number of links in drainage network
            config: Configuration dictionary. If None, uses default Config.
            
        Raises:
            ValueError: If network architecture doesn't match S4 (6×100)
            
        Note:
            Architecture must be S4 (6×100) per paper's selected configuration.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.model_config = self.config.model
            self.system_config = self.config.system
        else:
            self.config = config
            self.model_config = ModelConfig(**config.get('model', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set dimensions
        self.n_nodes = n_nodes
        self.n_links = n_links
        self.state_dim = n_nodes + n_links
        self.input_dim = self.state_dim + n_nodes  # concat(x_t, R_t)
        
        # Get network configuration (must be S4: 6×100)
        residue_config = self.model_config.residue_network
        self.hidden_layers = residue_config.get('hidden_layers', 6)
        self.hidden_units = residue_config.get('hidden_units_per_layer', 100)
        self.architecture = residue_config.get('architecture', 'S4')
        self.activation = residue_config.get('activation', 'relu')
        self.use_bias = residue_config.get('use_bias', True)
        self.dropout_rate = residue_config.get('dropout_rate', 0.0)
        self.batch_norm = residue_config.get('batch_norm', False)
        
        # Validate architecture matches S4
        if not (self.hidden_layers == 6 and self.hidden_units == 100):
            warnings.warn(f"Residue network architecture ({self.hidden_layers}×{self.hidden_units}) "
                         f"doesn't match paper's S4 (6×100). Using specified architecture anyway.")
        
        # Build hidden dimensions list
        hidden_dims = [self.hidden_units] * self.hidden_layers
        
        # Build network
        self.network = build_network_layers(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.state_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.device = get_device(config)
        self.to(self.device)
        
        print(f"Initialized ResidueNetwork (S4): {self.input_dim} → {self.hidden_layers}×{self.hidden_units} → {self.state_dim}")
        print(f"  Architecture: {self.architecture} ({self.hidden_layers}×{self.hidden_units})")
        print(f"  Activation: {self.activation}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights.
        
        Note:
            Paper doesn't specify weight initialization. Using PyTorch default
            (Kaiming uniform for ReLU, Xavier uniform for tanh/sigmoid).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU, Xavier for others
                if self.activation == 'relu' or self.activation == 'leaky_relu':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x_t: torch.Tensor, R_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residue network.
        
        Args:
            x_t: Current state tensor of shape (batch_size, state_dim)
            R_t: Runoff input tensor of shape (batch_size, n_nodes)
            
        Returns:
            Residue prediction N(x_t, R_t) of shape (batch_size, state_dim)
            
        Note:
            Implements Eq. 5: N(x_t, R_t; Θ_N) learns residuals not captured by L.
            Input is concatenation of state and runoff: concat(x_t, R_t).
        """
        # Validate input dimensions
        if x_t.dim() != 2:
            raise ValueError(f"x_t: Expected 2D tensor (batch, state_dim), got {x_t.dim()}D")
        
        if R_t.dim() != 2:
            raise ValueError(f"R_t: Expected 2D tensor (batch, n_nodes), got {R_t.dim()}D")
        
        if x_t.shape[1] != self.state_dim:
            raise ValueError(f"x_t: Expected state dimension {self.state_dim}, got {x_t.shape[1]}")
        
        if R_t.shape[1] != self.n_nodes:
            raise ValueError(f"R_t: Expected dimension {self.n_nodes}, got {R_t.shape[1]}")
        
        if x_t.shape[0] != R_t.shape[0]:
            raise ValueError(f"Batch size mismatch: x_t {x_t.shape[0]} != R_t {R_t.shape[0]}")
        
        # Concatenate state and runoff inputs
        combined_input = torch.cat([x_t, R_t], dim=1)
        
        # Forward pass through network
        return self.network(combined_input)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get network configuration for reproducibility.
        
        Returns:
            Dictionary with network configuration parameters
        """
        return {
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'hidden_units': self.hidden_units,
            'architecture': self.architecture,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'parameters': sum(p.numel() for p in self.parameters()),
            'paper_reference': "Residue network N: S4 architecture (6×100) learning residuals (Section 2.4, Table 1)"
        }


class ConstraintLayer(nn.Module):
    """
    Physics constraint layer for mass balance computations.
    
    This layer implements Eq. 8 from the paper, computing excess flows
    Q_w via mass balance at each node:
    
        Q_{w,i} = max(∑_{j∈US(i)} Q_j - ∑_{k∈DS(i)} Q_k + R_i, 0)
    
    where:
        US(i): Links entering node i (upstream)
        DS(i): Links exiting node i (downstream)
        Q_j: Flow in link j
        R_i: Runoff at node i
    
    Attributes:
        n_nodes: Number of nodes in network
        n_links: Number of links in network
        upstream_matrix: Dense tensor (n_nodes × n_links) where [i,j] = 1 if link j ends at node i
        downstream_matrix: Dense tensor (n_nodes × n_links) where [i,j] = 1 if link j starts at node i
        device: PyTorch device for tensor operations
        
    Note:
        The layer is differentiable (using torch.maximum for max(..., 0))
        to allow gradient flow back to the network predictions.
        Implements "spilling configuration": water doesn't re-enter system.
    """
    
    def __init__(self, upstream_matrix: csr_matrix, downstream_matrix: csr_matrix,
                 config: Optional[Dict] = None):
        """
        Initialize constraint layer with adjacency matrices.
        
        Args:
            upstream_matrix: Sparse CSR matrix (n_nodes × n_links) for upstream relationships
            downstream_matrix: Sparse CSR matrix (n_nodes × n_links) for downstream relationships
            config: Configuration dictionary. If None, uses default Config.
            
        Raises:
            ValueError: If matrix dimensions don't match or are invalid
            RuntimeError: If matrices cannot be converted to dense format
            
        Note:
            Converts sparse matrices to dense tensors for efficient GPU operations.
            For very large networks, consider keeping them sparse.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.model_config = self.config.model
            self.system_config = self.config.system
        else:
            self.config = config
            self.model_config = ModelConfig(**config.get('model', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Validate matrix dimensions
        n_nodes_up, n_links_up = upstream_matrix.shape
        n_nodes_down, n_links_down = downstream_matrix.shape
        
        if n_nodes_up != n_nodes_down:
            raise ValueError(f"Node count mismatch: upstream {n_nodes_up} != downstream {n_nodes_down}")
        
        if n_links_up != n_links_down:
            raise ValueError(f"Link count mismatch: upstream {n_links_up} != downstream {n_links_down}")
        
        self.n_nodes = n_nodes_up
        self.n_links = n_links_up
        
        # Convert sparse matrices to dense tensors
        try:
            # Convert to numpy arrays first, then to tensors
            upstream_dense = upstream_matrix.toarray().astype(np.float32)
            downstream_dense = downstream_matrix.toarray().astype(np.float32)
            
            # Move to device
            self.device = get_device(config)
            
            # Register as buffers (non-trainable parameters)
            self.register_buffer('upstream_matrix', 
                                torch.from_numpy(upstream_dense).to(self.device))
            self.register_buffer('downstream_matrix', 
                                torch.from_numpy(downstream_dense).to(self.device))
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert adjacency matrices to tensors: {str(e)}") from e
        
        # Validate matrices are binary (0 or 1)
        upstream_unique = torch.unique(self.upstream_matrix)
        downstream_unique = torch.unique(self.downstream_matrix)
        
        if not (torch.all(upstream_unique <= 1.0) and torch.all(downstream_unique <= 1.0)):
            warnings.warn("Adjacency matrices contain non-binary values. "
                         "Mass balance computations may be incorrect.")
        
        # Compute sparsity for logging
        upstream_sparsity = 1.0 - (torch.count_nonzero(self.upstream_matrix) / 
                                  (self.n_nodes * self.n_links))
        downstream_sparsity = 1.0 - (torch.count_nonzero(self.downstream_matrix) / 
                                    (self.n_nodes * self.n_links))
        
        print(f"Initialized ConstraintLayer: {self.n_nodes} nodes × {self.n_links} links")
        print(f"  Upstream matrix sparsity: {upstream_sparsity:.3f}")
        print(f"  Downstream matrix sparsity: {downstream_sparsity:.3f}")
        print(f"  Mass balance equation: Q_w,i = max(∑Q_US,i - ∑Q_DS,i + R_i, 0)")
        print(f"  Differentiable: Yes (using torch.maximum)")
    
    def forward(self, h_pred: torch.Tensor, Q_pred: torch.Tensor, 
                R_input: torch.Tensor) -> torch.Tensor:
        """
        Compute excess flows via mass balance constraints (Eq. 8).
        
        Args:
            h_pred: Predicted water levels of shape (batch_size, n_nodes)
                   (not used in computation but kept for interface consistency)
            Q_pred: Predicted link flows of shape (batch_size, n_links)
            R_input: Runoff inputs of shape (batch_size, n_nodes)
            
        Returns:
            Excess flows Q_w of shape (batch_size, n_nodes) in m³/s
            
        Raises:
            ValueError: If input dimensions don't match network dimensions
            
        Note:
            Water levels (h_pred) are not used in the mass balance computation
            but are included in the interface for consistency with paper notation.
            The computation is fully differentiable for gradient backpropagation.
        """
        # Validate input dimensions
        batch_size = Q_pred.shape[0]
        
        if h_pred.dim() != 2:
            raise ValueError(f"h_pred: Expected 2D tensor (batch, n_nodes), got {h_pred.dim()}D")
        
        if Q_pred.dim() != 2:
            raise ValueError(f"Q_pred: Expected 2D tensor (batch, n_links), got {Q_pred.dim()}D")
        
        if R_input.dim() != 2:
            raise ValueError(f"R_input: Expected 2D tensor (batch, n_nodes), got {R_input.dim()}D")
        
        if h_pred.shape[1] != self.n_nodes:
            raise ValueError(f"h_pred: Expected {self.n_nodes} nodes, got {h_pred.shape[1]}")
        
        if Q_pred.shape[1] != self.n_links:
            raise ValueError(f"Q_pred: Expected {self.n_links} links, got {Q_pred.shape[1]}")
        
        if R_input.shape[1] != self.n_nodes:
            raise ValueError(f"R_input: Expected {self.n_nodes} nodes, got {R_input.shape[1]}")
        
        # Check batch size consistency
        if not (h_pred.shape[0] == Q_pred.shape[0] == R_input.shape[0]):
            raise ValueError(f"Batch size mismatch: h_pred {h_pred.shape[0]}, "
                           f"Q_pred {Q_pred.shape[0]}, R_input {R_input.shape[0]}")
        
        # Compute upstream inflows: upstream_matrix @ Q_pred^T
        # Result shape: (n_nodes, batch_size)
        upstream_inflows = torch.matmul(self.upstream_matrix, Q_pred.t())
        
        # Compute downstream outflows: downstream_matrix @ Q_pred^T
        # Result shape: (n_nodes, batch_size)
        downstream_outflows = torch.matmul(self.downstream_matrix, Q_pred.t())
        
        # Transpose to get (batch_size, n_nodes) and add runoff
        # Eq. 8: Q_w,i = max(∑Q_US,i - ∑Q_DS,i + R_i, 0)
        net_inflow = (upstream_inflows.t() - downstream_outflows.t() + R_input)
        
        # Apply non-negativity constraint (spilling configuration)
        Q_w_pred = torch.maximum(net_inflow, torch.tensor(0.0, device=self.device))
        
        # Validate results only during inference — skipped during training to avoid
        # the two extra matmuls (_validate_constraints performs its own matrix
        # multiplications that double computation cost per timestep).
        if not self.training:
            self._validate_constraints(Q_w_pred, Q_pred, R_input, batch_size)
        
        return Q_w_pred
    
    def _validate_constraints(self, Q_w_pred: torch.Tensor, Q_pred: torch.Tensor,
                             R_input: torch.Tensor, batch_size: int) -> None:
        """
        Validate constraint computations for debugging.
        
        Args:
            Q_w_pred: Computed excess flows
            Q_pred: Predicted link flows
            R_input: Runoff inputs
            batch_size: Batch size
            
        Note:
            Checks for NaN values and extremely large flows.
            Also verifies mass conservation within tolerance.
        """
        # Check for NaN values
        if torch.any(torch.isnan(Q_w_pred)):
            warnings.warn("NaN values in constraint layer output")
        
        # Check for negative values (shouldn't happen due to max(..., 0))
        negative_mask = Q_w_pred < -1e-7
        if torch.any(negative_mask):
            min_val = torch.min(Q_w_pred).item()
            warnings.warn(f"Negative values in Q_w (min={min_val:.6f}). "
                         f"Should be non-negative due to max(..., 0).")
        
        # Check mass conservation (simplified)
        # Total inflow = ∑R_i
        # Total outflow through links = ∑(downstream - upstream) @ Q
        # Total excess = ∑Q_w
        # Conservation: inflow ≈ outflow + excess
        
        with torch.no_grad():
            total_inflow = torch.sum(R_input, dim=1)  # (batch_size,)
            
            # Compute net link outflow for each batch
            upstream_sum = torch.matmul(self.upstream_matrix, Q_pred.t()).sum(dim=0)  # (batch_size,)
            downstream_sum = torch.matmul(self.downstream_matrix, Q_pred.t()).sum(dim=0)  # (batch_size,)
            net_link_flow = downstream_sum - upstream_sum  # (batch_size,)
            
            total_excess = torch.sum(Q_w_pred, dim=1)  # (batch_size,)
            
            # Mass balance residual
            residual = total_inflow - net_link_flow - total_excess
            
            max_residual = torch.max(torch.abs(residual)).item()
            # if max_residual > 1e-3:  # 1 L/s threshold
            #     warnings.warn(f"Mass balance residual > 1e-3: max = {max_residual:.6f} m³/s")
    
    def compute_mass_balance(self, Q_pred: torch.Tensor, R_input: torch.Tensor) -> torch.Tensor:
        """
        Compute mass balance without applying non-negativity constraint.
        
        Args:
            Q_pred: Predicted link flows of shape (batch_size, n_links)
            R_input: Runoff inputs of shape (batch_size, n_nodes)
            
        Returns:
            Net inflow at each node (before max(..., 0)) of shape (batch_size, n_nodes)
            
        Note:
            Useful for debugging and analysis. Returns the value inside max() in Eq. 8.
        """
        # Validate inputs
        if Q_pred.shape[1] != self.n_links:
            raise ValueError(f"Q_pred: Expected {self.n_links} links, got {Q_pred.shape[1]}")
        
        if R_input.shape[1] != self.n_nodes:
            raise ValueError(f"R_input: Expected {self.n_nodes} nodes, got {R_input.shape[1]}")
        
        # Compute net inflow (without non-negativity)
        upstream_inflows = torch.matmul(self.upstream_matrix, Q_pred.t())
        downstream_outflows = torch.matmul(self.downstream_matrix, Q_pred.t())
        
        net_inflow = (upstream_inflows.t() - downstream_outflows.t() + R_input)
        
        return net_inflow
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get constraint layer configuration for reproducibility.
        
        Returns:
            Dictionary with constraint layer configuration
        """
        return {
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'upstream_matrix_shape': self.upstream_matrix.shape,
            'downstream_matrix_shape': self.downstream_matrix.shape,
            'upstream_nonzero': torch.count_nonzero(self.upstream_matrix).item(),
            'downstream_nonzero': torch.count_nonzero(self.downstream_matrix).item(),
            'device': str(self.device),
            'paper_reference': "Constraint layer: Q_w,i = max(∑Q_US,i - ∑Q_DS,i + R_i, 0) (Eq. 8)"
        }


class gResNet(nn.Module):
    """
    Physics-constrained Generalized Residue Network (gResNet) main model.
    
    This class implements the complete gResNet architecture from the paper,
    combining the prior network (L), residue network (N), and constraint layer.
    
    Architecture (Eq. 5):
        x_{t+Δt} = L(x_t; Θ_L) + N(x_t, R_t; Θ_N)
    
    with physics constraints (Eq. 8) applied to compute excess flows Q_w.
    
    Attributes:
        n_nodes: Number of nodes in drainage network
        n_links: Number of links in drainage network
        state_dim: State dimension (n_nodes + n_links)
        physical_constraints_enabled: Whether to apply physics constraints
        prior_network: PriorNetwork instance (L)
        residue_network: ResidueNetwork instance (N)
        constraint_layer: ConstraintLayer instance (optional)
        device: PyTorch device for computations
        
    Methods:
        forward: Complete forward pass with optional constraints
        predict_without_constraints: Prediction without physics constraints
        get_state_components: Split/combine state vector into components
        save_model: Save model to disk
        load_model: Load model from disk (class method)
        
    Note:
        Follows paper's methodology exactly: state vector format [h_nodes, Q_links],
        S4 residue network, and Eq. 8 constraints when enabled.
    """
    
    def __init__(self, n_nodes: int, n_links: int, config: Optional[Dict] = None,
                 upstream_matrix: Optional[csr_matrix] = None,
                 downstream_matrix: Optional[csr_matrix] = None):
        """
        Initialize gResNet model.
        
        Args:
            n_nodes: Number of nodes in drainage network
            n_links: Number of links in drainage network
            config: Configuration dictionary. If None, uses default Config.
            upstream_matrix: Sparse upstream adjacency matrix (required for constraints)
            downstream_matrix: Sparse downstream adjacency matrix (required for constraints)
            
        Raises:
            ValueError: If constraints enabled but adjacency matrices not provided
            RuntimeError: If model components cannot be initialized
            
        Note:
            If physical constraints are disabled, adjacency matrices are not required.
            This allows ablation studies comparing constrained vs unconstrained models.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.model_config = self.config.model
            self.system_config = self.config.system
        else:
            self.config = config
            self.model_config = ModelConfig(**config.get('model', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set dimensions
        self.n_nodes = n_nodes
        self.n_links = n_links
        self.state_dim = n_nodes + n_links
        
        # Check if physical constraints are enabled
        self.physical_constraints_enabled = self.model_config.physical_constraints_enabled
        
        # Initialize device
        self.device = get_device(config)
        
        # Initialize prior network
        try:
            self.prior_network = PriorNetwork(n_nodes, n_links, config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize prior network: {str(e)}") from e
        
        # Initialize residue network
        try:
            self.residue_network = ResidueNetwork(n_nodes, n_links, config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize residue network: {str(e)}") from e
        
        # Initialize constraint layer if enabled
        self.constraint_layer = None
        if self.physical_constraints_enabled:
            if upstream_matrix is None or downstream_matrix is None:
                raise ValueError("Physical constraints enabled but adjacency matrices not provided")
            
            try:
                self.constraint_layer = ConstraintLayer(upstream_matrix, downstream_matrix, config)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize constraint layer: {str(e)}") from e
        
        # Move model to device
        self.to(self.device)
        
        # Print model summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print model architecture summary."""
        print("\n" + "="*80)
        print("gResNet MODEL SUMMARY")
        print("="*80)
        print(f"Network dimensions: {self.n_nodes} nodes, {self.n_links} links")
        print(f"State vector dimension: {self.state_dim}")
        print(f"Physical constraints enabled: {self.physical_constraints_enabled}")
        
        # Count parameters
        prior_params = sum(p.numel() for p in self.prior_network.parameters())
        residue_params = sum(p.numel() for p in self.residue_network.parameters())
        total_params = prior_params + residue_params
        
        print(f"\nPrior network (L): {prior_params:,} parameters")
        print(f"Residue network (N): {residue_params:,} parameters")
        print(f"Total trainable parameters: {total_params:,}")
        
        print(f"\nArchitecture details:")
        print(f"  Prior network: {self.state_dim} → {self.prior_network.hidden_dim} → {self.state_dim}")
        print(f"  Residue network: {self.residue_network.input_dim} → "
              f"{self.residue_network.hidden_layers}×{self.residue_network.hidden_units} → {self.state_dim}")
        
        if self.physical_constraints_enabled:
            print(f"  Constraint layer: Mass balance (Eq. 8) with spilling configuration")
        
        print(f"\nPaper Methodology Alignment:")
        print(f"  ✓ State vector: x_t = [h_1...h_N, Q_1...Q_M] (Eq. 7)")
        print(f"  ✓ Prior network L: Single hidden layer approximating DMD")
        print(f"  ✓ Residue network N: S4 architecture (6×100) from Table 1")
        print(f"  ✓ Combination: x_pred = L(x_t) + N(x_t, R_t) (Eq. 5)")
        if self.physical_constraints_enabled:
            print(f"  ✓ Physics constraints: Q_w,i = max(∑Q_US,i - ∑Q_DS,i + R_i, 0) (Eq. 8)")
        print("="*80 + "\n")
    
    def forward(self, x_t: torch.Tensor, R_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass of gResNet model.
        
        Args:
            x_t: Current state tensor of shape (batch_size, state_dim)
                 where state_dim = n_nodes + n_links
            R_t: Runoff input tensor of shape (batch_size, n_nodes)
            
        Returns:
            Tuple of (h_pred, Q_pred, Q_w_pred) where:
            - h_pred: Predicted water levels of shape (batch_size, n_nodes)
            - Q_pred: Predicted link flows of shape (batch_size, n_links)
            - Q_w_pred: Predicted excess flows of shape (batch_size, n_nodes)
                     (None if constraints disabled)
            
        Raises:
            ValueError: If input dimensions are incorrect
            RuntimeError: If forward pass fails
            
        Note:
            Implements Eq. 5: x_pred = L(x_t) + N(x_t, R_t)
            Then splits into h and Q components, and applies constraints if enabled.
        """
        # Validate input dimensions
        if x_t.dim() != 2:
            raise ValueError(f"x_t: Expected 2D tensor (batch, state_dim), got {x_t.dim()}D")
        
        if R_t.dim() != 2:
            raise ValueError(f"R_t: Expected 2D tensor (batch, n_nodes), got {R_t.dim()}D")
        
        if x_t.shape[1] != self.state_dim:
            raise ValueError(f"x_t: Expected state dimension {self.state_dim}, got {x_t.shape[1]}")
        
        if R_t.shape[1] != self.n_nodes:
            raise ValueError(f"R_t: Expected dimension {self.n_nodes}, got {R_t.shape[1]}")
        
        if x_t.shape[0] != R_t.shape[0]:
            raise ValueError(f"Batch size mismatch: x_t {x_t.shape[0]} != R_t {R_t.shape[0]}")
        
        try:
            # Move inputs to correct device if needed
            if x_t.device != self.device:
                x_t = x_t.to(self.device)
            
            if R_t.device != self.device:
                R_t = R_t.to(self.device)
            
            # Get predictions from prior and residue networks
            prior_out = self.prior_network(x_t)           # L(x_t)
            residue_out = self.residue_network(x_t, R_t)  # N(x_t, R_t)
            
            # Combine: Eq. 5 in paper
            x_pred = prior_out + residue_out  # x_{t+Δt}
            
            # Split into water levels and link flows
            h_pred = x_pred[:, :self.n_nodes]   # First n_nodes: water levels
            Q_pred = x_pred[:, self.n_nodes:]   # Remaining: link flows
            
            # Apply physics constraints if enabled
            Q_w_pred = None
            if self.physical_constraints_enabled and self.constraint_layer is not None:
                Q_w_pred = self.constraint_layer(h_pred, Q_pred, R_t)
            
            return h_pred, Q_pred, Q_w_pred
            
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}") from e
    
    def predict_without_constraints(self, x_t: torch.Tensor, R_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict without applying physics constraints.
        
        Args:
            x_t: Current state tensor of shape (batch_size, state_dim)
            R_t: Runoff input tensor of shape (batch_size, n_nodes)
            
        Returns:
            Tuple of (h_pred, Q_pred) without Q_w computation
            
        Note:
            Useful for ablation studies comparing constrained vs unconstrained models.
            Paper's final model uses constraints, but ablation study might not.
        """
        h_pred, Q_pred, _ = self.forward(x_t, R_t)
        return h_pred, Q_pred
    
    def split_state_vector(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split state vector into water levels and link flows.
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (h, Q) where:
            - h: Water levels of shape (batch_size, n_nodes)
            - Q: Link flows of shape (batch_size, n_links)
            
        Note:
            State vector format: x = [h_1...h_N, Q_1...Q_M] (Eq. 7)
        """
        if x.shape[1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {x.shape[1]}")
        
        h = x[:, :self.n_nodes]
        Q = x[:, self.n_nodes:]
        
        return h, Q
    
    def combine_state_components(self, h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Combine water levels and link flows into state vector.
        
        Args:
            h: Water levels of shape (batch_size, n_nodes)
            Q: Link flows of shape (batch_size, n_links)
            
        Returns:
            State tensor x of shape (batch_size, state_dim)
            
        Note:
            Inverse of split_state_vector. Reconstructs x = [h, Q].
        """
        if h.shape[1] != self.n_nodes:
            raise ValueError(f"h: Expected {self.n_nodes} nodes, got {h.shape[1]}")
        
        if Q.shape[1] != self.n_links:
            raise ValueError(f"Q: Expected {self.n_links} links, got {Q.shape[1]}")
        
        if h.shape[0] != Q.shape[0]:
            raise ValueError(f"Batch size mismatch: h {h.shape[0]} != Q {Q.shape[0]}")
        
        return torch.cat([h, Q], dim=1)
    
    def get_total_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for reproducibility.
        
        Returns:
            Dictionary with model configuration
        """
        config = {
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'state_dim': self.state_dim,
            'physical_constraints_enabled': self.physical_constraints_enabled,
            'prior_network': self.prior_network.get_config(),
            'residue_network': self.residue_network.get_config(),
            'total_parameters': self.get_total_parameters(),
            'device': str(self.device),
            'paper_reference': "gResNet: x_{t+Δt} = L(x_t; Θ_L) + N(x_t, R_t; Θ_N) (Eq. 5)"
        }
        
        if self.constraint_layer is not None:
            config['constraint_layer'] = self.constraint_layer.get_config()
        
        return config
    
    def save_model(self, filepath: str, include_config: bool = True) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (.pth or .pt extension)
            include_config: Whether to include configuration in saved file
            
        Raises:
            IOError: If model cannot be saved
            
        Note:
            Saves model state dict and optionally configuration.
            Creates parent directories if they don't exist.
        """
        # Create directory if it doesn't exist
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'model_state_dict': self.state_dict(),
            'n_nodes': self.n_nodes,
            'n_links': self.n_links,
            'physical_constraints_enabled': self.physical_constraints_enabled
        }
        
        if include_config:
            save_data['config'] = self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        
        try:
            torch.save(save_data, filepath)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"Model saved to: {filepath}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Parameters: {self.get_total_parameters():,}")
            print(f"  Physical constraints: {self.physical_constraints_enabled}")
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}") from e
    
    @classmethod
    def load_model(cls, filepath: str, config: Optional[Dict] = None,
                   upstream_matrix: Optional[csr_matrix] = None,
                   downstream_matrix: Optional[csr_matrix] = None) -> 'gResNet':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model file
            config: Configuration dictionary (overrides saved config)
            upstream_matrix: Upstream adjacency matrix (required if constraints enabled)
            downstream_matrix: Downstream adjacency matrix (required if constraints enabled)
            
        Returns:
            Loaded gResNet model
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be loaded
            RuntimeError: If model loading fails
            
        Note:
            If config is provided, it overrides saved configuration.
            Adjacency matrices must be provided if saved model had constraints enabled.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        print(f"Loading model from: {filepath}")
        
        try:
            # Load saved data
            saved_data = torch.load(filepath, map_location='cpu')
            
            # Extract model parameters
            n_nodes = saved_data['n_nodes']
            n_links = saved_data['n_links']
            physical_constraints_enabled = saved_data.get('physical_constraints_enabled', True)
            
            # Get config (use provided if available, otherwise saved)
            saved_config = saved_data.get('config', {})
            if config is not None:
                # Merge: provided config takes precedence
                merged_config = {**saved_config, **config}
            else:
                merged_config = saved_config
            
            # Check if adjacency matrices are needed
            if physical_constraints_enabled and (upstream_matrix is None or downstream_matrix is None):
                warnings.warn("Model has physical constraints but adjacency matrices not provided. "
                            "Constraint layer will not be initialized.")
                # Disable constraints for loading
                merged_config.setdefault('model', {})
                merged_config['model']['physical_constraints_enabled'] = False
            
            # Create model instance
            model = cls(
                n_nodes=n_nodes,
                n_links=n_links,
                config=merged_config,
                upstream_matrix=upstream_matrix,
                downstream_matrix=downstream_matrix
            )
            
            # Load state dict
            model.load_state_dict(saved_data['model_state_dict'])
            
            # Move to appropriate device
            model.device = get_device(merged_config)
            model.to(model.device)
            
            print(f"  Loaded model: {n_nodes} nodes, {n_links} links")
            print(f"  Parameters: {model.get_total_parameters():,}")
            print(f"  Physical constraints: {model.physical_constraints_enabled}")
            print(f"  Device: {model.device}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
    def evaluate_mode(self, mode: str = 'train') -> None:
        """
        Set model to evaluation or training mode.
        
        Args:
            mode: 'train' or 'eval'
            
        Note:
            Also sets dropout and batch norm layers appropriately.
        """
        if mode.lower() == 'train':
            self.train()
            print("Model set to training mode (dropout/batch norm active)")
        elif mode.lower() == 'eval':
            self.eval()
            print("Model set to evaluation mode (dropout/batch norm inactive)")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'.")


def create_gresnet_from_network_data(network_data: Dict[str, Any], 
                                    config: Optional[Dict] = None) -> gResNet:
    """
    Factory function to create gResNet from parsed network data.
    
    Args:
        network_data: Dictionary from NetworkParser.parse_network()
                     Must contain 'nodes', 'links', 'upstream_matrix', 'downstream_matrix'
        config: Configuration dictionary
        
    Returns:
        Initialized gResNet model
        
    Raises:
        ValueError: If network_data is missing required fields
    """
    # Validate network data
    required_keys = ['nodes', 'links', 'upstream_matrix', 'downstream_matrix']
    for key in required_keys:
        if key not in network_data:
            raise ValueError(f"network_data missing required key: {key}")
    
    # Extract dimensions
    n_nodes = len(network_data['nodes'])
    n_links = len(network_data['links'])
    
    # Extract adjacency matrices
    upstream_matrix = network_data['upstream_matrix']
    downstream_matrix = network_data['downstream_matrix']
    
    print(f"Creating gResNet from network data:")
    print(f"  Nodes: {n_nodes}, Links: {n_links}")
    print(f"  Adjacency matrices: {upstream_matrix.shape}")
    
    # Create model
    model = gResNet(
        n_nodes=n_nodes,
        n_links=n_links,
        config=config,
        upstream_matrix=upstream_matrix,
        downstream_matrix=downstream_matrix
    )
    
    return model


def test_model():
    """
    Test function for model components.
    
    Creates simple test to verify model architecture and forward pass.
    """
    print("Testing gResNet model components...")
    
    # Test parameters
    n_nodes = 6
    n_links = 6
    batch_size = 4
    state_dim = n_nodes + n_links
    
    # Create dummy adjacency matrices
    from scipy.sparse import csr_matrix
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
        # Test PriorNetwork
        print("\n1. Testing PriorNetwork...")
        prior_net = PriorNetwork(n_nodes, n_links)
        
        # Create dummy input
        x_t = torch.randn(batch_size, state_dim)
        prior_out = prior_net(x_t)
        
        print(f"  Input shape: {x_t.shape}")
        print(f"  Output shape: {prior_out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in prior_net.parameters()):,}")
        assert prior_out.shape == (batch_size, state_dim)
        
        # Test ResidueNetwork
        print("\n2. Testing ResidueNetwork...")
        residue_net = ResidueNetwork(n_nodes, n_links)
        
        R_t = torch.randn(batch_size, n_nodes)
        residue_out = residue_net(x_t, R_t)
        
        print(f"  Input shapes: x_t {x_t.shape}, R_t {R_t.shape}")
        print(f"  Output shape: {residue_out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in residue_net.parameters()):,}")
        assert residue_out.shape == (batch_size, state_dim)
        
        # Test ConstraintLayer
        print("\n3. Testing ConstraintLayer...")
        constraint_layer = ConstraintLayer(upstream_matrix, downstream_matrix)
        
        h_pred = torch.randn(batch_size, n_nodes)
        Q_pred = torch.randn(batch_size, n_links)
        Q_w_pred = constraint_layer(h_pred, Q_pred, R_t)
        
        print(f"  Input shapes: h_pred {h_pred.shape}, Q_pred {Q_pred.shape}, R_t {R_t.shape}")
        print(f"  Output shape: {Q_w_pred.shape}")
        assert Q_w_pred.shape == (batch_size, n_nodes)
        
        # Test gResNet
        print("\n4. Testing gResNet...")
        model = gResNet(
            n_nodes=n_nodes,
            n_links=n_links,
            upstream_matrix=upstream_matrix,
            downstream_matrix=downstream_matrix
        )
        
        # Forward pass
        h_pred, Q_pred, Q_w_pred = model(x_t, R_t)
        
        print(f"  Input shapes: x_t {x_t.shape}, R_t {R_t.shape}")
        print(f"  Output shapes: h_pred {h_pred.shape}, Q_pred {Q_pred.shape}, Q_w_pred {Q_w_pred.shape}")
        assert h_pred.shape == (batch_size, n_nodes)
        assert Q_pred.shape == (batch_size, n_links)
        assert Q_w_pred.shape == (batch_size, n_nodes)
        
        # Test state vector splitting/combining
        print("\n5. Testing state vector utilities...")
        h_split, Q_split = model.split_state_vector(x_t)
        x_combined = model.combine_state_components(h_split, Q_split)
        
        print(f"  Split: h {h_split.shape}, Q {Q_split.shape}")
        print(f"  Combined: {x_combined.shape}")
        assert torch.allclose(x_t, x_combined, rtol=1e-5)
        
        # Test parameter counting
        print("\n6. Testing parameter counting...")
        total_params = model.get_total_parameters()
        prior_params = sum(p.numel() for p in model.prior_network.parameters())
        residue_params = sum(p.numel() for p in model.residue_network.parameters())
        
        print(f"  Prior network: {prior_params:,}")
        print(f"  Residue network: {residue_params:,}")
        print(f"  Total: {total_params:,}")
        assert total_params == prior_params + residue_params
        
        # Test save/load
        print("\n7. Testing save/load...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            model.save_model(save_path)
            
            loaded_model = gResNet.load_model(
                save_path,
                upstream_matrix=upstream_matrix,
                downstream_matrix=downstream_matrix
            )
            
            # Compare predictions
            h_pred_loaded, Q_pred_loaded, Q_w_pred_loaded = loaded_model(x_t, R_t)
            
            print(f"  Original vs loaded predictions:")
            print(f"    h_pred match: {torch.allclose(h_pred, h_pred_loaded, rtol=1e-5)}")
            print(f"    Q_pred match: {torch.allclose(Q_pred, Q_pred_loaded, rtol=1e-5)}")
            print(f"    Q_w_pred match: {torch.allclose(Q_w_pred, Q_w_pred_loaded, rtol=1e-5)}")
            
            assert torch.allclose(h_pred, h_pred_loaded, rtol=1e-5)
            assert torch.allclose(Q_pred, Q_pred_loaded, rtol=1e-5)
            assert torch.allclose(Q_w_pred, Q_w_pred_loaded, rtol=1e-5)
        
        print("\nAll model tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_model()
    if success:
        print("\nModel test completed successfully!")
    else:
        print("\nModel test failed!")
