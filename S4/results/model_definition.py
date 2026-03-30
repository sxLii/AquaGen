import torch
import torch.nn as nn
import torch.autograd as autograd


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Poisson equation on unit disk.
    
    Architecture: 3 hidden layers with 50 neurons each, tanh activation.
    Input: (x, y) coordinates
    Output: Predicted solution u(x, y)
    """
    def __init__(self, num_layers: int = 3, num_neurons: int = 50):
        """
        Initialize the PINN model.
        
        Args:
            num_layers (int): Number of hidden layers (excluding input/output)
            num_neurons (int): Number of neurons in each hidden layer
        """
        super(PINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(2, num_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(num_neurons, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2) containing (x,y) coordinates
        
        Returns:
            torch.Tensor: Predicted solution u(x, y), shape (N, 1)
        """
        return self.network(x)


class PhysicsInformedLoss:
    """
    Compute physics-informed loss for Poisson equation.
    
    Loss = λ_pde * PDE_loss + λ_bc * BC_loss + λ_data * data_loss
    """
    def __init__(self, lambda_pde: float = 0.4, lambda_bc: float = 0.6, 
                 lambda_data: float = 0.5):
        """
        Initialize loss function with weighting factors.
        
        Args:
            lambda_pde (float): Weight for PDE residual loss
            lambda_bc (float): Weight for boundary condition loss
            lambda_data (float): Weight for data loss
        """
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_data = lambda_data
        
    def compute_pde_loss(self, model: nn.Module, points: torch.Tensor, 
                         c: torch.Tensor, f: float = 1.0, a: float = 0.0) -> torch.Tensor:
        """
        Compute PDE residual loss for Poisson equation: -∇·(c∇u) - f = 0.
        
        Args:
            model (nn.Module): PINN model
            points (torch.Tensor): Collocation points, shape (N, 2)
            c (torch.Tensor): Diffusion coefficient (trainable parameter)
            f (float): Source term (fixed)
            a (float): Coefficient for u term (fixed)
        
        Returns:
            torch.Tensor: PDE loss
        """
        # Ensure points require gradients for autodiff
        points.requires_grad_(True)
        
        # Forward pass to get u predictions
        u = model(points)
        
        # Compute gradients ∇u = (∂u/∂x, ∂u/∂y)
        grad_u = autograd.grad(u, points, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
        
        # Compute divergence of (c * ∇u): ∂/∂x(c * ∂u/∂x) + ∂/∂y(c * ∂u/∂y)
        # Since c is scalar, this simplifies to c * (∂²u/∂x² + ∂²u/∂y²)
        grad_u_x = grad_u[:, 0].reshape(-1, 1)
        grad_u_y = grad_u[:, 1].reshape(-1, 1)
        
        # Compute second derivatives
        grad_u_x_x = autograd.grad(grad_u_x.sum(), points, create_graph=True)[0][:, 0].reshape(-1, 1)
        grad_u_y_y = autograd.grad(grad_u_y.sum(), points, create_graph=True)[0][:, 1].reshape(-1, 1)
        
        # Laplacian: Δu = ∂²u/∂x² + ∂²u/∂y²
        laplacian = grad_u_x_x + grad_u_y_y
        
        # PDE residual: -c * Δu - f + a*u = 0
        residual = -c * laplacian - f + a * u
        
        # Mean squared error of residual
        pde_loss = torch.mean(residual ** 2)
        
        return pde_loss
    
    def compute_bc_loss(self, model: nn.Module, boundary_points: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss (Dirichlet: u = 0 on boundary).
        
        Args:
            model (nn.Module): PINN model
            boundary_points (torch.Tensor): Boundary points, shape (M, 2)
        
        Returns:
            torch.Tensor: Boundary condition loss
        """
        u_pred = model(boundary_points)
        u_true = torch.zeros_like(u_pred)  # Zero Dirichlet BC
        bc_loss = torch.mean((u_pred - u_true) ** 2)
        return bc_loss
    
    def compute_data_loss(self, model: nn.Module, points: torch.Tensor, 
                         exact_solution: torch.Tensor) -> torch.Tensor:
        """
        Compute data loss between predicted and exact solution.
        
        Args:
            model (nn.Module): PINN model
            points (torch.Tensor): Data points, shape (N, 2)
            exact_solution (torch.Tensor): Exact solution values, shape (N, 1)
        
        Returns:
            torch.Tensor: Data loss
        """
        u_pred = model(points)
        data_loss = torch.mean((u_pred - exact_solution) ** 2)
        return data_loss
    
    def __call__(self, model: nn.Module, domain_points: torch.Tensor, 
                 boundary_points: torch.Tensor, c: torch.Tensor, 
                 exact_solution: torch.Tensor) -> torch.Tensor:
        """
        Compute total physics-informed loss.
        
        Args:
            model (nn.Module): PINN model
            domain_points (torch.Tensor): Interior collocation points
            boundary_points (torch.Tensor): Boundary points
            c (torch.Tensor): Diffusion coefficient
            exact_solution (torch.Tensor): Exact solution at domain_points
        
        Returns:
            torch.Tensor: Total weighted loss
        """
        pde_loss = self.compute_pde_loss(model, domain_points, c)
        bc_loss = self.compute_bc_loss(model, boundary_points)
        data_loss = self.compute_data_loss(model, domain_points, exact_solution)
        
        total_loss = (self.lambda_pde * pde_loss + 
                     self.lambda_bc * bc_loss + 
                     self.lambda_data * data_loss)
        
        return total_loss


if __name__ == "__main__":
    # Test the model and loss function
    model = PINN()
    print(model)
    
    # Test forward pass
    test_input = torch.randn(10, 2)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Test loss function
    loss_fn = PhysicsInformedLoss()
    c_param = torch.tensor(0.5, requires_grad=True)
    domain_pts = torch.randn(100, 2)
    boundary_pts = torch.randn(50, 2)
    exact = torch.randn(100, 1)
    loss = loss_fn(model, domain_pts, boundary_pts, c_param, exact)
    print(f"Test loss: {loss.item()}")
