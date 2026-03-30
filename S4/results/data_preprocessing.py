import torch
import numpy as np


def generate_disk_points(num_interior: int = 2000, num_boundary: int = 400):
    """
    Generate collocation points for training PINN on unit disk.
    
    Args:
        num_interior (int): Number of interior points (domain collocation)
        num_boundary (int): Number of boundary points (for BC enforcement)
    
    Returns:
        dict: Dictionary containing:
            - 'domain': torch.Tensor of shape (num_interior, 2) for interior points
            - 'boundary': torch.Tensor of shape (num_boundary, 2) for boundary points
    """
    # Generate interior points uniformly in unit disk using rejection sampling
    interior_points = []
    while len(interior_points) < num_interior:
        x = np.random.uniform(-1, 1, num_interior * 2)
        y = np.random.uniform(-1, 1, num_interior * 2)
        mask = x**2 + y**2 <= 1.0
        interior_points.extend(np.column_stack([x[mask], y[mask]]))
    
    interior_points = np.array(interior_points[:num_interior])
    
    # Generate boundary points uniformly on unit circle
    theta = np.random.uniform(0, 2 * np.pi, num_boundary)
    boundary_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Convert to PyTorch tensors
    data = {
        'domain': torch.tensor(interior_points, dtype=torch.float32),
        'boundary': torch.tensor(boundary_points, dtype=torch.float32)
    }
    
    return data


def get_solution_data(points: torch.Tensor) -> torch.Tensor:
    """
    Compute exact solution u = (1 - x^2 - y^2) / 4 for given points.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 2) containing (x,y) coordinates
    
    Returns:
        torch.Tensor: Exact solution values at points, shape (N, 1)
    """
    x = points[:, 0]
    y = points[:, 1]
    u_exact = (1 - x**2 - y**2) / 4
    return u_exact.reshape(-1, 1)


if __name__ == "__main__":
    # Test the functions
    data = generate_disk_points(num_interior=1000, num_boundary=200)
    print(f"Domain points shape: {data['domain'].shape}")
    print(f"Boundary points shape: {data['boundary'].shape}")
    
    # Test exact solution
    test_points = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    u_test = get_solution_data(test_points)
    print(f"Test solutions: {u_test}")
