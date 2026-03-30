import torch
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import get_solution_data


def predict_solution(model: torch.nn.Module, points: torch.Tensor, 
                    device: str = 'cpu') -> torch.Tensor:
    """
    Predict solution u(x,y) using trained PINN model.
    
    Args:
        model (torch.nn.Module): Trained PINN model
        points (torch.Tensor): Points to evaluate at, shape (N, 2)
        device (str): Device to run inference on
    
    Returns:
        torch.Tensor: Predicted solution values, shape (N, 1)
    """
    model.eval()
    device = torch.device(device)
    model.to(device)
    points = points.to(device)
    
    with torch.no_grad():
        predictions = model(points)
    
    return predictions.cpu()


def generate_evaluation_grid(num_points: int = 10000) -> torch.Tensor:
    """
    Generate points for evaluation visualization.
    
    Args:
        num_points (int): Number of points to generate
    
    Returns:
        torch.Tensor: Grid points within unit disk
    """
    # Generate points uniformly in unit disk
    points = []
    while len(points) < num_points:
        x = np.random.uniform(-1, 1, num_points)
        y = np.random.uniform(-1, 1, num_points)
        mask = x**2 + y**2 <= 1.0
        valid_points = np.column_stack([x[mask], y[mask]])
        points.extend(valid_points)
    
    points = np.array(points[:num_points])
    return torch.tensor(points, dtype=torch.float32)


def plot_solution_comparison(model: torch.nn.Module, c_value: float, 
                            device: str = 'cpu'):
    """
    Plot PINN solution and compare with exact solution.
    
    Args:
        model (torch.nn.Module): Trained PINN model
        c_value (float): Recovered diffusion coefficient
        device (str): Device to run inference on
    """
    # Generate evaluation points
    eval_points = generate_evaluation_grid(num_points=10000)
    
    # Get predictions
    u_pred = predict_solution(model, eval_points, device=device)
    
    # Get exact solution
    u_exact = get_solution_data(eval_points)
    
    # Compute absolute error
    error = torch.abs(u_pred - u_exact)
    
    # Convert to numpy for plotting
    points_np = eval_points.numpy()
    u_pred_np = u_pred.numpy().flatten()
    u_exact_np = u_exact.numpy().flatten()
    error_np = error.numpy().flatten()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot PINN solution
    sc1 = axes[0].scatter(points_np[:, 0], points_np[:, 1], c=u_pred_np, 
                         cmap='viridis', s=1, alpha=0.8)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'PINN Solution (c = {c_value:.4f})')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0])
    
    # Plot exact solution
    sc2 = axes[1].scatter(points_np[:, 0], points_np[:, 1], c=u_exact_np, 
                         cmap='viridis', s=1, alpha=0.8)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Exact Solution (c = 1.0)')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1])
    
    # Plot error
    sc3 = axes[2].scatter(points_np[:, 0], points_np[:, 1], c=error_np, 
                         cmap='hot', s=1, alpha=0.8)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Absolute Error')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('solution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"Recovered c value: {c_value:.4f}")
    print(f"Mean absolute error: {error_np.mean():.6f}")
    print(f"Max absolute error: {error_np.max():.6f}")
    print(f"Relative L2 error: {np.linalg.norm(u_pred_np - u_exact_np) / np.linalg.norm(u_exact_np):.6f}")


if __name__ == "__main__":
    # Example usage
    from model_definition import PINN
    
    # Load a model (or create a dummy one for testing)
    model = PINN()
    c_value = 1.0  # Dummy value
    
    # Test on CPU
    plot_solution_comparison(model, c_value, device='cpu')
