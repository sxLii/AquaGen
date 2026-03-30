import torch
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import get_solution_data


def evaluate_model_performance(model: torch.nn.Module, c_value: float, 
                              num_test_points: int = 5000, 
                              device: str = 'cpu'):
    """
    Evaluate PINN model performance and generate comprehensive visualizations.
    
    Args:
        model (torch.nn.Module): Trained PINN model
        c_value (float): Recovered diffusion coefficient
        num_test_points (int): Number of test points for evaluation
        device (str): Device to run evaluation on
    """
    # Generate test points
    points = []
    while len(points) < num_test_points:
        x = np.random.uniform(-1, 1, num_test_points)
        y = np.random.uniform(-1, 1, num_test_points)
        mask = x**2 + y**2 <= 1.0
        valid_points = np.column_stack([x[mask], y[mask]])
        points.extend(valid_points)
    
    test_points = torch.tensor(points[:num_test_points], dtype=torch.float32)
    
    # Get predictions and exact solutions
    with torch.no_grad():
        model.eval()
        u_pred = model(test_points.to(device)).cpu()
    
    u_exact = get_solution_data(test_points)
    
    # Compute errors
    absolute_error = torch.abs(u_pred - u_exact)
    relative_error = absolute_error / (torch.abs(u_exact) + 1e-8)
    
    # Convert to numpy
    points_np = test_points.numpy()
    u_pred_np = u_pred.numpy().flatten()
    u_exact_np = u_exact.numpy().flatten()
    abs_err_np = absolute_error.numpy().flatten()
    rel_err_np = relative_error.numpy().flatten()
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. PINN Solution
    ax1 = plt.subplot(2, 3, 1)
    sc1 = ax1.scatter(points_np[:, 0], points_np[:, 1], c=u_pred_np, 
                     cmap='viridis', s=5, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'PINN Solution (c = {c_value:.4f})')
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1)
    
    # 2. Exact Solution
    ax2 = plt.subplot(2, 3, 2)
    sc2 = ax2.scatter(points_np[:, 0], points_np[:, 1], c=u_exact_np, 
                     cmap='viridis', s=5, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Exact Solution (c = 1.0)')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2)
    
    # 3. Absolute Error
    ax3 = plt.subplot(2, 3, 3)
    sc3 = ax3.scatter(points_np[:, 0], points_np[:, 1], c=abs_err_np, 
                     cmap='hot', s=5, alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Absolute Error')
    ax3.set_aspect('equal')
    plt.colorbar(sc3, ax=ax3)
    
    # 4. Pointwise comparison scatter plot
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(u_exact_np, u_pred_np, s=5, alpha=0.6)
    ax4.plot([u_exact_np.min(), u_exact_np.max()], 
             [u_exact_np.min(), u_exact_np.max()], 
             'r--', linewidth=2, label='Perfect prediction')
    ax4.set_xlabel('Exact Solution')
    ax4.set_ylabel('PINN Prediction')
    ax4.set_title('Pointwise Comparison')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Error histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(abs_err_np, bins=50, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Absolute Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.grid(True, alpha=0.3)
    
    # 6. Radial profile comparison
    ax6 = plt.subplot(2, 3, 6)
    r = np.sqrt(points_np[:, 0]**2 + points_np[:, 1]**2)
    
    # Sort by radius for better visualization
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    u_pred_sorted = u_pred_np[sort_idx]
    u_exact_sorted = u_exact_np[sort_idx]
    
    ax6.plot(r_sorted, u_exact_sorted, 'b-', linewidth=2, label='Exact', alpha=0.7)
    ax6.plot(r_sorted, u_pred_sorted, 'r.', markersize=1, label='PINN', alpha=0.5)
    ax6.set_xlabel('Radius (r)')
    ax6.set_ylabel('Solution u(r)')
    ax6.set_title('Radial Profile')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive statistics
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    print(f"Recovered diffusion coefficient c: {c_value:.6f}")
    print(f"True diffusion coefficient c: 1.000000")
    print(f"Absolute error in c: {abs(c_value - 1.0):.6f}")
    print(f"Relative error in c: {abs(c_value - 1.0):.6%}")
    print()
    print(f"Mean absolute error: {abs_err_np.mean():.6e}")
    print(f"Max absolute error: {abs_err_np.max():.6e}")
    print(f"RMSE: {np.sqrt(np.mean(abs_err_np**2)):.6e}")
    print(f"Relative L2 error: {np.linalg.norm(u_pred_np - u_exact_np) / np.linalg.norm(u_exact_np):.6e}")
    print(f"R-squared: {1 - np.sum((u_pred_np - u_exact_np)**2) / np.sum((u_exact_np - u_exact_np.mean())**2):.6f}")
    print("="*60)


def save_model_and_results(model: torch.nn.Module, c_value: float, 
                          loss_history: list, c_history: list):
    """
    Save model and training results to files.
    
    Args:
        model (torch.nn.Module): Trained PINN model
        c_value (float): Recovered diffusion coefficient
        loss_history (list): Training loss history
        c_history (list): c value history during training
    """
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'c_value': c_value,
        'loss_history': loss_history,
        'c_history': c_history
    }, 'pinn_model.pth')
    
    # Save results as text file
    with open('training_results.txt', 'w') as f:
        f.write("PINN Training Results\n")
        f.write("="*50 + "\n")
        f.write(f"Final c value: {c_value:.6f}\n")
        f.write(f"Error in c: {abs(c_value - 1.0):.6f}\n")
        f.write(f"Final training loss: {loss_history[-1]:.6e}\n")
        f.write("\nTraining History:\n")
        for i, (loss, c_val) in enumerate(zip(loss_history, c_history)):
            f.write(f"Epoch {i+1}: loss={loss:.6e}, c={c_val:.6f}\n")
    
    print("\nModel and results saved to:\n")
    print("  - pinn_model.pth (PyTorch model checkpoint)")
    print("  - training_results.txt (training history)")
    print("  - training_history.png (loss and c evolution)")
    print("  - solution_comparison.png (solution plots)")
    print("  - performance_evaluation.png (comprehensive evaluation)")


if __name__ == "__main__":
    # Test evaluation with dummy data
    from model_definition import PINN
    
    model = PINN()
    c_value = 0.99
    loss_history = [0.1 * np.exp(-0.01*i) for i in range(100)]
    c_history = [0.5 + 0.5 * (1 - np.exp(-0.02*i)) for i in range(100)]
    
    evaluate_model_performance(model, c_value, num_test_points=1000)
    save_model_and_results(model, c_value, loss_history, c_history)
