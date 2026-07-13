import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(predictions, exact):
    mae = np.mean(np.abs(predictions - exact))
    rmse = np.sqrt(np.mean((predictions - exact)**2))
    max_abs = np.max(np.abs(predictions - exact))
    return {'MAE': mae, 'RMSE': rmse, 'Max_Abs_Error': max_abs}

def save_solution_plot(X, T_grid, predictions, exact, gH, output_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(X, T_grid, predictions, levels=50, cmap='viridis')
    plt.colorbar(cp1)
    plt.title('PINN Prediction')
    plt.xlabel('x (m)')
    plt.ylabel('t (s)')

    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(X, T_grid, exact, levels=50, cmap='viridis')
    plt.colorbar(cp2)
    plt.title('Exact Solution')
    plt.xlabel('x (m)')
    plt.ylabel('t (s)')

    plt.suptitle(f'Solution with gH = {gH:.4f}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()