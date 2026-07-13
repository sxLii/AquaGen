import os
import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Set headless backend before importing pyplot
import matplotlib.pyplot as plt

from data_preprocessing import generate_training_data, exact_solution
from model_definition import PINN
from model_training import train
from model_inference import predict_on_grid
from performance_evaluation import compute_metrics, save_solution_plot

def main(argv=None):
    parser = argparse.ArgumentParser(description='Inverse PINN for shallow water wave equation')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--n-domain', type=int, default=10000, help='Number of domain collocation points')
    parser.add_argument('--n-boundary', type=int, default=2000, help='Number of boundary points')
    parser.add_argument('--batch-size', type=int, default=4096, help='Mini-batch size')
    parser.add_argument('--n-x', type=int, default=100, help='Number of grid points in x')
    parser.add_argument('--n-t', type=int, default=100, help='Number of grid points in t')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to run on')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args(argv)

    # Validate positive integers
    for key in ['epochs', 'n_domain', 'n_boundary', 'batch_size', 'n_x', 'n_t']:
        val = getattr(args, key)
        if val <= 0:
            raise ValueError(f"--{key} must be a positive integer, got {val}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate training data
    training_data = generate_training_data(args.n_domain, args.n_boundary, args.seed, device)
    print(f"Generated {args.n_domain} domain points and {args.n_boundary} boundary points")

    # Initialize model
    pinn = PINN(hidden_layers=3, hidden_width=50, initial_gH=0.5).to(device)

    # Train
    trained_pinn, history = train(pinn, training_data,
                                  epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  initial_lr=0.01,
                                  lr_decay=0.001,
                                  device=device)
    print("Training completed")

    # Inference on grid
    L = 1.0
    T = 2.0
    X, T_grid, predictions = predict_on_grid(trained_pinn, L, T, args.n_x, args.n_t)
    print("Prediction on grid computed")

    # Compute exact solution on grid
    x_np = np.linspace(0, L, args.n_x)
    t_np = np.linspace(0, T, args.n_t)
    T_mesh, X_mesh = np.meshgrid(t_np, x_np, indexing='ij')
    coords = np.stack([X_mesh.ravel(), T_mesh.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    exact_flat = exact_solution(coords_tensor)
    exact_grid = exact_flat.reshape(args.n_t, args.n_x).numpy()

    # Metrics
    metrics = compute_metrics(predictions, exact_grid)
    print("Metrics:", metrics)

    # Save plot
    gH_recovered = trained_pinn.gH.item()
    recovered_depth = gH_recovered / 9.81
    print(f"Recovered gH: {gH_recovered:.6f}, Recovered depth H: {recovered_depth:.6f} m")
    plot_path = os.path.join(args.output_dir, 'solution_plot.png')
    save_solution_plot(X, T_grid, predictions, exact_grid, gH_recovered, plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()