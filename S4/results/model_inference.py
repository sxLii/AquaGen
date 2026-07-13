import torch
import numpy as np
from model_definition import PINN  # needed for type hint if desired

def predict_on_grid(pinn, L, T, n_x, n_t):
    model_device = next(pinn.parameters()).device
    pinn.eval()
    with torch.no_grad():
        # create meshgrid
        x = torch.linspace(0, L, n_x, device=model_device)
        t = torch.linspace(0, T, n_t, device=model_device)
        T_grid, X = torch.meshgrid(t, x, indexing='ij')  # shapes (n_t, n_x)
        coords = torch.stack([X.reshape(-1), T_grid.reshape(-1)], dim=1)  # (n_t*n_x, 2)
        pred_flat = pinn(coords)
        predictions = pred_flat.reshape(n_t, n_x)
    return X.cpu().numpy(), T_grid.cpu().numpy(), predictions.cpu().numpy()