import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_width=50, initial_gH=0.5):
        super().__init__()
        layers = []
        in_features = 2
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_width))
            layers.append(nn.Tanh())
            in_features = hidden_width
        # final layer
        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)
        self.gH = nn.Parameter(torch.tensor(initial_gH, dtype=torch.float32))

    def forward(self, points):
        # points: (N,2)
        return self.net(points)

def compute_loss(pinn, domain_points, domain_targets, boundary_points, boundary_targets):
    model_device = next(pinn.parameters()).device
    # create differentiable domain tensor
    domain = domain_points.detach().clone().to(model_device).requires_grad_(True)
    domain_targets = domain_targets.to(model_device)
    boundary_points = boundary_points.to(model_device)
    boundary_targets = boundary_targets.to(model_device)

    # Prediction on domain (used for PDE and data loss)
    eta = pinn(domain)

    # Data loss
    data_loss = F.mse_loss(eta, domain_targets)

    # First derivatives
    grad_eta = torch.autograd.grad(eta, domain, grad_outputs=torch.ones_like(eta),
                                   create_graph=True)[0]  # (N,2)
    eta_x = grad_eta[:, 0:1]
    eta_t = grad_eta[:, 1:2]

    # Second derivatives
    grad_eta_x = torch.autograd.grad(eta_x, domain, grad_outputs=torch.ones_like(eta_x),
                                     create_graph=True)[0]
    eta_xx = grad_eta_x[:, 0:1]

    grad_eta_t = torch.autograd.grad(eta_t, domain, grad_outputs=torch.ones_like(eta_t),
                                     create_graph=True)[0]
    eta_tt = grad_eta_t[:, 1:2]

    # PDE residual
    residual = eta_tt - pinn.gH * eta_xx
    pde_loss = torch.mean(residual**2)

    # Boundary loss
    predicted_bc = pinn(boundary_points)
    bc_loss = F.mse_loss(predicted_bc, boundary_targets)

    # Weighted total
    lambdaPDE = 0.4
    lambdaBC = 0.6
    lambdaData = 0.5
    total = lambdaPDE * pde_loss + lambdaBC * bc_loss + lambdaData * data_loss

    return total, {'total': total, 'pde': pde_loss, 'bc': bc_loss, 'data': data_loss}