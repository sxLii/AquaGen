import torch
import math

def exact_solution(points):
    # points: (N,2) tensor, first column x, second t
    x = points[:, 0]
    t = points[:, 1]
    omega = math.pi * math.sqrt(0.981)
    eta = torch.sin(math.pi * x) * torch.cos(omega * t)
    return eta.unsqueeze(1)  # (N,1)

def generate_training_data(n_domain, n_boundary, seed, device):
    torch.manual_seed(seed)
    # Domain points
    L = 1.0
    T = 2.0
    # random points in [0,L] x [0,T]
    x_domain = torch.rand(n_domain, 1, device=device, dtype=torch.float32) * L
    t_domain = torch.rand(n_domain, 1, device=device, dtype=torch.float32) * T
    domain_points = torch.cat([x_domain, t_domain], dim=1)  # (n_domain,2)
    domain_targets = exact_solution(domain_points).detach()  # no grad

    # Boundary points: uniform along each of the four edges
    # allocate points evenly, remainder on last edge
    n_per_edge = n_boundary // 4
    remainder = n_boundary % 4
    counts = [n_per_edge] * 4
    counts[-1] += remainder  # add remainder to last edge (top)
    edge_points_list = []
    # left edge: x=0, t random
    t_left = torch.rand(counts[0], 1, device=device, dtype=torch.float32) * T
    edge_points_list.append(torch.cat([torch.zeros_like(t_left), t_left], dim=1))
    # right edge: x=L, t random
    t_right = torch.rand(counts[1], 1, device=device, dtype=torch.float32) * T
    edge_points_list.append(torch.cat([torch.full_like(t_right, L), t_right], dim=1))
    # bottom edge: t=0, x random
    x_bottom = torch.rand(counts[2], 1, device=device, dtype=torch.float32) * L
    edge_points_list.append(torch.cat([x_bottom, torch.zeros_like(x_bottom)], dim=1))
    # top edge: t=T, x random
    x_top = torch.rand(counts[3], 1, device=device, dtype=torch.float32) * L
    edge_points_list.append(torch.cat([x_top, torch.full_like(x_top, T)], dim=1))
    boundary_points = torch.cat(edge_points_list, dim=0)  # (n_boundary,2)
    boundary_targets = exact_solution(boundary_points).detach()

    return {
        'domain_points': domain_points,
        'domain_targets': domain_targets,
        'boundary_points': boundary_points,
        'boundary_targets': boundary_targets,
    }