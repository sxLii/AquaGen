import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_definition import compute_loss

def train(pinn, training_data, *, epochs, batch_size, initial_lr, lr_decay, device):
    # Move data to device
    domain_pts = training_data['domain_points'].to(device)
    domain_tgts = training_data['domain_targets'].to(device)
    boundary_pts = training_data['boundary_points'].to(device)
    boundary_tgts = training_data['boundary_targets'].to(device)

    # Create DataLoader for domain points
    dataset = TensorDataset(domain_pts, domain_tgts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Separate optimizers: one for network weights (exclude gH) and one for gH
    pinn_params = [p for name, p in pinn.named_parameters() if name != 'gH']
    optimizer_pinn = optim.Adam(pinn_params, lr=initial_lr)
    optimizer_gH = optim.Adam([pinn.gH], lr=initial_lr)

    history = []
    iteration = 0
    pinn.train()
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        num_batches = 0
        for batch_pts, batch_tgts in dataloader:
            iteration += 1
            # learning rate decay per batch
            lr = initial_lr / (1 + lr_decay * iteration)
            for param_group in optimizer_pinn.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_gH.param_groups:
                param_group['lr'] = lr

            optimizer_pinn.zero_grad()
            optimizer_gH.zero_grad()

            total_loss, _ = compute_loss(pinn, batch_pts, batch_tgts, boundary_pts, boundary_tgts)
            total_loss.backward()

            optimizer_pinn.step()
            if (epoch + 1) > epochs / 10:  # zero-indexed epoch, MATLAB condition epoch > numEpochs/10
                optimizer_gH.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_total_loss / num_batches
        gH_val = pinn.gH.item()
        history.append({'epoch_loss': avg_loss, 'gH_value': gH_val})

    return pinn, history