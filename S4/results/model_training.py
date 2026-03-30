import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_definition import PINN, PhysicsInformedLoss
from data_preprocessing import generate_disk_points, get_solution_data


def train_pinn(num_epochs: int = 1500, mini_batch_size: int = 4096, 
               initial_lr: float = 0.01, lr_decay: float = 0.001, 
               device: str = 'cpu'):
    """
    Train the PINN model to recover diffusion coefficient c.
    
    Args:
        num_epochs (int): Number of training epochs
        mini_batch_size (int): Batch size for training
        initial_lr (float): Initial learning rate
        lr_decay (float): Learning rate decay factor
        device (str): Device to train on ('cpu' or 'cuda')
    
    Returns:
        tuple: (trained_model, c_value, loss_history, c_history)
    """
    # Move to device
    device = torch.device(device)
    
    # Generate training data
    print("Generating training data...")
    data = generate_disk_points(num_interior=10000, num_boundary=1000)
    domain_points = data['domain'].to(device)
    boundary_points = data['boundary'].to(device)
    exact_solution = get_solution_data(domain_points).to(device)
    
    # Initialize model and parameters
    model = PINN().to(device)
    c = torch.tensor(0.5, requires_grad=True, device=device)  # Initial guess
    
    # Initialize loss function
    loss_fn = PhysicsInformedLoss()
    
    # Setup optimizers
    model_optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    c_optimizer = optim.Adam([c], lr=initial_lr)
    
    # Learning rate scheduler
    def lr_lambda(iteration):
        return 1.0 / (1.0 + lr_decay * iteration)
    
    model_scheduler = LambdaLR(model_optimizer, lr_lambda=lr_lambda)
    c_scheduler = LambdaLR(c_optimizer, lr_lambda=lr_lambda)
    
    # Training history
    loss_history = []
    c_history = []
    
    # Create data loaders for mini-batching
    domain_dataset = torch.utils.data.TensorDataset(domain_points, exact_solution)
    domain_loader = torch.utils.data.DataLoader(
        domain_dataset, batch_size=mini_batch_size, shuffle=True
    )
    
    boundary_dataset = torch.utils.data.TensorDataset(boundary_points)
    boundary_loader = torch.utils.data.DataLoader(
        boundary_dataset, batch_size=mini_batch_size, shuffle=True
    )
    
    print("Starting training...")
    progress_bar = tqdm(range(num_epochs), desc='Training PINN')
    
    for epoch in progress_bar:
        epoch_loss = 0.0
        num_batches = 0
        
        # Create iterators for both loaders
        domain_iter = iter(domain_loader)
        boundary_iter = iter(boundary_loader)
        
        # Iterate through batches
        try:
            while True:
                # Get batches
                domain_batch, exact_batch = next(domain_iter)
                (boundary_batch,) = next(boundary_iter)
                
                # Compute loss
                loss = loss_fn(model, domain_batch, boundary_batch, c, exact_batch)
                
                # Backward pass and optimization
                model_optimizer.zero_grad()
                c_optimizer.zero_grad()
                loss.backward()
                
                # Update model parameters
                model_optimizer.step()
                
                # Update c parameter only after 1/10 of epochs
                if epoch > num_epochs // 10:
                    c_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
        except StopIteration:
            pass
        
        # Update learning rate
        model_scheduler.step()
        c_scheduler.step()
        
        # Record history
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        c_history.append(c.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'c': f'{c.item():.4f}'
        })
    
    print(f"Training completed. Final c value: {c.item():.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.grid(True)
    
    ax2.plot(c_history)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='True c = 1.0')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('c value')
    ax2.set_title('Diffusion Coefficient Evolution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return model, c, loss_history, c_history


if __name__ == "__main__":
    # Test training on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run shorter training for testing
    model, c, loss_history, c_history = train_pinn(
        num_epochs=100,
        mini_batch_size=512,
        initial_lr=0.01,
        device=device
    )
