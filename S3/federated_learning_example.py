import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Data preprocessing
def load_and_preprocess_data():
    """Load and preprocess data"""
    # Load data
    encodings = ['utf-8', 'gbk', 'gb2312', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv('original_data.csv', encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding, trying next encoding")
            continue
    
    if df is None:
        # If all encodings fail, try ignoring errors
        try:
            df = pd.read_csv('original_data.csv', encoding='utf-8', errors='ignore')
            print("Read file with utf-8 encoding and ignore errors")
        except Exception as e:
            print(f"All encoding attempts failed: {e}")
            raise
    
    # Check data basic information
    print("Data shape:", df.shape)
    print("Column names:", df.columns.tolist())
    
    # Select feature columns and target columns
    # Columns containing "进水" (inflow) as input features
    input_cols = [col for col in df.columns if '进水' in col and col != 'time']
    # Other columns as output targets
    output_cols = [col for col in df.columns if '进水' not in col and col != 'time']
    
    print(f"Input feature columns: {input_cols}")
    print(f"Output target columns: {output_cols}")
    
    # Handle missing values (if any)
    df = df.dropna()
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Normalize input features
    if input_cols:
        df[input_cols] = scaler.fit_transform(df[input_cols])
    
    # Normalize output targets
    if output_cols:
        output_scaler = StandardScaler()
        df[output_cols] = output_scaler.fit_transform(df[output_cols])
    
    # Prepare features and targets
    X = df[input_cols].values if input_cols else np.array([]).reshape(len(df), 0)
    y = df[output_cols].values if output_cols else np.array([]).reshape(len(df), 0)
    
    return X, y, scaler

# 2. Define model
class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron model"""
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 3. Federated learning setup
def setup_clients(X, y, num_clients=3):
    """Distribute data to multiple clients"""
    # Randomly shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Average distribution of data to clients
    client_data = []
    data_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i < num_clients - 1 else len(X)
        client_data.append((X[start:end], y[start:end]))
    
    return client_data

# 4. Federated learning training
def federated_learning(client_data, num_rounds=50, local_epochs=300, lr=0.01, batch_size=32):
    """Execute federated learning training"""
    # Get input and output dimensions
    input_dim = client_data[0][0].shape[1]
    output_dim = client_data[0][1].shape[1] if len(client_data[0][1].shape) > 1 else 1
    
    # Initialize global model
    global_model = SimpleMLP(input_dim, output_dim)
    print(f"Global model initialized, input dimension: {input_dim}, output dimension: {output_dim}")
    
    # Training rounds
    for round_num in range(num_rounds):
        print(f"\n===== Federated Learning Round {round_num + 1} =====")
        
        # Collect client model parameters
        client_models = []
        client_losses = []
        
        # Local training for each client
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"Client {i + 1} training...")
            
            # Copy global model to local
            local_model = copy.deepcopy(global_model)
            
            # Prepare data loader
            dataset = TensorDataset(torch.tensor(X_client, dtype=torch.float32), 
                                  torch.tensor(y_client, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Define optimizer and loss function
            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Local training
            local_model.train()
            local_loss = 0.0
            
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = local_model(batch_X)
                    # Ensure output and target shapes match
                    if output_dim == 1:
                        loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    else:
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_loss /= len(loader)
                local_loss += epoch_loss
                if np.mod(epoch, 100) == 0:
                    print(f"  Local epoch {epoch + 1}, loss: {epoch_loss:.4f}")
            
            local_loss /= local_epochs
            client_losses.append(local_loss)
            print(f"Client {i + 1} average loss: {local_loss:.4f}")
            
            # Collect local model
            client_models.append(local_model)
        
        # Model aggregation (federated averaging)
        print("\nAggregating client models...")
        
        # Initialize global model parameters to 0
        global_params = copy.deepcopy(list(global_model.parameters()))
        for param in global_params:
            param.data.zero_()
        
        # Average all client parameters
        for model in client_models:
            local_params = list(model.parameters())
            for i, param in enumerate(global_params):
                param.data += local_params[i].data / len(client_models)
        
        # Update global model
        for i, param in enumerate(global_model.parameters()):
            param.data.copy_(global_params[i].data)
        
        # Calculate average loss for this round
        avg_loss = sum(client_losses) / len(client_losses)
        print(f"Round {round_num + 1} average loss: {avg_loss:.4f}")
    
    return global_model

# 5. Model evaluation
def evaluate_model(model, X, y):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        predictions = outputs.numpy()
    
    # Calculate MSE and RMSE
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nModel evaluation results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return mse, rmse

# 6. Main function
def main():
    """Main function"""
    print("Starting federated learning example...")
    start_time = time.time()
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data()
    
    # Split into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Set up clients
    print("2. Setting up federated learning clients...")
    client_data = setup_clients(X_train, y_train, num_clients=3)
    print(f"Data distributed to {len(client_data)} clients")
    
    # Execute federated learning
    print("3. Executing federated learning training...")
    global_model = federated_learning(client_data, num_rounds=50, local_epochs=300, lr=0.01)
    
    # Evaluate model
    print("4. Evaluating model performance...")
    evaluate_model(global_model, X_test, y_test)
    
    # Save model
    print("5. Saving model...")
    torch.save(global_model.state_dict(), 'federated_model.pth')
    print("Model saved as 'federated_model.pth'")
    
    end_time = time.time()
    print(f"\nFederated learning example completed, total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
