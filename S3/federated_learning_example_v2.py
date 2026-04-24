import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler  # 移到顶部
import copy
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data preprocessing
def load_and_preprocess_data(file_path="original_data.csv"):
    """Load and preprocess data"""
    # Load data
    encodings = ['utf-8', 'gbk', 'gb2312', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading with {encoding} encoding: {e}")
            continue
    
    if df is None:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            print("Read file with utf-8 encoding and ignore errors")
        except Exception as e:
            raise RuntimeError(f"All encoding attempts failed: {e}")
    
    # Check data basic information
    print("Data shape:", df.shape)
    print("Column names:", df.columns.tolist())
    
    # Select feature columns and target columns
    input_cols = [col for col in df.columns if '进水' in col and col != 'time']
    output_cols = [col for col in df.columns if '进水' not in col and col != 'time']
    
    print(f"Input feature columns: {input_cols}")
    print(f"Output target columns: {output_cols}")
    
    # Handle missing values (if any)
    df = df.dropna()
    if len(df) == 0:
        raise ValueError("No valid data left after dropping missing values!")
    
    # Normalize data
    scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    # Normalize input features
    X = np.array([]).reshape(len(df), 0)
    if input_cols:
        X = scaler.fit_transform(df[input_cols])
    
    # Normalize output targets
    y = np.array([]).reshape(len(df), 0)
    if output_cols:
        y = output_scaler.fit_transform(df[output_cols])
    
    # 确保y是二维数组（统一维度）
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    return X, y, scaler, output_scaler

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
    if len(X) == 0:
        raise ValueError("No data to distribute to clients!")
    
    # 确保客户端数≤样本数，且至少1个客户端
    num_clients = max(1, min(num_clients, len(X)))
    # Randomly shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Average distribution of data to clients
    client_data = []
    data_per_client = max(1, len(X) // num_clients)  # 至少分配1条数据
    
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i < num_clients - 1 else len(X)
        client_data.append((X[start:end], y[start:end]))
    
    print(f"Distributed data to {num_clients} clients, data sizes: {[len(x) for x, _ in client_data]}")
    return client_data

# 4. Federated learning training
def federated_learning(client_data, num_rounds=50, local_epochs=300, lr=0.01, batch_size=32):
    """Execute federated learning training (standard FedAvg)"""
    # Get input and output dimensions
    input_dim = client_data[0][0].shape[1]
    y_client = client_data[0][1]
    output_dim = y_client.shape[1] if y_client.ndim == 2 else 1
    
    # Initialize global model
    global_model = SimpleMLP(input_dim, output_dim).to(DEVICE)
    print(f"Global model initialized (device: {DEVICE}), input dim: {input_dim}, output dim: {output_dim}")
    
    # 统计客户端数据量（用于加权平均）
    client_sizes = [len(X_client) for X_client, _ in client_data]
    total_client_size = sum(client_sizes)
    
    # Training rounds
    for round_num in range(num_rounds):
        print(f"\n===== Federated Learning Round {round_num + 1}/{num_rounds} =====")
        
        # Collect client model parameters and losses
        client_models = []
        client_losses = []
        
        # Local training for each client
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"\nClient {i + 1} training (data size: {len(X_client)})...")
            
            # Skip empty client data (defensive check)
            if len(X_client) == 0:
                print(f"Client {i + 1} has no data, skipping...")
                continue
            
            # Copy global model to local
            local_model = copy.deepcopy(global_model)
            
            # Prepare data loader
            X_tensor = torch.tensor(X_client, dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(y_client, dtype=torch.float32).to(DEVICE)
            dataset = TensorDataset(X_tensor, y_tensor)
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
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_loss /= len(loader)
                local_loss += epoch_loss
                if np.mod(epoch + 1, 100) == 0:  # 每100轮打印（含最后一轮）
                    print(f"  Local epoch {epoch + 1}, loss: {epoch_loss:.4f}")
            
            # Calculate average loss for client
            avg_local_loss = local_loss / local_epochs
            client_losses.append(avg_local_loss)
            print(f"Client {i + 1} average loss: {avg_local_loss:.4f}")
            
            # Collect local model
            client_models.append(local_model)
        
        # Skip aggregation if no valid client models
        if len(client_models) == 0:
            print("No valid client models to aggregate, skipping round...")
            continue
        
        # Model aggregation (standard FedAvg: weighted by client data size)
        print("\nAggregating client models (weighted by data size)...")
        
        # Initialize global model parameters to 0
        global_params = list(global_model.parameters())
        for param in global_params:
            param.data.zero_()
        
        # Weighted average of client parameters
        for model_idx, model in enumerate(client_models):
            local_params = list(model.parameters())
            # 计算当前客户端的权重
            weight = client_sizes[model_idx] / total_client_size
            for param_idx, param in enumerate(global_params):
                param.data += local_params[param_idx].data * weight
        
        # Update global model
        for param_idx, param in enumerate(global_model.parameters()):
            param.data.copy_(global_params[param_idx].data)
        
        # Calculate average loss for this round
        avg_round_loss = sum(client_losses) / len(client_losses)
        print(f"Round {round_num + 1} average loss: {avg_round_loss:.4f}")
    
    return global_model

# 5. Model evaluation
def evaluate_model(model, X, y, output_scaler=None):
    """Evaluate model performance (support denormalization)"""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        outputs = model(inputs)
        predictions = outputs.cpu().numpy()
    
    # 反归一化（若提供output_scaler）
    if output_scaler is not None:
        predictions = output_scaler.inverse_transform(predictions)
        y = output_scaler.inverse_transform(y)
    
    # 统一维度（避免广播问题）
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze()
    
    # Calculate MSE and RMSE
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nModel evaluation results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return mse, rmse

# 6. Main function
def main():
    """Main function with error handling"""
    try:
        print("Starting federated learning example...")
        start_time = time.time()
        
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        X, y, scaler, output_scaler = load_and_preprocess_data()
        print(f"Total data size: {len(X)}")
        
        # Split into training and testing sets (80/20)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        
        # 2. Set up federated learning clients
        print("\n2. Setting up federated learning clients...")
        client_data = setup_clients(X_train, y_train, num_clients=3)
        
        # 3. Execute federated learning
        print("\n3. Executing federated learning training...")
        global_model = federated_learning(
            client_data, 
            num_rounds=50, 
            local_epochs=300, 
            lr=0.01, 
            batch_size=32
        )
        
        # 4. Evaluate model
        print("\n4. Evaluating model performance...")
        # 评估归一化后的结果 + 反归一化后的真实结果
        print("\n[Normalized Results]")
        evaluate_model(global_model, X_test, y_test)
        print("\n[Denormalized Results (Real Scale)]")
        evaluate_model(global_model, X_test, y_test, output_scaler)
        
        # 5. Save model and related info
        print("\n5. Saving model and config...")
        model_config = {
            "input_dim": X_train.shape[1],
            "output_dim": y_train.shape[1] if y_train.ndim == 2 else 1,
            "state_dict": global_model.state_dict(),
            "scaler": scaler,
            "output_scaler": output_scaler
        }
        torch.save(model_config, 'federated_model_config.pth')
        print("Model and config saved as 'federated_model_config.pth'")
        
        end_time = time.time()
        print(f"\nFederated learning example completed successfully!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise

if __name__ == "__main__":
    main()