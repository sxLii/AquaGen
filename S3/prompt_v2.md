You are a senior Python code analyst and Data scientist.
Write a complete Python implementation of **Federated Averaging (FedAvg)** for a regression task using PyTorch. The code must meet the following specifications:

#### 1. Core Task
Train a multi-layer perceptron (MLP) on industrial CSV data to predict target columns from feature columns, using federated learning across multiple clients.

#### 2. Data Preprocessing
- Load a CSV file named `original_data.csv` with robust encoding handling: try `utf-8`, `gbk`, `gb2312`, `cp1252` in order; if all fail, use `utf-8` with `errors='ignore'`.
- Select feature columns: those containing "进水" (excluding "time").
- Select target columns: those *not* containing "进水" (excluding "time").
- Drop rows with missing values (raise an error if no valid data remains).
- Normalize features and targets separately using `StandardScaler` (return both scalers).
- Ensure targets are a 2D array (reshape if 1D).

#### 3. Model Architecture
Define a `SimpleMLP` class with:
- Input layer → Linear(64) → ReLU → Linear(32) → ReLU → Output layer (matches target dimension).

#### 4. Federated Learning Setup
- Split training data into 3 clients (shuffle first).
- Allocate data evenly (minimum 1 sample per client; last client gets remaining data).
- Print client data sizes after distribution.

#### 5. Training Process
- Use CUDA if available, else CPU.
- Set random seeds (numpy=42, torch=42, CUDA seeds) for reproducibility.
- Train for 50 global rounds:
  - For each client: copy the global model, train locally for 300 epochs with SGD (lr=0.01), MSE loss, and batch size=32.
  - Print local loss every 100 epochs.
- Aggregate client models via **weighted average** (weight = client data size / total training data size).
- Skip aggregation if no valid client models exist.

#### 6. Evaluation
- Split data into 80% training / 20% testing.
- Evaluate the global model on the test set:
  - Calculate MSE and RMSE for both normalized and denormalized (using the target scaler) results.
  - Print results clearly.

#### 7. Additional Requirements
- Save the model state dict, input/output scalers, and model dimensions to `federated_model_config.pth`.
- Add error handling in a `main()` function.
- Print detailed logs (data shape, column names, training progress, total runtime).