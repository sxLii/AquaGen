You are a senior Python code analyst and Data scientist.

Write a PyTorch FedAvg code for regression on CSV data:
- Load `original_data.csv` with multi-encoding support; select features (including "") and targets (not including ""); normalize with `StandardScaler`.
- Build a 3-layer MLP (64→32→output, ReLU).
- Split data into 3 clients; train 50 global rounds (300 local epochs, SGD lr=0.01, MSE loss; aggregate via weighted FedAvg).
- Evaluate on 20% test set (MSE/RMSE, normalized + denormalized).
- Save model/scalers; add reproducibility seeds, CUDA support, error handling, and logs.