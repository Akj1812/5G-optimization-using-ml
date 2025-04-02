import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import explained_variance_score, mean_squared_log_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 5)  # 5 features
y = X @ np.array([1.5, -2, 3, 0.5, 4]) + np.random.randn(1000) * 0.5  # Linear relationship with noise

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors for LSTM
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Initialize LSTM model
lstm_model = LSTMRegressor(input_size=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

# Train LSTM model
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = lstm_model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Machine": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Prediction and timing
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = (end_time - start_time) / len(y_test)
    
    # Calculate performance metrics
    evs = explained_variance_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, np.abs(y_pred))  # Ensure positive values for MSLE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
    
    results.append([name, evs, msle, mape, adj_r2, inference_time])

# LSTM evaluation
with torch.no_grad():
    lstm_start = time.time()
    y_lstm_pred = lstm_model(X_test_tensor).numpy().flatten()
    lstm_end = time.time()
    lstm_inference_time = (lstm_end - lstm_start) / len(y_test)

    lstm_evs = explained_variance_score(y_test, y_lstm_pred)
    lstm_msle = mean_squared_log_error(y_test, np.abs(y_lstm_pred))
    lstm_mape = mean_absolute_percentage_error(y_test, y_lstm_pred)
    lstm_r2 = r2_score(y_test, y_lstm_pred)
    lstm_adj_r2 = 1 - (1 - lstm_r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
    
    results.append(["LSTM", lstm_evs, lstm_msle, lstm_mape, lstm_adj_r2, lstm_inference_time])

# Create DataFrame
columns = ["Model", "Explained Variance", "MSLE", "MAPE", "Adjusted R²", "Inference Time (s/sample)"]
df_results = pd.DataFrame(results, columns=columns)

# Display results
print(df_results)
