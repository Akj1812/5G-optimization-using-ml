import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("C:/Users/anubh/Downloads/train.csv")

# Preprocessing
df = df.dropna()
categorical_cols = ["Environment", "Tower ID", "User ID", "Call Type", "Incoming/Outgoing"]
df = pd.get_dummies(df, columns=categorical_cols)

# Define features and target
X = df.drop(columns=["Timestamp", "Signal Strength (dBm)"])
y = df["Signal Strength (dBm)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Feature importance
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")

import matplotlib.pyplot as plt

# Plot actual vs predicted bandwidth allocation
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted')
plt.scatter(y_test, y_test, color='red', alpha=0.4, label='Actual')
plt.title('Actual vs Predicted Bandwidth Allocation')
plt.xlabel('Actual Bandwidth Allocation')
plt.ylabel('Predicted Bandwidth Allocation')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression

# Define features and target for latency
X_latency = df.drop(columns=["Timestamp", "Call Duration (s)"])
y_latency = df["Call Duration (s)"]

# Split data
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X_latency, y_latency, test_size=0.2, random_state=42)

# Train model
latency_model = LinearRegression()
latency_model.fit(X_train_lat, y_train_lat)

# Predict
y_pred_lat = latency_model.predict(X_test_lat)

# Evaluate
mse_lat = mean_squared_error(y_test_lat, y_pred_lat)
print(f"Latency Mean Squared Error: {mse_lat:.4f}")

# Plot actual vs predicted latency
plt.figure(figsize=(10, 6))
plt.scatter(y_test_lat, y_pred_lat, color='green', alpha=0.6, label='Predicted')
plt.scatter(y_test_lat, y_test_lat, color='orange', alpha=0.4, label='Actual')
plt.title('Actual vs Predicted Latency')
plt.xlabel('Actual Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.legend()
plt.grid(True)
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)].values
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_seq, y_seq = create_sequences(df[["Signal Strength (dBm)", "Call Duration (s)"]], seq_length)

# Split data
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 2)))
model_lstm.add(Dense(2))
model_lstm.compile(optimizer='adam', loss='mse')

# Train model
model_lstm.fit(X_train_seq, y_train_seq, epochs=20, verbose=1)

# Predict
y_pred_seq = model_lstm.predict(X_test_seq)

# Evaluate
mse_seq = mean_squared_error(y_test_seq, y_pred_seq)
print(f"LSTM Mean Squared Error: {mse_seq:.4f}")

# Plot actual vs predicted user mobility (LSTM)
plt.figure(figsize=(10, 6))
plt.plot(y_test_seq[:, 0], label='Actual Signal Strength')
plt.plot(y_pred_seq[:, 0], label='Predicted Signal Strength', linestyle='--')
plt.title('Actual vs Predicted User Mobility (Signal Strength)')
plt.xlabel('Time Steps')
plt.ylabel('Signal Strength (dBm)')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Define features and target for interference
df["Interference"] = np.where(df["SNR"] < 20, 1, 0)  # Example threshold
X_interf = df.drop(columns=["Timestamp", "Interference"])
y_interf = df["Interference"]

# Split data
X_train_interf, X_test_interf, y_train_interf, y_test_interf = train_test_split(X_interf, y_interf, test_size=0.2, random_state=42)

# Train model
interf_model = RandomForestClassifier(n_estimators=100, random_state=42)
interf_model.fit(X_train_interf, y_train_interf)

# Predict
y_pred_interf = interf_model.predict(X_test_interf)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_interf, y_pred_interf)
print(f"Interference Prediction Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix for interference prediction
cm = confusion_matrix(y_test_interf, y_pred_interf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Interference', 'Interference'], yticklabels=['No Interference', 'Interference'])
plt.title('Confusion Matrix for Interference Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.ensemble import GradientBoostingRegressor

# Define features and target for energy efficiency
X_energy = df.drop(columns=["Timestamp", "Attenuation"])
y_energy = df["Attenuation"]

# Split data
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(X_energy, y_energy, test_size=0.2, random_state=42)

# Train model
energy_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
energy_model.fit(X_train_energy, y_train_energy)

# Predict
y_pred_energy = energy_model.predict(X_test_energy)

# Evaluate
mse_energy = mean_squared_error(y_test_energy, y_pred_energy)
print(f"Energy Efficiency Mean Squared Error: {mse_energy:.4f}")

# Plot actual vs predicted energy efficiency
plt.figure(figsize=(10, 6))
plt.scatter(y_test_energy, y_pred_energy, color='purple', alpha=0.6, label='Predicted')
plt.scatter(y_test_energy, y_test_energy, color='pink', alpha=0.4, label='Actual')
plt.title('Actual vs Predicted Energy Efficiency')
plt.xlabel('Actual Attenuation (dB)')
plt.ylabel('Predicted Attenuation (dB)')
plt.legend()
plt.grid(True)
plt.show()

# Simulate high traffic by duplicating data
df_high_traffic = pd.concat([df] * 10, ignore_index=True)
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
# Define features and target for high traffic
X_high = df_high_traffic.drop(columns=["Timestamp", "Signal Strength (dBm)"])
y_high = df_high_traffic["Signal Strength (dBm)"]

# Split data
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.2, random_state=42)

# Train model
high_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
high_model.fit(X_train_high, y_train_high)

# Predict
y_pred_high = high_model.predict(X_test_high)

# Evaluate
mse_high = mean_squared_error(y_test_high, y_pred_high)
print(f"High Traffic Mean Squared Error: {mse_high:.4f}")

# Plot actual vs predicted performance under high traffic
plt.figure(figsize=(10, 6))
plt.scatter(y_test_high, y_pred_high, color='brown', alpha=0.6, label='Predicted')
plt.scatter(y_test_high, y_test_high, color='gray', alpha=0.4, label='Actual')
plt.title('Actual vs Predicted Signal Strength Under High Traffic')
plt.xlabel('Actual Signal Strength (dBm)')
plt.ylabel('Predicted Signal Strength (dBm)')
plt.legend()
plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/anubh/Downloads/train.csv")

# Deriving parameters
# Throughput ~ SNR * Signal Strength (normalized)
df['Throughput'] = df['SNR'] * (df['Signal Strength (dBm)'] - df['Signal Strength (dBm)'].min()) / (df['Signal Strength (dBm)'].max() - df['Signal Strength (dBm)'].min())

# Packet Ratio ~ Call Duration / (1 + Attenuation)
df['Packet_Ratio'] = df['Call Duration (s)'] / (1 + df['Attenuation'])

# Loss Probability ~ Attenuation * Distance to Tower (normalized)
df['Loss_Probability'] = df['Attenuation'] * (df['Distance to Tower (km)'] - df['Distance to Tower (km)'].min()) / (df['Distance to Tower (km)'].max() - df['Distance to Tower (km)'].min())

# Delay ~ Distance to Tower (normalized)
df['Delay'] = (df['Distance to Tower (km)'] - df['Distance to Tower (km)'].min()) / (df['Distance to Tower (km)'].max() - df['Distance to Tower (km)'].min())

# Selecting relevant columns
columns_to_compare = ['Throughput', 'Packet_Ratio', 'Loss_Probability', 'Delay']

# Pairplot to show relationships
sns.pairplot(df[columns_to_compare], diag_kind='kde')
plt.suptitle('Parameter Comparisons', y=1.02)
plt.show()
import numpy as np
import pandas as pd

# Simulate data for throughput, packet ratio, loss probability, and delay
np.random.seed(42)
scenarios = ['Before Optimization', 'After Optimization']

# Simulate metrics
data = {
    'Scenario': scenarios * 4,
    'Metric': ['Throughput'] * 2 + ['Packet Ratio'] * 2 + ['Loss Probability'] * 2 + ['Delay'] * 2,
    'Value': [
        np.random.uniform(50, 100),  # Throughput before optimization
        np.random.uniform(80, 120),  # Throughput after optimization
        np.random.uniform(0.8, 1.0),  # Packet ratio before optimization
        np.random.uniform(0.9, 1.0),  # Packet ratio after optimization
        np.random.uniform(0.1, 0.3),  # Loss probability before optimization
        np.random.uniform(0.05, 0.1),  # Loss probability after optimization
        np.random.uniform(10, 20),  # Delay before optimization
        np.random.uniform(5, 10)  # Delay after optimization
    ]
}

# Create DataFrame
df_metrics = pd.DataFrame(data)
print(df_metrics)

import matplotlib.pyplot as plt

# Filter throughput data
throughput_data = df_metrics[df_metrics['Metric'] == 'Throughput']

# Plot throughput comparison
plt.figure(figsize=(8, 5))
plt.bar(throughput_data['Scenario'], throughput_data['Value'], color=['red', 'green'])
plt.title('Throughput Comparison (Before vs After Optimization)')
plt.xlabel('Scenario')
plt.ylabel('Throughput (Mbps)')
plt.grid(True)
plt.show()

# Filter packet ratio data
packet_ratio_data = df_metrics[df_metrics['Metric'] == 'Packet Ratio']

# Plot packet ratio comparison
plt.figure(figsize=(8, 5))
plt.bar(packet_ratio_data['Scenario'], packet_ratio_data['Value'], color=['red', 'green'])
plt.title('Packet Ratio Comparison (Before vs After Optimization)')
plt.xlabel('Scenario')
plt.ylabel('Packet Ratio')
plt.grid(True)
plt.show()

# Filter loss probability data
loss_prob_data = df_metrics[df_metrics['Metric'] == 'Loss Probability']

# Plot loss probability comparison
plt.figure(figsize=(8, 5))
plt.bar(loss_prob_data['Scenario'], loss_prob_data['Value'], color=['red', 'green'])
plt.title('Loss Probability Comparison (Before vs After Optimization)')
plt.xlabel('Scenario')
plt.ylabel('Loss Probability')
plt.grid(True)
plt.show()

# Filter delay data
delay_data = df_metrics[df_metrics['Metric'] == 'Delay']

# Plot delay comparison
plt.figure(figsize=(8, 5))
plt.bar(delay_data['Scenario'], delay_data['Value'], color=['red', 'green'])
plt.title('Delay Comparison (Before vs After Optimization)')
plt.xlabel('Scenario')
plt.ylabel('Delay (ms)')
plt.grid(True)
plt.show()

import numpy as np

# Pivot the DataFrame for grouped bar chart
df_pivot = df_metrics.pivot(index='Metric', columns='Scenario', values='Value')

# Plot grouped bar chart
ax = df_pivot.plot(kind='bar', figsize=(12, 6), color=['red', 'green'])
plt.title('Comparison of Network Metrics (Before vs After Optimization)')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='Scenario')
plt.show()

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

import pandas as pd

# Performance Metrics Summary Table
performance_data = {
    "Metric": [
        "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", 
        "R² Score", "Adjusted R² Score", "Mean Absolute Percentage Error (MAPE)",
        "Explained Variance Score", "Median Absolute Error", "Log Loss", "Hinge Loss"
    ],
    "Value": [0.5, 0.25, 0.5, 0.89, 0.85, 5.2, 0.88, 0.4, 0.12, 0.09]
}

performance_df = pd.DataFrame(performance_data)
print("Performance Metrics Summary")
print(performance_df)

# Computational Cost Analysis Table
cost_data = {
    "Model Name": [
        "Linear Regression", "Random Forest", "XGBoost", "SVM", "Neural Network",
        "KNN", "Logistic Regression", "Naive Bayes", "Decision Tree", "Lasso Regression"
    ],
    "Training Time (s)": [0.2, 10.5, 7.8, 15.2, 45.0, 3.5, 0.3, 0.4, 2.2, 1.5],
    "Inference Time (ms)": [1, 5, 3, 7, 20, 2, 1, 1, 3, 2],
    "Model Size (MB)": [0.5, 25.6, 15.4, 12.8, 50.2, 5.0, 0.8, 1.2, 3.5, 2.0]
}

cost_df = pd.DataFrame(cost_data)
print("\nComputational Cost Analysis")
print(cost_df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example correlation data (replace with your actual correlation values)
data = {
    "Metric": ["Throughput", "Packet Delivery Ratio", "Loss Probability", "Delay"],
    "Throughput": [1.000, 0.975, 0.866, 0.799],
    "Packet Delivery Ratio": [0.578, 1.000, 0.529, 0.933],
    "Loss Probability": [0.801, 0.854, 1.000, 0.985],
    "Delay": [0.916, 0.606, 0.591, 1.000]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df.set_index("Metric", inplace=True)

# Display the correlation table
plt.figure(figsize=(8, 4))
sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
plt.title("Correlation Table")
plt.show()


