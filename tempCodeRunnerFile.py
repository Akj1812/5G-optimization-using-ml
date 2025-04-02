import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("File exists:", os.path.exists("train.csv"))
# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Display basic information
df.info()
print(df.head())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Environment', 'Tower ID', 'User ID', 'Call Type', 'Incoming/Outgoing']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Handling missing values
df = df.dropna()
print("DataFrame shape after dropping NA:", df.shape)

# Outlier detection and removal (Using IQR Method)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print("DataFrame shape after outlier removal:", df.shape)

# Ensure data is not empty
if df.shape[0] == 0:
    raise ValueError("Dataset is empty after outlier removal. Adjust outlier handling.")

# Define features (X) and target variables (Y)
X = df.drop(columns=['Timestamp', 'Signal Strength (dBm)'])
Y = df['Signal Strength (dBm)']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Display evaluation results
print(f"\nPerformance Metrics for Signal Strength (dBm):")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plotting all graphs in a single window

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 8))

# 1. Scatter plot: Actual vs Predicted values
axs[0, 0].scatter(Y_test, Y_pred, color='blue', alpha=0.6, label='Predicted')
axs[0, 0].scatter(Y_test, Y_test, color='red', alpha=0.4, label='Actual')
axs[0, 0].set_title('Actual vs Predicted Signal Strength')
axs[0, 0].set_xlabel('Actual Signal Strength (dBm)')
axs[0, 0].set_ylabel('Predicted Signal Strength (dBm)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Scatter plot: Residuals (Actual - Predicted)
residuals = Y_test - Y_pred

# Separate positive and negative residuals
positive_residuals = residuals[residuals >= 0]
negative_residuals = residuals[residuals < 0]

# Scatter plot for residuals with two different colors
axs[0, 1].scatter(positive_residuals.index, positive_residuals, color='green', alpha=0.6, label='Positive Residuals')
axs[0, 1].scatter(negative_residuals.index, negative_residuals, color='red', alpha=0.6, label='Negative Residuals')

# Set title and labels
axs[0, 1].set_title('Residuals (Actual signal strength  - Predicted signal strength) vs Index')
axs[0, 1].set_xlabel('Index(DataFrame)')
axs[0, 1].set_ylabel('Residual')

# Show legend
axs[0, 1].legend()

# Add grid
axs[0, 1].grid(True)

# 3. Bar plot: Feature importance (from the trained model)
feature_importance = model.feature_importances_
sns.barplot(x=feature_importance, y=X.columns, ax=axs[1, 0], color='green')
axs[1, 0].set_title('Feature Importance')
axs[1, 0].set_xlabel('Importance Score')
axs[1, 0].set_ylabel('Features')

# 4. Scatter plot: Mean Squared Error vs Number of Estimators (for model performance check)
estimators = np.arange(10, 201, 10)
mse_list = []
for n_estimators in estimators:
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.001, max_depth=6, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred_temp = model.predict(X_test)
    mse_temp = mean_squared_error(Y_test, Y_pred_temp)
    mse_list.append(mse_temp)

# Change scatter to line plot
axs[1, 1].plot(estimators, mse_list, color='purple', marker='o', linestyle='-', linewidth=2)
axs[1, 1].set_title('Mean Squared Error vs Number of Estimators')
axs[1, 1].set_xlabel('Number of Estimators')
axs[1, 1].set_ylabel('Mean Squared Error')
axs[1, 1].grid(True)


# Adjust layout to avoid overlapping of graphs
plt.tight_layout()

# Display all plots at once
plt.show()
