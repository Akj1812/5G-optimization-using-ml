import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Sample dataset (replace with actual data)
np.random.seed(42)
X = np.random.rand(100, 5)
y = 3*X[:, 0] + 2*X[:, 1] + np.random.rand(100) * 0.1  # Target variable with some noise
feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Machine": SVR(kernel='rbf')
}

# Results storage
results = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature Importance (only for models that support it)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = permutation_importance(model, X_test, y_test, random_state=42).importances_mean
    
    top_features = [feature_names[i] for i in np.argsort(importance)[-3:][::-1]]  # Top 3 features
    
    results.append({
        "Model": name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R² Score": round(r2, 4),
        "Training Time (s)": round(train_time, 4),
        "Top 3 Features": ', '.join(top_features)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)
