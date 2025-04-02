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
