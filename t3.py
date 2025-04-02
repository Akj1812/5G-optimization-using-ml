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
