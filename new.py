import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X1 = np.random.randn(100, 2) + np.array([2, 2])
X2 = np.random.randn(100, 2) + np.array([-2, -2])
X3 = np.random.randn(100, 2) + np.array([2, -2])

# Stack into a single dataset
X = np.vstack((X1, X2, X3))

# Create a DataFrame
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

# Save as CSV
file_path = "kmeans_large_dataset.csv"
df.to_csv(file_path, index=False)

print(f"CSV saved to: {file_path}")
print(df.head())
