import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("framingham_heart_disease.csv")

# Select correct features
X = df[["age", "male"]]
y = df["TenYearCHD"]

# Handle missing values
X = X.fillna(X.mean())

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ model.pkl created successfully!")
