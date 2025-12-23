import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("🧠 Social Network Ads – KNN Predictor")

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Social_Network_Ads.csv")

df = pd.read_csv(csv_path)

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("User Input")

age = st.sidebar.slider("Age", 18, 60, 30)
salary = st.sidebar.slider("Estimated Salary", 15000, 150000, 50000)

# Prediction
input_data = np.array([[age, salary]])
prediction = model.predict(input_data)

# Output
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("✅ User is likely to PURCHASE")
else:
    st.error("❌ User is NOT likely to purchase")

# Show data
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())
