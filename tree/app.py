import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("❤️ Heart Disease Prediction (Reduced Features)")
st.write("Enter patient details:")

@st.cache_resource
def train_model():
    import os

    BASE_DIR = os.path.dirname(__file__)
    data_path = os.path.join(BASE_DIR, "heart_dataset_5000.csv")

    df = pd.read_csv(data_path)

    X = df[["age", "cp", "chol", "bp"]]
    y = df["target"]

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )
    model.fit(X, y)

    return model

model = train_model()

# User inputs
age = st.number_input("Age", 1, 120, 45)
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
chol = st.number_input("Cholesterol", 100, 400, 200)
bp = st.number_input("Blood Pressure", 80, 200, 120)

# Predict
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, cp, chol, bp]],
        columns=["age", "cp", "chol", "bp"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.2f}%)")
