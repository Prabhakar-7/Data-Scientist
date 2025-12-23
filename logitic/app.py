import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Heart Disease Prediction (Reduced Features)")

# -------------------- User Inputs (pre-filled for high-risk example) --------------------
age = st.number_input("Age", min_value=1, max_value=120, value=65)
sex = st.selectbox("Sex", options=[0, 1], index=1, format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type", options=[0, 1, 2, 3], index=2)
trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=160)
chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=300)
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=120)
exang = st.selectbox("Exercise induced angina", options=[0, 1], index=1)

# -------------------- Prepare Input DataFrame --------------------
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, thalach, exang]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "thalach", "exang"])

st.write("Your input data:")
st.dataframe(input_data)

# -------------------- Predict Button --------------------
if st.button("Predict"):
    # Step 1: Rename 'sex' to 'male' if model expects it
    if 'male' in model.feature_names_in_:
        input_data.rename(columns={'sex': 'male'}, inplace=True)

    # Step 2: One-hot encode categorical variables
    input_data = pd.get_dummies(input_data)

    # Step 3: Add missing columns expected by the model
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # Step 4: Reorder columns exactly like training
    input_data = input_data[model.feature_names_in_]

    # Step 5: Predict probability
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"Predicted probability of heart disease: {prob:.2f}")

    # Step 6: Display risk level
    if prob >= 0.6:
        st.error("Warning: High risk of heart disease!")
    elif prob >= 0.3:
        st.warning("Moderate risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
