import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "rf_reg_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    return model, features

model, features = load_model()

st.set_page_config(page_title="Random Forest Regression", layout="centered")
st.title("🌲 Random Forest Regression")
st.write("Predict Height using Gender and Weight")

inputs = []

for feature in features:
    if feature.lower() == "gender":
        gender = st.selectbox("Gender", ["Female", "Male"])
        val = 1 if gender == "Male" else 0
    else:
        val = st.number_input(
            feature,
            min_value=0.0,
            step=0.1,
            value=0.0
        )
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)

if st.button("Predict"):
    prediction_in = model.predict(inputs)[0]   # inches
    prediction_cm = prediction_in * 2.54       # cm

    st.success(
    f"✅ Predicted Height: **{prediction_cm:.2f} cm** "
    f"({prediction_in:.2f} inches)"
)
