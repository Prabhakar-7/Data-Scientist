# app.py
import streamlit as st
import pickle
import os

# --- Load trained model and vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vector = pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Message Category Predictor", layout="centered")
st.title("📩 Message Category Prediction")
st.write("Enter a message below and the model will predict its category.")

# User input
user_input = st.text_area("Your Message:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform input and predict
        input_vector = vector.transform([user_input])
        prediction = clf.predict(input_vector)[0]
        prediction_proba = clf.predict_proba(input_vector)[0]

        st.success(f"**Predicted Category:** {prediction}")

        # Display probabilities for all categories
        st.write("**Prediction Probabilities:**")
        categories = clf.classes_
        for cat, proba in zip(categories, prediction_proba):
            st.write(f"- {cat}: {proba*100:.2f}%")
    else:
        st.warning("Please enter a message to predict.")
