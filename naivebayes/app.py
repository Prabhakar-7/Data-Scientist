import os
import pickle
import os

# --- Load trained model and vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)
import streamlit as st

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)
>>>>>>> a67cd5b3 (Add Streamlit app and model files)

st.title("📧 Email / SMS Spam Detection")
st.write("Naive Bayes Machine Learning Model")

# -------------------------------
# Safe Absolute Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Error loading model files")
    st.exception(e)
    st.stop()

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area(
    "Enter the message to classify:",
    height=150,
    placeholder="Type your email or SMS text here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
