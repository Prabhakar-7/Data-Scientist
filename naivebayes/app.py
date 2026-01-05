import os
import pickle
import streamlit as st

st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email / SMS Spam Detection")

# Absolute paths (Streamlit Cloud safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Model files not found")
    st.exception(e)
    st.stop()

text = st.text_area("Enter message")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
