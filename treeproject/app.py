import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Decision Tree App", page_icon="🌳")

MODEL_PATH = "decision_tree_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please check deployment.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
