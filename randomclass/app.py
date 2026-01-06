import streamlit as st
import pickle
import numpy as np
from collections import Counter
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INT_FEATURES = {"age", "children", "dependents", "count", "no_of_items"}
FLOAT_FEATURES = {"income", "balance", "salary", "score"}

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "rf_model.pkl")
    features_path = os.path.join(BASE_DIR, "features.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    return model, features

model, features = load_model()

def predict_tree(node, row):
    if not isinstance(node, dict):
        return node
    if row[node['index']] < node['value']:
        return predict_tree(node['left'], row)
    else:
        return predict_tree(node['right'], row)

def random_forest_predict(trees, row):
    preds = [predict_tree(tree, row) for tree in trees]
    return Counter(preds).most_common(1)[0][0]

st.set_page_config(page_title="Random Forest Classifier", layout="centered")
st.title("🌲 Random Forest Classifier (From Scratch)")
st.write("Enter feature values to get prediction")

inputs = []

for feature in features:
    val = st.number_input(feature, min_value=0, step=1, value=0)
    if feature.lower() in INT_FEATURES:
        val = int(val)
    elif feature.lower() in FLOAT_FEATURES:
        val = float(val)
    inputs.append(val)

inputs = np.array(inputs)

if st.button("Predict"):
    prediction = random_forest_predict(model, inputs)
    st.success(f"✅ Prediction: **{prediction}**")
