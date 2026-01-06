import streamlit as st
import pickle
import numpy as np
from collections import Counter

# ----------------------------
# Load model & features
# ----------------------------
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, features


model, features = load_model()

# ----------------------------
# Prediction logic (same as notebook)
# ----------------------------
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



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Random Forest Classifier", layout="centered")

st.title("🌲 Random Forest Classifier (From Scratch)")
st.write("Enter feature values to get prediction")

inputs = []

for feature in features:
    if feature.lower() in ["age", "years", "count", "children", "no_of_items"]:
        val = st.number_input(
            feature,
            min_value=0,
            step=1,
            value=0
        )
    else:
        val = st.number_input(
            feature,
            value=0,
            step=1
            # format="%.2f"
        )

    inputs.append(val)


inputs = np.array(inputs)

if st.button("Predict"):
    prediction = random_forest_predict(model, inputs)
    st.success(f"✅ Prediction: **{prediction}**")
