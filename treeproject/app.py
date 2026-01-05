import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Decision Tree App", page_icon="🌳")

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

model = load_model()

st.title("🌳 Decision Tree Prediction App")
st.write("Predict customer purchase based on Age & Salary")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Salary", min_value=1000, max_value=200000, value=35000)

if st.button("Predict"):
    input_df = pd.DataFrame([[age, salary]], columns=["Age", "Salary"])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Customer WILL Purchase")
    else:
        st.error("❌ Customer will NOT Purchase")
