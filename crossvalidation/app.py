import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

st.set_page_config(page_title="Cross Validation App", layout="wide")
st.title("🔁 Cross Validation Demo")
st.write("K-Fold and Stratified K-Fold Cross Validation")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Cross Validation Settings")
    cv_type = st.radio("CV Type", ["K-Fold", "Stratified K-Fold"])
    n_splits = st.slider("Number of Folds", 2, 10, 5)

    model = LogisticRegression(max_iter=1000)

    if cv_type == "K-Fold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_scaled, y, cv=cv)

    st.subheader("Cross Validation Results")
    st.write("Fold Scores:", scores)
    st.write("Mean Accuracy:", scores.mean())
