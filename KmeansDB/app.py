import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine DBSCAN Clustering", layout="wide")
st.title("🍷 Wine Clustering with DBSCAN")
st.write("Unsupervised clustering of wines based on chemical properties.")

# ---------------------------
# Step 1: Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Step 2: Feature Selection
    # ---------------------------
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose features:", options=df.columns.tolist(), default=df.columns.tolist())
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        X = df[features]

        # ---------------------------
        # Step 3: Standardize Features
        # ---------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------------------------
        # Step 4: DBSCAN Parameter Inputs
        # ---------------------------
        st.subheader("DBSCAN Parameters")
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        min_samples = st.slider("Minimum Samples (min_samples)", min_value=1, max_value=20, value=5, step=1)

        # ---------------------------
        # Step 5: Apply DBSCAN
        # ---------------------------
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        df['Cluster'] = clusters

        st.subheader("Cluster Assignment")
        st.dataframe(df[['Cluster'] + features].head(10))

        # ---------------------------
        # Step 6: Cluster Summary
        # ---------------------------
        st.subheader("Cluster Counts")
        st.write(df['Cluster'].value_counts().sort_index())

        # ---------------------------
        # Step 7: Visualizations
        # ---------------------------
        st.subheader("Cluster Visualizations")

        # Pairplot for first 2 features
        if len(features) >= 2:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(
                x=df[features[0]],
                y=df[features[1]],
                hue=df['Cluster'],
                palette="tab10",
                s=100,
                ax=ax
            )
            ax.set_title(f"DBSCAN Clusters: {features[0]} vs {features[1]}")
            st.pyplot(fig)

        # Histograms of features
        st.subheader("Feature Distributions by Cluster")
        for feature in features:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=feature, hue='Cluster', multiple='stack', palette="tab10", bins=20)
            st.pyplot(fig)
