import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Online Retail Hierarchical Clustering", layout="wide")
st.title("📦 Online Retail Clustering with Hierarchical Clustering")
st.write("Cluster transactions based on numeric features like Quantity and UnitPrice using Hierarchical Clustering.")

# ---------------------------
# Step 1: Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Step 2: Prepare Numeric Features
    # ---------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Select Numeric Features for Clustering")
    features = st.multiselect("Choose numeric features:", options=numeric_cols, default=['Quantity', 'UnitPrice'])
    
    if len(features) < 1:
        st.warning("Please select at least 1 numeric feature for clustering.")
    else:
        X = df[features].fillna(0)

        # ---------------------------
        # Step 3: Standardize Features
        # ---------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------------------------
        # Step 4: Hierarchical Clustering Parameters
        # ---------------------------
        st.subheader("Hierarchical Clustering Parameters")
        method = st.selectbox("Linkage Method", options=["single", "complete", "average", "ward"], index=3)
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)

        # ---------------------------
        # Step 5: Compute linkage and clusters
        # ---------------------------
        Z = linkage(X_scaled, method=method)
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        df['Cluster'] = cluster_labels

        # ---------------------------
        # Step 6: Dendrogram
        # ---------------------------
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(12,6))
        dendrogram(Z, labels=df.index.tolist(), leaf_rotation=90, leaf_font_size=6)
        plt.title(f"Hierarchical Clustering Dendrogram ({method} linkage)")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        fig.tight_layout()
        st.pyplot(fig)

        # ---------------------------
        # Step 7: Cluster Assignment
        # ---------------------------
        st.subheader("Cluster Assignment")
        st.dataframe(df[['InvoiceNo', 'CustomerID', 'Quantity', 'UnitPrice', 'Cluster']].head(10))

        # ---------------------------
        # Step 8: Cluster Counts
        # ---------------------------
        st.subheader("Cluster Counts")
        st.write(df['Cluster'].value_counts().sort_index())

        # ---------------------------
        # Step 9: Visualizations
        # ---------------------------
        st.subheader("Cluster Scatter Plot")
        if len(features) >= 2:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(
                x=df[features[0]],
                y=df[features[1]],
                hue=df['Cluster'],
                palette="tab10",
                s=50,
                ax=ax
            )
            ax.set_title(f"Clusters: {features[0]} vs {features[1]}")
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Feature Distributions by Cluster")
        for feature in features:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(data=df, x=feature, hue='Cluster', multiple='stack', palette='tab10', bins=20)
            fig.tight_layout()
            st.pyplot(fig)
