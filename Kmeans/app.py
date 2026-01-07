import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# App title
st.title("K-Means Clustering App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns
    X = df.select_dtypes(include=["int64", "float64"])
    X = X.fillna(X.mean())

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select number of clusters
    k = st.slider("Select number of clusters (K)", 2, 10, 3)

    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    plot_df["cluster"] = clusters

    # Plot
    fig, ax = plt.subplots()
    for c in sorted(plot_df["cluster"].unique()):
        cluster_data = plot_df[plot_df["cluster"] == c]
        ax.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {c}")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clustering (PCA View)")
    ax.legend()

    st.pyplot(fig)
