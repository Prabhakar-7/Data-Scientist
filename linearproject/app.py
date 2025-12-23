import od
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Spotify Popularity Predictor", layout="centered")

st.title("🎧 Spotify Song Popularity Predictor")
st.write("Predict song popularity using simple audio features")

# ---------------------------
# Load dataset
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spotify_song_popularity.csv")

df = pd.read_csv(csv_path)

features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
X = df[features]
y = df['popularity']

# ---------------------------
# Train model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("🎶 Song Features")

danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -10.0)
valence = st.sidebar.slider("Valence (Mood)", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("Tempo (BPM)", 60, 200, 120)

# ---------------------------
# Prediction
# ---------------------------
input_data = np.array([[danceability, energy, loudness, valence, tempo]])
prediction = model.predict(input_data)

# ---------------------------
# Output
# ---------------------------
st.subheader("🎯 Predicted Popularity")
st.success(f"{prediction[0]:.2f} / 100")

# ---------------------------
# Explanation
# ---------------------------
st.markdown("""
### 📌 Model Info
- Algorithm: **Linear Regression**
- Features used: Danceability, Energy, Loudness, Valence, Tempo
- No feature scaling (simpler & interpretable)
""")
