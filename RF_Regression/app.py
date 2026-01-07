import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("ebike_model.pkl", "rb"))

st.title("⚡ E-Bike Range Prediction")

battery = st.number_input("Battery Capacity (Wh)", 300, 700, 500)
speed = st.number_input("Average Speed (km/h)", 15, 40, 28)
weight = st.number_input("Rider Weight (kg)", 40, 120, 70)
terrain = st.selectbox("Terrain Type", [0, 1, 2])  # same encoding as training

input_df = pd.DataFrame({
    "battery_capacity_wh": [battery],
    "avg_speed_kmph": [speed],
    "rider_weight_kg": [weight],
    "terrain_type": [terrain]
})

prediction = model.predict(input_df)

st.success(f"🔋 Estimated Range: {prediction[0]:.2f} km")
