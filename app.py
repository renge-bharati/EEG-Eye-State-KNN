import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load Model and Scaler
# -----------------------------
with open("knn_eeg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection using KNN")
st.write("Predict whether eyes are **OPEN** or **CLOSED** using EEG signals")

# -----------------------------
# Input Fields (14 EEG Sensors)
# -----------------------------
st.subheader("Enter EEG Sensor Values")

features = []
sensor_names = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

for sensor in sensor_names:
    value = st.number_input(f"{sensor}", value=0.0)
    features.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Eye State"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("👁️ Eyes are OPEN")
    else:
        st.error("👁️ Eyes are CLOSED")
