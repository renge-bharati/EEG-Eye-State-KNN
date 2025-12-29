import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Helper function to load files safely
# -----------------------------
def load_pickle(file_name):
    if not os.path.exists(file_name):
        st.error(f"❌ File not found: {file_name}")
        st.stop()
    with open(file_name, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Load model and scaler
# -----------------------------
model = load_pickle("knn_eeg_model.pkl")
scaler = load_pickle("scaler.pkl")

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection using KNN")
st.write("Predict whether eyes are **OPEN** or **CLOSED** using EEG signals")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Enter EEG Sensor Values")

sensor_names = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

features = []
for sensor in sensor_names:
    value = st.number_input(sensor, value=0.0)
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
