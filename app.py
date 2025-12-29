import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="EEG Eye State Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Safe pickle loader
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
scaler = load_pickle("scalar(1).pkl")   # 👈 YOUR FILE NAME

# -----------------------------
# App UI
# -----------------------------
st.title("🧠 EEG Eye State Detection using KNN")
st.write(
    "Predict whether a person's eyes are **OPEN** or **CLOSED** "
    "using EEG brain signal values."
)

st.divider()

# -----------------------------
# EEG Input Fields
# -----------------------------
st.subheader("Enter EEG Sensor Values")

sensor_names = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

features = []
for sensor in sensor_names:
    value = st.number_input(sensor, value=0.0, format="%.4f")
    features.append(value)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Eye State"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("👁️ Eyes are **OPEN**")
    else:
        st.error("👁️ Eyes are **CLOSED**")

st.caption("Model: KNN | Dataset: EEG Eye State")
