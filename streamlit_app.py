import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load model and preprocessing
# -----------------------------
@st.cache_resource
def load_all():
    with open("lung_cancer_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_all()

st.set_page_config(page_title="Lung Cancer Risk Predictor", layout="centered")

st.title("Lung Cancer Risk Prediction")

st.write("Fill the details below to estimate lung cancer risk.")

# ---------------------------------------
# Input form
# ---------------------------------------
with st.form("input_form"):
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=1, max_value=120, value=35)

    smoking = st.selectbox("Smoking", [0, 1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])
    allergy = st.selectbox("Allergy", [0, 1])
    wheezing = st.selectbox("Wheezing", [0, 1])
    alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1])
    coughing = st.selectbox("Coughing", [0, 1])
    shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
    chest_pain = st.selectbox("Chest Pain", [0, 1])

    submitted = st.form_submit_button("Predict Risk")

# ---------------------------------------
# Prediction Logic
# ---------------------------------------
if submitted:
    try:
        gender_val = 1 if gender == "M" else 0

        features = np.array([
            gender_val, age, smoking, yellow_fingers, anxiety,
            peer_pressure, chronic_disease, fatigue, allergy,
            wheezing, alcohol_consumption, coughing,
            shortness_of_breath, swallowing_difficulty, chest_pain
        ]).reshape(1, -1)

        # scale
        scaled_features = scaler.transform(features)

        # model prediction
        prob = model.predict_proba(scaled_features)[0][1] * 100

        # --------------------------
        # Safe Progress Bar Handling
        # --------------------------
        progress_value = prob / 100

        try:
            if progress_value is None or isinstance(progress_value, str):
                progress_value = 0
            if progress_value != progress_value:  # NaN
                progress_value = 0
            progress_value = float(progress_value)
        except:
            progress_value = 0

        progress_value = max(0, min(progress_value, 1))

        st.subheader("Predicted Risk")
        st.progress(progress_value)

        st.write(f"### Risk Level: **{prob:.2f}%**")

        # --------------------------
        # Output message
        # --------------------------
        if prob < 20:
            st.success("Risk appears low. Maintain healthy habits.")
        elif prob < 50:
            st.warning("Moderate risk. Consider lifestyle improvements and screening.")
        else:
            st.error("High risk detected. A medical consultation is recommended.")

    except Exception as e:
        st.error("An error occurred while predicting. Please check inputs or files.")
