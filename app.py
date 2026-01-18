import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Predict the likelihood of heart disease using patient medical data.")

# -------------------------------
# Load Saved Artifacts
# -------------------------------
model = joblib.load("model.pkl")        # Pre-trained ML model
scaler = joblib.load("scaler.pkl")      # Saved StandardScaler
features = joblib.load("features.pkl")  # List of features

# -------------------------------
# Sidebar Input Fields
# -------------------------------
st.sidebar.header("🧑‍⚕️ Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 50)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
restecg = st.sidebar.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ("No", "Yes"))
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (1–3)", [1, 2, 3])

# -------------------------------
# Encode Categorical Inputs
# -------------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame([[ 
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]], columns=features)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Heart Disease"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2%}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
