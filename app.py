import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_detection_rf_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

st.markdown("Enter transaction details below to predict whether it's **fraudulent or not**.")

# Sidebar Instructions
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("- Input all 28 `V` features (from PCA).")
st.sidebar.markdown("- Provide the **scaled transaction amount** and **scaled time**.")
st.sidebar.markdown("- Set `is_outlier` to 1 if the transaction is considered an outlier, else 0.")
st.sidebar.markdown("- Click **Predict** to see results.")

# Input: V1 to V28
v_inputs = []
for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0, format="%.6f")
    v_inputs.append(value)

# Input: Scaled Amount and Scaled Time
scaled_amount = st.number_input("Scaled Amount", value=0.0, format="%.6f")
scaled_time = st.number_input("Scaled Time", value=0.0, format="%.6f")

# Input: is_outlier (0 or 1)
is_outlier = st.selectbox("Is Outlier", options=[0, 1], index=0)

# Combine all inputs
input_data = np.array(v_inputs + [scaled_amount, scaled_time, is_outlier]).reshape(1, -1)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Not Fraud (Confidence: {1 - prediction_proba:.2f})")
