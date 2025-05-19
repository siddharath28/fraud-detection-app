import streamlit as st
import numpy as np
import joblib
import random

# Load trained model
model = joblib.load("fraud_detection_rf_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.image("https://www.alamy.com/credit-card-security-banner-concept-of-protection-online-payments-and-money-from-fraud-and-mobile-scam-vector-landing-page-with-isometric-banking-cards-shield-and-padlock-image368461429.html")  # Replace with your banner URL if needed
st.title("💳 Credit Card Fraud Detection App")

st.markdown("_This app uses a trained machine learning model to predict whether a transaction is fraudulent based on user inputs._")

# Sidebar Instructions
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
- Input all 28 `V` features (from PCA).
- Provide the **scaled transaction amount** and **scaled time**.
- Set `is_outlier` to 1 if the transaction is an outlier.
- Click **Predict** to view the result.
""")

# Generate default keys for state
v_inputs = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", key=f"V{i}", value=0.0, format="%.6f")
    v_inputs.append(val)

scaled_amount = st.number_input("Scaled Amount", key="amount", value=0.0, format="%.6f")
scaled_time = st.number_input("Scaled Time", key="time", value=0.0, format="%.6f")
is_outlier = st.selectbox("Is Outlier", options=[0, 1], index=0, key="is_outlier")

# Example loader
if st.button("🔁 Load Example"):
    for i in range(1, 29):
        st.session_state[f"V{i}"] = round(random.uniform(-3, 3), 6)
    st.session_state["amount"] = 0.25
    st.session_state["time"] = -0.12
    st.session_state["is_outlier"] = 0

# Collect input for prediction
input_data = np.array(v_inputs + [scaled_amount, scaled_time, is_outlier]).reshape(1, -1)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Not Fraud (Confidence: {1 - prediction_proba:.2f})")

# Expander with model info
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    - **Model**: Random Forest Classifier
    - **Handling Imbalance**: SMOTE oversampling
    - **Key Metrics**:
        - F1 Score (Fraud): 0.86
        - ROC AUC: 0.96
    """)

# GitHub link
st.markdown("---")
st.markdown("🔗 [View Project on GitHub](https://github.com/siddharath28/fraud-detection-app)")
