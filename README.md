# 💳 Credit Card Fraud Detection App

A machine learning-powered Streamlit web application that detects fraudulent credit card transactions using a trained Random Forest model. Built with a focus on real-time prediction and model interpretability, this app is an excellent showcase of handling imbalanced data in a real-world setting.

---

## 🌐 Live Demo

👉 [Click here to try the app](https://new-fraud-detection-app.streamlit.app)  
_You can interact with the model by entering transaction feature values to check if it’s fraudulent._

---

## 🧠 Model Highlights

- **Model Used:** Random Forest Classifier
- **Dataset:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Techniques Used:**
  - SMOTE for class imbalance correction
  - Feature scaling and log transformation
  - Outlier flag as engineered feature
- **Performance Metrics:**
  - F1 Score (Fraud): **0.86**
  - ROC AUC: **0.96**

---

## 🖥️ Features

- Input fields for all PCA-based features `V1–V28`
- Scaled inputs for `Transaction Amount` and `Time`
- Outlier flag (`is_outlier`) selector
- Real-time fraud probability output
- Confidence level for every prediction
- Clean, interactive UI with Streamlit

---

## 📁 Project Structure

📦 fraud-detection-app/
├── app.py # Streamlit frontend
├── fraud_detection_rf_model.pkl # Trained ML model
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

Special thanks to the ULB Machine Learning Group for providing the credit card fraud dataset on Kaggle.
