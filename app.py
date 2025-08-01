# app.py

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgb_model.pkl")

st.title("💰 Credit Default Prediction App")

st.write("Enter customer information below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (₹)", min_value=10000, max_value=200000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, max_value=1000000, value=100000)

# Prediction
if st.button("Predict Default Risk"):
    input_data = np.array([[age, income, credit_score, loan_amount]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk: Likely to Default (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk: Unlikely to Default (Probability: {prob:.2f})")
