import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Telecom Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================

model_lr = joblib.load("logistic_model_churn.pkl")
model_rf = joblib.load("random_forest_churn.pkl")
scaler = joblib.load("scaler_churn.pkl")

# ==========================================
# UI TITLE
# ==========================================

st.title("📊 Telecom Customer Churn Prediction System")

st.markdown("---")

# ==========================================
# SIDEBAR INPUTS (ONLY IMPORTANT FEATURES)
# ==========================================

st.sidebar.header("👤 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# ==========================================
# ENCODING (IMPORTANT FIX)
# ==========================================

gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
contract = contract_map[contract]

# ==========================================
# INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame([[
    gender,
    senior,
    partner,
    tenure,
    monthly_charges,
    contract
]])

# scale input
input_scaled = scaler.transform(input_data)

# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Predict Churn"):

    lr_prob = model_lr.predict_proba(input_scaled)[0][1]
    rf_prob = model_rf.predict_proba(input_scaled)[0][1]

    final_prob = (lr_prob + rf_prob) / 2 * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if final_prob >= 70:
        st.error("🚨 High Risk: Customer likely to churn")
        risk = "HIGH"

    elif final_prob >= 40:
        st.warning("⚠ Medium Risk: Customer may churn")
        risk = "MEDIUM"

    else:
        st.success("✅ Low Risk: Customer will stay")
        risk = "LOW"

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{final_prob:.2f}%")

    with col2:
        st.metric("Risk Level", risk)

    st.markdown("---")

    st.subheader("💡 Retention Strategy")

    if risk == "HIGH":
        st.error("""
        - Offer discounts
        - Improve service quality
        - Provide loyalty benefits
        """)

    elif risk == "MEDIUM":
        st.warning("""
        - Send promotional offers
        - Monitor usage behavior
        """)

    else:
        st.success("""
        - Maintain service quality
        - Continue engagement programs
        """)