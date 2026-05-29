import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

model = joblib.load("fraud_random_forest.pkl")

scaler = joblib.load("fraud_scaler.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    color: white;
}

.metric-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1c1f26;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("💳 AI-Powered Credit Card Fraud Detection System")

st.markdown("""
### 🛡 Banking Fraud Analytics Dashboard

This system predicts:

- Fraudulent Transactions
- Safe Transactions
- Fraud Probability
- Transaction Risk Level
- Banking Recommendations

""")

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📥 Enter Transaction Details")
st.info("""
This system uses secure PCA-transformed banking features.
V1–V8 represent hidden transaction behavior patterns used for fraud detection.
""")
col1, col2, col3 = st.columns(3)

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        max_value=100000.0,
        value=500.0
    )

    v1 = st.slider(
        "V1",
        -30.0,
        30.0,
        0.0
    )

    v2 = st.slider(
        "V2",
        -30.0,
        30.0,
        0.0
    )

with col2:

    v3 = st.slider(
        "V3",
        -30.0,
        30.0,
        0.0
    )

    v4 = st.slider(
        "V4",
        -30.0,
        30.0,
        0.0
    )

    v5 = st.slider(
        "V5",
        -30.0,
        30.0,
        0.0
    )

with col3:

    v6 = st.slider(
        "V6",
        -30.0,
        30.0,
        0.0
    )

    v7 = st.slider(
        "V7",
        -30.0,
        30.0,
        0.0
    )

    v8 = st.slider(
        "V8",
        -30.0,
        30.0,
        0.0
    )

st.markdown("---")

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({

    'Amount': [amount],
    'V1': [v1],
    'V2': [v2],
    'V3': [v3],
    'V4': [v4],
    'V5': [v5],
    'V6': [v6],
    'V7': [v7],
    'V8': [v8]

})

# ==========================================
# SCALE INPUT
# ==========================================

input_scaled = scaler.transform(input_data)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Detect Fraud"):

    probability = model.predict_proba(
        input_scaled
    )[0][1] * 100

    st.markdown("---")

    st.subheader("📊 Detection Result")

    # ==========================================
    # FRAUD LOGIC
    # ==========================================

    if probability >= 70:

        st.error(
            "🚨 Fraudulent Transaction Detected!"
        )

        risk = "HIGH"

    elif probability >= 40:

        st.warning(
            "⚠ Suspicious Transaction Detected!"
        )

        risk = "MEDIUM"

    else:

        st.success(
            "✅ Legitimate Transaction"
        )

        risk = "LOW"

    # ==========================================
    # METRICS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Fraud Probability",
            value=f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            label="Risk Level",
            value=risk
        )

    st.markdown("---")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================

    st.subheader("💡 Banking Recommendations")

    if risk == "HIGH":

        st.error("""
        - Block transaction immediately
        - Notify account holder
        - Freeze card temporarily
        - Start fraud investigation
        """)

    elif risk == "MEDIUM":

        st.warning("""
        - Verify transaction manually
        - Send OTP confirmation
        - Monitor future activity
        """)

    else:

        st.success("""
        - Transaction appears safe
        - Allow transaction processing
        - Continue normal monitoring
        """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<center>

© 2026 AI Credit Card Fraud Detection System <br>
Built using Streamlit & Machine Learning

</center>
""", unsafe_allow_html=True)