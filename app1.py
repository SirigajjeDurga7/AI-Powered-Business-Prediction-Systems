import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

model = joblib.load("loan_random_forest.pkl")
scaler = joblib.load("loan_scaler.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to right, #0f172a, #020617);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #2563eb, #3b82f6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 14px;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #1d4ed8, #2563eb);
    color: white;
}

.input-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE SECTION
# ==========================================

st.markdown("""
# 🏦 AI-Powered Loan Approval Prediction System

### 📊 Smart Banking Risk Analytics Dashboard

Predict:
- Loan Approval
- Default Risk
- Customer Risk Level
- Banking Recommendations
""")

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📝 Enter Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    income = st.number_input(
        "Annual Income",
        min_value=10000,
        max_value=500000,
        value=50000,
        step=1000
    )

    credit_score = st.slider(
        "Credit Score",
        300,
        850,
        650
    )

with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        max_value=500000,
        value=100000,
        step=1000
    )

    months_employed = st.slider(
        "Months Employed",
        0,
        300,
        24
    )

    interest_rate = st.slider(
        "Interest Rate",
        1.0,
        30.0,
        10.0
    )

with col3:

    dti_ratio = st.slider(
        "Debt-To-Income Ratio",
        0.0,
        1.0,
        0.30
    )

    num_credit_lines = st.slider(
        "Number of Credit Lines",
        1,
        10,
        3
    )

st.markdown("")

# ==========================================
# PREDICT BUTTON
# ==========================================

predict_button = st.button("🔍 Predict Loan Status")

# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # CREATE DATAFRAME

    input_data = pd.DataFrame({

        'Age': [age],

        'Income': [income],

        'LoanAmount': [loan_amount],

        'CreditScore': [credit_score],

        'MonthsEmployed': [months_employed],

        'InterestRate': [interest_rate],

        'DTIRatio': [dti_ratio],

        'NumCreditLines': [num_credit_lines]

    })

    # SCALE INPUT

    input_scaled = scaler.transform(input_data)

    # MODEL PREDICTION

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1] * 100

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    # ==========================================
    # BUSINESS LOGIC
    # ==========================================

    if (
        prediction == 1 or
        credit_score < 500 or
        dti_ratio > 0.7 or
        income < 30000 or
        loan_amount > income * 4
    ):

        risk = "HIGH"

        st.error(
            "❌ Loan is likely to be Rejected."
        )

    elif (
        credit_score < 650 or
        dti_ratio > 0.5
    ):

        risk = "MEDIUM"

        st.warning(
            "⚠ Loan Approval is Risky."
        )

    else:

        risk = "LOW"

        st.success(
            "✅ Loan is likely to be Approved."
        )

    # ==========================================
    # METRICS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Default Probability",
            f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    st.markdown("---")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================

    st.subheader("💡 Banking Recommendations")

    if risk == "HIGH":

        st.error("""
        - Reject or manually review application
        - Verify employment and income proof
        - Reduce loan eligibility
        - Request collateral/security
        - High probability of default
        """)

    elif risk == "MEDIUM":

        st.warning("""
        - Approve with monitoring
        - Verify repayment capability
        - Offer moderate interest rates
        - Monitor customer transactions
        """)

    else:

        st.success("""
        - Applicant is financially stable
        - Safe candidate for loan approval
        - Low probability of default
        - Eligible for standard loan offers
        """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<center>

### 🤖 AI Banking Analytics Platform

Built using Streamlit & Machine Learning

</center>
""", unsafe_allow_html=True)
