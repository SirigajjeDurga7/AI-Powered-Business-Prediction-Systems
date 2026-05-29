import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="AI Employee Attrition Predictor",
    page_icon="🚀",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #0E1117;
    text-align: center;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #45a049;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL & SCALER
# =========================================

model = joblib.load("random_forest.pkl")

scaler = joblib.load("scaler.pkl")

# =========================================
# HEADER
# =========================================

st.title("🚀 AI-Powered Employee Attrition Prediction System")

st.markdown("""
### 📊 HR Analytics Dashboard

Predict:
- Employee Attrition Risk
- Risk Category
- Employee Retention Insights
- HR Recommendations
""")

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("🧑 Employee Details")

# =========================================
# INPUTS
# =========================================

age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=50000,
    value=5000
)

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,
    4,
    2
)

total_working_years = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    5
)

overtime = st.sidebar.selectbox(
    "OverTime",
    ["Yes", "No"]
)

work_life_balance = st.sidebar.slider(
    "Work Life Balance",
    1,
    4,
    2
)

environment_satisfaction = st.sidebar.slider(
    "Environment Satisfaction",
    1,
    4,
    2
)

years_at_company = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    5
)

distance_from_home = st.sidebar.slider(
    "Distance From Home",
    1,
    30,
    5
)

job_involvement = st.sidebar.slider(
    "Job Involvement",
    1,
    4,
    2
)

# =========================================
# ENCODE VALUES
# =========================================

overtime_value = 1 if overtime == "Yes" else 0

# =========================================
# CREATE INPUT DATA
# =========================================

input_data = pd.DataFrame({

    'Age': [age],

    'MonthlyIncome': [monthly_income],

    'JobSatisfaction': [job_satisfaction],

    'TotalWorkingYears': [total_working_years],

    'OverTime': [overtime_value],

    'WorkLifeBalance': [work_life_balance],

    'EnvironmentSatisfaction': [environment_satisfaction],

    'YearsAtCompany': [years_at_company],

    'DistanceFromHome': [distance_from_home],

    'JobInvolvement': [job_involvement]

})

# =========================================
# SCALE DATA
# =========================================

input_scaled = scaler.transform(input_data)

# =========================================
# PREDICTION
# =========================================

if st.button("🔍 Predict Attrition"):

    probability = model.predict_proba(
        input_scaled
    )[0][1] * 100

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    # =========================================
    # RESULT
    # =========================================

    if probability >= 70:

        st.error(
            "⚠ Employee is likely to leave the organization."
        )

        risk = "HIGH"

    elif probability >= 40:

        st.warning(
            "⚠ Employee has medium attrition risk."
        )

        risk = "MEDIUM"

    else:

        st.success(
            "✅ Employee is likely to stay in the organization."
        )

        risk = "LOW"

    # =========================================
    # DISPLAY METRICS
    # =========================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Attrition Probability",
            value=f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            label="Risk Level",
            value=risk
        )

    st.markdown("---")

    # =========================================
    # HR RECOMMENDATIONS
    # =========================================

    st.subheader("💡 HR Recommendations")

    if risk == "HIGH":

        st.error("""
        - Improve employee engagement
        - Reduce overtime workload
        - Increase salary benefits
        - Conduct HR counseling sessions
        - Improve work-life balance
        """)

    elif risk == "MEDIUM":

        st.warning("""
        - Monitor employee satisfaction
        - Provide career growth opportunities
        - Conduct regular feedback sessions
        """)

    else:

        st.success("""
        - Employee retention is stable
        - Continue employee support programs
        - Maintain healthy work environment
        """)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown("""
<center>
© 2026 AI Employee Attrition Prediction System <br>
Built with Streamlit & Machine Learning
</center>
""", unsafe_allow_html=True)