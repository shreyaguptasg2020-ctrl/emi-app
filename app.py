import streamlit as st
import pandas as pd
import numpy as np
import joblib
pipeline = joblib.load(open('C:/Users/Win 10/Desktop/emi_prediction_pipeline.pkl','rb'))

xgb_class = pipeline["classifier"]
xgb_reg = pipeline["regressor"]
le = pipeline["label_encoder"]
scaler = pipeline["scaler"]
import streamlit as st
st.set_page_config(page_title="EMI Prediction App", page_icon="💰", layout="wide")
st.title("💳 EMI Eligibility & EMI Amount Prediction")
st.markdown("Use this tool to predict whether an applicant is EMI eligible and estimate their monthly EMI amount.")

with st.form("emi_form"):
    st.header("🧾 Applicant Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 75, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate", "Other"])
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed", "Unemployed"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 5.0)
        company_type = st.selectbox("Company Type", ["Pvt Ltd", "Public Ltd", "Startup", "Other"])
        house_type = st.selectbox("House Type", ["Owned", "Rented", "Leased"])
        emi_scenario = st.selectbox("EMI Scenario", ["New Loan", "Top-Up Loan"])

    with col2:
        monthly_salary = st.number_input("Monthly Salary (₹)", 0.0, 1000000.0, 40000.0, step=1000.0)
        monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 200000.0, 10000.0, step=1000.0)
        family_size = st.number_input("Family Size", 1, 10, 3)
        dependents = st.number_input("Dependents", 0, 5, 1)
        school_fees = st.number_input("School Fees (₹)", 0.0, 100000.0, 5000.0, step=500.0)
        college_fees = st.number_input("College Fees (₹)", 0.0, 200000.0, 8000.0, step=500.0)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0.0, 50000.0, 2000.0, step=500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0, 100000.0, 10000.0, step=500.0)

    with col3:
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0.0, 100000.0, 2000.0, step=500.0)
        existing_loans = st.number_input("Existing Loans (count)", 0, 10, 1)
        current_emi_amount = st.number_input("Current EMI Amount (₹)", 0.0, 200000.0, 0.0, step=1000.0)
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        bank_balance = st.number_input("Bank Balance (₹)", 0.0, 10000000.0, 50000.0, step=1000.0)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0.0, 500000.0, 10000.0, step=1000.0)
        requested_amount = st.number_input("Requested Loan Amount (₹)", 0.0, 5000000.0, 200000.0, step=10000.0)
        requested_tenure = st.number_input("Requested Tenure (Months)", 6, 120, 36)

    submitted = st.form_submit_button("🔍 Predict EMI")
if submitted:
    # Convert to DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }])

    # Encode categorical columns safely
    for col in input_data.select_dtypes(include="object").columns:
        try:
            input_data[col] = le.transform(input_data[col])
        except ValueError:
            input_data[col] = 0  # for unseen category

    # Scale numerical columns
    input_scaled = scaler.transform(input_data)

    # Step 1: Predict Eligibility
    eligibility_pred = xgb_class.predict(input_scaled)[0]

    # Step 2: Predict EMI amount (if eligible)
    st.divider()
    if eligibility_pred == 1:  # assuming 1 = Eligible
        emi_pred = xgb_reg.predict(input_scaled)[0]
        st.success("✅ Applicant is **EMI Eligible**")
        st.metric(label="💰 Predicted EMI Amount", value=f"₹{emi_pred:,.2f}")
    else:
        st.error("❌ Applicant is **Not Eligible** for EMI.")

