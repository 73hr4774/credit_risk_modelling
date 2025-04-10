import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('model.joblib')
le_employment = joblib.load('employment_encoder.joblib')
le_marital = joblib.load('marital_encoder.joblib')

st.title(" Credit Risk Prediction App")
st.write("Enter applicant information below:")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, value=15000)
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)

employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])

# Preprocess input
emp_clean = employment_status.lower().strip()
mar_clean = marital_status.lower().strip()

try:
    emp_encoded = le_employment.transform([emp_clean])[0]
    mar_encoded = le_marital.transform([mar_clean])[0]
except ValueError:
    st.error("Selected value not recognized by the encoder. Check your training data.")
    st.stop()

# Predict
input_data = np.array([[age, income, loan_amount, credit_score, emp_encoded, mar_encoded]])
prediction = model.predict(input_data)[0]
pd_val = model.predict_proba(input_data)[0][1]
lgd = np.random.uniform(0.3, 0.6)
ead = loan_amount
expected_loss = pd_val * lgd * ead

# Display results
st.subheader(" Prediction Result")
st.write("Default Risk:", "Yes" if prediction == 1 else "No ✅")
st.write("Probability of Default (PD):", round(pd_val, 2))
st.write("Loss Given Default (LGD):", round(lgd, 2))
st.write("Exposure at Default (EAD):", ead)
st.write("Expected Loss:", round(expected_loss, 2))
