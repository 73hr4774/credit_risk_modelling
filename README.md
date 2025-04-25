# 💳 Credit Risk Modelling

This project focuses on assessing the creditworthiness of individuals using machine learning. It predicts the likelihood of a loan default based on key customer attributes — helping financial institutions minimize risk.

---

## 🧾 Objective

To build a credit scoring model that:
- Predicts whether a customer will default (`Defaulted`: 1 or 0)
- Calculates risk parameters: **PD (Probability of Default)**, **LGD (Loss Given Default)**, and **EAD (Exposure at Default)**
- Supports credit risk assessment for banks or NBFCs

---

## 📊 Dataset Overview

- **Custom dataset** with features like:
  - `Age`, `Income`, `Loan_Amount`, `Credit_Score`
  - `Employment_Status`, `Marital_Status`
  - `Defaulted` (Target)

- Categorical Features:
  - `Employment_Status` (e.g., Employed, Unemployed)
  - `Marital_Status` (e.g., Single, Married)

---

## ⚙️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **ML Models**: Logistic Regression, Random Forest
- **Risk Metrics**: PD, LGD, EAD Calculation
- **Deployment**: Streamlit

---

## 🚀 How to Run Locally

1. Clone the repo  
```bash
git clone https://github.com/73hr4774/credit_risk_modelling.git
cd credit_risk_modelling
pip install -r requirements.txt
streamlit run app.py

🌐 Deployed App
👉 Click here to view the deployed Streamlit app
https://creditriskmodelling-79oaqvzpbhfwcmheajvzyv.streamlit.app/

📈 Risk Metrics Logic
PD (Probability of Default)
Probability that the customer will default on the loan.

LGD (Loss Given Default)
Portion of the loan not recovered when a customer defaults.

EAD (Exposure at Default)
The total value at risk at the time of default.

🧠 Key Insights
Defaulting is influenced by factors like income, loan amount, and credit score.

Logistic Regression provides interpretable PD scores.

Random Forest improves classification accuracy.
