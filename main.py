import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('loan_default.csv')

# Clean column names
df.columns = df.columns.str.lower()

# Encode categorical variables
le_employment = LabelEncoder()
le_marital = LabelEncoder()

df['employment_status'] = df['employment_status'].str.lower().str.strip()
df['marital_status'] = df['marital_status'].str.lower().str.strip()

df['employment_status'] = le_employment.fit_transform(df['employment_status'])
df['marital_status'] = le_marital.fit_transform(df['marital_status'])

# Save encoders
joblib.dump(le_employment, 'employment_encoder.joblib')
joblib.dump(le_marital, 'marital_encoder.joblib')

# Define X and y
X = df.drop('defaulted', axis=1)
y = df['defaulted']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
