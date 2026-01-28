# =====================================================
# TELCO CUSTOMER CHURN SYSTEM (FINAL CLEAN VERSION)
# =====================================================

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
print("\nLoading Dataset...")
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customer ID
data.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges column
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.fillna(0, inplace=True)

# -----------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------
encoders = {}

for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
print("\nTraining AI Model...")
model = lgb.LGBMClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
print("\nMODEL ACCURACY:")
print(round(accuracy_score(y_test, y_pred), 3))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# -----------------------------
# BUILD OUTPUT TABLE
# -----------------------------
results = X_test.copy()
results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred
results["Risk_Level"] = y_prob

results["Churn_Status"] = results["Predicted_Churn"].apply(
    lambda x: "YES" if x == 1 else "NO"
)

# -----------------------------
# DECODE BUSINESS OUTCOMES
# -----------------------------
def decode(col, values):
    return encoders[col].inverse_transform(values.astype(int))

results["Contract_Type"] = decode("Contract", results["Contract"].values)
results["Internet_Service"] = decode("InternetService", results["InternetService"].values)
results["Tech_Support"] = decode("TechSupport", results["TechSupport"].values)
results["Payment_Method"] = decode("PaymentMethod", results["PaymentMethod"].values)

# Streaming (TV / Movies combined)
tv = decode("StreamingTV", results["StreamingTV"].values)
movies = decode("StreamingMovies", results["StreamingMovies"].values)
results["Streaming_Services"] = [
    f"TV: {t}, Movies: {m}" for t, m in zip(tv, movies)
]

# Real values
results["Customer_Tenure"] = results["tenure"]
results["Monthly_Charges"] = results["MonthlyCharges"]

# -----------------------------
# ANALYTICS OUTPUTS
# -----------------------------
print("\nTOTAL CUSTOMER STATUS:")
print(results["Churn_Status"].value_counts())

churned = results[results["Churn_Status"] == "YES"]
revenue_loss = churned["MonthlyCharges"].sum()

print("\nTOTAL REVENUE LOSS DUE TO CHURN:")
print("₹", round(revenue_loss, 2))

# -----------------------------
# SAMPLE BUSINESS OUTPUT (MODEL TEST DATA)
# -----------------------------
print("\nSAMPLE BUSINESS OUTPUT:")
print(results[[
    "Churn_Status",
    "Risk_Level",
    "Contract_Type",
    "Monthly_Charges",
    "Internet_Service",
    "Tech_Support",
    "Streaming_Services",
    "Customer_Tenure",
    "Payment_Method"
]].head(10))

# =====================================================
# INTERACTIVE CUSTOMER INPUT (LIVE PREDICTION)
# =====================================================
print("\nENTER CUSTOMER DETAILS FOR LIVE PREDICTION")

def clean_text(x):
    return x.strip().title()

user_input = {
    "Contract": clean_text(input("Enter Contract Type (Month-to-month / One year / Two year): ")),
    "MonthlyCharges": float(input("Enter Monthly Charges: ")),
    "InternetService": clean_text(input("Enter Internet Service (DSL / Fiber optic / No): ")),
    "TechSupport": clean_text(input("Enter Tech Support (Yes / No): ")),
    "StreamingTV": clean_text(input("Enter Streaming TV (Yes / No): ")),
    "StreamingMovies": clean_text(input("Enter Streaming Movies (Yes / No): ")),
    "tenure": int(input("Enter Customer Tenure (months): ")),
    "PaymentMethod": clean_text(input("Enter Payment Method: "))
}

# -----------------------------
# PREPARE INPUT FOR MODEL
# -----------------------------
sample = pd.DataFrame([user_input])

# Add missing columns with default values
for col in X.columns:
    if col not in sample.columns:
        sample[col] = 0

# Encode categorical values
for col, le in encoders.items():
    if col in sample.columns:
        try:
            sample[col] = le.transform(sample[col].astype(str))
        except:
            sample[col] = 0

# Reorder columns
sample = sample[X.columns]

# -----------------------------
# PREDICT
# -----------------------------
prob = model.predict_proba(sample)[:, 1][0]
pred = model.predict(sample)[0]

# -----------------------------
# DISPLAY BUSINESS OUTPUT
# -----------------------------
print("\nLIVE BUSINESS OUTPUT")
print("-----------------------")
print("Churn Prediction:", "YES" if pred == 1 else "NO")
risk_label = "LOW 🟢" if prob < 0.3 else "MEDIUM 🟡" if prob < 0.6 else "HIGH 🔴"
print("Risk Level:", round(prob, 2), f"({risk_label})")
print("Contract Type:", user_input["Contract"])
print("Monthly Charges: ₹", user_input["MonthlyCharges"])
print("Internet Service:", user_input["InternetService"])
print("Tech Support:", user_input["TechSupport"])
print("Streaming Services: TV =", user_input["StreamingTV"],
      ", Movies =", user_input["StreamingMovies"])
print("Customer Tenure:", user_input["tenure"], "months")
print("Payment Method:", user_input["PaymentMethod"])

print("\nPROJECT EXECUTED SUCCESSFULLY!")