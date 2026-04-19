import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
import os
app_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(app_dir, 'best_model.pkl'))
feature_names = joblib.load(os.path.join(app_dir, 'feature_names.pkl'))

# Page config
st.set_page_config(page_title="Insurance Premium Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Insurance Premium Predictor")
st.write("Enter your details below to get an estimated insurance premium.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
 age = st.slider("Age", min_value=18, max_value=66, value=30)
 height = st.slider("Height (cm)", min_value=145, max_value=188, value=168)
 weight = st.slider("Weight (kg)", min_value=51, max_value=132, value=77)
 num_surgeries = st.selectbox("Number of Major Surgeries", options=[0, 1, 2, 3])

with col2:
 diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
 bp = st.selectbox("Blood Pressure Problems", options=["No", "Yes"])
 transplants = st.selectbox("Any Transplants", options=["No", "Yes"])
 chronic = st.selectbox("Any Chronic Diseases", options=["No", "Yes"])
 allergies = st.selectbox("Known Allergies", options=["No", "Yes"])
 cancer_history = st.selectbox("History of Cancer in Family", options=["No", "Yes"])

# Convert Yes/No to 1/0
def yes_no(val):
 return 1 if val == "Yes" else 0

# Calculate BMI
bmi = weight / (height / 100) ** 2

# Build input dataframe
input_data = pd.DataFrame([[
 age,
 yes_no(diabetes),
 yes_no(bp),
 yes_no(transplants),
 yes_no(chronic),
 height,
 weight,
 yes_no(allergies),
 yes_no(cancer_history),
 num_surgeries,
 bmi
]], columns=feature_names)

# Predict
st.divider()
if st.button("Predict Premium", type="primary", use_container_width=True):
 prediction = model.predict(input_data)[0]
 
 st.success(f"### Estimated Premium: ₹{prediction:,.0f}")
 
 st.write("**Input Summary:**")
 summary = {
 "Age": age,
 "BMI": f"{bmi:.1f}",
 "Diabetes": diabetes,
 "Blood Pressure": bp,
 "Transplants": transplants,
 "Chronic Diseases": chronic,
 "Allergies": allergies,
 "Cancer History": cancer_history,
 "Major Surgeries": num_surgeries
 }
 st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))