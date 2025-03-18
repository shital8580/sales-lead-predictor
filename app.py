import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load trained model
model = joblib.load("sales_lead_model.pkl")  # Use the correct filename # Ensure your trained model is saved

# Load feature names used in training
feature_names = joblib.load("feature_columns.pkl")  # Ensure feature names are saved

# Set Streamlit page config
st.set_page_config(page_title="Sales Lead Conversion Prediction", page_icon="🔥", layout="centered")

# UI Design - Title & Description
st.markdown("""
    <h1 style='text-align: center; color: white;'>🔥 Sales Lead Conversion Prediction 🔥</h1>
    <p style='text-align: center;'>Enter lead details to predict the probability of conversion.</p>
    """, unsafe_allow_html=True)

# Input Fields
st.subheader("Lead Information")

requested_demo = st.radio("Requested a Demo?", ["No", "Yes"], horizontal=True)
competitor_usage = st.radio("Currently Using a Competitor?", ["No", "Yes"], horizontal=True)
industry = st.selectbox("Industry", ["IT", "Finance", "Retail", "Education", "Healthcare"])
lead_source = st.selectbox("Lead Source", ["Paid Ad", "LinkedIn", "Referral", "Event", "Cold Call"])

company_size = st.number_input("Company Size (Number of Employees)", min_value=1, value=500)
emails_sent = st.number_input("Emails Sent", min_value=0, value=10)
calls_made = st.number_input("Calls Made", min_value=0, value=5)
meetings_scheduled = st.number_input("Meetings Scheduled", min_value=0, value=2)
website_visits = st.number_input("Website Visits", min_value=0, value=10)

# Convert categorical inputs to match model training
input_data = pd.DataFrame({
    "Company_Size": [company_size],
    "Emails_Sent": [emails_sent],
    "Calls_Made": [calls_made],
    "Meetings_Scheduled": [meetings_scheduled],
    "Website_Visits": [website_visits],
    "Demo_Requests": [1 if requested_demo == "Yes" else 0],
    "Competitor_Usage": [1 if competitor_usage == "Yes" else 0],
    f"Industry_{industry}": [1],
    f"Lead_Source_{lead_source}": [1]
})

# Ensure missing columns from training are included with 0s
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Align column order
input_data = input_data[feature_names]

# Predict conversion probability
if st.button("Predict Conversion Probability"):
    prediction = model.predict_proba(input_data)[0][1] * 100
    st.success(f"🔥 Conversion Probability: {prediction:.2f}%")

    # Save input & prediction
    result_df = pd.DataFrame({
        "Company_Size": [company_size], "Emails_Sent": [emails_sent], "Calls_Made": [calls_made],
        "Meetings_Scheduled": [meetings_scheduled], "Website_Visits": [website_visits],
        "Demo_Requests": [requested_demo], "Competitor_Usage": [competitor_usage],
        "Industry": [industry], "Lead_Source": [lead_source], "Prediction": [prediction]
    })
    
    if os.path.exists("predictions.csv"):
        result_df.to_csv("predictions.csv", mode='a', header=False, index=False)
    else:
        result_df.to_csv("predictions.csv", index=False)
    
    st.write("✅ Prediction saved to history!")

    # Feature Importance Plot
    st.subheader("Feature Importance")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], ax=ax, palette="coolwarm")
    ax.set_title("Feature Impact on Conversion")
    st.pyplot(fig)

# Show saved predictions
if os.path.exists("predictions.csv"):
    st.subheader("📊 Prediction History")
    history_df = pd.read_csv("predictions.csv")
    st.dataframe(history_df)
