import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- File Paths ----------------
data_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\rawdata\technova_attrition_dataset.csv"
model_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\model.pkl"
preprocessor_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\preprocessor.pkl"

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="TechNova Attrition Predictor", layout="wide", initial_sidebar_state="expanded")

# Load the model and preprocessor
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    st.success("✅ Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("❌ Required files (model.pkl, preprocessor.pkl) not found. Please ensure they are in the 'artifacts' directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred while loading files: {e}")
    st.stop()

# ---------------- Streamlit App ----------------

st.title("TechNova Solutions Employee Attrition Predictor 🔮")

st.markdown("""
Welcome to the TechNova Solutions Attrition Predictor. This tool helps HR and management identify employees at risk of leaving the company. 
By entering an employee's details, you can get a real-time prediction and take proactive measures to improve retention.
""")

st.sidebar.header("Employee Details")

# Define input fields
def user_input_features():
    st.sidebar.subheader("General Information")
    age = st.sidebar.slider("Age", 18, 65, 30)
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    job_role = st.sidebar.selectbox("Job Role", ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                                                'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                                                'Sales Representative', 'Research Director', 'Human Resources', 
                                                'Manager R&D', 'Technical Staff'))
    department = st.sidebar.selectbox("Department", ('Sales', 'Research & Development', 'Human Resources'))
    
    st.sidebar.subheader("Work Experience & Compensation")
    job_level = st.sidebar.slider("Job Level", 1, 5, 1)
    salary = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000)
    tenure = st.sidebar.slider("Years at Company", 0, 40, 5)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
    stock_option_level = st.sidebar.slider("Stock Option Level", 0, 3, 0)
    
    st.sidebar.subheader("Job & Life Satisfaction")
    job_satisfaction = st.sidebar.selectbox("Job Satisfaction", (1, 2, 3, 4), format_func=lambda x: f"{x} - ({['Low', 'Medium', 'High', 'Very High'][x-1]})")
    work_env_satisfaction = st.sidebar.selectbox("Environment Satisfaction", (1, 2, 3, 4), format_func=lambda x: f"{x} - ({['Low', 'Medium', 'High', 'Very High'][x-1]})")
    job_involvement = st.sidebar.selectbox("Job Involvement", (1, 2, 3, 4), format_func=lambda x: f"{x} - ({['Low', 'Medium', 'High', 'Very High'][x-1]})")
    work_life_balance = st.sidebar.selectbox("Work-Life Balance", (1, 2, 3, 4), format_func=lambda x: f"{x} - ({['Bad', 'Good', 'Better', 'Best'][x-1]})")

    st.sidebar.subheader("Other Metrics")
    overtime = st.sidebar.selectbox("Over Time", ('Yes', 'No'))
    business_travel = st.sidebar.selectbox("Business Travel", ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
    
    # These columns are required by the model but are not user-facing inputs.
    # Provide plausible default values for them.
    marital_status = st.sidebar.selectbox("Marital Status", ('Single', 'Married', 'Divorced'))
    education = st.sidebar.selectbox("Education Level", (1, 2, 3, 4, 5), format_func=lambda x: f"{x} - ({['Below College', 'College', 'Bachelor', 'Master', 'Doctor'][x-1]})")
    promotion_last_5years = st.sidebar.slider("Promotions in Last 5 Years", 0, 1, 0)
    years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 0)
    training_hours = st.sidebar.slider("Training Hours", 0, 200, 80)

    data = {
        'age': age,
        'job_satisfaction': job_satisfaction,
        'salary': salary,
        'tenure': tenure,
        'work_env_satisfaction': work_env_satisfaction,
        'overtime': overtime,
        'marital_status': marital_status,
        'education': education,
        'department': department,
        'promotion_last_5years': promotion_last_5years,
        'years_since_last_promotion': years_since_last_promotion,
        'training_hours': training_hours,
        'work_life_balance': work_life_balance
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.header("Provided Employee Data")
st.dataframe(input_df, hide_index=True)

# Predict Attrition
if st.button("Predict Attrition Risk"):
    try:
        # Prepare the input data for prediction
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        attrition_risk = "High" if prediction[0] == 1 else "Low"
        emoji = "❗" if attrition_risk == "High" else "✅"
        
        st.subheader("Prediction Result")
        st.markdown(f"**Based on the provided data, the employee's attrition risk is: {attrition_risk} {emoji}**")
        
        attrition_probability = prediction_proba[0][1] * 100
        retention_probability = prediction_proba[0][0] * 100
        
        st.markdown(f"**Probability of Attrition:** `{attrition_probability:.2f}%`")
        st.markdown(f"**Probability of Retention:** `{retention_probability:.2f}%`")
        
        st.info("""
        **How to interpret this result?** A 'High' risk suggests the employee has characteristics similar to those who have left the company in the past. 
        A 'Low' risk indicates a higher likelihood of retention.
        """)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")

st.markdown("---")

# The section for 'Key Attrition Insights' has been removed to eliminate the KeyError.

st.markdown("""
<div style='text-align: center;'>
    <p>🚀 Powered by Machine Learning for a better workplace.</p>
    <p>TechNova Solutions © 2025</p>
</div>
""", unsafe_allow_html=True)
