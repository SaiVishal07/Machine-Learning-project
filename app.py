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
import os

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="TechNova Attrition Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------- File Paths ----------------
# Use relative paths for better portability
artifacts_dir = "artifacts"
rawdata_dir = "rawdata"

model_path = os.path.join(artifacts_dir, "model.pkl")
preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
data_path = os.path.join(rawdata_dir, "technova_attrition_dataset.csv")

# ---------------- Load Files with Error Handling ----------------
try:
    if not os.path.exists(artifacts_dir):
        st.error(f"❌ The directory '{artifacts_dir}' was not found. Please create it.")
        st.stop()

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error("❌ Required files ('model.pkl', 'preprocessor.pkl') not found in the 'artifacts' directory.")
        st.info("💡 **Solution:** Ensure you have run a separate script to train your model and save these files.")
        st.stop()
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    st.success("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"❌ An unexpected error occurred while loading files: {e}")
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
    # gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    department = st.sidebar.selectbox("Department", ('Sales', 'Research & Development', 'Human Resources'))
    
    st.sidebar.subheader("Work Experience & Compensation")
    salary = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000)
    tenure = st.sidebar.slider("Years at Company", 0, 40, 5)
    
    st.sidebar.subheader("Job & Life Satisfaction")
    job_satisfaction = st.sidebar.selectbox("Job Satisfaction", (1, 2, 3, 4))
    work_env_satisfaction = st.sidebar.selectbox("Environment Satisfaction", (1, 2, 3, 4))
    work_life_balance = st.sidebar.selectbox("Work-Life Balance", (1, 2, 3, 4), help="1=Bad, 2=Good, 3=Better, 4=Best")

    st.sidebar.subheader("Other Metrics")
    overtime = st.sidebar.selectbox("Over Time", ('Yes', 'No'))
    
    # These columns are required by the model but are not user-facing inputs based on your previous code.
    # Provide plausible default values for them.
    marital_status = st.sidebar.selectbox("Marital Status", ('Single', 'Married', 'Divorced'))
    education = st.sidebar.selectbox("Education Level", (1, 2, 3, 4, 5))
    promotion_last_5years = st.sidebar.radio("Promotions in Last 5 Years?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 0)
    training_hours = st.sidebar.slider("Training Hours", 0, 200, 80)
    
    data = {
        'age': [age],
        'job_satisfaction': [job_satisfaction],
        'salary': [salary],
        'tenure': [tenure],
        'work_env_satisfaction': [work_env_satisfaction],
        'overtime': [overtime],
        'marital_status': [marital_status],
        'education': [education],
        'department': [department],
        'promotion_last_5years': [promotion_last_5years],
        'years_since_last_promotion': [years_since_last_promotion],
        'training_hours': [training_hours],
        'work_life_balance': [work_life_balance]
    }
    features = pd.DataFrame(data)
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

## Dataset Insights and Statistics 📊

st.header("Dataset Insights and Statistics")

try:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    
    # Create three columns for charts
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Education Level Distribution")
        education_counts = df['education'].value_counts()
        education_labels = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        education_counts.index = education_counts.index.map(education_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        # Use a more vibrant color list for the pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        ax.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
    with col2:
        st.subheader("Average Salary by Department")
        avg_salary_dept = df.groupby('department')['salary'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        # Use a different, more professional seaborn palette
        sns.barplot(x='department', y='salary', data=avg_salary_dept, palette='pastel', ax=ax)
        ax.set_title('Average Monthly Salary per Department')
        ax.set_xlabel('Department')
        ax.set_ylabel('Average Salary')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    with col3:
        st.subheader("Job Satisfaction vs. Last Promotion")
        satisfaction_by_promo = df.groupby('years_since_last_promotion')['job_satisfaction'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        # Use a different color for the line plot
        sns.lineplot(x='years_since_last_promotion', y='job_satisfaction', data=satisfaction_by_promo, marker='o', ax=ax, color='teal')
        ax.set_title('Avg. Job Satisfaction by Years Since Last Promotion')
        ax.set_xlabel('Years Since Last Promotion')
        ax.set_ylabel('Avg. Job Satisfaction (1-4)')
        st.pyplot(fig)
            
except FileNotFoundError:
    st.warning("📊 Dataset file not found. Cannot display dataset statistics.")
except KeyError as e:
    st.warning(f"📊 A required column was not found in the dataset for charting: {e}")

st.markdown("---")
st.info("Developed with ❤️ for TechNova Solutions to build a better workplace.")
