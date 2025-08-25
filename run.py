import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and favicon
st.set_page_config(
    page_title="TechNova Attrition Predictor",
    page_icon="🤖",
    layout="wide"
)

# ---------------- File Paths ----------------
# Use relative paths for better portability
artifacts_dir = "artifacts"
rawdata_dir = "rawdata"

model_path = os.path.join(artifacts_dir, "model.pkl")
preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
data_path = os.path.join(rawdata_dir, "technova_attrition_dataset.csv")

# Load the model and preprocessor
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please ensure the 'artifacts' and 'rawdata' folders with all necessary files exist in the same directory. Missing file: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading model files: {e}")
    st.stop()

# ---------------- UI/UX Elements ----------------
st.title("👨‍💼 TechNova Employee Attrition Predictor")
st.markdown("""
Welcome to the **TechNova Solutions** Employee Attrition Predictor! 🚀 This app helps HR teams identify employees who may be at risk of leaving the company by using a machine learning model.
""")

st.markdown("---")

# ---------------- User Input Form ----------------
st.header("1. Employee Data Input")
st.markdown("Please enter the details of the employee to predict their attrition status.")

# Create input widgets for all features, aligning with the preprocessor's expected columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 65, 30)
    salary = st.slider("Salary", 0, 200000, 70000)
    tenure = st.slider("Tenure (Years at Company)", 0, 40, 5)
    training_hours = st.slider("Training Hours", 0, 100, 20)

with col2:
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    work_env_satisfaction = st.slider("Work Environment Satisfaction", 1, 4, 3)
    work_life_balance = st.slider("Work-Life Balance", 1, 4, 3, help="1=Bad, 2=Good, 3=Better, 4=Best")
    education = st.selectbox("Education Level", (1, 2, 3, 4, 5), help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")

with col3:
    department = st.selectbox("Department", ('Research & Development', 'Sales', 'Human Resources'))
    marital_status = st.selectbox("Marital Status", ('Single', 'Married', 'Divorced'))
    promotion_last_5years = st.radio("Promotions in Last 5 Years?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    overtime = st.radio("Overtime", ('Yes', 'No'))

# Create a DataFrame from the user inputs using the correct column names
user_data = pd.DataFrame({
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
})

st.markdown("---")

# ---------------- Prediction & Results ----------------
st.header("2. Prediction Results")
if st.button("Predict Attrition"):
    try:
        # Use the loaded preprocessor to transform user input
        transformed_data = preprocessor.transform(user_data)
        
        # Make a prediction
        prediction = model.predict(transformed_data)
        
        # Display the result
        if prediction[0] == 1:
            st.error("🚨 Attrition Predicted: This employee is **at high risk** of leaving.")
            st.markdown("Consider initiating a conversation with the employee to understand their concerns. Offer targeted retention strategies like career development, flexible work options, or a salary review. 💡")
            
        else:
            st.success("✅ Attrition Predicted: This employee is **not at high risk** of leaving.")
            st.markdown("This employee is likely to be a long-term asset. Continue to foster a positive work environment and provide opportunities for growth to maintain their satisfaction. 😊")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")

# ---------------- Dataset Information ----------------
st.header("3. Dataset Statistics")
st.markdown("Explore some key statistics from the dataset used for training the model.")

try:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    
    # Create three columns for charts
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Department Distribution")
        dept_counts = df['department'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(dept_counts, labels=dept_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
    with col2:
        st.subheader("Attrition Rate by Tenure")
        tenure_attrition_rate = df.groupby('tenure')['attrition'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(x='tenure', y='attrition', data=tenure_attrition_rate, marker='o', ax=ax, color='red')
        ax.set_title('Attrition Rate by Tenure')
        ax.set_xlabel('Tenure (Years)')
        ax.set_ylabel('Attrition Rate')
        st.pyplot(fig)
    
    with col3:
        st.subheader("Job Satisfaction vs. Attrition")
        satisfaction_avg = df.groupby('attrition')['job_satisfaction'].mean().reset_index()
        satisfaction_avg['attrition'] = satisfaction_avg['attrition'].map({0: 'Stayed', 1: 'Left'})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='attrition', y='job_satisfaction', data=satisfaction_avg, palette='viridis', ax=ax)
        ax.set_title('Average Job Satisfaction')
        ax.set_xlabel('Attrition Status')
        ax.set_ylabel('Avg. Job Satisfaction (1-4)')
        st.pyplot(fig)
            
except FileNotFoundError:
    st.warning("📊 Dataset file not found. Cannot display dataset statistics.")
except KeyError as e:
    st.warning(f"📊 A required column was not found in the dataset for charting: {e}")

st.markdown("---")
st.info("Developed with ❤️ for TechNova Solutions to build a better workplace.")
