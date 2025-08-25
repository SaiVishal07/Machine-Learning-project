import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Attrition Scenario Simulator 🌟", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e1f5fe, #ffffff);
    }
    .stButton>button {
        background-color: #0288d1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("TechNova Attrition Scenario Simulator 🌟")
st.subheader("Test employee interventions & see predicted risk 💼✨")

# ---------------- Paths ----------------
data_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\rawdata\technova_attrition_dataset.csv"
model_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\model.pkl"
preprocessor_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\preprocessor.pkl"

# Load dataset and artifacts
try:
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError:
    st.error("Dataset or artifacts not found. Please check the paths.")

# ---------------- Employee Input ----------------
st.header("Enter Base Employee Profile")
input_data = {}
for feature in df.drop(columns=['attrition']).columns:
    if df[feature].dtype == 'object':
        input_data[feature] = st.selectbox(feature, df[feature].unique(), key=feature)
    else:
        input_data[feature] = st.number_input(
            feature,
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].median()),
            key=feature
        )

st.header("Simulate Scenarios 🔄")
st.markdown("Adjust one or more features to see the impact on predicted attrition risk.")

# Scenario Inputs
salary_scenario = st.number_input("Simulate Salary", float(df['salary'].min()), float(df['salary'].max()), float(input_data['salary']))
overtime_scenario = st.selectbox("Simulate Overtime", ['Yes','No'], index=0 if input_data.get('overtime','No')=='Yes' else 1)
job_satisfaction_scenario = st.slider("Simulate Job Satisfaction (1-5)", 1, 5, input_data.get('job_satisfaction',3))
work_env_scenario = st.slider("Simulate Work Environment Satisfaction (1-5)", 1, 5, input_data.get('work_env_satisfaction',3))

if st.button("Run Scenario Analysis 🚀"):
    scenario_data = input_data.copy()
    scenario_data['salary'] = salary_scenario
    scenario_data['overtime'] = overtime_scenario
    scenario_data['job_satisfaction'] = job_satisfaction_scenario
    scenario_data['work_env_satisfaction'] = work_env_scenario

    scenario_df = pd.DataFrame([scenario_data])
    scenario_transformed = preprocessor.transform(scenario_df)
    scenario_prediction = model.predict(scenario_transformed)[0]

    st.success(f"Predicted Attrition Risk for this scenario: {'High 😢' if scenario_prediction==1 else 'Low 🎉'}")

    # Personalized Engagement Plan
    st.subheader("Scenario-Based Recommendations 💡")
    tips = []
    if scenario_prediction==1:
        if job_satisfaction_scenario <= 3:
            tips.append("Increase job satisfaction through mentoring or career growth.")
        if work_env_scenario <= 3:
            tips.append("Improve team environment and collaboration.")
        if overtime_scenario=='Yes':
            tips.append("Reduce overtime or provide flexible hours.")
        if salary_scenario < input_data['salary']:
            tips.append("Consider increasing salary to retain employee.")
        if not tips:
            tips.append("General engagement programs recommended.")
        for tip in tips:
            st.info(f"⚠️ {tip}")
    else:
        st.info("✅ Low risk scenario. Employee likely to stay under these conditions.")
