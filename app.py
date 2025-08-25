import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(page_title="TechNova Attrition Simulator 🌟", layout="wide")

# Background gradient
st.markdown("""
    <style>
    body {background: linear-gradient(to right, #e1f5fe, #ffffff);}
    .stButton>button {background-color: #0288d1; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("TechNova Attrition Scenario Simulator 🌟")
st.subheader("Enter employee profile and simulate scenarios 💼✨")

# ---------------- File Paths ----------------
data_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\rawdata\technova_attrition_dataset.csv"
model_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\model.pkl"
preprocessor_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\preprocessor.pkl"

# ---------------- Load Dataset and Artifacts ----------------
try:
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError:
    st.error("Dataset or artifacts not found. Please check the paths.")
    st.stop()

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation 🌟")
page = st.sidebar.radio("Select Page", ["Employee Input", "Scenario Simulator"])

# ---------------- Page 1: Employee Input ----------------
if page == "Employee Input":
    st.header("Employee Profile Input 💼")
    input_data = {}
    for feature in df.columns:
        if feature == 'attrition':
            continue  # skip target column
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
    if st.button("Save Employee Profile"):
        st.session_state['input_data'] = input_data
        st.success("Employee profile saved! Navigate to Scenario Simulator.")

# ---------------- Page 2: Scenario Simulator ----------------
elif page == "Scenario Simulator":
    if 'input_data' not in st.session_state:
        st.warning("Please enter employee data first on 'Employee Input' page.")
    else:
        st.header("Scenario Simulation 🔄")
        input_data = st.session_state['input_data'].copy()

        # Sliders / inputs for scenario simulation
        salary_scenario = st.number_input(
            "Salary",
            float(df['salary'].min()),
            float(df['salary'].max()),
            float(input_data.get('salary',0))
        )

        overtime_scenario = st.selectbox(
            "Overtime",
            ['Yes','No'],
            index=0 if input_data.get('overtime','No')=='Yes' else 1
        )

        job_satisfaction_scenario = st.slider(
            "Job Satisfaction (1-5)",
            min_value=1,
            max_value=5,
            value=int(input_data.get('job_satisfaction',3))
        )

        work_env_scenario = st.slider(
            "Work Env Satisfaction (1-5)",
            min_value=1,
            max_value=5,
            value=int(input_data.get('work_env_satisfaction',3))
        )

        if st.button("Run Simulation 🚀"):
            scenario_data = input_data.copy()
            scenario_data['salary'] = salary_scenario
            scenario_data['overtime'] = overtime_scenario
            scenario_data['job_satisfaction'] = job_satisfaction_scenario
            scenario_data['work_env_satisfaction'] = work_env_scenario

            scenario_df = pd.DataFrame([scenario_data])
            scenario_transformed = preprocessor.transform(scenario_df)
            prediction = model.predict(scenario_transformed)[0]

            st.success(f"Predicted Attrition Risk: {'High 😢' if prediction==1 else 'Low 🎉'}")

            # Personalized Recommendations
            st.subheader("Recommended Engagement Actions 💡")
            tips = []
            if prediction == 1:
                if job_satisfaction_scenario <= 3:
                    tips.append("Increase job satisfaction through mentoring or career growth.")
                if work_env_scenario <= 3:
                    tips.append("Improve work environment and collaboration.")
                if overtime_scenario=='Yes':
                    tips.append("Reduce overtime or provide flexible hours.")
                if salary_scenario < input_data.get('salary',0):
                    tips.append("Consider increasing salary to retain employee.")
                if not tips:
                    tips.append("General engagement programs recommended.")
                for tip in tips:
                    st.info(f"⚠️ {tip}")
            else:
                st.info("✅ Low risk scenario. Maintain current engagement practices.")
