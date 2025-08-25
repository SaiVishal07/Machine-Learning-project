import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="TechNova Attrition Dashboard 🌟", layout="wide")

# Background style
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
    }
    </style>
    """, unsafe_allow_html=True)

# Load dataset and artifacts
data_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\rawdata\technova_attrition_dataset.csv"
model_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\model.pkl"
preprocessor_path = r"C:\Users\chand\OneDrive\Desktop\MLproject\MLproject\artifacts\preprocessor.pkl"

df = pd.read_csv(data_path)
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Sidebar navigation
st.sidebar.title("Navigation 🌟")
page = st.sidebar.radio("Go to", ["Dashboard & Input", "Prediction Result"])

# -------------------- PAGE 1 --------------------
if page == "Dashboard & Input":
    st.title("TechNova Employee Attrition Dashboard 🌟")
    st.subheader("KPI & Employee Data Entry 💼✨")

    # KPIs
    total_employees = df.shape[0]
    attrition_count = df[df['attrition'] == 'Yes'].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2)
    avg_tenure = round(df['tenure'].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees 👥", total_employees)
    col2.metric("Attrition Rate 📉", f"{attrition_rate}%")
    col3.metric("Average Tenure 🕒", f"{avg_tenure} years")

    # Input form
    st.subheader("Enter Employee Details for Prediction")
    input_data = {}
    for feature in df.drop(columns=['attrition']).columns:
        if df[feature].dtype == 'object':
            options = df[feature].unique().tolist()
            input_data[feature] = st.selectbox(feature, options, key=feature)
        else:
            input_data[feature] = st.number_input(
                feature,
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].median()),
                key=feature
            )

    if st.button("Save Input for Prediction"):
        st.session_state['input_data'] = input_data
        st.success("Employee data saved! Navigate to 'Prediction Result' in the sidebar.")

# -------------------- PAGE 2 --------------------
if page == "Prediction Result":
    st.title("Prediction Result 📊")

    if 'input_data' in st.session_state:
        input_df = pd.DataFrame([st.session_state['input_data']])
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        st.success(f"Attrition Risk: {'Yes 😢' if prediction == 1 else 'No 🎉'}")

        # Feature-based dynamic tips
        tips = []
        data = st.session_state['input_data']

        if prediction == 1:
            if data.get('job_satisfaction', 5) <= 3:
                tips.append("Low job satisfaction – consider career development programs or mentoring.")
            if data.get('work_env_satisfaction', 5) <= 3:
                tips.append("Work environment dissatisfaction – improve workplace conditions or team dynamics.")
            if data.get('overtime', 'No') == 'Yes':
                tips.append("High overtime – consider flexible working hours or workload adjustment.")
            if data.get('tenure', 0) < 2:
                tips.append("Short tenure – focus on onboarding and engagement programs.")
            if data.get('promotion_last_5years', 0) == 0:
                tips.append("No recent promotion – consider role growth opportunities.")
            if not tips:
                tips.append("General engagement programs and recognition could help retain this employee.")

            for tip in tips:
                st.info(f"⚠️ Tip: {tip}")
        else:
            st.info("✅ Low risk of attrition. Keep monitoring satisfaction and engagement to maintain retention.")
    else:
        st.warning("No employee data found. Please enter data in 'Dashboard & Input' page first.")
