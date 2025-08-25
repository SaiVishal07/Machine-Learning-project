import streamlit as st
import pandas as pd
import joblib
import io

st.set_page_config(page_title="TechNova Attrition Dashboard 🌟", layout="wide")

# Background gradient
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("TechNova Employee Attrition Dashboard 🌟")
st.subheader("Predict & Analyze Employee Attrition 💼✨")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation 🌟")
page = st.sidebar.radio("Menu", ["Dashboard & Input", "Prediction Result"])

# ---------------- File Uploads ----------------
st.sidebar.markdown("### Upload Required Files")
dataset_file = st.sidebar.file_uploader("Upload Dataset CSV", type="csv")
model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl)", type="pkl")
preprocessor_file = st.sidebar.file_uploader("Upload Preprocessor (.pkl)", type="pkl")

if dataset_file and model_file and preprocessor_file:
    df = pd.read_csv(dataset_file)
    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)
    
    # ---------------- PAGE 1 ----------------
    if page == "Dashboard & Input":
        st.header("Dashboard & Employee Input 💼")
        
        # KPIs
        total_employees = df.shape[0]
        attrition_count = df[df['attrition'] == 'Yes'].shape[0]
        attrition_rate = round((attrition_count / total_employees) * 100, 2)
        avg_tenure = round(df['tenure'].mean(), 2)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employees 👥", total_employees)
        col2.metric("Attrition Rate 📉", f"{attrition_rate}%")
        col3.metric("Average Tenure 🕒", f"{avg_tenure} years")
        
        # Employee Input Form
        st.subheader("Enter Employee Details for Prediction")
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
        
        if st.button("Save Input for Prediction"):
            st.session_state['input_data'] = input_data
            st.success("Employee data saved! Navigate to 'Prediction Result' in the sidebar.")

    # ---------------- PAGE 2 ----------------
    if page == "Prediction Result":
        st.header("Prediction Result 📊")
        
        if 'input_data' in st.session_state:
            input_df = pd.DataFrame([st.session_state['input_data']])
            input_transformed = preprocessor.transform(input_df)
            prediction = model.predict(input_transformed)[0]
            
            st.success(f"Attrition Risk: {'High 😢' if prediction==1 else 'Low 🎉'}")
            
            # Dynamic Tips
            st.subheader("Recommended Actions 💡")
            tips = []
            data = st.session_state['input_data']
            
            if prediction == 1:
                if data.get('job_satisfaction',5) <= 3:
                    tips.append("Low job satisfaction – provide mentoring or career growth opportunities.")
                if data.get('work_env_satisfaction',5) <= 3:
                    tips.append("Improve work environment and team collaboration.")
                if data.get('overtime','No')=='Yes':
                    tips.append("Reduce overtime and workload stress.")
                if data.get('tenure',0) < 2:
                    tips.append("Focus on onboarding and early engagement programs.")
                if data.get('promotion_last_5years',0) == 0:
                    tips.append("Offer promotion or skill development opportunities.")
                if not tips:
                    tips.append("General engagement programs and recognition may help.")
                for tip in tips:
                    st.info(f"⚠️ {tip}")
            else:
                st.info("✅ Low risk of attrition. Maintain current engagement practices.")
        else:
            st.warning("No employee data found. Please enter data in 'Dashboard & Input' page first.")
else:
    st.warning("Please upload the dataset, model, and preprocessor files in the sidebar to continue.")
