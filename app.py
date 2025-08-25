import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="TechNova Attrition Heatmap 🌟", layout="wide")

# Background gradient
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #fffde7, #ffffff);
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("TechNova Attrition Risk & Engagement Planner 🌟")
st.subheader("Department Heatmap & Personalized Risk Analysis 💼✨")

# ---------------- File Uploads ----------------
st.sidebar.title("Upload Files 🌟")
dataset_file = st.sidebar.file_uploader("Upload Dataset CSV", type="csv")
model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type="pkl")
preprocessor_file = st.sidebar.file_uploader("Upload Preprocessor (.pkl)", type="pkl")

if dataset_file and model_file and preprocessor_file:
    df = pd.read_csv(dataset_file)
    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)
    
    # ---------------- Department Heatmap ----------------
    st.header("Department Attrition Heatmap 🔥")
    dept_summary = df.groupby('department')['attrition'].apply(lambda x: (x=='Yes').mean()*100)
    heatmap_df = pd.DataFrame(dept_summary).reset_index()
    heatmap_df.rename(columns={'attrition':'Attrition Rate (%)'}, inplace=True)
    
    # Color-code attrition rates
    def color_rate(val):
        if val > 25: return 'background-color: #ff8a80'  # High risk
        elif val > 10: return 'background-color: #fff176'  # Medium risk
        else: return 'background-color: #b9f6ca'           # Low risk
    
    st.dataframe(heatmap_df.style.applymap(color_rate, subset=['Attrition Rate (%)']))
    
    st.markdown("---")
    
    # ---------------- Individual Employee Risk ----------------
    st.header("Predict Individual Employee Attrition 🔮")
    st.subheader("Enter Employee Details")
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
    
    if st.button("Predict Risk 🚀"):
        input_df = pd.DataFrame([input_data])
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        
        st.success(f"Attrition Risk: {'High 😢' if prediction==1 else 'Low 🎉'}")
        
        # Personalized Engagement Plan
        st.subheader("Recommended Engagement Plan 💡")
        plan = []
        if prediction==1:
            if input_data.get('job_satisfaction',5)<=3:
                plan.append("Boost job satisfaction: mentoring or career growth opportunities.")
            if input_data.get('work_env_satisfaction',5)<=3:
                plan.append("Improve work environment and team collaboration.")
            if input_data.get('overtime','No')=='Yes':
                plan.append("Adjust workload and reduce overtime.")
            if input_data.get('tenure',0)<2:
                plan.append("Enhance onboarding and early engagement programs.")
            if input_data.get('promotion_last_5years',0)==0:
                plan.append("Offer skill development or promotion opportunities.")
            if not plan:
                plan.append("General engagement programs and recognition.")
        else:
            plan.append("Employee is low-risk. Maintain current engagement practices.")
        
        for p in plan:
            st.info(f"⚠️ {p}")
else:
    st.warning("Please upload dataset, model, and preprocessor files in the sidebar to continue.")
