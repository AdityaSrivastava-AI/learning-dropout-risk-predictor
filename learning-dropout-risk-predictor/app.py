import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model_dropout_rf.pkl")

st.set_page_config(page_title="Learning Dropout Risk Predictor")

st.title("Learning Dropout Risk Predictor")
st.write("Predict the risk of a student dropping out based on engagement behavior.")

st.sidebar.header("Student Engagement Inputs")

login_frequency = st.sidebar.slider("Login Frequency (per week)", 0, 10, 3)
time_spent_hours = st.sidebar.slider("Time Spent (hours/week)", 0, 20, 5)
assignments_completed = st.sidebar.slider("Assignments Completed (%)", 0, 100, 50)
quiz_score = st.sidebar.slider("Quiz Score (%)", 0, 100, 50)
days_inactive = st.sidebar.slider("Days Inactive", 0, 30, 5)

# Derived features
engagement_score = login_frequency * time_spent_hours
performance_index = 0.7 * assignments_completed + 0.3 * quiz_score
inactivity_flag = 1 if days_inactive >= 14 else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    "login_frequency": login_frequency,
    "time_spent_hours": time_spent_hours,
    "assignments_completed": assignments_completed,
    "quiz_score": quiz_score,
    "days_inactive": days_inactive,
    "engagement_score": engagement_score,
    "performance_index": performance_index,
    "inactivity_flag": inactivity_flag
}])

if st.button("Predict Dropout Risk"):
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.30:
        risk = "🟢 Low Risk"
    elif prob < 0.60:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader("Prediction Result")
    st.write(f"**Dropout Probability:** {prob:.2f}")
    st.write(f"**Risk Category:** {risk}")
