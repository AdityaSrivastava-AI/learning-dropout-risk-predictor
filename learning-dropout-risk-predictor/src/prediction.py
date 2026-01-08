import joblib
import pandas as pd

# Load saved model
model = joblib.load("model_dropout_rf.pkl")

# Sample new student data (simulating real input)
new_student = pd.DataFrame([{
    "login_frequency": 1,
    "time_spent_hours": 2,
    "assignments_completed": 30,
    "quiz_score": 40,
    "days_inactive": 18,
    "engagement_score": 1 * 2,
    "performance_index": 0.7 * 30 + 0.3 * 40,
    "inactivity_flag": 1
}])

# Predict dropout probability
dropout_prob = model.predict_proba(new_student)[0][1]

# Risk categorization
if dropout_prob < 0.30:
    risk = "Low Risk"
elif dropout_prob < 0.60:
    risk = "Medium Risk"
else:
    risk = "High Risk"

print(f"Dropout Probability: {dropout_prob:.2f}")
print(f"Risk Category: {risk}")
