import pandas as pd

# Load dataset
df = pd.read_csv("data/student_engagement_data.csv")

# Feature 1: Engagement Score
df["engagement_score"] = df["login_frequency"] * df["time_spent_hours"]

# Feature 2: Performance Index
df["performance_index"] = (
    0.7 * df["assignments_completed"] +
    0.3 * df["quiz_score"]
)


# Feature 3: Inactivity Flag
df["inactivity_flag"] = df["days_inactive"].apply(
    lambda x: 1 if x >= 14 else 0
)

print("Feature Engineering Completed\n")
print(df.head())

# Save engineered dataset
df.to_csv("data/student_engagement_engineered.csv", index=False)
