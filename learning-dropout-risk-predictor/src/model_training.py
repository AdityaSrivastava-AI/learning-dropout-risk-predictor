import pandas as pd

# Load engineered dataset
df = pd.read_csv("data/student_engagement_engineered.csv")

print("Dataset loaded for model training")
print(df.head())

##Separate Features & Target and then TTS
from sklearn.model_selection import train_test_split

X = df.drop("dropout", axis=1)
y = df["dropout"]
#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

##feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##Train Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

print("Logistic Regression trained successfully")

## evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred_lr = log_reg.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


##decision tree

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully")

#eval DT

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


##feature importance

import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()


##saving modelm and scaler

import joblib

# Save model
joblib.dump(rf_model, "model_dropout_rf.pkl")

# Save scaler (for future consistency if needed)
joblib.dump(scaler, "scaler.pkl")

print("\nFinal model and scaler saved successfully")
