#  Student Dropout Risk Predictor

An AI-powered web application that predicts the likelihood of a student dropping out based on their digital engagement and academic performance.

##  Overview
Identifying at-risk students early is crucial for academic success. This project uses **Machine Learning** [Predictive Analytics] to analyze student behavior and categorize them into risk levels (Low, Medium, High). The system is built using **Random Forest** [Ensemble Learning] and deployed via **Streamlit**.

##  Key Features
* **Real-time Prediction:** Get instant risk scores using interactive sliders.
* **Risk Categorization:** Visual indicators (🟢, 🟡, 🔴) for easy interpretation.
* **Engagement Analytics:** Features like `Inactivity Flag` and `Performance Index` to track behavior.
* **Feature Importance:** Insights into which factors (like quiz scores or inactive days) drive the dropout risk.

##  Tech Stack
* **Language:** Python
* **ML Library:** Scikit-learn (Random Forest Classifier)
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, Joblib
* **Visualization:** Matplotlib

##  Project Structure
```text
├── app.py                # Streamlit Web Application
├── model_dropout_rf.pkl  # Trained Random Forest Model
├── scaler.pkl            # Saved StandardScaler [Normalization Tool]
├── train_model.py        # Script used for training and evaluation
└── README.md             # Project documentation
