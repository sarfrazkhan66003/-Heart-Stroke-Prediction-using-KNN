# -Heart-Stroke-Prediction-using-KNN

📌 Repository Description
This project is a Heart Stroke Prediction System built using Machine Learning (K-Nearest Neighbors) and deployed with Streamlit.
It predicts whether a person is at high risk of heart stroke based on medical attributes like age, cholesterol, blood pressure, chest pain, and more.

📖 Project Details
🔑 Key Concepts
Dataset: Kaggle Heart Disease Dataset
Algorithm Used:
-K-Nearest Neighbors (KNN)
-Works by finding the “nearest” patients with similar medical profiles and predicting risk based on their outcomes.
Scaling: StandardScaler is applied to normalize features for better accuracy.
Deployment: Streamlit Web App with interactive UI and Light/Dark mode.
Data Loading → Import the Kaggle heart dataset.
Preprocessing → Handle features & target, normalize values using StandardScaler.
Train-Test Split → 80% training, 20% testing.
Model Training → KNN with k=5 neighbors.
Evaluation → Accuracy score tested on unseen data.
Save Artifacts → Model, Scaler, and Feature Columns saved via Joblib.
Deployment → A user-friendly Streamlit app where patients/doctors can input details and instantly get a risk prediction.


⌨️ Input Parameters
The user provides the following medical details:
👤 Age
⚧ Sex (Male/Female)
❤️ Chest Pain Type (0–3)
💉 Resting Blood Pressure
🩸 Serum Cholesterol (mg/dl)
🍬 Fasting Blood Sugar (>120 mg/dl)
📈 Resting ECG Results
🏃‍♂️ Maximum Heart Rate Achieved
💔 Exercise Induced Angina
📉 ST Depression (Oldpeak)
📊 Slope of ST Segment
🫀 Number of Major Vessels (0–3)
🧬 Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)

📤 Output
✅ Low Risk of Heart Stroke
⚠️ High Risk of Heart Stroke

📊 Accuracy
The trained KNN model achieved over 80% accuracy (depending on dataset split and preprocessing).
Performance can be further improved with hyperparameter tuning and additional medical data.

🏥 Medical Use Case & Solution
This system is not a replacement for doctors, but it can assist in:
Early Detection → Quickly flagging patients at risk of heart stroke.
Decision Support → Helping doctors prioritize patients needing urgent care.
Awareness Tool → Allowing individuals to self-check and consult professionals in time.

