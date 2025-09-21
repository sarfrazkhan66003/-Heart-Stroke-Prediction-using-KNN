import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# Theme Mode
mode = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"])

# Light Theme CSS
light_css = """
<style>
.stApp {
    background: linear-gradient(135deg, #f0f4f7, #c9e0ed);
    color: #1a1a2e;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
</style>
"""

# Dark Theme CSS
dark_css = """
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928DAB);
    color: white;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
</style>
"""

# Apply CSS
if mode == "Light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Stroke Prediction</h1>", unsafe_allow_html=True)

st.subheader("Provide the following details to check your risk:")

# Input fields (example)
age = st.slider("Age", 20, 80, 30)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

# Convert inputs into DataFrame
input_data = pd.DataFrame([[age, 1 if sex == "M" else 0, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]], columns=expected_columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Stroke")
    else:
        st.success("✅ Low Risk of Heart Stroke")
