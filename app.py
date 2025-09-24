import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical
sex = 1 if sex == "male" else 0
embarked = {"C":0, "Q":1, "S":2}[embarked]

# Input array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    if prediction == 1:
        st.success(f"✅ Survived! (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Did not Survive (Probability: {prob:.2f})")
