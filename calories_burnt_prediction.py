
# ===============================
# Calories Burnt Prediction App
# ===============================

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ===============================
# Load and Prepare Data
# ===============================

df1 = pd.read_csv('exercise.csv')
df2 = pd.read_csv('calories.csv')

# Merge datasets
df = df1.merge(df2, on='User_ID')

# Encode gender
df.replace({'male': 0, 'female': 1}, inplace=True)

# Features & Target
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ===============================
# Train Model
# ===============================

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

print("Training MAE:", mean_absolute_error(y_train, train_preds))
print("Validation MAE:", mean_absolute_error(y_val, val_preds))

# Save model & scaler (optional but recommended)
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ===============================
# Streamlit App (Enhanced UI)
# ===============================

st.set_page_config(page_title="🔥 Calories Predictor", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stButton>button {
        background-color: #22c55e;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    .stNumberInput, .stSelectbox {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 🔥 Calories Burnt Prediction")
st.caption("Estimate calories burned based on your workout and body metrics")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 100, 25)
    height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)
    weight = st.number_input("Weight (kg)", 10.0, 200.0, 70.0)

with col2:
    duration = st.number_input("Duration (min)", 1.0, 150.0, 30.0)
    heart_rate = st.number_input("Heart Rate", 40.0, 220.0, 100.0)
    body_temp = st.number_input("Body Temp (°C)", 30.0, 45.0, 37.0)

# Encode gender
gender_val = 0 if gender == "Male" else 1

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.markdown("---")

# Prediction Button
if st.button("🚀 Predict Calories Burned"):
    input_data = np.array([[
        gender_val,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    st.markdown(f"""
        <div style="
            background-color:#1e293b;
            padding:20px;
            border-radius:15px;
            text-align:center;
        ">
            <h2 style="color:#22c55e;">🔥 {prediction[0]:.2f} kcal</h2>
            <p>Estimated Calories Burnt</p>
        </div>
    """, unsafe_allow_html=True)
