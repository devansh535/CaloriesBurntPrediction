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
# Streamlit App
# ===============================

st.title("Calories Burnt Prediction App")

# Inputs
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 25)
height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)
weight = st.number_input("Weight (kg)", 10.0, 200.0, 70.0)
duration = st.number_input("Exercise Duration (min)", 1.0, 150.0, 30.0)
heart_rate = st.number_input("Heart Rate", 40.0, 220.0, 100.0)
body_temp = st.number_input("Body Temperature (°C)", 30.0, 45.0, 37.0)

# Encode gender
gender_val = 0 if gender == "Male" else 1

# Load saved model (for real apps)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction
if st.button("Predict Calories"):
    input_data = np.array([[
        gender_val,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]])

    # Scale input
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Estimated Calories Burnt: {prediction[0]:.2f} kcal")
