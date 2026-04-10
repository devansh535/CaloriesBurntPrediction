# =========================
# Calories Burnt Prediction - Single File App
# =========================

import numpy as np
import pandas as pd
import streamlit as st
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =========================
# TRAIN MODEL (only if not saved)
# =========================

if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):

    df1 = pd.read_csv('exercise.csv')
    df2 = pd.read_csv('calories.csv')

    df = df1.merge(df2, on='User_ID')

    # Encode gender
    df.replace({'male': 0, 'female': 1}, inplace=True)

    # Drop columns
    df.drop(['User_ID', 'Weight', 'Duration'], axis=1, inplace=True)

    # Features
    X = df.drop('Calories', axis=1)
    y = df['Calories']

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=22
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Calories Predictor", page_icon="🔥")

st.title("🔥 Calories Burnt Prediction App")

st.markdown("Enter your workout details:")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 25)
height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)
heart_rate = st.number_input("Heart Rate", 40.0, 220.0, 100.0)
body_temp = st.number_input("Body Temperature (°C)", 30.0, 45.0, 37.0)

# Encode gender
gender_val = 0 if gender == "Male" else 1

# Prediction
if st.button("Predict Calories"):

    input_data = np.array([[gender_val, age, height, heart_rate, body_temp]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"🔥 Estimated Calories Burnt: {prediction[0]:.2f} kcal")
