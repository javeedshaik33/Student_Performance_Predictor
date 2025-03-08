import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
st.set_page_config(page_title="Student Grade Prediction")
df = pd.read_csv('Studenr_Performance_Prediction12.csv')
df['Previous Grades'] = df['Previous Grades'].astype(str).str.replace('%', '').astype(float)
df['Attendance Rate'] = df['Attendance Rate'].astype(str).str.replace('%', '').astype(float)
df['Previous Grades'] = df['Previous Grades'].fillna(df['Previous Grades'].mean())
df['Attendance Rate'] = df['Attendance Rate'].fillna(df['Attendance Rate'].mean())
df['Study_Hours_per_Week'] = df['Study_Hours_per_Week'].fillna(df['Study_Hours_per_Week'].mean())
df['Participation- Extracurricular Activities'] = df['Participation- Extracurricular Activities'].fillna(
    df['Participation- Extracurricular Activities'].mode()[0]
)
df['Participation- Extracurricular Activities'] = df['Participation- Extracurricular Activities'].apply(
    lambda x: 1 if str(x).strip().lower() == 'yes' else 0
)
X = df[['Attendance Rate', 'Participation- Extracurricular Activities', 'Study_Hours_per_Week']]
y = df['Previous Grades']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
st.title("Student Grade Prediction")
st.subheader("Enter the student data to predict grade rate:")
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, step=1)
participation = st.selectbox("Participation in Extracurricular Activities", ["Yes", "No"])
study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=100, step=1)
participation_binary = 1 if participation == "Yes" else 0
if st.button("Predict Grade Rate"):
    input_data = np.array([[attendance_rate, participation_binary, study_hours]])
    input_scaled = scaler.transform(input_data)  
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Grade Rate: {prediction[0]:.2f}%")
st.write("Sample Data:")
st.dataframe(df.sample(n=50, random_state=1))
