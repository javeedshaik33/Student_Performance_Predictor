import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Call Service Predictor",
    page_icon="📞"
)

# Load the dataset
df = pd.read_csv('Call Center Data.csv')

# Print column names to check
st.write("Column names in the dataset:", df.columns)

# Data preprocessing
# Ensure the correct column names are used
if 'Answer Rate' in df.columns:
    df['Answer_Rate'] = df['Answer Rate'].str.replace('%', '').astype(float)
else:
    st.error("Column 'Answer Rate' not found in the dataset!")

# Convert time columns to seconds
def hms_to_seconds(hms):
    try:
        h, m, s = map(int, hms.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        return np.nan  # Return NaN if there's an error in the time format

df['Answer_Speed_AVG'] = df['Answer Speed (AVG)'].apply(hms_to_seconds)
df['Talk_Duration_AVG'] = df['Talk Duration (AVG)'].apply(hms_to_seconds)
df['Waiting_Time_AVG'] = df['Waiting Time (AVG)'].apply(hms_to_seconds)

# Clean the 'Service_Level' column by removing '%' and converting to float
df['Service_Level'] = df['Service Level (20 Seconds)'].str.replace('%', '').astype(float)

# Features and target variable
x = df[['Answer_Rate', 'Answer_Speed_AVG', 'Talk_Duration_AVG', 'Waiting_Time_AVG']]
y = df['Service_Level']

# Normalize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Create and train the model
model = LinearRegression()
model.fit(x_scaled, y)

st.markdown("""
    <style>
        .stApp {
            background-color: #000000;  /* Dark background */
            background-image: none;     /* No gradient */
        }
        .css-1cpxqw2 {
            color: white;  /* White color for text and headings */
            font-size: 1.2em;
            font-weight: bold;
        }
        .stTitle {
            color: white;  /* Ensure title is white */
        }
        .stSubheader {
            color: white;  /* Ensure subheaders are white */
        }
        .stButton>button {
            background-color: #555555;  /* Dark button with lighter text */
            color: white;
        }
        .stTextInput>div>input {
            background-color: #333333;  /* Dark input fields */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)



st.title("Service Level Prediction for Call Center")
st.subheader("Enter Details:")

Answer_Rate = st.number_input("Answer Rate (%)", min_value=0, max_value=100, step=1)
Answer_Speed_AVG = st.text_input("Answer Speed (AVG) in HH:MM:SS", "0:00:00")
Talk_Duration_AVG = st.text_input("Talk Duration (AVG) in HH:MM:SS", "0:00:00")
Waiting_Time_AVG = st.text_input("Waiting Time (AVG) in HH:MM:SS", "0:00:00")

# Convert the input times into seconds
def hms_to_seconds_input(hms):
    try:
        h, m, s = map(int, hms.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        st.error("Invalid time format! Please use HH:MM:SS.")
        return None

predict = st.button("Predict Service Level")

if predict:
    # Convert user inputs into seconds
    Answer_Speed_AVG_sec = hms_to_seconds_input(Answer_Speed_AVG)
    Talk_Duration_AVG_sec = hms_to_seconds_input(Talk_Duration_AVG)
    Waiting_Time_AVG_sec = hms_to_seconds_input(Waiting_Time_AVG)

    # Ensure all inputs are valid
    if None not in (Answer_Speed_AVG_sec, Talk_Duration_AVG_sec, Waiting_Time_AVG_sec):
        # Prepare input for prediction
        input_data = np.array([[Answer_Rate, Answer_Speed_AVG_sec, Talk_Duration_AVG_sec, Waiting_Time_AVG_sec]])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_scaled)[0]

        # Clip the prediction to ensure it's within the valid range (1% to 100%)
        prediction = np.clip(prediction, 1, 100)

        # Display the prediction
        st.write(f"Predicted Service Level: {prediction:.2f}%")
    else:
        st.error("Please correct the input errors before predicting.")

# Plotting (optional)
df_sample = df.sample(n=50, random_state=1)

fig, ax = plt.subplots()
ax.plot(df_sample['Answer_Rate'], df_sample['Service_Level'], label='Service Level by Answer Rate', color='black')
ax.set_xlabel("Answer Rate (%)")
ax.set_ylabel("Service Level (%)")
ax.set_title("Line Graph: Service Level vs Answer Rate")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.bar(df_sample['Answer_Rate'], df_sample['Service_Level'], color='lightgreen')
ax.set_xlabel("Answer Rate (%)")
ax.set_ylabel("Service Level (%)")
ax.set_title("Bar Graph: Service Level by Answer Rate")
st.pyplot(fig)
