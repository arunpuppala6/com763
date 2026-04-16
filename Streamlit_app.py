import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AirSense AI", layout="wide")
# Styling
st.markdown("""
<style>
h1 {font-size: 42px;}
h2 {font-size: 30px;}
h3 {font-size: 24px;}
.stButton>button {
    background-color: #2ecc71;
    color: white;
    font-size: 20px;
    height: 3em;
    width: 200px;
    border-radius: 8px;
    display: block;
    margin: auto;
}
.big-font {
    font-size: 32px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Title
st.title("AirSense AI")
st.write("Predict AQI using pollution and date inputs with machine learning")

# Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Enter Pollution Details")
    CO = st.number_input("CO", value=0.0)
    NO2 = st.number_input("NO2", value=0.0)
    SO2 = st.number_input("SO2", value=0.0)
    O3 = st.number_input("O3", value=0.0)
    PM25 = st.number_input("PM2.5", value=0.0)
    PM10 = st.number_input("PM10", value=0.0)
with col2:
    st.subheader("Date Information")
    Year = st.number_input("Year", value=2024)
    Month = st.slider("Month", 1, 12, 1)
    Day = st.slider("Day", 1, 31, 1)
    city_list = [col for col in columns if col.startswith("City_")]
    city_selected = st.selectbox("Select City", ["None"] + city_list)

# Input Summary 
st.subheader("Input Summary")
summary_df = pd.DataFrame({
    "CO": [CO],
    "NO2": [NO2],
    "SO2": [SO2],
    "O3": [O3],
    "PM2.5": [PM25],
    "PM10": [PM10],
    "Year": [Year],
    "Month": [Month],
    "Day": [Day],
    "City": [city_selected]
})
st.dataframe(summary_df)
# Prepare input
input_dict = {col: 0 for col in columns}
input_dict.update({
    "CO": CO,
    "NO2": NO2,
    "SO2": SO2,
    "O3": O3,
    "PM2.5": PM25,
    "PM10": PM10,
    "Year": Year,
    "Month": Month,
    "Day": Day
})
if city_selected != "None":
    input_dict[city_selected] = 1
input_df = pd.DataFrame([input_dict])

# Button
if st.button("Predict AQI"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    # BIG AQI VALUE
    st.markdown(f"<div class='big-font'>AQI: {round(prediction,2)}</div>", unsafe_allow_html=True)

    # Category
    if prediction <= 50:
        st.success("Good Air Quality")
    elif prediction <= 100:
        st.info("Moderate Air Quality")
    elif prediction <= 200:
        st.warning("Unhealthy Air Quality")
    else:
        st.error("Very Unhealthy Air Quality")