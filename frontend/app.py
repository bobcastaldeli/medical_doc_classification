"""This is the frontend of the app."""


import requests
import streamlit as st


st.title("App Medical Docs Classifier App")
st.write("This app predicts the type of cancer based on a medical document")


input_text = st.text_input("Enter your text here")

if st.button("Predict"):
    data = {"text": input_text}

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    prediction = response.json()["prediction"]

    if prediction == 0:
        st.write("This document is about colon cancer")
    elif prediction == 1:
        st.write("This document is about lung cancer")
    else:
        st.write("This document is about thyroid cancer")
