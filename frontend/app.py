"""This is the frontend of the app."""


import requests
import pandas as pd
import plotly.express as px
import streamlit as st


st.title("App Medical Docs Classifier App")
st.subheader(
    "This app predicts the type of cancer based on a medical document"
)
st.write(
    """
    For Biomedical text document classification, abstract and full papers
    available and used. This dataset focused on long research paper whose
    page size more than 6 pages. Dataset includes cancer documents to be
    classified into 3 categories like 'Thyroid_Cancer', 'Colon_Cancer', 'Lung_Cancer'.
    """
)


col1, col2, col3 = st.columns(3)
col1.metric("Size of the dataset", "7569")
col2.metric("Train size", "6055", "80%")
col3.metric("Test size", "1514", "20%")


input_text = st.text_area("Enter your text here")


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
