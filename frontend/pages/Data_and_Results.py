"""This page is used to display the results of the EDA and the results of the
model."""


import streamlit as st
import pandas as pd
import plotly.express as px


st.write(
    """
    Para classificação de documentos de texto biomédico, resumos e artigos completos
    disponíveis e usados. Este conjunto de dados se concentrou em artigos de pesquisa cujo
    tamanho da página fica em torno de 6 páginas por artigo. Conjunto de dados inclui
    documentos de câncer a serem classificado em 3 categorias.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Tamanho do conjunto de dados", "7569")
col2.metric("Tamanho do conjunto de treinamento", "6055", "80%")
col3.metric("Tamanho do conjunto de teste", "1514", "20%")


df = pd.read_csv("../data/interim/processed_data.csv", encoding="latin-1")

st.bar_chart(df["label"].value_counts())


st.image("../reports/UMAP.png", caption="UMAP para os dados de treinamento")


metrics = pd.read_csv(
    "../reports/metrics.txt", sep=":", names=["metric", "value"]
)

# create cards for each metric side by side
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", metrics.loc[0, "value"])
col2.metric("Acurácia Balanceada", metrics.loc[1, "value"])
col3.metric("Precisão", metrics.loc[2, "value"])
col4.metric("F1-Score", metrics.loc[3, "value"])


st.image("../reports/confusion_matrix.png", caption="Matriz de confusão")
