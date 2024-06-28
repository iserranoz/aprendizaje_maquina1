import streamlit as st
import pandas as pd
import joblib
from utils import *
import matplotlib.pyplot as plt



st.set_page_config(page_title='UP Aprendizaje de Maquina 1',page_icon="assets/logo2.png",layout='wide')


nav = st.sidebar.selectbox("Menu",['Presentación',"Panel 1"])

if nav == "Presentación":
    col1 , col2= st.columns([1,8])
    with col1:
        st.image("assets/logo1.jpeg",width=180)
    with col2:
        st.title("")
        st.title("")
        st.title("")
        st.write("")
        st.write("")
        st.title("Comparación de algoritmos de clasificación para la predicción del riesgo crediticio utilizando vectores esparsos")
        st.write("Maestría en Ciencia de Datos")
        st.write("""Alumnos: Iván Serrano, Patricia León, Miguel Canul""")
        st.write("Profesor: Hiram Ponce")
        st.write("Materia: Aprendizaje de Maquina 1")

        st.markdown("""
        ## **Introducción**

        El desarrollo de esta aplicación tiene como objetivo la demostración de resultados e inferencias  de modelos en un entorno 
                    que se asemeja al productivo referente al proyecto que puede consultarse en el siguiente
                    [link](https://github.com/iserranoz/aprendizaje_maquina1).

        Nuestro propsito fue entender qué tipos de modelos pueden ser útiles para predecir si un cliente pagará o no 
        utilizando vectores esparsos. En otras palabras, trabajaremos con listas de 0 y 1, donde 1 significa que el cliente tiene instalada la aplicación X y 0 si no la tiene. Se utilizarán modelos de aprendizaje automático supervisado para esta comparación, empleando diferentes métricas como F1-Score, AUC-ROC, entre otras

       """)
        
    
if nav == "Panel 1":
    st.title("Resultados")
    st.title("")
    st.title("")
    st.title("")
    st.markdown("""En esta sección se presentan los resultados obtenidos de dos de los modelos quye tuvieron mejor desempeño, 
                Regresión Loggística con PCA y SGD Classifier, las predicciones se ejcutan al espandir la sección la primera vez""")
    
    ### cargar datos test
    df_test = pd.read_csv('data/data_test.csv')
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    ### cargar pipelines
    
    pipeline_lrg = joblib.load('pipelines/model_lgr_PCA.joblib')
    pipeline_sgd = joblib.load('pipelines/model_SGD.joblib')
    with st.expander("Regresión Logística con PCA"):
        st.write("Se muestran los resultados obtenidos por el modelo, el cual tuvo el mejor performance")
        y_pred_lrg = pipeline_lrg.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_pred_lrg, y_test, "Regresión Logística con PCA")
        st.pyplot(plt)
        st.title("")
        st.title("")
        plot_density(y_pred_lrg, y_test, "Regresión Logística con PCA")
        st.pyplot(plt)

    with st.expander("Descenso de gradiente estocástico"):
        st.write("Se muestran los resultados obtenidos por el modelo, el cual tuvo el segundo mejor performance")
        y_pred_sgd = pipeline_sgd.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_pred_sgd, y_test, "Descenso de gradiente estocástico")
        st.pyplot(plt)
        st.title("")
        st.title("")
        plot_density(y_pred_sgd, y_test, "Descenso de gradiente estocástico")
        st.pyplot(plt)
        
        
