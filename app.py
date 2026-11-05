# app.py - Prototipo Web para Clasificación de Masas Mamarias con SVM
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, 1, 0) # 1: Maligno, 0: Benigno
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline con la mejor configuración (SVM RBF C=10)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, data.feature_names

model, feature_names = train_model()

st.title("🩺 Prototipo Experimental de IA: Clasificación de Masas Mamarias")
st.markdown("""
**Proyecto Final - Inteligencia Artificial (UPB 2026)**  
*Integrantes:* Juan Felipe Cardona, Daniel Villa y Valeria Gómez.  
⚠️ **Aviso importante:** Este sistema es un prototipo académico y experimental. No constituye una herramienta de diagnóstico médico oficial ni sustituye la valoración de un profesional de la salud.
""")

st.sidebar.header("Parámetros de Entrada (Características FNA)")
input_values = []
# Usamos el promedio real de cada columna del dataset en lugar de 0.0 por defecto
default_values = X.mean(axis=0) 

for i, name in enumerate(feature_names):
    val = st.sidebar.number_input(f"{name}", value=float(default_values[i]))
    input_values.append(val)

if st.button("Clasificar Muestra"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]
    
    st.subheader("Resultado del Análisis Experimental:")
    if prediction == 1:
        st.error(f"⚠️ **Resultado Predicho: MALIGNO** (Probabilidad: {proba[1]*100:.2f}%)")
        st.markdown("Se detectaron patrones numéricos asociados con características malignas en los núcleos celulares. Se requiere revisión clínica especializada inmediata.")
    else:
        st.success(f"✅ **Resultado Predicho: BENIGNO** (Probabilidad: {proba[0]*100:.2f}%)")
        st.markdown("La muestra presenta patrones numéricos dentro del rango característico de lesiones benignas.")
        
    st.info(f"Confianza del modelo — Benigno: {proba[0]*100:.2f}% | Maligno: {proba[1]*100:.2f}%")
