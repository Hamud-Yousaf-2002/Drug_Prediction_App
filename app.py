import os

import pandas as pd
import requests
import streamlit as st

# In Docker this is set to the API container's address; locally it falls back to port 8001
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/predict")

st.set_page_config(page_title="Drug Prediction", page_icon="💊")
st.title("💊 Drug Prediction")
st.write("Enter patient details to predict the suitable drug.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    sex = st.selectbox("Sex", ["M", "F"])
    bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
with col2:
    chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
    na_to_k = st.number_input("Na to K ratio", min_value=0.1, value=15.5, step=0.1)

if st.button("Predict", type="primary"):
    payload = {
        "Age": age, 
        "Sex": sex, 
        "BP": bp, 
        "Cholesterol": chol, 
        "Na_to_K": na_to_k
               }
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        st.success(f"Model prediction: **{data['prediction']}**")
        probs = pd.Series(data["probabilities"], name="Probability")
        # DISPLAY VALUES USING BARCHART
        st.bar_chart(probs)
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. Is FastAPI running?\n\n{e}")
