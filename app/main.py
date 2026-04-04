import streamlit as st
import os
import sys

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import ModelEngine
from app.manual_prediction import render_manual_prediction
from app.bulk_prediction import render_bulk_prediction

# ==========================
# Load Model Natively (No Joblib)
# ==========================
@st.cache_resource
def load_pure_model():
    model_path = "model.json"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please run 'python src/train.py' first.")
        st.stop()
        
    return ModelEngine(model_path)


# ==========================
# App Config
# ==========================
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide", page_icon="🏦")

# Initialize engine globally mapped in session_state
if 'model_engine' not in st.session_state:
    with st.spinner("Loading ultra-fast zero-dependency model..."):
        st.session_state.model_engine = load_pure_model()

st.title("🏦 Bank Term Deposit Prediction System")
st.markdown("Predict whether a client will subscribe to a term deposit based on their profile and campaign history.")

# ==========================
# Sidebar UI
# ==========================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Manual Prediction", "Bulk Prediction Scanner"], index=1)


# ==========================
# Route Management
# ==========================
if option == "Manual Prediction":
    render_manual_prediction()
else:
    render_bulk_prediction()
