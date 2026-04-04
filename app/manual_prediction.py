import streamlit as st
import pandas as pd
from src.predict import ModelEngine
from src.utils import EXPECTED_COLUMNS
import io

def render_manual_prediction():
    st.header("📝 Client Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'])
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
        balance = st.number_input("Yearly Average Balance (in EUR)", -10000, 200000, 1000)

    with col2:
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
        contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone', 'unknown'])
        day = st.number_input("Last Contact Day of the Month", 1, 31, 15)
        month = st.selectbox("Last Contact Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

    with col3:
        duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 200)
        campaign = st.number_input("Number of Contacts during this campaign", 1, 50, 1)
        pdays = st.number_input("Days since last contact (from previous campaign)", -1, 1000, -1)
        previous = st.number_input("Number of Contacts performed before this campaign", 0, 50, 0)
        poutcome = st.selectbox("Outcome of previous campaign", ['unknown', 'failure', 'success', 'other'])

    if st.button("🚀 Predict Conversion", use_container_width=True):
        try:
            model_engine = st.session_state.model_engine
            
            # Construct DataFrame exactly matching EXPECTED_COLUMNS
            input_dict = {
                'age': age, 'job': job, 'marital': marital, 'education': education, 
                'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
                'contact': contact, 'day': day, 'month': month, 'duration': duration,
                'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
            }
            
            input_df = pd.DataFrame([input_dict], columns=EXPECTED_COLUMNS)
            preds, probas = model_engine.predict(input_df)
            
            is_yes = preds[0]
            probability = probas[0]
            
            st.markdown("---")
            if is_yes:
                st.success(f"### Prediction: YES\n**Probability:** {probability:.2%}\n\nThis client is highly likely to subscribe to the term deposit.")
            else:
                st.error(f"### Prediction: NO\n**Probability:** {probability:.2%}\n\nThis client is unlikely to subscribe to the term deposit.")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
