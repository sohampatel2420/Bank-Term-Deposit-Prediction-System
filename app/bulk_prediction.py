import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from src.utils import load_uploaded_file, validate_dataframe, EXPECTED_COLUMNS

def get_sample_df():
    return pd.DataFrame([{
        "age": 30, "job": "admin.", "marital": "married", "education": "secondary",
        "default": "no", "balance": 1000, "housing": "yes", "loan": "no",
        "contact": "cellular", "day": 15, "month": "may", "duration": 200,
        "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
    }])

def render_bulk_prediction():
    st.header("🔍 Bulk Prediction Scanner")

    # --- Phase 1: Templates ---
    st.markdown("### 1. Download Sample Templates 🔗")
    st.write("Ensure your data matches these templates before uploading.")
    sample_df = get_sample_df()
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.download_button("📄 Download CSV Sample", sample_df.to_csv(index=False), "sample.csv", use_container_width=True)
    with t_col2:
        buffer = io.BytesIO()
        sample_df.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button("📊 Download Excel Sample", buffer.getvalue(), "sample.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with t_col3:
        st.download_button("📦 Download JSON Sample", sample_df.to_json(orient="records"), "sample.json", use_container_width=True)

    st.markdown("---")

    # --- Phase 2: Upload ---
    st.markdown("### 2. Upload File to Scan")
    uploaded_file = st.file_uploader("Choose a file (CSV, XLSX, JSON limit 200MB)", type=["csv", "xlsx", "json", "xls"])

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            
            # --- Phase 3: Preview & Validation ---
            is_valid, msg = validate_dataframe(df)
            
            st.markdown("#### Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
            if not is_valid:
                st.error(f"⚠️ Validation Failed: {msg}")
                st.stop()
            else:
                st.success("✅ Data validated successfully. All required columns are present.")
                
            # --- Phase 4: Prediction Execution ---
            if st.button("⚡ Run Bulk Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    # Progress bar bonus
                    progress_text = "Running batch inference..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    time.sleep(0.5) # Slight delay for visual UX
                    my_bar.progress(30, text="Scaling and Encoding...")
                    
                    model_engine = st.session_state.model_engine
                    
                    my_bar.progress(60, text="Executing Random Forest...")
                    preds, probas = model_engine.predict(df)
                    
                    df['prediction'] = np.where(preds, 'Yes', 'No')
                    df['probability'] = probas
                    
                    my_bar.progress(100, text="Finalizing results...")
                    time.sleep(0.5)
                    my_bar.empty()
                    
                st.markdown("---")
                st.markdown("### 📈 Prediction Summary")
                
                # Metrics
                yes_count = (df['prediction'] == 'Yes').sum()
                no_count = (df['prediction'] == 'No').sum()
                total = len(df)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", total)
                c2.metric("Predicted YES 👍", f"{yes_count} ({yes_count/total:.1%})")
                c3.metric("Predicted NO 👎", f"{no_count} ({no_count/total:.1%})")
                
                st.markdown("#### Model Results")
                # Highlight prediction column by bringing it to front
                cols = ['prediction', 'probability'] + [col for col in df.columns if col not in ['prediction', 'probability']]
                st.dataframe(df[cols].head(10), use_container_width=True)
                
                # --- Phase 5: Download ---
                st.markdown("### 💾 Export Results")
                out_col1, out_col2 = st.columns(2)
                
                with out_col1:
                    csv_res = df.to_csv(index=False)
                    st.download_button("Download Predictions (CSV)", csv_res, "predictions.csv", use_container_width=True)
                    
                with out_col2:
                    out_buffer = io.BytesIO()
                    df.to_excel(out_buffer, index=False, engine='openpyxl')
                    st.download_button("Download Predictions (Excel)", out_buffer.getvalue(), "predictions.xlsx", 
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
