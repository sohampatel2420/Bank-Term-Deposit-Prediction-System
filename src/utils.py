import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants matched exactly with original app.py
EXPECTED_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

def load_uploaded_file(file):
    """Loads an uploaded file (CSV, Excel, JSON) into a pandas dataframe."""
    name = file.name.lower()
    try:
        if name.endswith('.csv'):
            # Detect separator roughly
            sample = file.read(2048).decode('utf-8')
            file.seek(0)
            sep = ';' if sample.count(';') > sample.count(',') else ','
            return pd.read_csv(file, sep=sep)
        elif name.endswith('.xlsx') or name.endswith('.xls'):
            return pd.read_excel(file)
        elif name.endswith('.json'):
            return pd.read_json(file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON.")
    except Exception as e:
        raise ValueError(f"Could not read the file correctly. {str(e)}")

def validate_dataframe(df):
    """Validates dataframe for missing strings and missing columns."""
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
        
    return True, "Valid"

# ==========================================
# Custom JSON Serializer for Random Forest Pipeline
# ==========================================
def save_model_manual(pipeline, threshold, extract_path):
    """
    Extracts ColumnTransformer (StandardScaler + OneHotEncoder) and
    RandomForestClassifier configuration natively into a pure JSON.
    """
    logger.info(f"Extracting Pipeline into JSON payload: {extract_path}")
    col_transformer = pipeline.named_steps['preprocessor']
    rf_model = pipeline.named_steps['classifier']
    
    # 1. Preprocessor Extraction
    num_pipe = col_transformer.named_transformers_['num']
    cat_pipe = col_transformer.named_transformers_['cat']
    
    scaler = num_pipe.named_steps['scaler']
    ohe = cat_pipe.named_steps['onehot']
    
    preprocessor_state = {
        'num_cols': [x.replace('num__', '') for x in col_transformer.get_feature_names_out() if x.startswith('num__')],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'cat_cols': [c for c in EXPECTED_COLUMNS if df_type_mock(c) == 'cat'], # Reconstruct categorical column base manually or fetch
        # Actually it's safer to extract from memory
    }
    
    # Accurate cols
    preprocessor_state['num_cols'] = num_pipe.feature_names_in_.tolist()
    preprocessor_state['cat_cols'] = cat_pipe.feature_names_in_.tolist()
    preprocessor_state['ohe_categories'] = [cat.tolist() for cat in ohe.categories_]
    
    # 2. RF Extraction
    trees = []
    for dt in rf_model.estimators_:
        tree = dt.tree_
        trees.append({
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'value': tree.value.tolist()
        })
        
    rf_state = {
        'trees': trees,
        'classes': rf_model.classes_.tolist()
    }
    
    payload = {
        'preprocessor': preprocessor_state,
        'rf_model': rf_state,
        'threshold': float(threshold)
    }
    
    with open(extract_path, 'w') as f:
        json.dump(payload, f)
        
    logger.info("Manual JSON serialization successfully completed!")

def df_type_mock(col):
    if col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
        return 'num'
    return 'cat'
