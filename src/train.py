import os
import sys
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Add the root directory to path to allow src imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data, get_preprocessor
from src.utils import save_model_manual

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training(data_path="data.csv", output_path="model.json"):
    logger.info("Starting Training Pipeline...")
    
    # 1. Load Data
    X, y = load_data(data_path)
    
    # 2. Architect Preprocessor & Model 
    preprocessor = get_preprocessor(X)
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5, 
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    # 3. Fit Pipeline
    logger.info("Fitting Random Forest Model...")
    pipeline.fit(X, y)
    
    # Optional: calculate optimal threshold. For simplicity, we use 0.5.
    threshold = 0.5
    
    # 4. Save via Manual Engine (Zero Pickle approach)
    logger.info("Executing custom JSON serialization...")
    save_model_manual(pipeline, threshold, output_path)
    logger.info("Training and Extraction complete!")

if __name__ == "__main__":
    run_training()
