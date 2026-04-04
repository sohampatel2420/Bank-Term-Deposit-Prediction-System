from src.predict import ModelEngine
from src.preprocess import load_data
import os

if __name__ == "__main__":
    if not os.path.exists('model.json'):
        print("Missing model.json")
    else:
        df, _ = load_data('data.csv')
        sample = df.head(5).copy()
        
        engine = ModelEngine('model.json')
        preds, probas = engine.predict(sample)
        print("Manual Prediction:")
        print(preds)
        print(probas)
