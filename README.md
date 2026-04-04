# 🏦 Bank Marketing Term Deposit Prediction

An end-to-end Machine Learning project to predict whether a client will subscribe to a term deposit based on their demographic information and past interactions with the bank.

## 🎯 Project Overview
Running telemarketing campaigns is expensive and often yields low conversion rates. This project aims to reduce marketing costs and improve efficiency by targeting only those clients who are most likely to subscribe to a term deposit. 

We built a modular, production-ready machine learning pipeline that includes Data Preprocessing, Hyperparameter Tuning, Model Evaluation, and a Streamlit Web UI.

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Random Forest
- **Web Framework**: Streamlit
- **Data Visualization**: Matplotlib, Seaborn

---

## 🚀 How to Run Locally

### 1. Setup Environment
Ensure your dataset is placed at `data/data.csv`. Then, install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
cd notebooks
python 01_EDA.py
```
*This will output basic data insights to the console and generate visualization plots in the `notebooks/` directory.*

### 3. Model Training
To train the models and perform hyperparameter tuning:
```bash
cd src
python train.py
```
*This script will train Logistic Regression, Decision Tree, Random Forest, and XGBoost models. It then tunes the best performing model and saves the final serialized pipeline to `models/best_model.pkl`.*

### 4. Start the Streamlit App
To launch the interactive UI for making predictions:
```bash
streamlit run app/main.py
```

---

## 📊 Results Summary
After comparing several models, **Random Forest** proved to be the most effective, especially after tuning to handle the highly imbalanced nature of the dataset.

- **Accuracy**: ~85%
- **F1 Score**: ~0.61
- **ROC-AUC**: ~0.93

### Key Insights
- **Previous Interactions**: `poutcome` (outcome of previous marketing campaigns) is a strong predictor. 
- **Duration**: The interaction `duration` is heavily correlated with a successful deposit subscription. 
