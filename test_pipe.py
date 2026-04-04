import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.DataFrame({'age': [20, 30], 'job': ['admin.', 'student']})

# Training
scaler_orig = StandardScaler()
ohe_orig = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
pipe_orig = ColumnTransformer([
   ('num', scaler_orig, ['age']),
   ('cat', ohe_orig, ['job'])
])
res_orig = pipe_orig.fit_transform(df)
print("Original:", res_orig)

# Saved state (mock)
scaler_mean = scaler_orig.mean_
scaler_scale = scaler_orig.scale_
ohe_cats = ohe_orig.categories_

# Manual Restore
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
scaler.var_ = scaler_scale ** 2
# scaler.n_features_in_ = 1  # required in newer versions

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.categories_ = ohe_cats
# ohe.n_features_in_ = 1
# ohe._infrequent_enabled = False # and other private vars...

pipe_manual = ColumnTransformer([
   ('num', scaler, ['age']),
   ('cat', ohe, ['job'])
])

try:
    res_manual = pipe_manual.transform(df)
    print("Manual:", res_manual)
except Exception as e:
    print("Error:", e)
