import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

def rf_to_dict(rf):
    trees = []
    for dt in rf.estimators_:
        tree = dt.tree_
        trees.append({
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'value': tree.value.tolist()
        })
    return {'trees': trees, 'classes': rf.classes_.tolist()}

class ManualRF(BaseEstimator, ClassifierMixin):
    def __init__(self, model_dict):
        self.trees = model_dict['trees']
        self.classes_ = np.array(model_dict['classes'])
        
    def predict_proba(self, X):
        if isinstance(X, np.ndarray) == False:
            X = np.array(X)
        n_samples = X.shape[0]
        n_trees = len(self.trees)
        probas = np.zeros((n_samples, len(self.classes_)))
        
        for tree in self.trees:
            left = tree['children_left']
            right = tree['children_right']
            feature = tree['feature']
            threshold = tree['threshold']
            value = tree['value']
            
            for i in range(n_samples):
                node = 0
                while left[node] != -1:
                    if X[i, feature[node]] <= threshold[node]:
                        node = left[node]
                    else:
                        node = right[node]
                # node is a leaf
                val = np.array(value[node][0])
                probas[i] += val / np.sum(val)
                
        probas /= n_trees
        return probas
        
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
scaler = StandardScaler()
rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
pipe = Pipeline([('scaler', scaler), ('rf', rf)])
pipe.fit(X, y)

# Save
model_dict = rf_to_dict(pipe.named_steps['rf'])
with open('rf.json', 'w') as f:
    json.dump(model_dict, f)

# Load
with open('rf.json', 'r') as f:
    loaded_dict = json.load(f)

manual_rf = ManualRF(loaded_dict)
pipe2 = Pipeline([('scaler', scaler), ('rf', manual_rf)])

print("Original:", pipe.predict(X[:5]))
print("Manual:  ", pipe2.predict(X[:5]))
print("Proba Original:", pipe.predict_proba(X[:5]))
print("Proba Manual:  ", pipe2.predict_proba(X[:5]))
