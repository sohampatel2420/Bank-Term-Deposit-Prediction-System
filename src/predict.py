import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ManualPreprocessor:
    def __init__(self, prep_dict):
        self.num_cols = prep_dict['num_cols']
        self.cat_cols = prep_dict['cat_cols']
        self.scaler_mean = np.array(prep_dict['scaler_mean'])
        self.scaler_scale = np.array(prep_dict['scaler_scale'])
        self.ohe_categories = [np.array(cats) for cats in prep_dict['ohe_categories']]
        
    def transform(self, df):
        # 1. Scale num_cols
        num_data = df[self.num_cols].values
        num_scaled = (num_data - self.scaler_mean) / self.scaler_scale
        
        # 2. One Hot Encode cat_cols (manually)
        cat_encoded_list = []
        for i, col in enumerate(self.cat_cols):
            cats = self.ohe_categories[i]
            col_data = df[col].values
            
            # Create a localized array of zeros
            ohe_matrix = np.zeros((len(df), len(cats)))
            for j, val in enumerate(col_data):
                # Handle unknown by ignoring (all zeros for this row)
                idx = np.where(cats == val)[0]
                if len(idx) > 0:
                    ohe_matrix[j, idx[0]] = 1.0
            
            cat_encoded_list.append(ohe_matrix)
            
        if cat_encoded_list:
            cat_scaled = np.hstack(cat_encoded_list)
            return np.hstack([num_scaled, cat_scaled])
        return num_scaled

class ManualRF:
    def __init__(self, rf_dict):
        self.trees = rf_dict['trees']
        self.classes = np.array(rf_dict['classes'])
        
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_trees = len(self.trees)
        probas = np.zeros((n_samples, len(self.classes)))
        
        for tree in self.trees:
            left = tree['children_left']
            right = tree['children_right']
            feature = tree['feature']
            threshold = tree['threshold']
            value = tree['value']
            
            # Predict for each tree (vectorized where possible, but loop is fine for few elements. 
            # Actually, to avoid slow loops, we can do batch masks)
            # Batch masks vectorization for decision tree predictions:
            
            node_indices = np.zeros(n_samples, dtype=int)
            mask_active = np.ones(n_samples, dtype=bool)
            
            while np.any(mask_active):
                # get nodes for active samples
                current_nodes = node_indices[mask_active]
                
                # Check which ones are leaves
                is_leaf = np.array([left[n] == -1 for n in current_nodes])
                
                # Turn off mask for leaves
                active_indices = np.where(mask_active)[0]
                leaves_indices = active_indices[is_leaf]
                mask_active[leaves_indices] = False
                
                if not np.any(mask_active):
                    break
                    
                # For remaining active, advance them
                active_indices = np.where(mask_active)[0]
                current_nodes = node_indices[active_indices]
                
                features = np.array([feature[n] for n in current_nodes])
                thresholds = np.array([threshold[n] for n in current_nodes])
                
                go_left = X[active_indices, features] <= thresholds
                
                left_children = np.array([left[n] for n in current_nodes])
                right_children = np.array([right[n] for n in current_nodes])
                
                node_indices[active_indices] = np.where(go_left, left_children, right_children)

            # Node indices are now leaves
            for i in range(n_samples):
                leaf_node = node_indices[i]
                val = np.array(value[leaf_node][0])
                probas[i] += val / np.sum(val)
                
        probas /= n_trees
        return probas

class ModelEngine:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.preprocessor = ManualPreprocessor(data['preprocessor'])
        self.rf = ManualRF(data['rf_model'])
        self.threshold = data.get('threshold', 0.5)
        
    def predict(self, df):
        logger.info(f"Running bulk inference on {len(df)} records...")
        X_trans = self.preprocessor.transform(df)
        probas = self.rf.predict_proba(X_trans)[:, 1]
        preds = probas >= self.threshold
        return preds, probas
