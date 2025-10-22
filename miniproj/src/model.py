"""
src/model.py

Module for training and optimizing the classification model.
This includes:
1.  Data alignment (X, y, NaNs)
2.  A time-series Purged K-Fold class for cross-validation.
3.  A grid-search function to optimize model hyperparameters (L and H).
4.  A function to fit the final model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

# Import our custom functions
from data import load_aligned_data, STOCK_LIST
from features import engineer_features
from labels import create_labels

# --- Purged K-Fold Class ---
# This is a special CV for time-series data to prevent data leakage.
class PurgedKFold(KFold):
    """
    A KFold class that purges and embargoes samples to prevent
    data leakage from overlapping forward-looking labels.
    """
    def __init__(self, n_splits=5, purge_days=1, embargo_days=5):
        super().__init__(n_splits=n_splits, shuffle=False)
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        
        # Calculate split boundaries
        fold_sizes = np.full(self.n_splits, len(X) // self.n_splits, dtype=int)
        fold_sizes[:len(X) % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            
            # --- Purge logic ---
            # Purge data *before* the test set
            train_stop_purged = start - self.purge_days
            
            # --- Embargo logic ---
            # Embargo data *after* the test set
            train_start_embargoed = stop + self.embargo_days

            # Define train/test indices
            train_indices = np.concatenate([
                indices[:train_stop_purged], 
                indices[train_start_embargoed:]
            ])
            test_indices = indices[start:stop]
            
            yield train_indices, test_indices
            current = stop

# --- Helper Functions ---

def get_clean_data(X_features: pd.DataFrame, 
                   y_labels: pd.Series, 
                   feature_lag: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aligns X and y, applies an optional feature lag, and drops all NaNs.

    Args:
        X_features: The full DataFrame of engineered features.
        y_labels: The full Series of target labels.
        feature_lag: Number of days to lag features (L).

    Returns:
        A tuple (X_clean, y_clean) of aligned, NaN-free data.
    """
    # 1. Apply optional feature lag 'L'
    if feature_lag > 0:
        X_features = X_features.shift(feature_lag)
    
    # 2. Combine into one DataFrame for easy alignment
    combined = pd.concat([X_features, y_labels], axis=1)
    
    # 3. Drop all rows with ANY NaN values
    # This drops NaNs from:
    # - 5-year rolling z-score
    # - 6-month momentum
    # - 63-day forward labels
    # - Applied feature lag
    combined_cleaned = combined.dropna()
    
    # 4. Separate back into X and y
    X_clean = combined_cleaned.drop(columns=y_labels.name)
    y_clean = combined_cleaned[y_labels.name]
    
    return X_clean, y_clean


# --- Main Optimization Function ---

def run_optimization_grid() -> Dict[str, Any]:
    """
    Runs a grid search to find the optimal (H, L) parameters.
    
    H = Label Horizon (e.g., 21, 63, 126 days)
    L = Feature Lag (e.g., 0, 5, 10 days)
    
    Returns:
        A dictionary containing the best parameters and best score.
    """
    print("Loading data for optimization grid search...")
    # Load the base, un-engineered data
    prices_df, macro_df = load_aligned_data()
    
    # --- Define the Grid ---
    H_horizons = list(range(30, 150, 20))   # 30, 31, 32, ..., 60
    L_lags = list(range(5, 40,5)) 
     
    results = []
    
    # --- Engineer base features (we'll lag them later) ---
    X_features_base = engineer_features(macro_df)
    
    # Initialize the cross-validator
    # Purge 1 day, Embargo for 1 week (5 days)
    cv = PurgedKFold(n_splits=5, purge_days=1, embargo_days=5)

    print("🚀 Starting (H, L) optimization grid search...")
    
    for H in H_horizons:
        print(f"\nTesting Horizon H = {H} days")
        # 1. Create labels for this horizon
        y_labels_base = create_labels(prices_df, STOCK_LIST, horizon=H)
        
        for L in L_lags:
            # 2. Align data and apply feature lag L
            # This step also drops all NaNs
            try:
                X_clean, y_clean = get_clean_data(X_features_base, y_labels_base, feature_lag=L)
                
                if len(X_clean) < 500: # Not enough data to train
                    print(f"  L={L:2d}: Skipped (not enough data)")
                    continue

                fold_scores = []
                for train_idx, val_idx in cv.split(X_clean):
                    X_train, y_train = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
                    X_val, y_val = X_clean.iloc[val_idx], y_clean.iloc[val_idx]
                    
                    # 3. Scale and Train
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model = LogisticRegression(solver='lbfgs', 
                           max_iter=1000, 
                           C=0.1,
                           random_state=42)
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # 4. Predict and Score
                    y_pred = model.predict(X_val_scaled)
                    # Use 'macro' average F1-score to balance all 3 classes
                    score = f1_score(y_val, y_pred, average='macro')
                    fold_scores.append(score)
                
                avg_score = np.mean(fold_scores)
                results.append({'H': H, 'L': L, 'score': avg_score})
                print(f"  L={L:2d}: Avg F1 = {avg_score:.4f}")

            except Exception as e:
                print(f"  L={L:2d}: FAILED ({e})")

    # --- Find the best result ---
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['score'].idxmax()]
    
    print("\n--- Grid Search Complete ---")
    print(f"Best F1-Score: {best_result['score']:.4f}")
    print(f"Best Horizon (H): {int(best_result['H'])} days")
    print(f"Best Feature Lag (L): {int(best_result['L'])} days")
    
    return best_result.to_dict()


# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    print("Running model optimization module as standalone script...")
    
    best_params = run_optimization_grid()
    
    print("\nFinal Best Parameters:")
    print(best_params)