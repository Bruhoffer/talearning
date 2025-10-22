"""
src/features.py

Module for transforming raw, lagged macroeconomic data into
predictive features for the model.
"""

import pandas as pd
import numpy as np

# --- Default Parameters ---
# These are standard lookbacks for trading days
CPI_YOY_WINDOW = 252       # ~1 year
Z_SCORE_WINDOW = 5 * 252   # ~5 years (1260 days)
MOMENTUM_WINDOWS = [63, 126] # ~3 months and 6 months


def engineer_features(macro_df: pd.DataFrame, 
                      cpi_window: int = CPI_YOY_WINDOW, 
                      z_window: int = Z_SCORE_WINDOW, 
                      mom_windows: list = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """
    Engineers features from the raw, lagged macro data.
    
    Features created:
    1.  CPI Year-over-Year (YoY) percentage change.
    2.  Rolling Z-Scores (surprise) for all indicators.
    3.  Rolling Momentum (rate of change) for all indicators.

    Args:
        macro_df: DataFrame from data.py (daily, lagged).
        cpi_window: Lookback for YoY CPI calculation.
        z_window: Lookback for z-score calculation.
        mom_windows: List of lookbacks for momentum calculation.

    Returns:
        A DataFrame 'X' containing the engineered features.
    """
    print(f"Engineering features with z-window={z_window}...")
    
    # 1. Initialize the new 'X' (features) DataFrame
    X = pd.DataFrame(index=macro_df.index)
    
    # 2. Create the primary transformation: CPI YoY % Change
    # This is a crucial feature, so we create it first.
    X['CPI_YoY'] = macro_df['CPIAUCSL'].pct_change(periods=cpi_window)
    
    # 3. Create a dictionary of the series to process
    # We use the new 'CPI_YoY' and the original 'UNRATE' and 'CFNAI'
    series_to_process = {
        'UNRATE': macro_df['UNRATE'],
        'CFNAI': macro_df['CFNAI'],
        'CPI_YoY': X['CPI_YoY']  # Use the transformed series
    }
    
    # 4. Loop to create Z-Scores and Momentum features
    for name, series in series_to_process.items():
        
        # a. Create Z-Score (The "Surprise")
        # How many standard deviations from the 5-year mean?
        mean = series.rolling(window=z_window).mean()
        std = series.rolling(window=z_window).std()
        X[f'{name}_zscore'] = (series - mean) / std
        
        # b. Create Momentum (The "Trend")
        # What is the 3-month and 6-month change?
        for period in mom_windows:
            col_name = f'{name}_{period}d_change'
            X[col_name] = series.diff(periods=period)
            
    print("✅ Feature engineering complete.")
    # Note: We do NOT drop NaNs here. That happens after labels are created.
    return X


# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    # Import the data loader from our other file
    from data import load_aligned_data
    
    print("Running feature engineering module as standalone script...")
    
    # 1. Load the raw, aligned data
    try:
        prices_df, macro_df = load_aligned_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data.py script runs correctly.")
        exit()
        
    # 2. Engineer the features
    X_features = engineer_features(macro_df)
    
    print("\n--- Engineered Features (X_features) ---")
    
    # Show the start - will be full of NaNs from the 5-year z-score
    print("Head (expect NaNs from rolling windows):")
    print(X_features.head(5))
    
    # Show the end
    print("\nTail (should be populated):")
    print(X_features.tail(5))
    
    # Show info to see all 9 new columns and the NaNs
    print("\n--- DataFrame Info ---")
    X_features.info()