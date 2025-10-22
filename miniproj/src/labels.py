"""
src/labels.py

Module for creating the target variable (y) for the model.
This involves:
1. Creating an equal-weight portfolio.
2. Calculating 3-month (63-day) forward returns.
3. Assigning a 3-class label (Long, Short, Neutral).
"""

import pandas as pd
import numpy as np

# --- Configuration ---
HOLDING_PERIOD = 63  # ~3 months in trading days
LONG_THRESHOLD = 0.05  # 5% profit
SHORT_THRESHOLD = -0.05 # 5% loss

def create_labels(prices_df: pd.DataFrame, 
                  stock_list: list, 
                  horizon: int = HOLDING_PERIOD,
                  long_thresh: float = LONG_THRESHOLD,
                  short_thresh: float = SHORT_THRESHOLD) -> pd.Series:
    """
    Creates 3-class target labels based on 3-month forward returns
    of an equal-weight portfolio.

    Classes:
    - 2: Long (forward return > long_thresh)
    - 0: Short (forward return < short_thresh)
    - 1: Neutral (between thresholds)

    Args:
        prices_df: DataFrame of stock prices (from data.py).
        stock_list: List of tickers to include in the portfolio.
        horizon: Forward-looking period in trading days.
        long_thresh: Upper return threshold for a 'Long' signal.
        short_thresh: Lower return threshold for a 'Short' signal.

    Returns:
        A pandas Series 'y' containing the 3-class labels.
    """
    print(f"Creating labels for {len(stock_list)} stocks with {horizon}-day horizon...")
    
    # 1. Create an Equal-Weight Portfolio
    # .mean(axis=1) calculates the average price across all stocks for each day
    ew_portfolio = prices_df[stock_list].mean(axis=1)
    
    # 2. Calculate Forward Returns
    # .shift(-horizon) pulls future prices backward in time
    # This calculates (Price_t+63 / Price_t) - 1
    fwd_returns = (ew_portfolio.shift(-horizon) / ew_portfolio) - 1
    
    # 3. Create 3-Class Labels
    # Initialize 'y' with the 'Neutral' class (1)
    y = pd.Series(1, index=fwd_returns.index, name="Target_Label")
    
    # Set 'Long' class (2)
    y.loc[fwd_returns > long_thresh] = 2
    
    # Set 'Short' class (0)
    y.loc[fwd_returns < short_thresh] = 0
    
    y.loc[fwd_returns.isnull()] = np.nan
    
    # NaNs will be created at the end of the series because
    # we can't calculate a 63-day forward return.
    # We will drop these later when aligning with features.
    
    print("✅ Label creation complete.")
    return y


# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    # Import the data loader and config from other files
    from data import load_aligned_data, STOCK_LIST
    
    print("Running label creation module as standalone script...")
    
    # 1. Load the raw, aligned data
    try:
        prices_df, macro_df = load_aligned_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data.py script runs correctly.")
        exit()
        
    # 2. Create the labels
    y_labels = create_labels(prices_df, STOCK_LIST)
    
    print("\n--- Target Labels (y_labels) ---")
    
    # Show the start
    print("Head (should be populated):")
    print(y_labels.head(5))
    
    # Show the end
    print("\nTail (expect NaNs from forward-looking window):")
    print(y_labels.tail(5))
    
    # Show value counts to see the class distribution
    print("\n--- Label Distribution ---")
    print(y_labels.value_counts(dropna=False))