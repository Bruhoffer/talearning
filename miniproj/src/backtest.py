"""
src/backtest.py

Module for running the final backtest of the strategy.
This involves:
1.  Training the final model on the best (H, L) parameters.
2.  Generating daily position signals (Long, Short, Neutral).
3.  Simulating the portfolio's equity curve over the training period.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

# Import our custom functions
from data import load_aligned_data, STOCK_LIST
from features import engineer_features
from labels import create_labels
from model import get_clean_data

# --- Default Backtest Configuration ---
INITIAL_CAPITAL = 500_000
CONFIDENCE_THRESHOLD = 0.5 # Your 'τ*' (tau) threshold

# --- Main Backtest Functions ---

def train_final_model(best_params: Dict[str, Any]):
    """
    Trains one final model on the full dataset using the
    best H and L parameters.
    
    Returns the fitted model, scaler, and the full X/y data.
    """
    print("Training final model on best parameters...")
    
    # 1. Load all data
    prices_df, macro_df = load_aligned_data()
    X_features = engineer_features(macro_df)
    
    H = int(best_params['H'])
    L = int(best_params['L'])
    
    y_labels = create_labels(prices_df, STOCK_LIST, horizon=H)
    
    # 2. Get the final, clean, aligned data
    X_clean, y_clean = get_clean_data(X_features, y_labels, feature_lag=L)
    
    # 3. Fit the Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # 4. Fit the Final Model
    # We use the same settings as the grid search
    model = LogisticRegression(
        # multi_class='multinomial', # <-- Removed to silence warning
        solver='lbfgs',
        max_iter=1000,
        C=0.1,
        random_state=42
    )
    model.fit(X_scaled, y_clean)
    
    print("✅ Final model trained.")
    
    # We return X_clean.index so we know which dates our signals map to
    return model, scaler, X_clean, y_clean

def get_portfolio_returns(prices_df: pd.DataFrame, 
                          stock_list: list) -> pd.Series:
    """
    Calculates the daily returns of the equal-weight stock basket.
    """
    # 1. Create the equal-weight portfolio
    ew_portfolio = prices_df[stock_list].mean(axis=1)
    
    # 2. Calculate daily returns
    daily_returns = ew_portfolio.pct_change().fillna(0)
    return daily_returns

def run_backtest(model, 
                 scaler, 
                 X_clean: pd.DataFrame, 
                 prices_df: pd.DataFrame,
                 threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """
    Generates signals and simulates the portfolio's equity.
    """
    print(f"Running backtest with threshold={threshold}...")
    
    # 1. Scale all our feature data
    X_scaled = scaler.transform(X_clean)
    
    # 2. Get daily probabilities for all 3 classes
    # This gives [prob_0, prob_1, prob_2] for each day
    all_probs = model.predict_proba(X_scaled)
    
    # 3. Create a signal DataFrame
    signals_df = pd.DataFrame(
        all_probs, 
        index=X_clean.index, 
        columns=['prob_short', 'prob_neutral', 'prob_long']
    )
    
    # 4. Generate Position signals (-1, 0, +1)
    # Default position is 0 (Neutral)
    signals_df['position'] = 0
    
    # Go LONG (+1) if prob_long > threshold
    signals_df.loc[signals_df['prob_long'] > threshold, 'position'] = 1
    
    # Go SHORT (-1) if prob_short > threshold
    signals_df.loc[signals_df['prob_short'] > threshold, 'position'] = -1

    # --- Portfolio Accounting ---
    
    # 5. Get the daily returns of the asset we are trading
    portfolio_returns = get_portfolio_returns(prices_df, STOCK_LIST)
    
    # 6. Align signals with returns
    # This ensures we only use returns on days we have a signal
    backtest_df = signals_df.join(portfolio_returns.rename('asset_return'), how='inner')

    # 7. Calculate Strategy Returns
    # Our return is the asset_return * our position (-1, 0, or 1)
    # We must .shift(1) because we use today's signal to trade *tomorrow*
    backtest_df['strategy_return'] = backtest_df['asset_return'] * backtest_df['position'].shift(1)
    
    # The first day's return will be NaN, so fill with 0
    backtest_df['strategy_return'] = backtest_df['strategy_return'].fillna(0)
    
    # 8. Calculate the final Equity Curve
    backtest_df['equity'] = INITIAL_CAPITAL * (1 + backtest_df['strategy_return']).cumprod()
    
    print("✅ Backtest complete.")
    return backtest_df


# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    print("Running backtest module as standalone script...")
    
    # 1. Define the best parameters we found
    # (This would be passed from model.py in the real notebook)
    best_params = {'H': 43.0, 'L': 17.0}
    
    # 2. Train the final model
    model, scaler, X_clean, y_clean = train_final_model(best_params)
    
    # --- NEW CODE BLOCK: PRINT MODEL WEIGHTS ---
    print("\n--- Final Model Weights (Coefficients) ---")
    
    # Get the feature names from the X_clean DataFrame
    feature_names = X_clean.columns
    
    # Get the class names from the model (will be [0, 1, 2])
    class_names = ["Short (0)", "Neutral (1)", "Long (2)"]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # Create a DataFrame to show the weights cleanly
    weights_df = pd.DataFrame(
        model.coef_, 
        columns=feature_names, 
        index=class_names
    )
    print(weights_df)
    # --- END NEW CODE BLOCK ---
    
    # 3. Load the *full* price history for the backtest
    prices_df, _ = load_aligned_data()
    
    # 4. Run the backtest simulation
    backtest_results = run_backtest(model, scaler, X_clean, prices_df)
    
    print("\n--- Backtest Results ---")
    print(backtest_results[['prob_short', 'prob_neutral', 'prob_long', 'position', 'asset_return', 'strategy_return', 'equity']].tail(10))
    
    print("\n--- Final Equity ---")
    print(f"${backtest_results['equity'].iloc[-1]:,.2f}")
    
    print("\n--- Position Distribution ---")
    print(backtest_results['position'].value_counts(normalize=True))
    
    # Simple plot to visualize
    try:
        import matplotlib.pyplot as plt
        
        # --- Create Benchmark Equity Curve ---
        
        # 1. Get SPY prices from the loaded prices_df
        #    (We already loaded prices_df earlier)
        spy_prices = prices_df['SPY'] 
        
        # 2. Align SPY to our backtest's start/end dates
        #    (backtest_results.index is our clean, aligned index)
        spy_aligned = spy_prices.reindex(backtest_results.index)
        
        # 3. Create the benchmark equity curve (normalized to start capital)
        #    (spy_aligned / spy_aligned.iloc[0]) normalizes it to 1
        benchmark_equity = (spy_aligned / spy_aligned.iloc[0]) * INITIAL_CAPITAL
        benchmark_equity.name = "SPY Buy-and-Hold"
        
        # --- End Benchmark ---

        
        # 4. Plot both curves
        plt.figure(figsize=(12, 7))
        
        # Plot Strategy
        backtest_results['equity'].plot(label='Macro Strategy', lw=2)
        
        # Plot Benchmark
        benchmark_equity.plot(label='SPY Buy-and-Hold', lw=2, linestyle='--')
        
        plt.title(f"Strategy vs. S&P 500 (H={int(best_params['H'])}, L={int(best_params['L'])})")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.legend() # Add the legend
        plt.tight_layout()
        plt.show() # This will pop up a chart
        
    except ImportError:
        print("\n(Install matplotlib to see a plot of the equity curve)")