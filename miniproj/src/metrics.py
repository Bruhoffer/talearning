"""
src/metrics.py

Module for calculating all strategy performance metrics and
generating key visualizations for the final report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252

# --- Metric Calculation Functions ---

def calculate_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates a dictionary of all key performance metrics.
    
    Args:
        daily_returns: A pd.Series of daily strategy returns.
        risk_free_rate: The annualized risk-free rate (default 0.0).

    Returns:
        A dictionary containing all calculated metrics.
    """
    
    # 1. Annualized Return
    total_return = (1 + daily_returns).prod()
    num_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    annualized_return = (total_return ** (1 / num_years)) - 1
    
    # 2. Annualized Volatility
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # 3. Sharpe Ratio
    returns_over_rf = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (returns_over_rf.mean() * TRADING_DAYS_PER_YEAR) / annualized_volatility
    
    # 4. Max Drawdown
    equity_curve = (1 + daily_returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 5. Calmar Ratio
    # Handle case where max_drawdown is 0 (e.g., perfect run)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    
    # 6. Sortino Ratio
    negative_returns = returns_over_rf[returns_over_rf < 0]
    downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = (returns_over_rf.mean() * TRADING_DAYS_PER_YEAR) / downside_std
    
    return {
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_volatility:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": f"{calmar_ratio:.2f}",
        "Total Return": f"{(total_return - 1):.2%}"
    }

def generate_metrics_table(strategy_returns: pd.Series, 
                           benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Generates a clean DataFrame comparing strategy and benchmark metrics.
    """
    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    metrics_df = pd.DataFrame({
        "Strategy": strategy_metrics,
        "Benchmark (SPY)": benchmark_metrics
    })
    
    return metrics_df

# --- Plotting Functions ---

def plot_drawdown(daily_returns: pd.Series, ax=None):
    """
    Plots the drawdown curve for a given series of daily returns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    equity_curve = (1 + daily_returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    ax.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.5)
    ax.set_title("Strategy Drawdown", fontsize=14)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return ax

def plot_positions(backtest_df: pd.DataFrame, spy_prices: pd.Series, ax=None):
    """
    Plots the strategy positions (-1, 0, 1) overlaid on the SPY price.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Get the position data and align it
    positions = backtest_df['position'].shift(1).fillna(0) # Shift to show position held
    spy_aligned = spy_prices.reindex(positions.index)
    
    # 1. Plot SPY Price on primary Y-axis
    ax.plot(spy_aligned.index, spy_aligned, label='SPY Price', color='black', alpha=0.6)
    ax.set_ylabel("SPY Price ($)")
    ax.set_xlabel("Date")
    
    # 2. Create secondary Y-axis for positions
    ax2 = ax.twinx()
    
    # Plot positions as a "step" plot
    ax2.step(positions.index, positions, where='post', label='Strategy Position', color='blue', alpha=0.7)
    
    # Color the background based on position
    ax2.fill_between(positions.index, 0, 1, where=positions > 0, 
                     color='green', alpha=0.1, label='Long')
    ax2.fill_between(positions.index, -1, 0, where=positions < 0, 
                     color='red', alpha=0.1, label='Short')
    
    ax2.set_ylabel("Position (1=Long, -1=Short)")
    ax2.set_yticks([-1, 0, 1])
    
    # 3. Format plot
    ax.set_title("Strategy Positions vs. SPY Price", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    return ax

# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    print("Running metrics module as standalone script...")
    
    # Import functions from other files
    from data import load_aligned_data
    from backtest import train_final_model, run_backtest, get_portfolio_returns

    # 1. Define Best Params
    best_params = {'H': 43.0, 'L': 17.0}
    
    # 2. Get Data
    prices_df, _ = load_aligned_data()
    model, scaler, X_clean, y_clean = train_final_model(best_params)
    
    # 3. Run Backtest
    backtest_results = run_backtest(model, scaler, X_clean, prices_df)
    
    # 4. Get Strategy and Benchmark Returns
    strategy_returns = backtest_results['strategy_return']
    
    # Get benchmark returns
    spy_returns = get_portfolio_returns(prices_df, ['SPY'])
    # Align benchmark returns to our backtest's index
    benchmark_returns_aligned = spy_returns.reindex(strategy_returns.index).fillna(0)
    
    # 5. Generate Metrics Table
    print("\n--- Performance Metrics Table ---")
    metrics_table = generate_metrics_table(strategy_returns, benchmark_returns_aligned)
    print(metrics_table)
    
    # 6. Generate Plots
    print("\nGenerating plots...")
    
    # Plot 1: Drawdown
    plot_drawdown(strategy_returns)
    plt.show()
    
    # Plot 2: Positions
    plot_positions(backtest_results, prices_df['SPY'])
    plt.show()