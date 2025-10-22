"""
src/data.py

Module for ingesting and aligning equity price data and lagged 
macroeconomic data from yfinance and FRED.
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
from pandas.tseries.offsets import MonthEnd, BDay
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---

# Stock universe
# STOCK_LIST = ["XOM", "CVX", "JPM", "BAC", "UNH", "JNJ", "CAT", "DE", "WM", "RTX"]
STOCK_LIST = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "TSLA", "BRK-B"]
BENCHMARK = "SPY"
FULL_TICKER_LIST = STOCK_LIST + [BENCHMARK]

# Macro universe (simplified)
MACRO_SERIES = ["UNRATE", "CPIAUCSL", "CFNAI"]

# Date ranges
DATA_DOWNLOAD_START = "2005-01-01"  # Start far back for feature engineering
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2019-12-31"

# Lag configuration for macro data to prevent look-ahead bias
# We assume data for month 'M' is released 10 business days 
# into month 'M+1'.
RELEASE_LAG_DAYS = 10 

# --- FRED API Client ---

try:
    # Assumes your FRED API key is set as an environment variable
    FRED_API_KEY = os.environ.get("FRED_API_KEY")
    if FRED_API_KEY is None:
        raise ValueError("FRED_API_KEY environment variable not set.")
    fred_client = Fred(api_key=FRED_API_KEY)
except Exception as e:
    print(f"Error initializing FRED client: {e}")
    print("Please make sure your FRED_API_KEY is set as an environment variable.")
    fred_client = None

# --- Main Functions ---

def get_stock_data(tickers: list, 
                   start_date: str, 
                   end_date: str, 
                   train_start: str, 
                   train_end: str) -> pd.DataFrame:
    """
    Downloads daily adjusted close prices for a list of tickers
    and aligns them to a clean business-day index.

    Args:
        tickers: List of stock tickers.
        start_date: Start date for yfinance download (e.g., '2000-01-01').
        end_date: End date for yfinance download (e.g., '2020-01-01').
        train_start: The first day of the final indexed DataFrame.
        train_end: The last day of the final indexed DataFrame.

    Returns:
        A DataFrame of adjusted close prices, indexed by business days
        and forward-filled for holidays.
    """
    print(f"Downloading stock data for {len(tickers)} tickers...")
    
    # 1. Download data
    prices = yf.download(tickers, start=start_date, end=end_date)
    
    # 2. Isolate Adjusted Close
    if not prices.empty:
        prices_df = prices['Close']
    else:
        raise ValueError("yfinance download failed. No data returned.")
        
    # Handle single ticker download (yfinance returns Series)
    if len(tickers) == 1:
        prices_df = prices_df.to_frame(name=tickers[0])

    # 3. Create Master Business-Day Index for the training period
    bday_index = pd.bdate_range(start=train_start, end=train_end)
    
    # 4. Re-index and Fill
    prices_df_aligned = prices_df.reindex(bday_index)
    
    # 5. Forward-fill NaNs from market holidays
    prices_df_filled = prices_df_aligned.ffill()
    
    # Check for any remaining NaNs (e.g., if a stock didn't exist in 2010)
    if prices_df_filled.isnull().values.any():
        print("Warning: NaNs found in price data after ffill.")
        print("This may be due to stocks not existing at the start of the period.")
        # We'll drop NaNs later after merging with macro data
    
    print("✅ Stock price data ingestion complete.")
    return prices_df_filled


def get_macro_data(series_ids: list, 
                   start_date: str, 
                   bday_index: pd.DatetimeIndex, 
                   lag_days: int) -> pd.DataFrame:
    """
    Downloads monthly macro data from FRED and aligns it to a daily
    business-day index, applying a publication lag to prevent
    look-ahead bias.

    Args:
        series_ids: List of FRED series IDs.
        start_date: Start date for FRED download (e.g., '2000-01-01').
        bday_index: The target daily business-day index to align to.
        lag_days: The number of business days to shift data forward
                  to simulate release lag.

    Returns:
        A DataFrame of macro data, lagged and forward-filled to
        the daily business-day index.
    """
    if fred_client is None:
        raise ConnectionError("FRED client is not initialized.")
        
    print(f"Downloading {len(series_ids)} macro series from FRED...")
    
    # 1. Create an empty shell DataFrame with the target daily index
    macro_df_daily = pd.DataFrame(index=bday_index)
    
    # 2. Loop and Lag Each Series
    for series_id in series_ids:
        # a. Fetch Data (monthly, timestamped to first of month)
        try:
            series_monthly = fred_client.get_series(series_id, start_date)
            series_monthly.name = series_id
        except Exception as e:
            print(f"Warning: Could not fetch {series_id}. Error: {e}. Skipping.")
            continue
        
        # b. Apply the Lag (The "Anti-Look-Ahead" Logic)
        # Convert index to datetime (just in case)
        series_monthly.index = pd.to_datetime(series_monthly.index)
        
        # 1. Shift timestamp from '2010-09-01' to '2010-09-30'
        lagged_index = series_monthly.index + MonthEnd(0)
        
        # 2. Add the business day lag to get the release date
        # '2010-09-30' -> '~2010-10-14'
        lagged_index = lagged_index + BDay(lag_days)
        
        # c. Update the series' index with these new lagged dates
        series_monthly.index = lagged_index
        
        # d. Map to Daily Frame
        # Join with the daily frame, which will have NaNs on non-release days
        macro_df_daily = macro_df_daily.join(series_monthly)
        
    # 3. Forward-Fill the macro_df
    # Propagate the latest known value forward
    macro_df_filled = macro_df_daily.ffill()
    
    # 4. Handle NaNs at the Start
    # This removes the initial period before all data series have
    # at least one value (due to lags from 2000s data).
    macro_df_final = macro_df_filled.dropna()
    
    print("✅ Macro data ingestion and lagging complete.")
    return macro_df_final


def load_aligned_data():
    """
    Main wrapper function to load and align both stock and macro data.
    
    Returns:
        prices_df (pd.DataFrame): Daily stock prices.
        macro_df (pd.DataFrame): Daily, lagged macro data.
    """
    
    # 1. Get stock data and the master index
    prices_df = get_stock_data(
        tickers=FULL_TICKER_LIST,
        start_date=DATA_DOWNLOAD_START,
        end_date=TRAIN_END_DATE,  # Only need up to training end for yf
        train_start=TRAIN_START_DATE,
        train_end=TRAIN_END_DATE
    )
    
    # 2. Get macro data using the stock data's index
    macro_df = get_macro_data(
        series_ids=MACRO_SERIES,
        start_date=DATA_DOWNLOAD_START,
        bday_index=prices_df.index,
        lag_days=RELEASE_LAG_DAYS
    )
    
    # 3. Final Alignment
    # Ensure both dataframes have the exact same index
    # (e.g., macro_df.dropna() might have removed early 2010 rows)
    common_index = prices_df.index.intersection(macro_df.index)
    
    prices_df = prices_df.loc[common_index]
    macro_df = macro_df.loc[common_index]
    
    print(f"--- Data loading complete ---")
    print(f"Final aligned shape (prices): {prices_df.shape}")
    print(f"Final aligned shape (macro):  {macro_df.shape}")
    
    return prices_df, macro_df


# --- Example Usage (if run as a script) ---

if __name__ == "__main__":
    
    print("Running data ingestion module as standalone script...")
    
    try:
        prices_df, macro_df = load_aligned_data()
        
        print("\n--- Stock Prices (prices_df) ---")
        print(prices_df.head(3))
        print("...")
        print(prices_df.tail(3))
        
        print("\n--- Macro Data (macro_df) ---")
        print(macro_df.head(3))
        print("...")
        print(macro_df.tail(3))
        
        # Check the lag
        print("\n--- Lag Verification ---")
        unrate_series = macro_df['UNRATE']
        # Find the day the 'UNRATE' value changed
        unrate_changes = unrate_series.drop_duplicates()
        print("Recent changes in 'UNRATE' (showing release dates):")
        print(unrate_changes.tail(5))
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your FRED_API_KEY is correct and you have internet access.")