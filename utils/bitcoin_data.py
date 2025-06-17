"""
Bitcoin data fetching utilities using yfinance
"""
import pandas as pd
import yfinance as yf
from datetime import date
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_bitcoin_data():
    """
    Fetch Bitcoin OHLCV data from Yahoo Finance
    Returns processed DataFrame with Date index
    """
    try:
        TICKER = "BTC-USD"
        START_DATE = "2015-01-01"
        END_DATE = date.today().strftime("%Y-%m-%d")
        
        print(f"Downloading {TICKER} daily OHLCV from {START_DATE} to {END_DATE}...")
        
        # Download data using yfinance
        df = yf.download(
            TICKER,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True
        )
        
        # Check if download was successful
        if df is None or df.empty:
            raise RuntimeError("Download returned an empty DataFrame. "
                             "Check your internet connection or ticker symbol.")
        
        # Handle MultiIndex columns (yfinance returns hierarchical columns)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the columns - take the first level (price type)
            df.columns = [col[0] for col in df.columns]
        
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Ensure we have the expected columns
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f"Missing columns: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Convert Date column to datetime and ensure timezone-naive
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Remove timezone information if present for consistency
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Set Date as index
        df = df.set_index('Date')
        
        # Ensure numeric columns are properly typed (only process existing columns)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove rows with missing Close prices
        df = df.dropna(subset=["Close"])
        
        # Basic data validation
        if len(df) < 100:
            raise RuntimeError(f"Insufficient data: only {len(df)} rows available")
        
        print(f"Successfully downloaded {len(df)} rows of Bitcoin data")
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching Bitcoin data: {str(e)}")

def get_bitcoin_statistics(df):
    """
    Calculate basic Bitcoin statistics
    """
    prices = df['Close'].values
    returns = df['Close'].pct_change().dropna()
    
    stats = {
        'current_price': prices[-1],
        'daily_return_mean': returns.mean(),
        'daily_return_std': returns.std(),
        'annual_volatility': returns.std() * (365 ** 0.5),
        'max_drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
        'total_return': (prices[-1] / prices[0] - 1),
        'data_points': len(df)
    }
    
    return stats
