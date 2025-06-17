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
    TICKER = "BTC-USD"
    START_DATE = "2015-01-01"
    END_DATE = date.today().strftime("%Y-%m-%d")
    
    # Download data
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
    )
    
    if df.empty:
        raise RuntimeError("Download returned an empty DataFrame. "
                          "Check your internet connection or ticker symbol.")
    
    # Clean and process data
    df = df.rename_axis("Date").reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove rows with missing Close prices
    df = df.dropna(subset=["Close"])
    
    # Basic data validation
    if len(df) < 100:
        raise RuntimeError(f"Insufficient data: only {len(df)} rows available")
    
    return df

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
