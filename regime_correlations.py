"""
Regime-specific correlation matrices for different market conditions.
These correlations reflect how assets typically behave together during various market scenarios.
"""

# Daily frequency variables from the screenshot (as set for efficient lookup)
DAILY_VARIABLES = {
    "Bitcoin Daily Close Price",
    "Gold Daily Close Price", 
    "WTI Crude Oil Price (USD/Barrel)",
    "SPY Daily Close Price",
    "QQQ Daily Close Price",
    "10-Year Treasury Yield (%)",
    "10-Year TIPS Yield (%)",
    "5-Year TIPS Yield (%)", 
    "20-Year TIPS Yield (%)",
    "30-Year TIPS Yield (%)",
    "DefiLlama DEX Historical Volume",
    "Fear & Greed Index",
    "CoinGecko ETH Daily Volume",
    "US Equity Market Capitalization (Billions USD)",
    "US Federal Funds Rate",
    "Baltic Dry Index"
}

# Mapping from full names to short codes for correlation matrices
VARIABLE_NAME_MAPPING = {
    "Bitcoin Daily Close Price": "BTC",
    "Gold Daily Close Price": "GOLD",
    "WTI Crude Oil Price (USD/Barrel)": "WTI", 
    "SPY Daily Close Price": "SPY",
    "QQQ Daily Close Price": "QQQ",
    "10-Year Treasury Yield (%)": "TREASURY_10Y",
    "10-Year TIPS Yield (%)": "TIPS_10Y",
    "5-Year TIPS Yield (%)": "TIPS_5Y",
    "20-Year TIPS Yield (%)": "TIPS_20Y", 
    "30-Year TIPS Yield (%)": "TIPS_30Y",
    "CoinGecko ETH Daily Volume": "ETH_VOLUME",
    "DefiLlama DEX Historical Volume": "DEX_VOLUME",
    "Fear & Greed Index": "FEAR_GREED",
    "US Equity Market Capitalization (Billions USD)": "EQUITY_CAP",
    "US Federal Funds Rate": "FED_RATE",
    "Baltic Dry Index": "BALTIC",
    # Common alternative names
    "BTC": "BTC",
    "GOLD": "GOLD",
    "WTI": "WTI",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "TIPS_10Y": "TIPS_10Y",
    "TIPS_5Y": "TIPS_5Y",
    "TIPS_20Y": "TIPS_20Y",
    "TIPS_30Y": "TIPS_30Y",
    "TREASURY_10Y": "TREASURY_10Y",
    "VIX": "VIX",
    "DXY": "DXY"
}

def is_daily_only_strategy(strategy):
    """
    Check if a strategy only uses daily frequency variables
    
    Args:
        strategy: Strategy dictionary with excel_names field
        
    Returns:
        bool: True if all excel_names are in DAILY_VARIABLES
    """
    excel_names = strategy.get("excel_names", [])
    if not excel_names:
        return True  # No variables needed, consider it valid
    
    return all(var in DAILY_VARIABLES for var in excel_names)

REGIME_CORRELATIONS = {
    "HIGH_VOL_DOWN": {
        # Market crash scenario - flight to quality, risk-off behavior
        "BTC":  {"BTC": 1.0, "GOLD": -0.3, "WTI": 0.4, "SPY": 0.7, "QQQ": 0.8, "TREASURY_10Y": -0.4, "TIPS_10Y": -0.3, "TIPS_5Y": -0.2, "TIPS_20Y": -0.4, "TIPS_30Y": -0.3, "ETH_VOLUME": 0.6, "DEX_VOLUME": 0.5, "FEAR_GREED": -0.8},
        "GOLD": {"BTC": -0.3, "GOLD": 1.0, "WTI": -0.2, "SPY": -0.5, "QQQ": -0.6, "TREASURY_10Y": 0.3, "TIPS_10Y": 0.4, "TIPS_5Y": 0.3, "TIPS_20Y": 0.4, "TIPS_30Y": 0.4, "ETH_VOLUME": -0.4, "DEX_VOLUME": -0.3, "FEAR_GREED": 0.4},
        "WTI":  {"BTC": 0.4, "GOLD": -0.2, "WTI": 1.0, "SPY": 0.3, "QQQ": 0.2, "TREASURY_10Y": -0.3, "TIPS_10Y": -0.2, "TIPS_5Y": -0.1, "TIPS_20Y": -0.3, "TIPS_30Y": -0.2, "ETH_VOLUME": 0.3, "DEX_VOLUME": 0.2, "FEAR_GREED": -0.3},
        "SPY":  {"BTC": 0.7, "GOLD": -0.5, "WTI": 0.3, "SPY": 1.0, "QQQ": 0.9, "TREASURY_10Y": -0.6, "TIPS_10Y": -0.5, "TIPS_5Y": -0.4, "TIPS_20Y": -0.6, "TIPS_30Y": -0.5, "ETH_VOLUME": 0.4, "DEX_VOLUME": 0.3, "FEAR_GREED": -0.7},
        "QQQ":  {"BTC": 0.8, "GOLD": -0.6, "WTI": 0.2, "SPY": 0.9, "QQQ": 1.0, "TREASURY_10Y": -0.7, "TIPS_10Y": -0.6, "TIPS_5Y": -0.5, "TIPS_20Y": -0.7, "TIPS_30Y": -0.6, "ETH_VOLUME": 0.5, "DEX_VOLUME": 0.4, "FEAR_GREED": -0.8},
        "TREASURY_10Y": {"BTC": -0.4, "GOLD": 0.3, "WTI": -0.3, "SPY": -0.6, "QQQ": -0.7, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.8, "TIPS_5Y": 0.7, "TIPS_20Y": 0.9, "TIPS_30Y": 0.8, "ETH_VOLUME": -0.5, "DEX_VOLUME": -0.4, "FEAR_GREED": 0.5},
        "TIPS_10Y": {"BTC": -0.3, "GOLD": 0.4, "WTI": -0.2, "SPY": -0.5, "QQQ": -0.6, "TREASURY_10Y": 0.8, "TIPS_10Y": 1.0, "TIPS_5Y": 0.9, "TIPS_20Y": 0.9, "TIPS_30Y": 0.9, "ETH_VOLUME": -0.4, "DEX_VOLUME": -0.3, "FEAR_GREED": 0.4},
        "TIPS_5Y": {"BTC": -0.2, "GOLD": 0.3, "WTI": -0.1, "SPY": -0.4, "QQQ": -0.5, "TREASURY_10Y": 0.7, "TIPS_10Y": 0.9, "TIPS_5Y": 1.0, "TIPS_20Y": 0.8, "TIPS_30Y": 0.7, "ETH_VOLUME": -0.3, "DEX_VOLUME": -0.2, "FEAR_GREED": 0.3},
        "TIPS_20Y": {"BTC": -0.4, "GOLD": 0.4, "WTI": -0.3, "SPY": -0.6, "QQQ": -0.7, "TREASURY_10Y": 0.9, "TIPS_10Y": 0.9, "TIPS_5Y": 0.8, "TIPS_20Y": 1.0, "TIPS_30Y": 0.9, "ETH_VOLUME": -0.5, "DEX_VOLUME": -0.4, "FEAR_GREED": 0.5},
        "TIPS_30Y": {"BTC": -0.3, "GOLD": 0.4, "WTI": -0.2, "SPY": -0.5, "QQQ": -0.6, "TREASURY_10Y": 0.8, "TIPS_10Y": 0.9, "TIPS_5Y": 0.7, "TIPS_20Y": 0.9, "TIPS_30Y": 1.0, "ETH_VOLUME": -0.4, "DEX_VOLUME": -0.3, "FEAR_GREED": 0.4},
        "ETH_VOLUME": {"BTC": 0.6, "GOLD": -0.4, "WTI": 0.3, "SPY": 0.4, "QQQ": 0.5, "TREASURY_10Y": -0.5, "TIPS_10Y": -0.4, "TIPS_5Y": -0.3, "TIPS_20Y": -0.5, "TIPS_30Y": -0.4, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.7, "FEAR_GREED": -0.6},
        "DEX_VOLUME": {"BTC": 0.5, "GOLD": -0.3, "WTI": 0.2, "SPY": 0.3, "QQQ": 0.4, "TREASURY_10Y": -0.4, "TIPS_10Y": -0.3, "TIPS_5Y": -0.2, "TIPS_20Y": -0.4, "TIPS_30Y": -0.3, "ETH_VOLUME": 0.7, "DEX_VOLUME": 1.0, "FEAR_GREED": -0.5},
        "FEAR_GREED": {"BTC": -0.8, "GOLD": 0.4, "WTI": -0.3, "SPY": -0.7, "QQQ": -0.8, "TREASURY_10Y": 0.5, "TIPS_10Y": 0.4, "TIPS_5Y": 0.3, "TIPS_20Y": 0.5, "TIPS_30Y": 0.4, "ETH_VOLUME": -0.6, "DEX_VOLUME": -0.5, "FEAR_GREED": 1.0}
    },
    
    "HIGH_VOL_UP": {
        # Risk-on rally scenario - everything rallies together
        "BTC":  {"BTC": 1.0, "GOLD": -0.1, "WTI": 0.55, "SPY": 0.75, "QQQ": 0.8, "TREASURY_10Y": -0.4, "TIPS_10Y": -0.35, "TIPS_5Y": -0.3, "TIPS_20Y": -0.4, "TIPS_30Y": -0.35, "ETH_VOLUME": 0.85, "DEX_VOLUME": 0.8, "FEAR_GREED": 0.7},
        "GOLD": {"BTC": -0.1, "GOLD": 1.0, "WTI": 0.4, "SPY": -0.2, "QQQ": -0.25, "TREASURY_10Y": 0.3, "TIPS_10Y": 0.35, "TIPS_5Y": 0.3, "TIPS_20Y": 0.35, "TIPS_30Y": 0.35, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": -0.2},
        "WTI":  {"BTC": 0.55, "GOLD": 0.4, "WTI": 1.0, "SPY": 0.6, "QQQ": 0.55, "TREASURY_10Y": -0.3, "TIPS_10Y": -0.25, "TIPS_5Y": -0.2, "TIPS_20Y": -0.3, "TIPS_30Y": -0.25, "ETH_VOLUME": 0.5, "DEX_VOLUME": 0.45, "FEAR_GREED": 0.5},
        "SPY":  {"BTC": 0.75, "GOLD": -0.2, "WTI": 0.6, "SPY": 1.0, "QQQ": 0.95, "TREASURY_10Y": -0.3, "TIPS_10Y": -0.25, "TIPS_5Y": -0.2, "TIPS_20Y": -0.3, "TIPS_30Y": -0.25, "ETH_VOLUME": 0.7, "DEX_VOLUME": 0.65, "FEAR_GREED": 0.65},
        "QQQ":  {"BTC": 0.8, "GOLD": -0.25, "WTI": 0.55, "SPY": 0.95, "QQQ": 1.0, "TREASURY_10Y": -0.35, "TIPS_10Y": -0.3, "TIPS_5Y": -0.25, "TIPS_20Y": -0.35, "TIPS_30Y": -0.3, "ETH_VOLUME": 0.75, "DEX_VOLUME": 0.7, "FEAR_GREED": 0.7},
        "TREASURY_10Y": {"BTC": -0.4, "GOLD": 0.3, "WTI": -0.3, "SPY": -0.3, "QQQ": -0.35, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.85, "TIPS_5Y": 0.75, "TIPS_20Y": 0.9, "TIPS_30Y": 0.85, "ETH_VOLUME": -0.35, "DEX_VOLUME": -0.3, "FEAR_GREED": -0.3},
        "TIPS_10Y": {"BTC": -0.35, "GOLD": 0.35, "WTI": -0.25, "SPY": -0.25, "QQQ": -0.3, "TREASURY_10Y": 0.85, "TIPS_10Y": 1.0, "TIPS_5Y": 0.9, "TIPS_20Y": 0.95, "TIPS_30Y": 0.9, "ETH_VOLUME": -0.3, "DEX_VOLUME": -0.25, "FEAR_GREED": -0.25},
        "TIPS_5Y": {"BTC": -0.3, "GOLD": 0.3, "WTI": -0.2, "SPY": -0.2, "QQQ": -0.25, "TREASURY_10Y": 0.75, "TIPS_10Y": 0.9, "TIPS_5Y": 1.0, "TIPS_20Y": 0.85, "TIPS_30Y": 0.8, "ETH_VOLUME": -0.25, "DEX_VOLUME": -0.2, "FEAR_GREED": -0.2},
        "TIPS_20Y": {"BTC": -0.4, "GOLD": 0.35, "WTI": -0.3, "SPY": -0.3, "QQQ": -0.35, "TREASURY_10Y": 0.9, "TIPS_10Y": 0.95, "TIPS_5Y": 0.85, "TIPS_20Y": 1.0, "TIPS_30Y": 0.95, "ETH_VOLUME": -0.35, "DEX_VOLUME": -0.3, "FEAR_GREED": -0.3},
        "TIPS_30Y": {"BTC": -0.35, "GOLD": 0.35, "WTI": -0.25, "SPY": -0.25, "QQQ": -0.3, "TREASURY_10Y": 0.85, "TIPS_10Y": 0.9, "TIPS_5Y": 0.8, "TIPS_20Y": 0.95, "TIPS_30Y": 1.0, "ETH_VOLUME": -0.3, "DEX_VOLUME": -0.25, "FEAR_GREED": -0.25},
        "ETH_VOLUME": {"BTC": 0.85, "GOLD": -0.15, "WTI": 0.5, "SPY": 0.7, "QQQ": 0.75, "TREASURY_10Y": -0.35, "TIPS_10Y": -0.3, "TIPS_5Y": -0.25, "TIPS_20Y": -0.35, "TIPS_30Y": -0.3, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.9, "FEAR_GREED": 0.75},
        "DEX_VOLUME": {"BTC": 0.8, "GOLD": -0.1, "WTI": 0.45, "SPY": 0.65, "QQQ": 0.7, "TREASURY_10Y": -0.3, "TIPS_10Y": -0.25, "TIPS_5Y": -0.2, "TIPS_20Y": -0.3, "TIPS_30Y": -0.25, "ETH_VOLUME": 0.9, "DEX_VOLUME": 1.0, "FEAR_GREED": 0.7},
        "FEAR_GREED": {"BTC": 0.7, "GOLD": -0.2, "WTI": 0.5, "SPY": 0.65, "QQQ": 0.7, "TREASURY_10Y": -0.3, "TIPS_10Y": -0.25, "TIPS_5Y": -0.2, "TIPS_20Y": -0.3, "TIPS_30Y": -0.25, "ETH_VOLUME": 0.75, "DEX_VOLUME": 0.7, "FEAR_GREED": 1.0}
    },
    
    "HIGH_VOL_STABLE": {
        # Choppy/uncertain market - lower correlations overall
        "BTC":  {"BTC": 1.0, "GOLD": 0.1, "WTI": 0.25, "SPY": 0.4, "QQQ": 0.45, "TREASURY_10Y": 0.0, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.7, "DEX_VOLUME": 0.65, "FEAR_GREED": 0.0},
        "GOLD": {"BTC": 0.1, "GOLD": 1.0, "WTI": 0.3, "SPY": 0.0, "QQQ": -0.05, "TREASURY_10Y": 0.2, "TIPS_10Y": 0.25, "TIPS_5Y": 0.2, "TIPS_20Y": 0.25, "TIPS_30Y": 0.25, "ETH_VOLUME": 0.05, "DEX_VOLUME": 0.1, "FEAR_GREED": 0.1},
        "WTI":  {"BTC": 0.25, "GOLD": 0.3, "WTI": 1.0, "SPY": 0.4, "QQQ": 0.35, "TREASURY_10Y": -0.1, "TIPS_10Y": -0.05, "TIPS_5Y": 0.0, "TIPS_20Y": -0.1, "TIPS_30Y": -0.05, "ETH_VOLUME": 0.2, "DEX_VOLUME": 0.15, "FEAR_GREED": 0.15},
        "SPY":  {"BTC": 0.4, "GOLD": 0.0, "WTI": 0.4, "SPY": 1.0, "QQQ": 0.85, "TREASURY_10Y": -0.1, "TIPS_10Y": -0.05, "TIPS_5Y": 0.0, "TIPS_20Y": -0.1, "TIPS_30Y": -0.05, "ETH_VOLUME": 0.35, "DEX_VOLUME": 0.3, "FEAR_GREED": 0.2},
        "QQQ":  {"BTC": 0.45, "GOLD": -0.05, "WTI": 0.35, "SPY": 0.85, "QQQ": 1.0, "TREASURY_10Y": -0.15, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.15, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.4, "DEX_VOLUME": 0.35, "FEAR_GREED": 0.25},
        "TREASURY_10Y": {"BTC": 0.0, "GOLD": 0.2, "WTI": -0.1, "SPY": -0.1, "QQQ": -0.15, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.8, "TIPS_5Y": 0.7, "TIPS_20Y": 0.85, "TIPS_30Y": 0.8, "ETH_VOLUME": -0.05, "DEX_VOLUME": 0.0, "FEAR_GREED": 0.05},
        "TIPS_10Y": {"BTC": -0.1, "GOLD": 0.25, "WTI": -0.05, "SPY": -0.05, "QQQ": -0.1, "TREASURY_10Y": 0.8, "TIPS_10Y": 1.0, "TIPS_5Y": 0.85, "TIPS_20Y": 0.9, "TIPS_30Y": 0.85, "ETH_VOLUME": -0.1, "DEX_VOLUME": -0.05, "FEAR_GREED": 0.0},
        "TIPS_5Y": {"BTC": -0.05, "GOLD": 0.2, "WTI": 0.0, "SPY": 0.0, "QQQ": -0.05, "TREASURY_10Y": 0.7, "TIPS_10Y": 0.85, "TIPS_5Y": 1.0, "TIPS_20Y": 0.8, "TIPS_30Y": 0.75, "ETH_VOLUME": -0.05, "DEX_VOLUME": 0.0, "FEAR_GREED": 0.05},
        "TIPS_20Y": {"BTC": -0.1, "GOLD": 0.25, "WTI": -0.1, "SPY": -0.1, "QQQ": -0.15, "TREASURY_10Y": 0.85, "TIPS_10Y": 0.9, "TIPS_5Y": 0.8, "TIPS_20Y": 1.0, "TIPS_30Y": 0.9, "ETH_VOLUME": -0.1, "DEX_VOLUME": -0.05, "FEAR_GREED": 0.0},
        "TIPS_30Y": {"BTC": -0.1, "GOLD": 0.25, "WTI": -0.05, "SPY": -0.05, "QQQ": -0.1, "TREASURY_10Y": 0.8, "TIPS_10Y": 0.85, "TIPS_5Y": 0.75, "TIPS_20Y": 0.9, "TIPS_30Y": 1.0, "ETH_VOLUME": -0.1, "DEX_VOLUME": -0.05, "FEAR_GREED": 0.0},
        "ETH_VOLUME": {"BTC": 0.7, "GOLD": 0.05, "WTI": 0.2, "SPY": 0.35, "QQQ": 0.4, "TREASURY_10Y": -0.05, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.8, "FEAR_GREED": 0.1},
        "DEX_VOLUME": {"BTC": 0.65, "GOLD": 0.1, "WTI": 0.15, "SPY": 0.3, "QQQ": 0.35, "TREASURY_10Y": 0.0, "TIPS_10Y": -0.05, "TIPS_5Y": 0.0, "TIPS_20Y": -0.05, "TIPS_30Y": -0.05, "ETH_VOLUME": 0.8, "DEX_VOLUME": 1.0, "FEAR_GREED": 0.05},
        "FEAR_GREED": {"BTC": 0.0, "GOLD": 0.1, "WTI": 0.15, "SPY": 0.2, "QQQ": 0.25, "TREASURY_10Y": 0.05, "TIPS_10Y": 0.0, "TIPS_5Y": 0.05, "TIPS_20Y": 0.0, "TIPS_30Y": 0.0, "ETH_VOLUME": 0.1, "DEX_VOLUME": 0.05, "FEAR_GREED": 1.0}
    },
    
    "STABLE_VOL_UP": {
        # Healthy bull market - moderate positive correlations
        "BTC":  {"BTC": 1.0, "GOLD": -0.3, "WTI": 0.45, "SPY": 0.6, "QQQ": 0.65, "TREASURY_10Y": -0.2, "TIPS_10Y": -0.25, "TIPS_5Y": -0.2, "TIPS_20Y": -0.25, "TIPS_30Y": -0.25, "ETH_VOLUME": 0.8, "DEX_VOLUME": 0.75, "FEAR_GREED": 0.6},
        "GOLD": {"BTC": -0.3, "GOLD": 1.0, "WTI": 0.35, "SPY": -0.4, "QQQ": -0.45, "TREASURY_10Y": 0.2, "TIPS_10Y": 0.25, "TIPS_5Y": 0.2, "TIPS_20Y": 0.25, "TIPS_30Y": 0.25, "ETH_VOLUME": -0.25, "DEX_VOLUME": -0.2, "FEAR_GREED": -0.35},
        "WTI":  {"BTC": 0.45, "GOLD": 0.35, "WTI": 1.0, "SPY": 0.5, "QQQ": 0.45, "TREASURY_10Y": -0.15, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.15, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.4, "DEX_VOLUME": 0.35, "FEAR_GREED": 0.4},
        "SPY":  {"BTC": 0.6, "GOLD": -0.4, "WTI": 0.5, "SPY": 1.0, "QQQ": 0.9, "TREASURY_10Y": 0.1, "TIPS_10Y": 0.05, "TIPS_5Y": 0.1, "TIPS_20Y": 0.05, "TIPS_30Y": 0.05, "ETH_VOLUME": 0.55, "DEX_VOLUME": 0.5, "FEAR_GREED": 0.55},
        "QQQ":  {"BTC": 0.65, "GOLD": -0.45, "WTI": 0.45, "SPY": 0.9, "QQQ": 1.0, "TREASURY_10Y": 0.05, "TIPS_10Y": 0.0, "TIPS_5Y": 0.05, "TIPS_20Y": 0.0, "TIPS_30Y": 0.0, "ETH_VOLUME": 0.6, "DEX_VOLUME": 0.55, "FEAR_GREED": 0.6},
        "TREASURY_10Y": {"BTC": -0.2, "GOLD": 0.2, "WTI": -0.15, "SPY": 0.1, "QQQ": 0.05, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.85, "TIPS_5Y": 0.75, "TIPS_20Y": 0.9, "TIPS_30Y": 0.85, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": -0.05},
        "TIPS_10Y": {"BTC": -0.25, "GOLD": 0.25, "WTI": -0.1, "SPY": 0.05, "QQQ": 0.0, "TREASURY_10Y": 0.85, "TIPS_10Y": 1.0, "TIPS_5Y": 0.9, "TIPS_20Y": 0.95, "TIPS_30Y": 0.9, "ETH_VOLUME": -0.2, "DEX_VOLUME": -0.15, "FEAR_GREED": -0.1},
        "TIPS_5Y": {"BTC": -0.2, "GOLD": 0.2, "WTI": -0.05, "SPY": 0.1, "QQQ": 0.05, "TREASURY_10Y": 0.75, "TIPS_10Y": 0.9, "TIPS_5Y": 1.0, "TIPS_20Y": 0.85, "TIPS_30Y": 0.8, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": -0.05},
        "TIPS_20Y": {"BTC": -0.25, "GOLD": 0.25, "WTI": -0.15, "SPY": 0.05, "QQQ": 0.0, "TREASURY_10Y": 0.9, "TIPS_10Y": 0.95, "TIPS_5Y": 0.85, "TIPS_20Y": 1.0, "TIPS_30Y": 0.95, "ETH_VOLUME": -0.2, "DEX_VOLUME": -0.15, "FEAR_GREED": -0.1},
        "TIPS_30Y": {"BTC": -0.25, "GOLD": 0.25, "WTI": -0.1, "SPY": 0.05, "QQQ": 0.0, "TREASURY_10Y": 0.85, "TIPS_10Y": 0.9, "TIPS_5Y": 0.8, "TIPS_20Y": 0.95, "TIPS_30Y": 1.0, "ETH_VOLUME": -0.2, "DEX_VOLUME": -0.15, "FEAR_GREED": -0.1},
        "ETH_VOLUME": {"BTC": 0.8, "GOLD": -0.25, "WTI": 0.4, "SPY": 0.55, "QQQ": 0.6, "TREASURY_10Y": -0.15, "TIPS_10Y": -0.2, "TIPS_5Y": -0.15, "TIPS_20Y": -0.2, "TIPS_30Y": -0.2, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.85, "FEAR_GREED": 0.65},
        "DEX_VOLUME": {"BTC": 0.75, "GOLD": -0.2, "WTI": 0.35, "SPY": 0.5, "QQQ": 0.55, "TREASURY_10Y": -0.1, "TIPS_10Y": -0.15, "TIPS_5Y": -0.1, "TIPS_20Y": -0.15, "TIPS_30Y": -0.15, "ETH_VOLUME": 0.85, "DEX_VOLUME": 1.0, "FEAR_GREED": 0.6},
        "FEAR_GREED": {"BTC": 0.6, "GOLD": -0.35, "WTI": 0.4, "SPY": 0.55, "QQQ": 0.6, "TREASURY_10Y": -0.05, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.65, "DEX_VOLUME": 0.6, "FEAR_GREED": 1.0}
    },
    
    "STABLE_VOL_DOWN": {
        # Slow bear market - moderate correlations, some flight to safety
        "BTC":  {"BTC": 1.0, "GOLD": 0.15, "WTI": 0.6, "SPY": 0.7, "QQQ": 0.72, "TREASURY_10Y": 0.3, "TIPS_10Y": 0.25, "TIPS_5Y": 0.2, "TIPS_20Y": 0.3, "TIPS_30Y": 0.25, "ETH_VOLUME": 0.65, "DEX_VOLUME": 0.6, "FEAR_GREED": -0.7},
        "GOLD": {"BTC": 0.15, "GOLD": 1.0, "WTI": -0.3, "SPY": -0.1, "QQQ": -0.15, "TREASURY_10Y": 0.5, "TIPS_10Y": 0.55, "TIPS_5Y": 0.5, "TIPS_20Y": 0.55, "TIPS_30Y": 0.55, "ETH_VOLUME": 0.1, "DEX_VOLUME": 0.15, "FEAR_GREED": 0.3},
        "WTI":  {"BTC": 0.6, "GOLD": -0.3, "WTI": 1.0, "SPY": 0.55, "QQQ": 0.5, "TREASURY_10Y": 0.1, "TIPS_10Y": 0.05, "TIPS_5Y": 0.0, "TIPS_20Y": 0.1, "TIPS_30Y": 0.05, "ETH_VOLUME": 0.5, "DEX_VOLUME": 0.45, "FEAR_GREED": -0.5},
        "SPY":  {"BTC": 0.7, "GOLD": -0.1, "WTI": 0.55, "SPY": 1.0, "QQQ": 0.92, "TREASURY_10Y": -0.4, "TIPS_10Y": -0.45, "TIPS_5Y": -0.4, "TIPS_20Y": -0.45, "TIPS_30Y": -0.45, "ETH_VOLUME": 0.6, "DEX_VOLUME": 0.55, "FEAR_GREED": -0.65},
        "QQQ":  {"BTC": 0.72, "GOLD": -0.15, "WTI": 0.5, "SPY": 0.92, "QQQ": 1.0, "TREASURY_10Y": -0.45, "TIPS_10Y": -0.5, "TIPS_5Y": -0.45, "TIPS_20Y": -0.5, "TIPS_30Y": -0.5, "ETH_VOLUME": 0.62, "DEX_VOLUME": 0.57, "FEAR_GREED": -0.68},
        "TREASURY_10Y": {"BTC": 0.3, "GOLD": 0.5, "WTI": 0.1, "SPY": -0.4, "QQQ": -0.45, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.88, "TIPS_5Y": 0.78, "TIPS_20Y": 0.92, "TIPS_30Y": 0.88, "ETH_VOLUME": 0.25, "DEX_VOLUME": 0.3, "FEAR_GREED": 0.4},
        "TIPS_10Y": {"BTC": 0.25, "GOLD": 0.55, "WTI": 0.05, "SPY": -0.45, "QQQ": -0.5, "TREASURY_10Y": 0.88, "TIPS_10Y": 1.0, "TIPS_5Y": 0.92, "TIPS_20Y": 0.96, "TIPS_30Y": 0.92, "ETH_VOLUME": 0.2, "DEX_VOLUME": 0.25, "FEAR_GREED": 0.45},
        "TIPS_5Y": {"BTC": 0.2, "GOLD": 0.5, "WTI": 0.0, "SPY": -0.4, "QQQ": -0.45, "TREASURY_10Y": 0.78, "TIPS_10Y": 0.92, "TIPS_5Y": 1.0, "TIPS_20Y": 0.88, "TIPS_30Y": 0.82, "ETH_VOLUME": 0.15, "DEX_VOLUME": 0.2, "FEAR_GREED": 0.4},
        "TIPS_20Y": {"BTC": 0.3, "GOLD": 0.55, "WTI": 0.1, "SPY": -0.45, "QQQ": -0.5, "TREASURY_10Y": 0.92, "TIPS_10Y": 0.96, "TIPS_5Y": 0.88, "TIPS_20Y": 1.0, "TIPS_30Y": 0.96, "ETH_VOLUME": 0.25, "DEX_VOLUME": 0.3, "FEAR_GREED": 0.45},
        "TIPS_30Y": {"BTC": 0.25, "GOLD": 0.55, "WTI": 0.05, "SPY": -0.45, "QQQ": -0.5, "TREASURY_10Y": 0.88, "TIPS_10Y": 0.92, "TIPS_5Y": 0.82, "TIPS_20Y": 0.96, "TIPS_30Y": 1.0, "ETH_VOLUME": 0.2, "DEX_VOLUME": 0.25, "FEAR_GREED": 0.45},
        "ETH_VOLUME": {"BTC": 0.65, "GOLD": 0.1, "WTI": 0.5, "SPY": 0.6, "QQQ": 0.62, "TREASURY_10Y": 0.25, "TIPS_10Y": 0.2, "TIPS_5Y": 0.15, "TIPS_20Y": 0.25, "TIPS_30Y": 0.2, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.75, "FEAR_GREED": -0.55},
        "DEX_VOLUME": {"BTC": 0.6, "GOLD": 0.15, "WTI": 0.45, "SPY": 0.55, "QQQ": 0.57, "TREASURY_10Y": 0.3, "TIPS_10Y": 0.25, "TIPS_5Y": 0.2, "TIPS_20Y": 0.3, "TIPS_30Y": 0.25, "ETH_VOLUME": 0.75, "DEX_VOLUME": 1.0, "FEAR_GREED": -0.5},
        "FEAR_GREED": {"BTC": -0.7, "GOLD": 0.3, "WTI": -0.5, "SPY": -0.65, "QQQ": -0.68, "TREASURY_10Y": 0.4, "TIPS_10Y": 0.45, "TIPS_5Y": 0.4, "TIPS_20Y": 0.45, "TIPS_30Y": 0.45, "ETH_VOLUME": -0.55, "DEX_VOLUME": -0.5, "FEAR_GREED": 1.0}
    },
    
    "STABLE_VOL_STABLE": {
        # Low volatility ranging market - weak correlations overall
        "BTC":  {"BTC": 1.0, "GOLD": 0.0, "WTI": 0.2, "SPY": 0.3, "QQQ": 0.35, "TREASURY_10Y": -0.05, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.75, "DEX_VOLUME": 0.7, "FEAR_GREED": 0.4},
        "GOLD": {"BTC": 0.0, "GOLD": 1.0, "WTI": 0.25, "SPY": -0.2, "QQQ": -0.25, "TREASURY_10Y": 0.25, "TIPS_10Y": 0.3, "TIPS_5Y": 0.25, "TIPS_20Y": 0.3, "TIPS_30Y": 0.3, "ETH_VOLUME": -0.05, "DEX_VOLUME": 0.0, "FEAR_GREED": -0.15},
        "WTI":  {"BTC": 0.2, "GOLD": 0.25, "WTI": 1.0, "SPY": 0.35, "QQQ": 0.3, "TREASURY_10Y": -0.05, "TIPS_10Y": 0.0, "TIPS_5Y": 0.05, "TIPS_20Y": -0.05, "TIPS_30Y": 0.0, "ETH_VOLUME": 0.15, "DEX_VOLUME": 0.1, "FEAR_GREED": 0.25},
        "SPY":  {"BTC": 0.3, "GOLD": -0.2, "WTI": 0.35, "SPY": 1.0, "QQQ": 0.88, "TREASURY_10Y": 0.0, "TIPS_10Y": -0.05, "TIPS_5Y": 0.0, "TIPS_20Y": -0.05, "TIPS_30Y": -0.05, "ETH_VOLUME": 0.25, "DEX_VOLUME": 0.2, "FEAR_GREED": 0.45},
        "QQQ":  {"BTC": 0.35, "GOLD": -0.25, "WTI": 0.3, "SPY": 0.88, "QQQ": 1.0, "TREASURY_10Y": -0.05, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.3, "DEX_VOLUME": 0.25, "FEAR_GREED": 0.5},
        "TREASURY_10Y": {"BTC": -0.05, "GOLD": 0.25, "WTI": -0.05, "SPY": 0.0, "QQQ": -0.05, "TREASURY_10Y": 1.0, "TIPS_10Y": 0.82, "TIPS_5Y": 0.72, "TIPS_20Y": 0.87, "TIPS_30Y": 0.82, "ETH_VOLUME": -0.1, "DEX_VOLUME": -0.05, "FEAR_GREED": 0.1},
        "TIPS_10Y": {"BTC": -0.1, "GOLD": 0.3, "WTI": 0.0, "SPY": -0.05, "QQQ": -0.1, "TREASURY_10Y": 0.82, "TIPS_10Y": 1.0, "TIPS_5Y": 0.88, "TIPS_20Y": 0.92, "TIPS_30Y": 0.88, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": 0.05},
        "TIPS_5Y": {"BTC": -0.05, "GOLD": 0.25, "WTI": 0.05, "SPY": 0.0, "QQQ": -0.05, "TREASURY_10Y": 0.72, "TIPS_10Y": 0.88, "TIPS_5Y": 1.0, "TIPS_20Y": 0.82, "TIPS_30Y": 0.77, "ETH_VOLUME": -0.1, "DEX_VOLUME": -0.05, "FEAR_GREED": 0.1},
        "TIPS_20Y": {"BTC": -0.1, "GOLD": 0.3, "WTI": -0.05, "SPY": -0.05, "QQQ": -0.1, "TREASURY_10Y": 0.87, "TIPS_10Y": 0.92, "TIPS_5Y": 0.82, "TIPS_20Y": 1.0, "TIPS_30Y": 0.92, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": 0.05},
        "TIPS_30Y": {"BTC": -0.1, "GOLD": 0.3, "WTI": 0.0, "SPY": -0.05, "QQQ": -0.1, "TREASURY_10Y": 0.82, "TIPS_10Y": 0.88, "TIPS_5Y": 0.77, "TIPS_20Y": 0.92, "TIPS_30Y": 1.0, "ETH_VOLUME": -0.15, "DEX_VOLUME": -0.1, "FEAR_GREED": 0.05},
        "ETH_VOLUME": {"BTC": 0.75, "GOLD": -0.05, "WTI": 0.15, "SPY": 0.25, "QQQ": 0.3, "TREASURY_10Y": -0.1, "TIPS_10Y": -0.15, "TIPS_5Y": -0.1, "TIPS_20Y": -0.15, "TIPS_30Y": -0.15, "ETH_VOLUME": 1.0, "DEX_VOLUME": 0.8, "FEAR_GREED": 0.45},
        "DEX_VOLUME": {"BTC": 0.7, "GOLD": 0.0, "WTI": 0.1, "SPY": 0.2, "QQQ": 0.25, "TREASURY_10Y": -0.05, "TIPS_10Y": -0.1, "TIPS_5Y": -0.05, "TIPS_20Y": -0.1, "TIPS_30Y": -0.1, "ETH_VOLUME": 0.8, "DEX_VOLUME": 1.0, "FEAR_GREED": 0.4},
        "FEAR_GREED": {"BTC": 0.4, "GOLD": -0.15, "WTI": 0.25, "SPY": 0.45, "QQQ": 0.5, "TREASURY_10Y": 0.1, "TIPS_10Y": 0.05, "TIPS_5Y": 0.1, "TIPS_20Y": 0.05, "TIPS_30Y": 0.05, "ETH_VOLUME": 0.45, "DEX_VOLUME": 0.4, "FEAR_GREED": 1.0}
    }
}

def get_regime_correlation(regime_name, asset1, asset2):
    """
    Get correlation between two assets for a specific market regime.
    
    Args:
        regime_name (str): Market regime (e.g., "HIGH_VOL_DOWN")
        asset1 (str): First asset (e.g., "BTC")
        asset2 (str): Second asset (e.g., "SPY")
        
    Returns:
        float: Correlation coefficient between the two assets in the given regime
    """
    if regime_name not in REGIME_CORRELATIONS:
        raise ValueError(f"Unknown regime: {regime_name}")
    
    regime_matrix = REGIME_CORRELATIONS[regime_name]
    
    if asset1 not in regime_matrix or asset2 not in regime_matrix[asset1]:
        raise ValueError(f"Assets {asset1} or {asset2} not found in regime {regime_name}")
    
    return regime_matrix[asset1][asset2]

def validate_correlation_matrices():
    """
    Validate that all correlation matrices are symmetric and have valid values.
    
    Returns:
        bool: True if all matrices are valid
    """
    assets = ["BTC", "GOLD", "SPY", "TIPS_10Y"]
    
    for regime_name, matrix in REGIME_CORRELATIONS.items():
        print(f"Validating {regime_name}...")
        
        # Check symmetry
        for asset1 in assets:
            for asset2 in assets:
                corr12 = matrix[asset1][asset2]
                corr21 = matrix[asset2][asset1]
                
                if abs(corr12 - corr21) > 1e-6:
                    print(f"  ERROR: Matrix not symmetric for {asset1}-{asset2}")
                    return False
                
                # Check correlation bounds
                if not (-1.0 <= corr12 <= 1.0):
                    print(f"  ERROR: Invalid correlation {corr12} for {asset1}-{asset2}")
                    return False
        
        # Check diagonal elements are 1.0
        for asset in assets:
            if abs(matrix[asset][asset] - 1.0) > 1e-6:
                print(f"  ERROR: Diagonal element for {asset} is not 1.0")
                return False
        
        print(f"  ✓ {regime_name} matrix is valid")
    
    print("All correlation matrices are valid!")
    return True

if __name__ == "__main__":
    # Run validation
    validate_correlation_matrices()
    
    # Example usage
    print("\nExample correlations:")
    print(f"BTC-SPY in HIGH_VOL_DOWN: {get_regime_correlation('HIGH_VOL_DOWN', 'BTC', 'SPY')}")
    print(f"GOLD-TIPS in STABLE_VOL_UP: {get_regime_correlation('STABLE_VOL_UP', 'GOLD', 'TIPS_10Y')}")