"""
Regime-specific correlation matrices for different market conditions.
These correlations reflect how assets typically behave together during various market scenarios.
"""

# Daily frequency variables from the screenshot
DAILY_VARIABLES = [
    "BTC", "GOLD", "WTI", "SPY", "QQQ", 
    "TREASURY_10Y", "TIPS_10Y", "TIPS_5Y", "TIPS_20Y", "TIPS_30Y",
    "ETH_VOLUME", "DEX_VOLUME", "FEAR_GREED"
]

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
    assets = ["BTC", "GOLD", "SPY", "TIPS"]
    
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
    print(f"GOLD-TIPS in STABLE_VOL_UP: {get_regime_correlation('STABLE_VOL_UP', 'GOLD', 'TIPS')}")