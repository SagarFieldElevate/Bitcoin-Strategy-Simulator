"""
Regime-specific correlation matrices for different market conditions.
These correlations reflect how assets typically behave together during various market scenarios.
"""

REGIME_CORRELATIONS = {
    "HIGH_VOL_DOWN": {
        # Market crash scenario - flight to quality, risk-off behavior
        "BTC":  {"BTC": 1.0, "GOLD": -0.3, "SPY": 0.7, "TIPS": -0.4},
        "GOLD": {"BTC": -0.3, "GOLD": 1.0, "SPY": -0.5, "TIPS": 0.3},
        "SPY":  {"BTC": 0.7, "GOLD": -0.5, "SPY": 1.0, "TIPS": -0.6},
        "TIPS": {"BTC": -0.4, "GOLD": 0.3, "SPY": -0.6, "TIPS": 1.0}
    },
    
    "HIGH_VOL_UP": {
        # Bull market with high volatility - risk-on behavior with momentum
        "BTC":  {"BTC": 1.0, "GOLD": -0.2, "SPY": 0.6, "TIPS": -0.5},
        "GOLD": {"BTC": -0.2, "GOLD": 1.0, "SPY": -0.1, "TIPS": 0.2},
        "SPY":  {"BTC": 0.6, "GOLD": -0.1, "SPY": 1.0, "TIPS": -0.4},
        "TIPS": {"BTC": -0.5, "GOLD": 0.2, "SPY": -0.4, "TIPS": 1.0}
    },
    
    "HIGH_VOL_STABLE": {
        # High volatility sideways - choppy markets, mixed correlations
        "BTC":  {"BTC": 1.0, "GOLD": 0.1, "SPY": 0.3, "TIPS": -0.2},
        "GOLD": {"BTC": 0.1, "GOLD": 1.0, "SPY": -0.1, "TIPS": 0.1},
        "SPY":  {"BTC": 0.3, "GOLD": -0.1, "SPY": 1.0, "TIPS": -0.2},
        "TIPS": {"BTC": -0.2, "GOLD": 0.1, "SPY": -0.2, "TIPS": 1.0}
    },
    
    "STABLE_VOL_UP": {
        # Stable bull market - steady risk-on, moderate correlations
        "BTC":  {"BTC": 1.0, "GOLD": 0.0, "SPY": 0.4, "TIPS": -0.3},
        "GOLD": {"BTC": 0.0, "GOLD": 1.0, "SPY": 0.1, "TIPS": 0.2},
        "SPY":  {"BTC": 0.4, "GOLD": 0.1, "SPY": 1.0, "TIPS": -0.2},
        "TIPS": {"BTC": -0.3, "GOLD": 0.2, "SPY": -0.2, "TIPS": 1.0}
    },
    
    "STABLE_VOL_DOWN": {
        # Stable bear market - gradual decline, moderate flight to quality
        "BTC":  {"BTC": 1.0, "GOLD": -0.1, "SPY": 0.5, "TIPS": -0.2},
        "GOLD": {"BTC": -0.1, "GOLD": 1.0, "SPY": -0.3, "TIPS": 0.4},
        "SPY":  {"BTC": 0.5, "GOLD": -0.3, "SPY": 1.0, "TIPS": -0.4},
        "TIPS": {"BTC": -0.2, "GOLD": 0.4, "SPY": -0.4, "TIPS": 1.0}
    },
    
    "STABLE_VOL_STABLE": {
        # Calm sideways market - low correlations, independent movements
        "BTC":  {"BTC": 1.0, "GOLD": 0.1, "SPY": 0.2, "TIPS": -0.1},
        "GOLD": {"BTC": 0.1, "GOLD": 1.0, "SPY": 0.0, "TIPS": 0.1},
        "SPY":  {"BTC": 0.2, "GOLD": 0.0, "SPY": 1.0, "TIPS": -0.1},
        "TIPS": {"BTC": -0.1, "GOLD": 0.1, "SPY": -0.1, "TIPS": 1.0}
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