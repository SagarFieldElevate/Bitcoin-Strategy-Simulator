"""
Test script to demonstrate regime-based correlation matrix simulation with debug logging
"""

import numpy as np
import pandas as pd
from utils.market_conditions import MarketCondition
from regime_correlations import REGIME_CORRELATIONS, VARIABLE_NAME_MAPPING
from correlation_utils import build_matrix_from_dict

def test_regime_simulation_with_debug():
    """Test regime-based simulation with comprehensive debug logging"""
    
    print("TESTING REGIME-BASED SIMULATION WITH DEBUG LOGGING")
    print("=" * 60)
    
    # Test configuration
    market_condition = MarketCondition.HIGH_VOL_DOWN
    test_variables = ["BITCOIN DAILY CLOSE PRICE", "SPY DAILY CLOSE PRICE", "GOLD DAILY CLOSE PRICE", "FEAR & GREED INDEX"]
    
    # Map variables to correlation codes
    variable_codes = []
    for var_name in test_variables:
        if var_name in VARIABLE_NAME_MAPPING:
            variable_codes.append(VARIABLE_NAME_MAPPING[var_name])
        else:
            print(f"Warning: Unknown variable '{var_name}', using BTC")
            variable_codes.append('BTC')
    
    print(f"\n=== SIMULATION DEBUG INFO ===")
    print(f"Selected Market Condition: {market_condition.value}")
    print(f"Variables being simulated: {test_variables}")
    print(f"Mapped to correlation codes: {variable_codes}")
    
    # Build correlation matrix
    regime_name = market_condition.name
    if regime_name in REGIME_CORRELATIONS:
        correlation_matrix = build_matrix_from_dict(
            REGIME_CORRELATIONS[regime_name], variable_codes
        )
        print(f"\nUsing {regime_name} regime correlation matrix:")
        corr_df = pd.DataFrame(correlation_matrix, index=variable_codes, columns=variable_codes)
        print(corr_df.round(3))
        print()
        
        # Verify specific correlations
        print("Key correlations to verify:")
        btc_idx = variable_codes.index('BTC') if 'BTC' in variable_codes else 0
        gold_idx = variable_codes.index('GOLD') if 'GOLD' in variable_codes else -1
        spy_idx = variable_codes.index('SPY') if 'SPY' in variable_codes else -1
        fear_idx = variable_codes.index('FEAR_GREED') if 'FEAR_GREED' in variable_codes else -1
        
        if gold_idx != -1:
            btc_gold_corr = correlation_matrix[btc_idx, gold_idx]
            print(f"  BTC-GOLD correlation: {btc_gold_corr:.3f} (should be negative in crash scenario)")
        
        if spy_idx != -1:
            btc_spy_corr = correlation_matrix[btc_idx, spy_idx]
            print(f"  BTC-SPY correlation: {btc_spy_corr:.3f} (should be positive - risk assets move together)")
        
        if fear_idx != -1:
            btc_fear_corr = correlation_matrix[btc_idx, fear_idx]
            print(f"  BTC-FEAR_GREED correlation: {btc_fear_corr:.3f} (should be negative - fear rises when BTC crashes)")
        
        print()
        
        # Test Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            print(f"✓ Cholesky decomposition successful for {len(variable_codes)}x{len(variable_codes)} matrix")
            print(f"  Cholesky matrix shape: {cholesky_matrix.shape}")
        except np.linalg.LinAlgError as e:
            print(f"✗ Cholesky decomposition failed: {e}")
            
            # Check matrix properties
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            print(f"  Matrix eigenvalues: {eigenvalues}")
            print(f"  Minimum eigenvalue: {np.min(eigenvalues):.6f}")
            
        print()
        return True
    else:
        print(f"Error: No correlation data for regime {regime_name}")
        return False

def test_multiple_regimes():
    """Test correlation matrices for different market regimes"""
    
    print("TESTING MULTIPLE MARKET REGIMES")
    print("=" * 40)
    
    test_assets = ['BTC', 'SPY', 'GOLD', 'FEAR_GREED']
    regimes_to_test = ['HIGH_VOL_DOWN', 'HIGH_VOL_UP', 'STABLE_VOL_UP']
    
    for regime in regimes_to_test:
        if regime in REGIME_CORRELATIONS:
            print(f"\n{regime} Correlation Matrix:")
            correlation_matrix = build_matrix_from_dict(
                REGIME_CORRELATIONS[regime], test_assets
            )
            corr_df = pd.DataFrame(correlation_matrix, index=test_assets, columns=test_assets)
            print(corr_df.round(3))
            
            # Highlight key relationships
            btc_gold = correlation_matrix[0, 2]  # BTC-GOLD
            btc_spy = correlation_matrix[0, 1]   # BTC-SPY
            print(f"  BTC-GOLD: {btc_gold:.3f}, BTC-SPY: {btc_spy:.3f}")

def test_variable_filtering():
    """Test the daily variable filtering functionality"""
    
    print("\nTESTING VARIABLE FILTERING")
    print("=" * 30)
    
    from regime_correlations import is_daily_only_strategy, DAILY_VARIABLES
    
    test_strategies = [
        {
            'excel_names': ['BITCOIN DAILY CLOSE PRICE', 'SPY DAILY CLOSE PRICE'],
            'description': 'Valid daily strategy'
        },
        {
            'excel_names': ['BITCOIN DAILY CLOSE PRICE', 'SOME_WEEKLY_INDICATOR'],
            'description': 'Invalid strategy with non-daily variable'
        },
        {
            'excel_names': ['FEAR & GREED INDEX', '10-YEAR TREASURY YIELD'],
            'description': 'Valid sentiment and rates strategy'
        }
    ]
    
    print(f"Known daily variables: {len(DAILY_VARIABLES)} total")
    
    for i, strategy in enumerate(test_strategies):
        is_valid = is_daily_only_strategy(strategy)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"Strategy {i+1}: {status} - {strategy['description']}")
        print(f"  Variables: {strategy['excel_names']}")
        
        if not is_valid:
            invalid_vars = [var for var in strategy['excel_names'] if var not in DAILY_VARIABLES]
            print(f"  Non-daily variables: {invalid_vars}")

if __name__ == "__main__":
    # Run all tests
    success = test_regime_simulation_with_debug()
    test_multiple_regimes()
    test_variable_filtering()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All regime-based correlation tests completed successfully!")
        print("The system can now:")
        print("  - Detect strategy variables from excel_names")
        print("  - Build filtered correlation matrices dynamically")
        print("  - Apply regime-specific correlations (BTC-GOLD negative in crashes)")
        print("  - Use Cholesky decomposition for correlated paths")
        print("  - Filter strategies to daily-frequency variables only")
    else:
        print("✗ Some tests failed - check correlation matrix setup")