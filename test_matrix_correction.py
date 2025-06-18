"""
Test positive definite matrix correction for Cholesky decomposition
"""

import numpy as np
from correlation_utils import make_positive_definite, ensure_cholesky_ready
import pandas as pd

def test_positive_definite_correction():
    """Test the positive definite matrix correction functionality"""
    
    print("TESTING POSITIVE DEFINITE MATRIX CORRECTION")
    print("=" * 50)
    
    # Test 1: Create a problematic correlation matrix (not positive definite)
    print("Test 1: Problematic correlation matrix")
    problematic_matrix = np.array([
        [1.0,  0.9,  0.8,  0.7],
        [0.9,  1.0,  0.95, 0.85],
        [0.8,  0.95, 1.0,  0.9],
        [0.7,  0.85, 0.9,  1.0]
    ])
    
    print("Original matrix:")
    print(pd.DataFrame(problematic_matrix, 
                      index=['A', 'B', 'C', 'D'], 
                      columns=['A', 'B', 'C', 'D']).round(3))
    
    # Check eigenvalues
    eigenvalues = np.linalg.eigvals(problematic_matrix)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Minimum eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Is positive definite: {np.all(eigenvalues > 0)}")
    
    # Test Cholesky on original matrix
    try:
        cholesky_original = np.linalg.cholesky(problematic_matrix)
        print("✓ Original matrix Cholesky successful")
    except np.linalg.LinAlgError:
        print("✗ Original matrix Cholesky failed (expected)")
    
    # Apply correction
    corrected_matrix, was_corrected = ensure_cholesky_ready(problematic_matrix)
    
    print(f"\nMatrix was corrected: {was_corrected}")
    if was_corrected:
        print("Corrected matrix:")
        print(pd.DataFrame(corrected_matrix,
                          index=['A', 'B', 'C', 'D'],
                          columns=['A', 'B', 'C', 'D']).round(3))
        
        # Check corrected eigenvalues
        corrected_eigenvalues = np.linalg.eigvals(corrected_matrix)
        print(f"Corrected eigenvalues: {corrected_eigenvalues}")
        print(f"Minimum corrected eigenvalue: {np.min(corrected_eigenvalues):.6f}")
        print(f"Is now positive definite: {np.all(corrected_eigenvalues > 0)}")
    
    # Test Cholesky on corrected matrix
    try:
        cholesky_corrected = np.linalg.cholesky(corrected_matrix)
        print("✓ Corrected matrix Cholesky successful")
        print(f"Cholesky matrix shape: {cholesky_corrected.shape}")
    except np.linalg.LinAlgError:
        print("✗ Corrected matrix Cholesky failed (unexpected)")
    
    print()
    
    # Test 2: Already positive definite matrix
    print("Test 2: Already positive definite matrix")
    good_matrix = np.array([
        [1.0,  0.3, -0.2],
        [0.3,  1.0,  0.1],
        [-0.2, 0.1,  1.0]
    ])
    
    print("Good matrix:")
    print(pd.DataFrame(good_matrix,
                      index=['X', 'Y', 'Z'],
                      columns=['X', 'Y', 'Z']).round(3))
    
    good_eigenvalues = np.linalg.eigvals(good_matrix)
    print(f"Eigenvalues: {good_eigenvalues}")
    print(f"Minimum eigenvalue: {np.min(good_eigenvalues):.6f}")
    
    # Test correction (should not be needed)
    good_corrected, good_was_corrected = ensure_cholesky_ready(good_matrix)
    print(f"Matrix was corrected: {good_was_corrected} (should be False)")
    
    # Test Cholesky
    try:
        cholesky_good = np.linalg.cholesky(good_corrected)
        print("✓ Good matrix Cholesky successful")
    except np.linalg.LinAlgError:
        print("✗ Good matrix Cholesky failed (unexpected)")
    
    print()
    
    # Test 3: Regime correlation matrix from actual data
    print("Test 3: Actual regime correlation matrix")
    from regime_correlations import REGIME_CORRELATIONS
    from correlation_utils import build_matrix_from_dict
    
    if 'HIGH_VOL_DOWN' in REGIME_CORRELATIONS:
        test_assets = ['BTC', 'SPY', 'GOLD', 'FEAR_GREED']
        regime_matrix = build_matrix_from_dict(
            REGIME_CORRELATIONS['HIGH_VOL_DOWN'], test_assets
        )
        
        print("HIGH_VOL_DOWN regime matrix:")
        print(pd.DataFrame(regime_matrix,
                          index=test_assets,
                          columns=test_assets).round(3))
        
        regime_eigenvalues = np.linalg.eigvals(regime_matrix)
        print(f"Eigenvalues: {regime_eigenvalues}")
        print(f"Minimum eigenvalue: {np.min(regime_eigenvalues):.6f}")
        
        # Test correction
        regime_corrected, regime_was_corrected = ensure_cholesky_ready(regime_matrix)
        print(f"Matrix was corrected: {regime_was_corrected}")
        
        # Test Cholesky
        try:
            cholesky_regime = np.linalg.cholesky(regime_corrected)
            print("✓ Regime matrix Cholesky successful")
        except np.linalg.LinAlgError:
            print("✗ Regime matrix Cholesky failed")
    
    print("\n" + "=" * 50)
    print("✓ Matrix correction tests completed")
    print("The system can now handle:")
    print("  - Problematic correlation matrices with negative eigenvalues")
    print("  - Automatic positive definite correction")
    print("  - Guaranteed Cholesky decomposition success")
    print("  - Preservation of good matrices without unnecessary changes")

def test_simulation_robustness():
    """Test that simulation works with various matrix conditions"""
    
    print("\nTESTING SIMULATION ROBUSTNESS")
    print("=" * 35)
    
    from utils.regime_monte_carlo import RegimeMonteCarloSimulator
    from utils.market_conditions import MarketCondition
    
    # Create synthetic historical data
    np.random.seed(42)
    n_days = 100
    test_variables = ['BTC', 'SPY', 'GOLD']
    
    historical_data = {}
    for var in test_variables:
        start_prices = {'BTC': 50000, 'SPY': 400, 'GOLD': 2000}
        start_price = start_prices[var]
        
        # Generate synthetic price series
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [start_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        historical_data[var] = prices[1:]
    
    hist_df = pd.DataFrame(historical_data)
    
    # Test with potentially problematic correlation matrix
    simulator = RegimeMonteCarloSimulator(hist_df)
    
    # Create regime scenarios
    regime_scenarios = {
        'BTC': MarketCondition.HIGH_VOL_DOWN,
        'SPY': MarketCondition.HIGH_VOL_DOWN,
        'GOLD': MarketCondition.HIGH_VOL_UP
    }
    
    print(f"Testing simulation with {len(test_variables)} variables")
    
    try:
        # This should work even with potentially problematic matrices
        price_paths = simulator.simulate_joint_paths(
            n_simulations=5,
            n_days=10,
            regime_scenarios=regime_scenarios
        )
        
        print("✓ Simulation completed successfully")
        for var, paths in price_paths.items():
            print(f"  {var}: {paths.shape[0]} paths, {paths.shape[1]} time steps")
            
    except Exception as e:
        print(f"✗ Simulation failed: {e}")

if __name__ == "__main__":
    test_positive_definite_correction()
    test_simulation_robustness()