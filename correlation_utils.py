"""
Utility functions for working with correlation matrices and regime data.
"""

import numpy as np
from typing import Dict, List, Any

def build_matrix_from_dict(correlation_dict: Dict[str, Dict[str, float]], 
                          asset_list: List[str]) -> np.ndarray:
    """
    Build a 2D numpy correlation matrix from a nested dictionary.
    
    Args:
        correlation_dict: Nested dictionary with correlation values
                         Format: {"ASSET1": {"ASSET1": 1.0, "ASSET2": 0.5, ...}, ...}
        asset_list: List of asset names in desired matrix order
                   
    Returns:
        np.ndarray: n_assets × n_assets correlation matrix where rows/columns 
                   correspond to the order in asset_list
                   
    Example:
        >>> corr_dict = {
        ...     "BTC": {"BTC": 1.0, "SPY": 0.7, "GOLD": -0.3},
        ...     "SPY": {"BTC": 0.7, "SPY": 1.0, "GOLD": -0.2},
        ...     "GOLD": {"BTC": -0.3, "SPY": -0.2, "GOLD": 1.0}
        ... }
        >>> assets = ["BTC", "SPY", "GOLD"]
        >>> matrix = build_matrix_from_dict(corr_dict, assets)
        >>> matrix.shape
        (3, 3)
    """
    n_assets = len(asset_list)
    matrix = np.zeros((n_assets, n_assets))
    
    for i, asset1 in enumerate(asset_list):
        for j, asset2 in enumerate(asset_list):
            if asset1 in correlation_dict and asset2 in correlation_dict[asset1]:
                matrix[i, j] = correlation_dict[asset1][asset2]
            elif asset1 == asset2:
                # Diagonal elements should be 1.0 if not specified
                matrix[i, j] = 1.0
            else:
                # If correlation not found, assume 0.0
                matrix[i, j] = 0.0
    
    return matrix

def validate_matrix_properties(matrix: np.ndarray, asset_names: List[str] = None) -> bool:
    """
    Validate that a correlation matrix has proper mathematical properties.
    
    Args:
        matrix: Correlation matrix to validate
        asset_names: Optional list of asset names for better error messages
        
    Returns:
        bool: True if matrix is valid, False otherwise
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(matrix.shape[0])]
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print(f"ERROR: Matrix is not square: {matrix.shape}")
        return False
    
    # Check diagonal elements are 1.0
    diagonal = np.diag(matrix)
    if not np.allclose(diagonal, 1.0, atol=1e-6):
        print(f"ERROR: Diagonal elements are not 1.0: {diagonal}")
        return False
    
    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        print("ERROR: Matrix is not symmetric")
        return False
    
    # Check correlation bounds [-1, 1]
    if np.any(matrix < -1.0) or np.any(matrix > 1.0):
        print("ERROR: Correlation values outside [-1, 1] bounds")
        return False
    
    # Check if matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues < -1e-8):  # Small tolerance for numerical precision
        print(f"WARNING: Matrix may not be positive semi-definite. Min eigenvalue: {np.min(eigenvalues)}")
        # Don't return False as this might be fixable
    
    return True

def extract_regime_matrix(regime_correlations: Dict[str, Any], 
                         regime_name: str, 
                         asset_list: List[str]) -> np.ndarray:
    """
    Extract correlation matrix for a specific regime from REGIME_CORRELATIONS.
    
    Args:
        regime_correlations: Full regime correlations dictionary
        regime_name: Name of the regime (e.g., "HIGH_VOL_DOWN")
        asset_list: List of assets to include in the matrix
        
    Returns:
        np.ndarray: Correlation matrix for the specified regime
    """
    if regime_name not in regime_correlations:
        raise ValueError(f"Regime '{regime_name}' not found in correlations")
    
    regime_dict = regime_correlations[regime_name]
    return build_matrix_from_dict(regime_dict, asset_list)

def test_correlation_utils():
    """Test the correlation utility functions."""
    print("TESTING CORRELATION UTILITIES")
    print("=" * 40)
    
    # Test data
    test_dict = {
        "BTC": {"BTC": 1.0, "SPY": 0.7, "GOLD": -0.3, "TIPS": -0.4},
        "SPY": {"BTC": 0.7, "SPY": 1.0, "GOLD": -0.2, "TIPS": -0.3},
        "GOLD": {"BTC": -0.3, "SPY": -0.2, "GOLD": 1.0, "TIPS": 0.3},
        "TIPS": {"BTC": -0.4, "SPY": -0.3, "GOLD": 0.3, "TIPS": 1.0}
    }
    
    assets = ["BTC", "SPY", "GOLD", "TIPS"]
    
    # Test matrix building
    print("Building correlation matrix...")
    matrix = build_matrix_from_dict(test_dict, assets)
    print(f"Matrix shape: {matrix.shape}")
    print("Matrix:")
    print(matrix)
    print()
    
    # Test validation
    print("Validating matrix properties...")
    is_valid = validate_matrix_properties(matrix, assets)
    print(f"Matrix is valid: {is_valid}")
    print()
    
    # Test different asset order
    reordered_assets = ["GOLD", "BTC", "TIPS", "SPY"]
    print(f"Testing with reordered assets: {reordered_assets}")
    reordered_matrix = build_matrix_from_dict(test_dict, reordered_assets)
    print("Reordered matrix:")
    print(reordered_matrix)
    print()
    
    # Verify that reordering gives same correlations in different positions
    btc_spy_original = matrix[0, 1]  # BTC-SPY in original
    btc_spy_reordered = reordered_matrix[1, 3]  # BTC-SPY in reordered
    print(f"BTC-SPY correlation consistency: {btc_spy_original} == {btc_spy_reordered}: {abs(btc_spy_original - btc_spy_reordered) < 1e-10}")

if __name__ == "__main__":
    test_correlation_utils()