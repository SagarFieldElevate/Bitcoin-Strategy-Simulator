"""
Regime-aware Monte Carlo simulation with correlation matrices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.market_conditions import MarketCondition, adjust_mu_sigma_for_condition
from regime_correlations import REGIME_CORRELATIONS, DAILY_VARIABLES
from correlation_utils import build_matrix_from_dict

class RegimeMonteCarloSimulator:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize regime-aware Monte Carlo simulator
        
        Args:
            historical_data: DataFrame with historical price data for all variables
        """
        self.historical_data = historical_data
        self.variables = historical_data.columns.tolist()
        self.n_vars = len(self.variables)
        
        # Calculate base statistics from historical data
        self.returns = historical_data.pct_change().dropna()
        self.mean_returns = self.returns.mean().to_numpy()
        self.volatilities = self.returns.std().to_numpy()
        
        print(f"Initialized regime Monte Carlo with {self.n_vars} variables: {self.variables}")
    
    def simulate_joint_paths(self, n_simulations: int, n_days: int, 
                           regime_scenarios: Dict[str, MarketCondition],
                           correlation_matrix: Optional[np.ndarray] = None,
                           seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Simulate correlated price paths using regime-specific parameters and correlations
        
        Args:
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate
            regime_scenarios: Dictionary mapping variable names to their market scenarios
            correlation_matrix: Regime-specific correlation matrix (optional)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping variable names to price path arrays (n_simulations, n_days+1)
        """
        np.random.seed(seed)
        
        # Build correlation matrix if not provided
        if correlation_matrix is None:
            # Use the regime from the first variable to determine correlation matrix
            primary_regime = list(regime_scenarios.values())[0]
            regime_name = primary_regime.name
            
            if regime_name in REGIME_CORRELATIONS:
                correlation_matrix = build_matrix_from_dict(
                    REGIME_CORRELATIONS[regime_name], self.variables
                )
            else:
                # Fallback to identity matrix
                correlation_matrix = np.eye(self.n_vars)
                print(f"Warning: No correlation matrix for regime {regime_name}, using identity")
        
        # Ensure correlation matrix is positive semi-definite
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        # Ensure matrix is positive definite before Cholesky decomposition
        from correlation_utils import ensure_cholesky_ready
        correlation_matrix, was_corrected = ensure_cholesky_ready(correlation_matrix)
        
        if was_corrected:
            print("Matrix corrected for positive definiteness")
        
        # Apply Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            print(f"Successfully computed Cholesky decomposition for {self.n_vars}x{self.n_vars} matrix")
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed even after correction, using identity matrix")
            cholesky_matrix = np.eye(self.n_vars)
        
        # Calculate regime-adjusted parameters for each variable
        adjusted_params = self._calculate_regime_parameters(regime_scenarios)
        
        # Initialize price path arrays (n_simulations, n_days+1)
        price_paths = {}
        for i, var in enumerate(self.variables):
            price_paths[var] = np.zeros((n_simulations, n_days + 1))
            # Set initial prices to last historical values
            price_paths[var][:, 0] = self.historical_data[var].iloc[-1]
        
        print(f"Generating {n_simulations} correlated paths for {n_days} days...")
        
        # Generate correlated random shocks for all simulations and days at once
        random_shocks = np.random.normal(0, 1, (n_simulations, n_days, self.n_vars))
        
        # Apply Cholesky transformation to create correlations
        for sim in range(n_simulations):
            for day in range(n_days):
                # Get independent random variables for this day
                independent_shocks = random_shocks[sim, day, :]
                
                # Apply Cholesky transformation for correlation
                correlated_shocks = cholesky_matrix @ independent_shocks
                
                # Update each variable's price
                for i, var in enumerate(self.variables):
                    drift, volatility = adjusted_params[var]
                    
                    # Calculate return: drift + volatility * correlated_shock
                    daily_return = drift + volatility * correlated_shocks[i]
                    
                    # Update price using geometric Brownian motion
                    prev_price = price_paths[var][sim, day]
                    new_price = prev_price * np.exp(daily_return)
                    
                    # Ensure positive prices
                    price_paths[var][sim, day + 1] = max(new_price, 0.01)
        
        print("Joint path simulation completed successfully")
        return price_paths
    
    def _calculate_regime_parameters(self, regime_scenarios: Dict[str, MarketCondition]) -> Dict[str, tuple]:
        """
        Calculate regime-adjusted drift and volatility for each variable
        
        Returns:
            Dictionary mapping variable names to (drift, volatility) tuples
        """
        adjusted_params = {}
        
        print("Applying regime adjustments:")
        for i, var in enumerate(self.variables):
            base_drift = self.mean_returns[i]
            base_volatility = self.volatilities[i]
            
            if var in regime_scenarios:
                scenario = regime_scenarios[var]
                adjusted_drift, adjusted_volatility = adjust_mu_sigma_for_condition(
                    base_drift, base_volatility, scenario
                )
                print(f"  {var}: {scenario.value}")
                print(f"    Drift: {base_drift:.6f} → {adjusted_drift:.6f}")
                print(f"    Vol:   {base_volatility:.4f} → {adjusted_volatility:.4f}")
            else:
                # Use base parameters if no scenario specified
                adjusted_drift, adjusted_volatility = base_drift, base_volatility
                print(f"  {var}: Using base parameters (no scenario)")
            
            adjusted_params[var] = (adjusted_drift, adjusted_volatility)
        
        return adjusted_params
    
    def _ensure_positive_definite(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure correlation matrix is positive semi-definite by adjusting eigenvalues
        """
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        
        # Set minimum eigenvalue threshold
        min_eigenvalue = 1e-8
        adjusted_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        if not np.allclose(eigenvalues, adjusted_eigenvalues):
            print(f"Adjusted {np.sum(eigenvalues < min_eigenvalue)} negative eigenvalues")
            # Reconstruct matrix with adjusted eigenvalues
            correlation_matrix = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
            
            # Ensure diagonal elements are 1.0
            diag_sqrt = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return correlation_matrix
    
    def get_regime_correlation_matrix(self, regime_name: str) -> np.ndarray:
        """
        Get correlation matrix for a specific regime
        
        Args:
            regime_name: Name of the market regime (e.g., "HIGH_VOL_DOWN")
            
        Returns:
            Correlation matrix for the specified regime
        """
        if regime_name in REGIME_CORRELATIONS:
            return build_matrix_from_dict(REGIME_CORRELATIONS[regime_name], self.variables)
        else:
            print(f"Warning: No correlation data for regime {regime_name}")
            return np.eye(self.n_vars)

def test_regime_monte_carlo():
    """Test the regime-aware Monte Carlo simulation"""
    print("TESTING REGIME MONTE CARLO SIMULATION")
    print("=" * 50)
    
    # Create synthetic historical data for testing
    np.random.seed(42)
    n_days_hist = 252  # 1 year of data
    test_variables = ["BTC", "SPY", "GOLD", "TREASURY_10Y", "FEAR_GREED"]
    
    # Generate synthetic historical data
    historical_data = {}
    for var in test_variables:
        # Different starting prices for different assets
        start_prices = {"BTC": 50000, "SPY": 400, "GOLD": 2000, "TREASURY_10Y": 4.5, "FEAR_GREED": 50}
        start_price = start_prices.get(var, 100)
        
        # Generate random walk
        returns = np.random.normal(0.0005, 0.02, n_days_hist)  # 0.05% daily drift, 2% daily vol
        prices = [start_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        historical_data[var] = prices[1:]  # Remove initial price
    
    hist_df = pd.DataFrame(historical_data)
    
    # Initialize simulator
    simulator = RegimeMonteCarloSimulator(hist_df)
    
    # Define regime scenarios
    regime_scenarios = {
        "BTC": MarketCondition.HIGH_VOL_DOWN,
        "SPY": MarketCondition.HIGH_VOL_DOWN, 
        "GOLD": MarketCondition.HIGH_VOL_STABLE,
        "TREASURY_10Y": MarketCondition.HIGH_VOL_UP,
        "FEAR_GREED": MarketCondition.HIGH_VOL_UP
    }
    
    print(f"Testing with regime scenarios: {regime_scenarios}")
    
    # Run simulation
    price_paths = simulator.simulate_joint_paths(
        n_simulations=10,
        n_days=30,
        regime_scenarios=regime_scenarios
    )
    
    # Verify results
    print(f"\nSimulation Results:")
    for var, paths in price_paths.items():
        print(f"  {var}: shape {paths.shape}, price range [{paths.min():.2f}, {paths.max():.2f}]")
    
    return price_paths

if __name__ == "__main__":
    test_regime_monte_carlo()