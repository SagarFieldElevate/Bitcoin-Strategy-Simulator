"""
Multi-factor data fetching and simulation for macro-economic variables
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MultiFactorDataFetcher:
    def __init__(self):
        """Initialize the multi-factor data fetcher"""
        # Yahoo Finance symbols for macro variables
        self.symbol_map = {
            'BTC': 'BTC-USD',
            'WTI': 'CL=F',  # WTI Crude Oil Futures
            'TIPS_10Y': '^TNX',  # 10-Year Treasury Note Yield (proxy for TIPS)
            'CPI': '^GSPC',  # S&P 500 as CPI proxy (will use calculated inflation)
            'DXY': 'DX-Y.NYB',  # US Dollar Index
            'GOLD': 'GC=F',  # Gold Futures
            'VIX': '^VIX',  # VIX Volatility Index
            'SPY': 'SPY'  # S&P 500 ETF
        }
    
    def fetch_multi_factor_data(self, variables, start_date=None, end_date=None):
        """
        Fetch historical data for multiple variables
        
        Args:
            variables: List of variable codes (e.g., ['BTC', 'WTI', 'TIPS_10Y'])
            start_date: Start date for data (default: 3 years ago)
            end_date: End date for data (default: today)
            
        Returns:
            pd.DataFrame: Multi-column DataFrame with aligned data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching multi-factor data for {variables} from {start_date} to {end_date}")
        
        data_frames = {}
        
        for var in variables:
            if var not in self.symbol_map:
                print(f"Warning: Unknown variable {var}, skipping")
                continue
                
            symbol = self.symbol_map[var]
            print(f"Downloading {var} ({symbol})...")
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    print(f"Warning: No data found for {var}")
                    continue
                
                # Use Close price for most variables
                if var == 'VIX':
                    # VIX is already a volatility measure
                    data_frames[var] = data['Close']
                elif var == 'TIPS_10Y':
                    # Treasury yield is already in percentage
                    data_frames[var] = data['Close']
                else:
                    # Use Close price
                    data_frames[var] = data['Close']
                    
            except Exception as e:
                print(f"Error fetching {var}: {e}")
                continue
        
        if not data_frames:
            raise ValueError("No data could be fetched for any variables")
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(data_frames)
        
        # Forward fill missing values and drop any remaining NaN rows
        combined_df = combined_df.fillna(method='ffill').dropna()
        
        print(f"Successfully fetched data: {len(combined_df)} days, {len(combined_df.columns)} variables")
        return combined_df
    
    def calculate_correlation_matrix(self, data_df):
        """
        Calculate correlation matrix from historical data
        
        Args:
            data_df: DataFrame with historical price data
            
        Returns:
            np.ndarray: Correlation matrix
        """
        # Calculate daily returns
        returns = data_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr().values
        
        print("Correlation matrix calculated:")
        for i, var1 in enumerate(data_df.columns):
            for j, var2 in enumerate(data_df.columns):
                if i < j:  # Only show upper triangle
                    corr = correlation_matrix[i, j]
                    print(f"  {var1}-{var2}: {corr:.3f}")
        
        return correlation_matrix
    
    def simulate_multi_factor_series(self, historical_data, n_simulations, simulation_days, correlation_matrix=None):
        """
        Generate correlated synthetic paths for multiple variables using Cholesky decomposition
        
        Args:
            historical_data: DataFrame with historical data
            n_simulations: Number of simulation paths
            simulation_days: Number of days to simulate
            correlation_matrix: Pre-calculated correlation matrix (optional)
            
        Returns:
            dict: Dictionary of simulated paths for each variable
        """
        variables = historical_data.columns.tolist()
        n_vars = len(variables)
        
        print(f"Simulating {n_simulations} paths for {simulation_days} days with {n_vars} variables")
        
        # Calculate returns and statistics
        returns = historical_data.pct_change().dropna()
        
        # Calculate mean returns and volatilities
        mean_returns = returns.mean().values
        volatilities = returns.std().values
        
        # Use provided correlation matrix or calculate from data
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix(historical_data)
        
        # Ensure correlation matrix is valid
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        # Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            print("Warning: Correlation matrix not positive definite, using identity matrix")
            cholesky_matrix = np.eye(n_vars)
        
        # Generate correlated random variables
        simulated_paths = {}
        
        for var in variables:
            simulated_paths[var] = np.zeros((n_simulations, simulation_days + 1))
            # Set initial values to last known prices
            simulated_paths[var][:, 0] = historical_data[var].iloc[-1]
        
        # Generate correlated paths
        for sim in range(n_simulations):
            for day in range(simulation_days):
                # Generate independent random variables
                random_vars = np.random.normal(0, 1, n_vars)
                
                # Apply Cholesky transformation for correlation
                correlated_vars = cholesky_matrix @ random_vars
                
                # Generate returns for each variable
                for i, var in enumerate(variables):
                    # Calculate return with drift and volatility
                    drift = mean_returns[i]
                    volatility = volatilities[i]
                    
                    # Apply the correlated random shock
                    daily_return = drift + volatility * correlated_vars[i]
                    
                    # Update price
                    prev_price = simulated_paths[var][sim, day]
                    new_price = prev_price * (1 + daily_return)
                    simulated_paths[var][sim, day + 1] = max(new_price, 0.01)  # Prevent negative prices
        
        # Remove initial column (keep only simulated days)
        for var in variables:
            simulated_paths[var] = simulated_paths[var][:, 1:]
        
        print(f"Multi-factor simulation completed successfully")
        return simulated_paths
    
    def _ensure_positive_definite(self, correlation_matrix):
        """
        Ensure correlation matrix is positive definite by adjusting eigenvalues
        """
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        
        # If any eigenvalue is negative or too small, adjust
        min_eigenvalue = 1e-8
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct matrix
        adjusted_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Ensure diagonal is 1 (correlation property)
        diagonal = np.sqrt(np.diag(adjusted_matrix))
        adjusted_matrix = adjusted_matrix / np.outer(diagonal, diagonal)
        
        return adjusted_matrix