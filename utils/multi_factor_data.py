"""
Multi-factor data fetching and simulation for macro-economic variables
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import streamlit as st

class MultiFactorDataFetcher:
    def __init__(self, pinecone_client=None):
        """Initialize the multi-factor data fetcher"""
        self.pinecone_client = pinecone_client
        self.intelligence_index_name = "intelligence-main"
        
        # Cache for found vectors to avoid repeated searches
        self.vector_cache = {}
        
        # Cache for historical data - this is the KEY optimization
        self._historical_data_cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=24)  # Cache for 24 hours
    
    @st.cache_data(ttl=3600*24, show_spinner=False)  # Cache for 24 hours
    def get_cached_historical_data(self, variables_tuple, start_date, end_date):
        """
        Static method for Streamlit caching of historical data.
        This is separate from instance cache to leverage Streamlit's caching.
        """
        # This will be called by fetch_multi_factor_data
        return None  # Placeholder - actual fetching happens in fetch_multi_factor_data
    
    def find_vector_for_variable(self, variable_name):
        """
        Find all vectors for a specific variable using Pinecone metadata filtering
        """
        if variable_name in self.vector_cache:
            print(f"🔄 Using cached results for {variable_name}")
            return self.vector_cache[variable_name]
        
        if not self.pinecone_client:
            print(f"❌ No Pinecone client available for {variable_name}")
            return None
        
        # Connect to intelligence-main index
        try:
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
        except Exception as e:
            print(f"❌ Could not connect to intelligence-main index: {e}")
            return None
        
        print(f"🔍 Searching for '{variable_name}' using metadata filter...")
        
        # Map excel_names from bitcoin-strategies to exact names in intelligence-main
        # This mapping covers ALL excel_names found in the bitcoin-strategies index
        excel_name_to_intelligence_main = {
            # Direct matches (excel_name same as intelligence-main name)
            '10-Year TIPS Yield (%)': '10-Year TIPS Yield (%)',
            '10-Year Treasury Yield (%)': '10-Year Treasury Yield (%)',
            '20-Year TIPS Yield (%)': '20-Year TIPS Yield (%)',
            '30-Year TIPS Yield (%)': '30-Year TIPS Yield (%)',
            '5-Year TIPS Yield (%)': '5-Year TIPS Yield (%)',
            'Baa Corporate Credit Spread (%)': 'Baa Corporate Credit Spread (%)',
            'Baltic Dry Index': 'Baltic Dry Index',
            'Bank Prime Loan Rate (%)': 'Bank Prime Loan Rate (%)',
            'COIN50 Perpetual Index (365 Days)': 'COIN50 Perpetual Index (365 Days)',
            'CoinGecko BTC Daily Volume': 'CoinGecko BTC Daily Volume',
            'CoinGecko ETH Daily Volume': 'CoinGecko ETH Daily Volume',
            'DXY Daily Close Price': 'DXY Daily Close Price',
            'DefiLlama DEX Historical Volume': 'DefiLlama DEX Historical Volume',
            'Fear & Greed Index': 'Fear & Greed Index',
            'Gold Daily Close Price': 'Gold Daily Close Price',
            'Growth ETF': 'Growth ETF',
            'IWM Daily Close Price': 'IWM Daily Close Price',
            'Japan ETF': 'Japan ETF',
            'Momentum ETF': 'Momentum ETF',
            'NASDAQ 100 Index': 'NASDAQ 100 Index',
            'QQQ Daily Close Price': 'QQQ Daily Close Price',
            'SPY Daily Close Price': 'SPY Daily Close Price',
            'US Equity Market Capitalization (Billions USD)': 'US Equity Market Capitalization (Billions USD)',
            'US Federal Funds Rate': 'US Federal Funds Rate',
            'Ultra Short 20+ Year Treasury': 'Ultra Short 20+ Year Treasury',
            'WTI Crude Oil Price (USD/Barrel)': 'WTI Crude Oil Price (USD/Barrel)',
            
            # Common variable codes for backward compatibility
            'BTC': 'Bitcoin Daily Close Price',
            'BITCOIN': 'Bitcoin Daily Close Price',
            'ETH': 'CoinGecko ETH Daily Volume',
            'ETHEREUM': 'CoinGecko ETH Daily Volume',
            'GOLD': 'Gold Daily Close Price',
            'XAU': 'Gold Daily Close Price',
            'WTI': 'WTI Crude Oil Price (USD/Barrel)',
            'OIL': 'WTI Crude Oil Price (USD/Barrel)',
            'CRUDE': 'WTI Crude Oil Price (USD/Barrel)',
            'SPY': 'SPY Daily Close Price',
            'SP500': 'S&P 500 Index',
            'SPX': 'S&P 500 Index',
            'QQQ': 'QQQ Daily Close Price',
            'NASDAQ': 'NASDAQ 100 Index',
            'NDX': 'NASDAQ 100 Index',
            'IWM': 'IWM Daily Close Price',
            'RUSSELL': 'Russell 2000 Index',
            'DXY': 'DXY Daily Close Price',
            'DOLLAR': 'DXY Daily Close Price',
            'VIX': 'CBOE Volatility Index (VIX)',
            'VOLATILITY': 'CBOE Volatility Index (VIX)',
            'TIPS': '10-Year TIPS Yield (%)',
            'TIPS_10Y': '10-Year TIPS Yield (%)',
            'TIPS_5Y': '5-Year TIPS Yield (%)',
            'TIPS_20Y': '20-Year TIPS Yield (%)',
            'TIPS_30Y': '30-Year TIPS Yield (%)',
            'TREASURY_10Y': '10-Year Treasury Yield (%)',
            'FED_FUNDS': 'US Federal Funds Rate',
            'BANK_PRIME': 'Bank Prime Loan Rate (%)',
            'CORPORATE_CREDIT': 'Baa Corporate Credit Spread (%)',
            'FEAR_GREED': 'Fear & Greed Index',
            'BTC_VOLUME': 'CoinGecko BTC Daily Volume',
            'ETH_VOLUME': 'CoinGecko ETH Daily Volume'
        }
        
        # Get the exact name to search for
        exact_name = excel_name_to_intelligence_main.get(variable_name, variable_name)
        
        if exact_name != variable_name:
            print(f"📝 Variable mapping: '{variable_name}' -> '{exact_name}'")
        else:
            print(f"📝 Using exact variable name: '{exact_name}'")
        
        try:
            # Use metadata filter to find all vectors for this variable
            print(f"🔎 Querying Pinecone with filter: excel_name = '{exact_name}'")
            results = intelligence_index.query(
                vector=[0.0] * 1536,  # Placeholder vector
                filter={"excel_name": {"$eq": exact_name}},
                top_k=10000,  # Get as many results as possible
                include_metadata=True
            )
            
            if not hasattr(results, 'matches') or not results.matches:
                print(f"❌ No vectors found for '{variable_name}' with exact name '{exact_name}'")
                return None
            
            print(f"✅ Found {len(results.matches)} vectors for '{variable_name}'")
            
            # Cache the results
            cache_entry = {
                'variable_name': variable_name,
                'exact_name': exact_name,
                'vectors': results.matches
            }
            self.vector_cache[variable_name] = cache_entry
            
            return cache_entry
            
        except Exception as e:
            print(f"❌ Error searching for '{variable_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fetch_data_from_pinecone(self, vector_info):
        """
        Extract time series data from multiple Pinecone vectors for a variable
        """
        if not vector_info or 'vectors' not in vector_info:
            print(f"❌ Invalid vector_info provided")
            return None
        
        vectors = vector_info['vectors']
        if not vectors:
            print(f"❌ No vectors in vector_info")
            return None
        
        variable_name = vector_info.get('variable_name', 'Unknown')
        print(f"📊 Extracting time series data from {len(vectors)} vectors for '{variable_name}'...")
        
        all_data_points = []
        vectors_processed = 0
        vectors_with_data = 0
        
        for vector in vectors:
            vectors_processed += 1
            if not hasattr(vector, 'metadata') or not vector.metadata:
                continue
            
            metadata = vector.metadata
            found_data_in_vector = False
            
            # Extract date and value from metadata
            # Try multiple approaches based on how data might be stored
            
            # Approach 1: Direct date and value fields
            if 'date' in metadata and 'value' in metadata:
                try:
                    date = pd.to_datetime(metadata['date'])
                    value = float(metadata['value'])
                    all_data_points.append({'date': date, 'value': value})
                    found_data_in_vector = True
                    continue
                except (ValueError, TypeError, pd.errors.ParserError):
                    pass
            
            # Approach 2: Parse from raw_text
            if 'raw_text' in metadata:
                raw_text = metadata['raw_text']
                if '|' in raw_text:
                    try:
                        parts = raw_text.split('|')
                        if len(parts) >= 2:
                            # Extract date
                            date_part = parts[0].strip()
                            if 'Date:' in date_part:
                                date_str = date_part.replace('Date:', '').strip()
                            else:
                                date_str = date_part
                            
                            date = pd.to_datetime(date_str)
                            
                            # Extract value
                            value_part = parts[1].strip()
                            import re
                            numbers = re.findall(r'[-+]?\d*\.?\d+', value_part)
                            if numbers:
                                value = float(numbers[-1])
                                all_data_points.append({'date': date, 'value': value})
                                found_data_in_vector = True
                                continue
                    except Exception as e:
                        pass
            
            # Approach 3: Check for time series data in metadata
            for key in ['time_series', 'data', 'values']:
                if key in metadata:
                    ts_data = metadata[key]
                    if isinstance(ts_data, dict):
                        for date_str, value in ts_data.items():
                            try:
                                date = pd.to_datetime(date_str)
                                value = float(value)
                                all_data_points.append({'date': date, 'value': value})
                                found_data_in_vector = True
                            except (ValueError, TypeError, pd.errors.ParserError):
                                pass
                    elif isinstance(ts_data, list):
                        # Assume it's a list of [date, value] pairs or dicts
                        for item in ts_data:
                            if isinstance(item, dict) and 'date' in item and 'value' in item:
                                try:
                                    date = pd.to_datetime(item['date'])
                                    value = float(item['value'])
                                    all_data_points.append({'date': date, 'value': value})
                                    found_data_in_vector = True
                                except (ValueError, TypeError, pd.errors.ParserError):
                                    pass
                    break
            
            if found_data_in_vector:
                vectors_with_data += 1
        
        print(f"📈 Processed {vectors_processed} vectors, {vectors_with_data} contained extractable data")
        
        if not all_data_points:
            print(f"❌ No data points extracted from any vectors for '{variable_name}'")
            return None
        
        print(f"📊 Extracted {len(all_data_points)} raw data points")
        
        # Convert to DataFrame and clean
        df = pd.DataFrame(all_data_points)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        
        # Create pandas Series
        series = pd.Series(df['value'].values, index=df['date'], name=variable_name)
        
        print(f"✅ Successfully created time series with {len(series)} unique data points")
        print(f"📅 Date range: {series.index.min().strftime('%Y-%m-%d')} to {series.index.max().strftime('%Y-%m-%d')}")
        
        return series
    


    def fetch_multi_factor_data(self, variables, start_date=None, end_date=None):
        """
        Fetch historical data for multiple variables with intelligent caching
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')  # 10 years
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create cache key
        cache_key = f"{','.join(sorted(variables))}_{start_date}_{end_date}"
        
        # Check if we have valid cached data
        if cache_key in self._historical_data_cache:
            cache_time = self._cache_expiry.get(cache_key, datetime.min)
            if datetime.now() < cache_time:
                print(f"Using cached multi-factor data for {variables}")
                return self._historical_data_cache[cache_key]
        
        print(f"Fetching fresh multi-factor data for {variables} (10 years)")
        print(f"Date range: {start_date} to {end_date}")
        print("=" * 80)
        
        # Use Streamlit's progress bar for better UX
        progress_bar = st.progress(0, text="Fetching historical data...")
        
        data_frames = {}
        total_vars = len(variables)
        
        # Track success/failure for each variable
        variable_results = {}
        
        for idx, var in enumerate(variables):
            progress_bar.progress((idx) / total_vars, text=f"Fetching {var} data...")
            
            print(f"\n[{idx+1}/{total_vars}] Processing variable: {var}")
            print("-" * 50)
            
            variable_results[var] = {
                'status': 'FAILED',
                'vectors_found': 0,
                'data_points_extracted': 0,
                'final_data_points': 0,
                'error_message': None,
                'source': None
            }
            
            if var == 'BTC':
                # Handle Bitcoin separately using yfinance
                print(f"📊 Fetching {var} from yfinance...")
                variable_results[var]['source'] = 'yfinance'
                try:
                    ticker = yf.Ticker('BTC-USD')
                    btc_data = ticker.history(start=start_date, end=end_date)
                    if not btc_data.empty:
                        data_frames[var] = btc_data['Close']
                        variable_results[var]['status'] = 'SUCCESS'
                        variable_results[var]['final_data_points'] = len(btc_data)
                        print(f"✅ Successfully fetched {var}: {len(btc_data)} days")
                    else:
                        variable_results[var]['error_message'] = "No Bitcoin data found in yfinance"
                        print(f"❌ No Bitcoin data found")
                except Exception as e:
                    variable_results[var]['error_message'] = f"yfinance error: {str(e)}"
                    print(f"❌ Error fetching Bitcoin data: {e}")
                continue
            
            # Find and fetch data from Pinecone
            variable_results[var]['source'] = 'pinecone'
            print(f"🔍 Finding vectors for {var}...")
            vector_info = self.find_vector_for_variable(var)
            
            if vector_info:
                variable_results[var]['vectors_found'] = len(vector_info.get('vectors', []))
                print(f"✅ Found {variable_results[var]['vectors_found']} vectors for {var}")
                
                # Fetch data from all vectors
                print(f"📈 Extracting time series data...")
                series_data = self.fetch_data_from_pinecone(vector_info)
                
                if series_data is not None and len(series_data) > 0:
                    variable_results[var]['data_points_extracted'] = len(series_data)
                    print(f"✅ Extracted {len(series_data)} raw data points")
                    
                    # Filter by date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    filtered_data = series_data[
                        (series_data.index >= start_dt) & 
                        (series_data.index <= end_dt)
                    ]
                    
                    if len(filtered_data) > 0:
                        data_frames[var] = filtered_data
                        variable_results[var]['status'] = 'SUCCESS'
                        variable_results[var]['final_data_points'] = len(filtered_data)
                        print(f"✅ Successfully fetched {var}: {len(filtered_data)} days in date range")
                    else:
                        variable_results[var]['error_message'] = f"No data points in date range {start_date} to {end_date}"
                        print(f"❌ No data in date range for {var}")
                else:
                    variable_results[var]['error_message'] = "Could not extract time series from vectors"
                    print(f"❌ Could not extract time series for {var}")
            else:
                variable_results[var]['error_message'] = "No matching vectors found in Pinecone"
                print(f"❌ Could not find matching vectors for {var}")
        
        progress_bar.progress(1.0, text="Data fetching complete!")
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📊 VARIABLE FETCHING SUMMARY")
        print("=" * 80)
        
        successful_vars = []
        failed_vars = []
        
        for var, result in variable_results.items():
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_icon} {var:<35} | {result['status']:<7} | Source: {result['source']:<10}")
            
            if result['status'] == 'SUCCESS':
                successful_vars.append(var)
                if result['source'] == 'pinecone':
                    print(f"    └─ Vectors: {result['vectors_found']:<6} | Raw Points: {result['data_points_extracted']:<6} | Final: {result['final_data_points']}")
                else:
                    print(f"    └─ Final Data Points: {result['final_data_points']}")
            else:
                failed_vars.append(var)
                if result['vectors_found'] > 0:
                    print(f"    └─ Vectors: {result['vectors_found']:<6} | Raw Points: {result['data_points_extracted']:<6} | Error: {result['error_message']}")
                else:
                    print(f"    └─ Error: {result['error_message']}")
        
        print("-" * 80)
        success_rate = len(successful_vars) / len(variables) * 100
        print(f"📈 SUCCESS RATE: {len(successful_vars)}/{len(variables)} variables ({success_rate:.1f}%)")
        
        if successful_vars:
            print(f"✅ SUCCESSFUL: {', '.join(successful_vars)}")
        
        if failed_vars:
            print(f"❌ FAILED: {', '.join(failed_vars)}")
        
        print("=" * 80)
        
        if not data_frames:
            raise ValueError("No data could be fetched for any variables")
        
        # Normalize all datetime indexes to be timezone-naive before combining
        normalized_data = {}
        for var, data in data_frames.items():
            # Convert timezone-aware indexes to timezone-naive
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            normalized_data[var] = data
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(normalized_data)
        
        # Forward fill missing values and drop any remaining NaN rows
        combined_df = combined_df.ffill().dropna()
        
        print(f"🎯 FINAL RESULT: Combined DataFrame with {len(combined_df)} days and {len(combined_df.columns)} variables")
        
        # Cache the results
        self._historical_data_cache[cache_key] = combined_df
        self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
        
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
    
    def simulate_multi_factor_series(self, historical_data, n_simulations, simulation_days, correlation_matrix=None, market_condition=None, regime_scenarios=None):
        """
        Generate correlated synthetic paths for multiple variables using Cholesky decomposition
        
        Args:
            historical_data: DataFrame with historical data
            n_simulations: Number of simulation paths
            simulation_days: Number of days to simulate
            correlation_matrix: Pre-calculated correlation matrix (optional)
            market_condition: Market condition to apply consistent adjustments across all variables (legacy)
            regime_scenarios: Dictionary mapping variable names to MarketCondition scenarios (new)
            
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
        
        # Apply regime-specific adjustments per variable
        if regime_scenarios is not None:
            from .market_conditions import adjust_mu_sigma_for_condition
            print(f"Applying regime-specific scenarios to variables")
            
            adjusted_params = []
            for i, var in enumerate(variables):
                if var in regime_scenarios:
                    scenario = regime_scenarios[var]
                    mu_adjusted, sigma_adjusted = adjust_mu_sigma_for_condition(
                        mean_returns[i], volatilities[i], scenario
                    )
                    print(f"  {var}: {scenario.value}")
                    print(f"    μ {mean_returns[i]:.4f} -> {mu_adjusted:.4f}, σ {volatilities[i]:.4f} -> {sigma_adjusted:.4f}")
                else:
                    # Use base parameters if no scenario specified
                    mu_adjusted, sigma_adjusted = mean_returns[i], volatilities[i]
                    print(f"  {var}: Using base parameters (no scenario)")
                
                adjusted_params.append((mu_adjusted, sigma_adjusted))
            
            mean_returns = np.array([p[0] for p in adjusted_params])
            volatilities = np.array([p[1] for p in adjusted_params])
            
        # Fallback to legacy uniform market condition application
        elif market_condition is not None:
            from .market_conditions import adjust_mu_sigma_for_condition
            print(f"Applying uniform market condition: {market_condition.value}")
            
            adjusted_params = []
            for i, var in enumerate(variables):
                mu_adjusted, sigma_adjusted = adjust_mu_sigma_for_condition(
                    mean_returns[i], volatilities[i], market_condition
                )
                adjusted_params.append((mu_adjusted, sigma_adjusted))
                print(f"  {var}: μ {mean_returns[i]:.4f} -> {mu_adjusted:.4f}, σ {volatilities[i]:.4f} -> {sigma_adjusted:.4f}")
            
            mean_returns = np.array([p[0] for p in adjusted_params])
            volatilities = np.array([p[1] for p in adjusted_params])
        
        # Use provided correlation matrix or calculate from data
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix(historical_data)
        
        # Ensure correlation matrix is valid
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        # Ensure matrix is positive definite before Cholesky decomposition
        from correlation_utils import ensure_cholesky_ready
        correlation_matrix, was_corrected = ensure_cholesky_ready(correlation_matrix)
        
        if was_corrected:
            print("Matrix corrected for positive definiteness")
        
        # Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            print(f"Cholesky decomposition successful for {n_vars}x{n_vars} matrix")
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed even after correction, using identity matrix")
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