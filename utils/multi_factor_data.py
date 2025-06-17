"""
Multi-factor data fetching and simulation for macro-economic variables
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from openai import OpenAI

class MultiFactorDataFetcher:
    def __init__(self, pinecone_client=None):
        """Initialize the multi-factor data fetcher"""
        self.pinecone_client = pinecone_client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.intelligence_index_name = "intelligence-main"
        
        # Cache for found vectors to avoid repeated LLM calls
        self.vector_cache = {}
    
    def find_vector_for_variable(self, variable_name):
        """
        Use LLM to find the best matching vector in intelligence-main index for a given variable
        """
        if variable_name in self.vector_cache:
            return self.vector_cache[variable_name]
        
        if not self.pinecone_client:
            print(f"No Pinecone client available for {variable_name}")
            return None
        
        # Connect to intelligence-main index
        try:
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
        except Exception as e:
            print(f"Could not connect to intelligence-main index: {e}")
            return None
        
        # Search for vectors with semantic similarity to the variable name
        try:
            # Generate embedding for the variable name using OpenAI
            search_text = f"{variable_name} price data time series economic indicator"
            
            # Get sample vectors to understand the data structure
            dummy_vector = [0.0] * 1536  # intelligence-main uses 1536 dimensions
            response = intelligence_index.query(
                vector=dummy_vector,
                top_k=50,
                include_metadata=True
            )
            
            if not hasattr(response, 'matches') or not response.matches:
                print(f"No vectors found in intelligence-main for {variable_name}")
                return None
            
            # Collect vector descriptions for LLM analysis
            vector_options = []
            for match in response.matches[:20]:  # Analyze top 20
                if hasattr(match, 'metadata') and match.metadata:
                    vector_id = match.id
                    metadata = match.metadata
                    name = metadata.get('name', metadata.get('symbol', metadata.get('title', vector_id)))
                    description = metadata.get('description', metadata.get('info', 'No description'))
                    
                    vector_options.append({
                        'id': vector_id,
                        'name': name,
                        'description': description,
                        'metadata': metadata
                    })
            
            if not vector_options:
                print(f"No valid vectors with metadata found for {variable_name}")
                return None
            
            # Use LLM to find the best match
            prompt = f"""Given this list of available financial data vectors, find the best match for "{variable_name}":

Available vectors:
{json.dumps(vector_options[:10], indent=2)}

Find the vector that best matches "{variable_name}". Look for:
- WTI/Oil: crude oil, petroleum, energy commodities
- TIPS/Treasury: treasury bonds, government bonds, interest rates, yields
- CPI: consumer price index, inflation, price levels
- DXY: dollar index, currency, USD strength
- Gold: precious metals, commodities
- VIX: volatility index, market fear
- Bitcoin/BTC: cryptocurrency, digital assets

Respond with JSON:
{{
  "best_match_id": "vector_id_here",
  "confidence": 0.8,
  "reasoning": "explanation"
}}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at matching financial variables to data sources. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            best_match_id = result.get('best_match_id')
            confidence = result.get('confidence', 0)
            
            print(f"LLM Variable Matching for {variable_name}:")
            print(f"  Best match: {best_match_id}")
            print(f"  Confidence: {confidence}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
            
            if confidence >= 0.6:  # Only use if confident
                # Find the full vector data
                matched_vector = next((v for v in vector_options if v['id'] == best_match_id), None)
                if matched_vector:
                    self.vector_cache[variable_name] = matched_vector
                    return matched_vector
            
            print(f"No confident match found for {variable_name}")
            return None
            
        except Exception as e:
            print(f"Error finding vector for {variable_name}: {e}")
            return None
    
    def fetch_data_from_pinecone(self, vector_info):
        """
        Extract time series data from a Pinecone vector
        """
        try:
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
            
            # Fetch the specific vector
            vector_response = intelligence_index.fetch([vector_info['id']])
            
            if not hasattr(vector_response, 'vectors') or not vector_response.vectors:
                print(f"Could not fetch vector {vector_info['id']}")
                return None
            
            vector_data = vector_response.vectors[vector_info['id']]
            metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else {}
            
            # Extract time series data from metadata
            time_series_data = metadata.get('time_series', metadata.get('data', metadata.get('values')))
            
            if not time_series_data:
                print(f"No time series data found in vector {vector_info['id']}")
                return None
            
            # Convert to pandas DataFrame
            if isinstance(time_series_data, dict):
                df = pd.DataFrame.from_dict(time_series_data, orient='index')
            elif isinstance(time_series_data, list):
                df = pd.DataFrame(time_series_data)
            else:
                print(f"Unrecognized time series format for {vector_info['id']}")
                return None
            
            # Ensure proper date index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    # If index conversion fails, use dates from first column if available
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    else:
                        print(f"Could not parse dates for {vector_info['id']}")
                        return None
            
            # Get the main value column
            value_col = None
            for col in ['close', 'value', 'price', 'rate', 'level']:
                if col in df.columns:
                    value_col = col
                    break
            
            if value_col is None and len(df.columns) == 1:
                value_col = df.columns[0]
            
            if value_col is None:
                print(f"Could not identify value column for {vector_info['id']}")
                return None
            
            return df[value_col].dropna()
            
        except Exception as e:
            print(f"Error fetching data from Pinecone for {vector_info['id']}: {e}")
            return None
    
    def fetch_multi_factor_data(self, variables, start_date=None, end_date=None):
        """
        Fetch historical data for multiple variables from intelligence-main Pinecone index
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching multi-factor data for {variables} from intelligence-main")
        
        data_frames = {}
        
        for var in variables:
            if var == 'BTC':
                # Handle Bitcoin separately using yfinance as fallback
                print(f"Fetching {var} from yfinance...")
                try:
                    ticker = yf.Ticker('BTC-USD')
                    btc_data = ticker.history(start=start_date, end=end_date)
                    if not btc_data.empty:
                        data_frames[var] = btc_data['Close']
                        print(f"Successfully fetched {var}: {len(btc_data)} days")
                    else:
                        print(f"No Bitcoin data found")
                except Exception as e:
                    print(f"Error fetching Bitcoin data: {e}")
                continue
            
            # Find matching vector in intelligence-main
            print(f"Finding vector for {var}...")
            vector_info = self.find_vector_for_variable(var)
            
            if vector_info:
                # Fetch data from Pinecone
                print(f"Fetching {var} data from Pinecone vector {vector_info['id']}...")
                series_data = self.fetch_data_from_pinecone(vector_info)
                
                if series_data is not None and len(series_data) > 0:
                    # Filter by date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    filtered_data = series_data[
                        (series_data.index >= start_dt) & 
                        (series_data.index <= end_dt)
                    ]
                    
                    if len(filtered_data) > 0:
                        data_frames[var] = filtered_data
                        print(f"Successfully fetched {var}: {len(filtered_data)} days")
                    else:
                        print(f"No data in date range for {var}")
                else:
                    print(f"Could not extract time series for {var}")
            else:
                print(f"Could not find matching vector for {var}")
        
        if not data_frames:
            raise ValueError("No data could be fetched for any variables")
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(data_frames)
        
        # Forward fill missing values and drop any remaining NaN rows
        combined_df = combined_df.fillna(method='ffill').dropna()
        
        print(f"Successfully fetched multi-factor data: {len(combined_df)} days, {len(combined_df.columns)} variables")
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