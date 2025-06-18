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
        """Initialize the multi-factor data fetcher with enhanced discovery capabilities"""
        self.pinecone_client = pinecone_client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.intelligence_index_name = "intelligence-main"
        
        # Cache for found vectors to avoid repeated LLM calls
        self.vector_cache = {}
        
        # Enhanced variable configurations for comprehensive LLM access
        self.enhanced_configs = {
            'WTI': {
                'search_terms': ['wti crude oil', 'west texas intermediate', 'crude oil price', 'oil futures'],
                'metadata_patterns': ['wti', 'crude', 'oil', 'petroleum', 'energy'],
                'excel_names': ['WTI', 'CRUDE_OIL', 'OIL_PRICE'],
                'symbols': ['WTI', 'CL', 'CRUDE']
            },
            'OIL': {
                'search_terms': ['crude oil', 'oil price', 'petroleum price', 'energy commodity'],
                'metadata_patterns': ['oil', 'crude', 'petroleum', 'energy'],
                'excel_names': ['OIL', 'CRUDE_OIL', 'OIL_PRICE'],
                'symbols': ['OIL', 'WTI', 'CL']
            },
            'GOLD': {
                'search_terms': ['gold price', 'gold commodity', 'precious metal gold', 'bullion'],
                'metadata_patterns': ['gold', 'precious', 'metal', 'bullion'],
                'excel_names': ['GOLD', 'GOLD_PRICE', 'XAU'],
                'symbols': ['GOLD', 'GLD', 'XAU']
            },
            'SPY': {
                'search_terms': ['s&p 500', 'spy etf', 'stock market index', 'equity index'],
                'metadata_patterns': ['spy', 'sp500', 'stock', 'index'],
                'excel_names': ['SPY', 'SP500', 'S&P500'],
                'symbols': ['SPY', 'SPX']
            },
            'VIX': {
                'search_terms': ['volatility index', 'vix index', 'fear gauge', 'market volatility'],
                'metadata_patterns': ['vix', 'volatility', 'fear'],
                'excel_names': ['VIX', 'VOLATILITY'],
                'symbols': ['VIX']
            },
            'TIPS': {
                'search_terms': ['treasury inflation protected securities', 'tips bonds', 'inflation bonds'],
                'metadata_patterns': ['tips', 'treasury', 'inflation', 'bond'],
                'excel_names': ['TIPS', 'TIPS_10Y'],
                'symbols': ['TIPS', 'TIP']
            }
        }
    
    def find_vector_for_variable_enhanced(self, variable_name):
        """
        Enhanced LLM-powered variable discovery with comprehensive search strategies
        """
        if not self.pinecone_client or not self.pinecone_client.pc:
            print(f"No Pinecone client available for {variable_name}")
            return None
        
        variable_upper = variable_name.upper()
        
        # Check cache first
        if variable_upper in self.vector_cache:
            return self.vector_cache[variable_upper]
        
        try:
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
        except Exception as e:
            print(f"Could not connect to intelligence-main index: {e}")
            return None
        
        print(f"Enhanced LLM search for {variable_name}...")
        
        # Get enhanced configuration
        config = self.enhanced_configs.get(variable_upper, {
            'search_terms': [variable_name.lower()],
            'metadata_patterns': [variable_name.lower()],
            'excel_names': [variable_upper],
            'symbols': [variable_upper]
        })
        
        all_candidates = []
        
        # Strategy 1: Exact metadata matching
        for excel_name in config['excel_names']:
            try:
                response = intelligence_index.query(
                    vector=[0.0] * 1536,
                    top_k=50,
                    include_metadata=True,
                    filter={"excel_name": {"$eq": excel_name}}
                )
                all_candidates.extend(self._process_search_response(response, "excel_exact"))
            except:
                pass
        
        # Strategy 2: Symbol matching
        for symbol in config['symbols']:
            try:
                response = intelligence_index.query(
                    vector=[0.0] * 1536,
                    top_k=50,
                    include_metadata=True,
                    filter={"symbol": {"$eq": symbol}}
                )
                all_candidates.extend(self._process_search_response(response, "symbol_exact"))
            except:
                pass
        
        # Strategy 3: LLM semantic search
        if self.openai_client:
            for search_term in config['search_terms']:
                try:
                    # Generate embedding
                    embedding_response = self.openai_client.embeddings.create(
                        input=f"{search_term} daily price data time series economic financial indicator",
                        model="text-embedding-ada-002"
                    )
                    
                    query_vector = embedding_response.data[0].embedding
                    
                    response = intelligence_index.query(
                        vector=query_vector,
                        top_k=50,
                        include_metadata=True
                    )
                    
                    all_candidates.extend(self._process_search_response(response, "semantic"))
                    
                except Exception as e:
                    print(f"LLM semantic search failed for {search_term}: {e}")
                    continue
        
        # Strategy 4: Pattern matching in metadata
        for pattern in config['metadata_patterns']:
            try:
                response = intelligence_index.query(
                    vector=[0.0] * 1536,
                    top_k=100,
                    include_metadata=True,
                    filter={"description": {"$regex": f".*{pattern}.*"}}
                )
                all_candidates.extend(self._process_search_response(response, "pattern"))
            except:
                pass
        
        # Strategy 5: Comprehensive metadata scan
        try:
            response = intelligence_index.query(
                vector=[0.0] * 1536,
                top_k=500,
                include_metadata=True
            )
            
            for match in response.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    
                    # Check all text fields for our patterns
                    all_text = ' '.join([
                        str(metadata.get('excel_name', '')),
                        str(metadata.get('name', '')),
                        str(metadata.get('symbol', '')),
                        str(metadata.get('description', '')),
                        str(metadata.get('title', '')),
                        str(metadata.get('raw_text', ''))
                    ]).lower()
                    
                    # Check if any of our search patterns match
                    for pattern in config['metadata_patterns'] + [variable_name.lower()]:
                        if pattern in all_text:
                            all_candidates.append({
                                'id': match.id,
                                'metadata': metadata,
                                'score': getattr(match, 'score', 0),
                                'match_type': 'comprehensive',
                                'matched_pattern': pattern
                            })
                            break
        
        except Exception as e:
            print(f"Comprehensive scan failed: {e}")
        
        if not all_candidates:
            print(f"No vectors found for {variable_name}")
            return None
        
        # Rank candidates by relevance
        best_candidate = self._rank_and_select_best(all_candidates, config, variable_name)
        
        if best_candidate:
            # Cache the result
            self.vector_cache[variable_upper] = best_candidate
            print(f"Best match for {variable_name}: {best_candidate['id']} (confidence: {best_candidate.get('confidence', 0):.2f})")
            return best_candidate
        
        return None
    
    def _process_search_response(self, response, match_type):
        """Process Pinecone search response"""
        candidates = []
        
        if hasattr(response, 'matches') and response.matches:
            for match in response.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    candidates.append({
                        'id': match.id,
                        'metadata': match.metadata,
                        'score': getattr(match, 'score', 0),
                        'match_type': match_type
                    })
        
        return candidates
    
    def _rank_and_select_best(self, candidates, config, variable_name):
        """Rank candidates and select the best match"""
        if not candidates:
            return None
        
        # Remove duplicates
        unique_candidates = {}
        for candidate in candidates:
            vector_id = candidate['id']
            if vector_id not in unique_candidates or candidate['score'] > unique_candidates[vector_id]['score']:
                unique_candidates[vector_id] = candidate
        
        candidates = list(unique_candidates.values())
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_relevance_score(candidate, config, variable_name)
            scored_candidates.append({
                **candidate,
                'confidence': score
            })
        
        # Sort by confidence
        scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return best if above threshold
        if scored_candidates and scored_candidates[0]['confidence'] > 0.3:
            return scored_candidates[0]
        
        return None
    
    def _calculate_relevance_score(self, candidate, config, variable_name):
        """Calculate relevance score for a candidate"""
        score = 0.0
        metadata = candidate.get('metadata', {})
        
        # Base score from match type
        match_type_scores = {
            'excel_exact': 1.0,
            'symbol_exact': 0.9,
            'semantic': 0.8,
            'pattern': 0.6,
            'comprehensive': 0.4
        }
        
        score += match_type_scores.get(candidate.get('match_type', ''), 0.2)
        
        # Exact matches bonus
        excel_name = str(metadata.get('excel_name', '')).upper()
        symbol = str(metadata.get('symbol', '')).upper()
        
        if excel_name in config['excel_names']:
            score += 0.4
        if symbol in config['symbols']:
            score += 0.3
        
        # Time series data bonus
        has_time_series = any(key in metadata for key in [
            'time_series', 'data', 'values', 'price_data', 'historical_data'
        ])
        if has_time_series:
            score += 0.3
        
        # Variable name exact match bonus
        if variable_name.upper() in excel_name or variable_name.upper() in symbol:
            score += 0.2
        
        return min(score, 1.0)

    def find_vector_for_variable(self, variable_name):
        """
        Enhanced LLM-powered variable discovery - now uses the comprehensive search system
        """
        # Use the enhanced method for all variables
        return self.find_vector_for_variable_enhanced(variable_name)
    
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
    
    def fetch_wti_data_direct(self):
        """
        Direct extraction of WTI/crude oil data using multiple search strategies
        """
        try:
            intelligence_index = self.pinecone_client.client.Index("intelligence-main")
            
            # Multiple search strategies for WTI/crude oil data
            search_strategies = [
                # Strategy 1: Direct metadata search for oil-related terms
                {"filter": {"excel_name": {"$in": ["WTI", "CRUDE", "OIL", "BRENT"]}}},
                {"filter": {"name": {"$regex": ".*oil.*"}}},
                {"filter": {"symbol": {"$in": ["WTI", "CRUDE", "CLF", "CL"]}}},
                {"filter": {"description": {"$regex": ".*crude.*"}}},
                {"filter": {"title": {"$regex": ".*petroleum.*"}}},
            ]
            
            all_matches = []
            
            # Try each search strategy
            for strategy in search_strategies:
                try:
                    response = intelligence_index.query(
                        vector=[0.0] * 1536,  # Dummy vector for metadata search
                        top_k=100,
                        include_metadata=True,
                        **strategy
                    )
                    
                    if hasattr(response, 'matches') and response.matches:
                        for match in response.matches:
                            if hasattr(match, 'metadata') and match.metadata:
                                metadata = match.metadata
                                # Look for oil/crude related keywords
                                text_fields = [
                                    metadata.get('excel_name', ''),
                                    metadata.get('name', ''),
                                    metadata.get('symbol', ''),
                                    metadata.get('description', ''),
                                    metadata.get('title', ''),
                                    metadata.get('raw_text', '')
                                ]
                                
                                combined_text = ' '.join(str(field).lower() for field in text_fields)
                                
                                # Check for oil-related keywords
                                oil_keywords = ['wti', 'crude', 'oil', 'petroleum', 'brent', 'energy']
                                if any(keyword in combined_text for keyword in oil_keywords):
                                    all_matches.append({
                                        'id': match.id,
                                        'metadata': metadata,
                                        'score': match.score if hasattr(match, 'score') else 0,
                                        'text': combined_text
                                    })
                except Exception as e:
                    print(f"Search strategy failed: {e}")
                    continue
            
            if not all_matches:
                print("No WTI/crude oil data found in intelligence-main index")
                return None
            
            # Sort by relevance and try to extract data
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            for match in all_matches[:5]:  # Try top 5 matches
                try:
                    # Try to extract time series data
                    metadata = match['metadata']
                    
                    # Look for time series data in various formats
                    time_series_sources = [
                        metadata.get('time_series'),
                        metadata.get('data'),
                        metadata.get('values'),
                        metadata.get('price_data'),
                        metadata.get('historical_data')
                    ]
                    
                    for ts_data in time_series_sources:
                        if ts_data:
                            # Try to parse the time series data
                            if isinstance(ts_data, dict):
                                df = pd.DataFrame.from_dict(ts_data, orient='index')
                            elif isinstance(ts_data, list):
                                df = pd.DataFrame(ts_data)
                            else:
                                continue
                            
                            # Convert index to datetime if needed
                            if not isinstance(df.index, pd.DatetimeIndex):
                                try:
                                    df.index = pd.to_datetime(df.index)
                                except:
                                    if 'date' in df.columns:
                                        df['date'] = pd.to_datetime(df['date'])
                                        df = df.set_index('date')
                                    else:
                                        continue
                            
                            # Find value column
                            value_col = None
                            for col in ['close', 'value', 'price', 'rate', 'level', 'wti', 'crude']:
                                if col.lower() in [c.lower() for c in df.columns]:
                                    value_col = next(c for c in df.columns if c.lower() == col.lower())
                                    break
                            
                            if value_col is None and len(df.columns) == 1:
                                value_col = df.columns[0]
                            
                            if value_col and len(df) > 10:  # Ensure we have sufficient data
                                series = df[value_col].dropna()
                                if len(series) > 10:
                                    print(f"Successfully extracted WTI data: {len(series)} points from {match['id']}")
                                    return series
                                    
                except Exception as e:
                    print(f"Failed to extract data from {match['id']}: {e}")
                    continue
            
            print("Could not extract valid WTI time series data from any matches")
            return None
            
        except Exception as e:
            print(f"Error in WTI data extraction: {e}")
            return None

    def fetch_gold_data_direct(self):
        """
        Direct extraction of gold data using confirmed working patterns
        """
        try:
            intelligence_index = self.pinecone_client.pc.Index('intelligence-main')
            
            # Comprehensive search patterns to find all gold vectors
            search_patterns = [
                [0.001 if i % 2 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 3 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 5 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 7 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 11 == 0 else 0.0 for i in range(1536)], 
                [0.001 if i % 13 == 0 else 0.0 for i in range(1536)], 
                [0.001 if i % 17 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 19 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 23 == 0 else 0.0 for i in range(1536)],
                [0.001 if i % 29 == 0 else 0.0 for i in range(1536)],
                [0.0] * 1536,  # Also try zero vector
            ]
            
            all_gold_data = []
            
            for query_vector in search_patterns:
                response = intelligence_index.query(
                    vector=query_vector,
                    top_k=300,  # Cast wide net
                    include_metadata=True
                )
                
                if hasattr(response, 'matches') and response.matches:
                    for match in response.matches:
                        if hasattr(match, 'metadata') and match.metadata:
                            metadata = match.metadata
                            excel_name = metadata.get('excel_name', '').lower()
                            raw_text = metadata.get('raw_text', '')
                            
                            if ('gold' in excel_name or 'gold' in raw_text.lower()) and 'Gold Close Price' in raw_text:
                                try:
                                    # Parse: "Date: 2023-09-15 00:00:00 | Gold Close Price (USD): 1923.7"
                                    if '|' in raw_text:
                                        parts = raw_text.split('|')
                                        date_part = parts[0].strip().replace('Date:', '').strip()
                                        price_part = parts[1].strip()
                                        
                                        date = pd.to_datetime(date_part)
                                        price_str = price_part.split(':')[-1].strip()
                                        price = float(price_str)
                                        
                                        all_gold_data.append({'date': date, 'price': price})
                                except Exception:
                                    continue
            
            if all_gold_data:
                # Convert to DataFrame and clean
                df = pd.DataFrame(all_gold_data)
                df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
                
                # Create pandas Series with proper index
                series = pd.Series(df['price'].values, index=df['date'], name='GOLD')
                print(f"Successfully extracted gold data: {len(series)} points")
                return series
            else:
                print("No gold data found in comprehensive search")
                return None
                
        except Exception as e:
            print(f"Error in direct gold extraction: {e}")
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
            
            # Special handling for WTI/Oil variables
            if var.upper() in ['WTI', 'CRUDE', 'OIL', 'CL']:
                print(f"Fetching {var} using direct WTI extraction...")
                try:
                    wti_data = self.fetch_wti_data_direct()
                    if wti_data is not None and len(wti_data) > 0:
                        # Filter by date range
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        
                        filtered_data = wti_data[
                            (wti_data.index >= start_dt) & 
                            (wti_data.index <= end_dt)
                        ]
                        
                        if len(filtered_data) > 0:
                            data_frames[var] = filtered_data
                            print(f"Successfully fetched {var}: {len(filtered_data)} days")
                        else:
                            print(f"No WTI data in date range for {var}")
                    else:
                        print(f"Could not extract WTI data for {var}")
                except Exception as e:
                    print(f"Error fetching WTI data for {var}: {e}")
                continue
            
            # Special handling for GOLD variables
            if var.upper() in ['GOLD', 'GLD', 'XAU']:
                print(f"Fetching {var} using direct gold extraction...")
                try:
                    gold_data = self.fetch_gold_data_direct()
                    if gold_data is not None and len(gold_data) > 0:
                        # Filter by date range
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        
                        filtered_data = gold_data[
                            (gold_data.index >= start_dt) & 
                            (gold_data.index <= end_dt)
                        ]
                        
                        if len(filtered_data) > 0:
                            data_frames[var] = filtered_data
                            print(f"Successfully fetched {var}: {len(filtered_data)} days")
                        else:
                            print(f"No gold data in date range for {var}")
                    else:
                        print(f"Could not extract gold data for {var}")
                except Exception as e:
                    print(f"Error fetching gold data for {var}: {e}")
                continue
            
            # Find matching vector in intelligence-main for other variables
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