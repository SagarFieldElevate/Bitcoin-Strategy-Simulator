"""
Enhanced data discovery system for comprehensive variable access in intelligence-main index
Ensures all daily variables including WTI, GOLD, SPY, VIX, etc. can be properly accessed by LLM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import openai
import os
from typing import Dict, List, Optional, Tuple, Any


class EnhancedDataDiscovery:
    def __init__(self, pinecone_client=None):
        """Initialize enhanced data discovery with comprehensive variable mapping"""
        self.pinecone_client = pinecone_client
        self.intelligence_index_name = "intelligence-main"
        
        # Initialize OpenAI for LLM-powered searches
        self.openai_client = None
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        
        # Comprehensive variable configurations for all daily data
        self.variable_configs = {
            # Energy & Commodities
            'WTI': {
                'search_terms': ['wti', 'crude oil', 'west texas intermediate', 'petroleum', 'oil price'],
                'symbols': ['WTI', 'CL', 'CLF', 'CRUDE', 'OIL', 'USO'],
                'excel_names': ['WTI', 'CRUDE_OIL', 'OIL_PRICE', 'PETROLEUM'],
                'metadata_fields': ['crude', 'oil', 'petroleum', 'energy', 'commodity'],
                'data_types': ['price', 'close', 'value', 'rate'],
                'priority_keywords': ['daily', 'spot', 'futures', 'front month']
            },
            'OIL': {
                'search_terms': ['oil', 'crude oil', 'petroleum', 'energy commodity'],
                'symbols': ['OIL', 'WTI', 'CL', 'USO', 'CRUDE'],
                'excel_names': ['OIL', 'CRUDE_OIL', 'OIL_PRICE'],
                'metadata_fields': ['oil', 'crude', 'petroleum', 'energy'],
                'data_types': ['price', 'close', 'value'],
                'priority_keywords': ['daily', 'spot']
            },
            'CRUDE': {
                'search_terms': ['crude oil', 'crude', 'petroleum', 'wti', 'brent'],
                'symbols': ['CRUDE', 'WTI', 'CL', 'OIL'],
                'excel_names': ['CRUDE', 'CRUDE_OIL', 'WTI'],
                'metadata_fields': ['crude', 'oil', 'petroleum'],
                'data_types': ['price', 'close', 'value'],
                'priority_keywords': ['daily', 'spot']
            },
            
            # Precious Metals
            'GOLD': {
                'search_terms': ['gold', 'gold price', 'precious metal', 'bullion'],
                'symbols': ['GOLD', 'GLD', 'XAU', 'XAUUSD', 'GC'],
                'excel_names': ['GOLD', 'GOLD_PRICE', 'XAU', 'GLD'],
                'metadata_fields': ['gold', 'precious', 'metal', 'bullion'],
                'data_types': ['price', 'close', 'value'],
                'priority_keywords': ['daily', 'spot', 'london fix']
            },
            'GLD': {
                'search_terms': ['gld', 'gold etf', 'gold fund', 'gold'],
                'symbols': ['GLD', 'GOLD', 'XAU'],
                'excel_names': ['GLD', 'GOLD', 'GOLD_ETF'],
                'metadata_fields': ['gld', 'gold', 'etf'],
                'data_types': ['price', 'close', 'nav'],
                'priority_keywords': ['daily', 'etf']
            },
            'SILVER': {
                'search_terms': ['silver', 'silver price', 'precious metal'],
                'symbols': ['SILVER', 'SLV', 'XAG', 'XAGUSD'],
                'excel_names': ['SILVER', 'SILVER_PRICE', 'XAG'],
                'metadata_fields': ['silver', 'precious', 'metal'],
                'data_types': ['price', 'close', 'value'],
                'priority_keywords': ['daily', 'spot']
            },
            
            # Equity Indices
            'SPY': {
                'search_terms': ['spy', 's&p 500', 'sp500', 'stock index'],
                'symbols': ['SPY', 'SP500', 'SPX', 'GSPC'],
                'excel_names': ['SPY', 'SP500', 'S&P500', 'SPX'],
                'metadata_fields': ['spy', 'sp500', 'stock', 'index'],
                'data_types': ['price', 'close', 'value', 'index'],
                'priority_keywords': ['daily', 'index', 'equity']
            },
            'QQQ': {
                'search_terms': ['qqq', 'nasdaq 100', 'tech index', 'nasdaq'],
                'symbols': ['QQQ', 'NASDAQ', 'NDX', 'IXIC'],
                'excel_names': ['QQQ', 'NASDAQ', 'NASDAQ100'],
                'metadata_fields': ['qqq', 'nasdaq', 'tech', 'index'],
                'data_types': ['price', 'close', 'value', 'index'],
                'priority_keywords': ['daily', 'index', 'tech']
            },
            
            # Volatility
            'VIX': {
                'search_terms': ['vix', 'volatility index', 'fear index', 'cboe vix'],
                'symbols': ['VIX', 'VOLATILITY', 'FEAR'],
                'excel_names': ['VIX', 'VOLATILITY', 'FEAR_INDEX'],
                'metadata_fields': ['vix', 'volatility', 'fear', 'cboe'],
                'data_types': ['index', 'value', 'level'],
                'priority_keywords': ['daily', 'volatility', 'index']
            },
            
            # Fixed Income
            'TIPS': {
                'search_terms': ['tips', 'treasury inflation protected', 'inflation bonds'],
                'symbols': ['TIPS', 'TREASURY', 'TIP', 'SCHP'],
                'excel_names': ['TIPS', 'TIPS_10Y', 'TREASURY_TIPS'],
                'metadata_fields': ['tips', 'treasury', 'inflation', 'bond'],
                'data_types': ['yield', 'price', 'value', 'rate'],
                'priority_keywords': ['daily', '10 year', 'real yield']
            },
            'TNX': {
                'search_terms': ['10 year treasury', 'treasury yield', 'government bond'],
                'symbols': ['TNX', '10Y', 'TREASURY', 'DGS10'],
                'excel_names': ['TNX', '10Y', 'TREASURY_10Y'],
                'metadata_fields': ['treasury', '10 year', 'yield', 'bond'],
                'data_types': ['yield', 'rate', 'value'],
                'priority_keywords': ['daily', '10 year', 'yield']
            },
            
            # Currencies
            'DXY': {
                'search_terms': ['dollar index', 'usd index', 'dxy', 'us dollar'],
                'symbols': ['DXY', 'USD', 'DOLLAR', 'USDX'],
                'excel_names': ['DXY', 'USD_INDEX', 'DOLLAR_INDEX'],
                'metadata_fields': ['dxy', 'dollar', 'usd', 'currency'],
                'data_types': ['index', 'value', 'rate'],
                'priority_keywords': ['daily', 'index', 'currency']
            },
            'EUR': {
                'search_terms': ['euro', 'eurusd', 'eur usd', 'european currency'],
                'symbols': ['EUR', 'EURUSD', 'EURO'],
                'excel_names': ['EUR', 'EURUSD', 'EUR_USD'],
                'metadata_fields': ['eur', 'euro', 'eurusd', 'currency'],
                'data_types': ['rate', 'price', 'value'],
                'priority_keywords': ['daily', 'fx', 'currency']
            }
        }
    
    def comprehensive_variable_search(self, variable_name: str) -> Optional[Dict]:
        """
        Comprehensive search for any variable using multiple strategies
        """
        if not self.pinecone_client:
            print(f"No Pinecone client available for {variable_name}")
            return None
        
        try:
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
        except Exception as e:
            print(f"Could not connect to intelligence-main index: {e}")
            return None
        
        variable_upper = variable_name.upper()
        print(f"Comprehensive search for {variable_name}...")
        
        # Get configuration for the variable
        config = self.variable_configs.get(variable_upper, self._create_generic_config(variable_name))
        
        # Strategy 1: Direct metadata filtering
        candidates = self._search_by_metadata_filters(intelligence_index, config)
        
        # Strategy 2: Semantic search with LLM embeddings
        if not candidates and self.openai_client:
            candidates.extend(self._search_by_semantic_similarity(intelligence_index, config))
        
        # Strategy 3: Comprehensive text search
        if not candidates:
            candidates.extend(self._search_by_text_patterns(intelligence_index, config))
        
        # Strategy 4: Brute force metadata exploration
        if not candidates:
            candidates.extend(self._search_by_brute_force(intelligence_index, config))
        
        if not candidates:
            print(f"No candidates found for {variable_name}")
            return None
        
        # Rank and select best candidate
        best_candidate = self._rank_candidates(candidates, config)
        
        if best_candidate:
            print(f"Best match for {variable_name}: {best_candidate['id']} (confidence: {best_candidate['confidence']:.2f})")
            return best_candidate
        
        return None
    
    def _create_generic_config(self, variable_name: str) -> Dict:
        """Create generic configuration for unknown variables"""
        return {
            'search_terms': [variable_name.lower(), variable_name.upper()],
            'symbols': [variable_name.upper(), variable_name],
            'excel_names': [variable_name.upper(), variable_name],
            'metadata_fields': [variable_name.lower()],
            'data_types': ['price', 'value', 'close', 'rate', 'index'],
            'priority_keywords': ['daily', 'price', 'data']
        }
    
    def _search_by_metadata_filters(self, index, config: Dict) -> List[Dict]:
        """Search using direct metadata filtering"""
        candidates = []
        
        # Search by excel_name
        for excel_name in config['excel_names']:
            try:
                response = index.query(
                    vector=[0.0] * 1536,
                    top_k=50,
                    include_metadata=True,
                    filter={"excel_name": {"$eq": excel_name}}
                )
                candidates.extend(self._process_query_response(response, "excel_name_exact"))
            except:
                pass
        
        # Search by symbol
        for symbol in config['symbols']:
            try:
                response = index.query(
                    vector=[0.0] * 1536,
                    top_k=50,
                    include_metadata=True,
                    filter={"symbol": {"$eq": symbol}}
                )
                candidates.extend(self._process_query_response(response, "symbol_exact"))
            except:
                pass
        
        return candidates
    
    def _search_by_semantic_similarity(self, index, config: Dict) -> List[Dict]:
        """Search using LLM-generated embeddings for semantic similarity"""
        candidates = []
        
        if not self.openai_client:
            return candidates
        
        for search_term in config['search_terms']:
            try:
                # Generate embedding using OpenAI
                embedding_response = self.openai_client.embeddings.create(
                    input=f"{search_term} daily price data time series economic indicator",
                    model="text-embedding-ada-002"
                )
                
                query_vector = embedding_response.data[0].embedding
                
                response = index.query(
                    vector=query_vector,
                    top_k=50,
                    include_metadata=True
                )
                
                candidates.extend(self._process_query_response(response, "semantic"))
                
            except Exception as e:
                print(f"Semantic search failed for {search_term}: {e}")
                continue
        
        return candidates
    
    def _search_by_text_patterns(self, index, config: Dict) -> List[Dict]:
        """Search using text pattern matching in metadata"""
        candidates = []
        
        # Search in description field
        for field in config['metadata_fields']:
            try:
                response = index.query(
                    vector=[0.0] * 1536,
                    top_k=100,
                    include_metadata=True,
                    filter={"description": {"$regex": f".*{field}.*"}}
                )
                candidates.extend(self._process_query_response(response, "description_pattern"))
            except:
                pass
        
        # Search in name field
        for field in config['metadata_fields']:
            try:
                response = index.query(
                    vector=[0.0] * 1536,
                    top_k=100,
                    include_metadata=True,
                    filter={"name": {"$regex": f".*{field}.*"}}
                )
                candidates.extend(self._process_query_response(response, "name_pattern"))
            except:
                pass
        
        return candidates
    
    def _search_by_brute_force(self, index, config: Dict) -> List[Dict]:
        """Brute force search through all metadata"""
        candidates = []
        
        try:
            # Get a large sample of vectors
            response = index.query(
                vector=[0.0] * 1536,
                top_k=1000,
                include_metadata=True
            )
            
            for match in response.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    
                    # Check all metadata fields for any of our search terms
                    all_text = ' '.join([
                        str(metadata.get('excel_name', '')),
                        str(metadata.get('name', '')),
                        str(metadata.get('symbol', '')),
                        str(metadata.get('description', '')),
                        str(metadata.get('title', '')),
                        str(metadata.get('raw_text', ''))
                    ]).lower()
                    
                    # Check if any search terms match
                    for term in config['search_terms'] + config['symbols'] + config['excel_names']:
                        if term.lower() in all_text:
                            candidates.append({
                                'id': match.id,
                                'metadata': metadata,
                                'score': getattr(match, 'score', 0),
                                'match_type': 'brute_force',
                                'matched_term': term
                            })
                            break
        
        except Exception as e:
            print(f"Brute force search failed: {e}")
        
        return candidates
    
    def _process_query_response(self, response, match_type: str) -> List[Dict]:
        """Process Pinecone query response into standardized format"""
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
    
    def _rank_candidates(self, candidates: List[Dict], config: Dict) -> Optional[Dict]:
        """Rank candidates and return the best match"""
        if not candidates:
            return None
        
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_candidate_score(candidate, config)
            scored_candidates.append({
                **candidate,
                'confidence': score
            })
        
        # Sort by confidence score
        scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return best candidate if confidence is above threshold
        best = scored_candidates[0]
        if best['confidence'] > 0.3:  # Minimum confidence threshold
            return best
        
        return None
    
    def _calculate_candidate_score(self, candidate: Dict, config: Dict) -> float:
        """Calculate confidence score for a candidate"""
        score = 0.0
        metadata = candidate.get('metadata', {})
        
        # Base score from match type
        match_type_scores = {
            'excel_name_exact': 1.0,
            'symbol_exact': 0.9,
            'semantic': 0.8,
            'description_pattern': 0.7,
            'name_pattern': 0.6,
            'brute_force': 0.4
        }
        
        score += match_type_scores.get(candidate.get('match_type', ''), 0.2)
        
        # Bonus for exact symbol matches
        symbol = metadata.get('symbol', '').upper()
        if symbol in config['symbols']:
            score += 0.3
        
        # Bonus for excel_name matches
        excel_name = metadata.get('excel_name', '').upper()
        if excel_name in config['excel_names']:
            score += 0.4
        
        # Bonus for having time series data
        if any(key in metadata for key in ['time_series', 'data', 'values', 'price_data']):
            score += 0.3
        
        # Bonus for priority keywords
        all_text = ' '.join([
            str(metadata.get('description', '')),
            str(metadata.get('name', '')),
            str(metadata.get('raw_text', ''))
        ]).lower()
        
        for keyword in config['priority_keywords']:
            if keyword in all_text:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_time_series_data(self, vector_info: Dict) -> Optional[pd.Series]:
        """Extract time series data from a discovered vector"""
        if not vector_info or not self.pinecone_client:
            return None
        
        try:
            # Get the vector data
            intelligence_index = self.pinecone_client.pc.Index(self.intelligence_index_name)
            response = intelligence_index.fetch(ids=[vector_info['id']])
            
            if not response.vectors or vector_info['id'] not in response.vectors:
                print(f"Could not fetch vector {vector_info['id']}")
                return None
            
            vector_data = response.vectors[vector_info['id']]
            metadata = vector_data.metadata
            
            # Look for time series data in various formats
            time_series_sources = [
                metadata.get('time_series'),
                metadata.get('data'),
                metadata.get('values'),
                metadata.get('price_data'),
                metadata.get('historical_data'),
                metadata.get('daily_data')
            ]
            
            for ts_data in time_series_sources:
                if ts_data:
                    series = self._parse_time_series(ts_data, vector_info['id'])
                    if series is not None and len(series) > 10:
                        return series
            
            # Try parsing raw_text if structured data not found
            raw_text = metadata.get('raw_text', '')
            if raw_text:
                series = self._parse_raw_text_series(raw_text)
                if series is not None and len(series) > 10:
                    return series
            
            print(f"No valid time series data found in {vector_info['id']}")
            return None
            
        except Exception as e:
            print(f"Error extracting time series from {vector_info['id']}: {e}")
            return None
    
    def _parse_time_series(self, ts_data, vector_id: str) -> Optional[pd.Series]:
        """Parse time series data from various formats"""
        try:
            if isinstance(ts_data, dict):
                df = pd.DataFrame.from_dict(ts_data, orient='index')
            elif isinstance(ts_data, list):
                df = pd.DataFrame(ts_data)
            elif isinstance(ts_data, str):
                # Try to parse JSON string
                import json
                data = json.loads(ts_data)
                if isinstance(data, dict):
                    df = pd.DataFrame.from_dict(data, orient='index')
                else:
                    df = pd.DataFrame(data)
            else:
                return None
            
            # Convert index to datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    else:
                        return None
            
            # Find value column
            value_col = self._identify_value_column(df)
            if value_col is None:
                return None
            
            series = df[value_col].dropna()
            if len(series) > 0:
                print(f"Successfully parsed time series: {len(series)} points from {vector_id}")
                return series
            
        except Exception as e:
            print(f"Error parsing time series from {vector_id}: {e}")
        
        return None
    
    def _parse_raw_text_series(self, raw_text: str) -> Optional[pd.Series]:
        """Parse time series from raw text format"""
        try:
            # Try various parsing strategies for raw text
            lines = raw_text.strip().split('\n')
            data_points = []
            
            for line in lines:
                # Look for date-value patterns
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # Try to parse date and value
                        date_str = parts[0]
                        value_str = parts[-1]  # Last part is usually the value
                        
                        date = pd.to_datetime(date_str)
                        value = float(value_str.replace(',', '').replace('$', ''))
                        
                        data_points.append((date, value))
                    except:
                        continue
            
            if len(data_points) > 10:
                dates, values = zip(*data_points)
                series = pd.Series(values, index=dates)
                return series.sort_index()
            
        except Exception as e:
            print(f"Error parsing raw text series: {e}")
        
        return None
    
    def _identify_value_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the main value column in a DataFrame"""
        # Priority order for column names
        priority_cols = ['close', 'value', 'price', 'rate', 'level', 'index', 'adj_close', 'last']
        
        # Check for exact matches first
        for col_name in priority_cols:
            if col_name in df.columns:
                return col_name
        
        # Check for case-insensitive matches
        df_cols_lower = [col.lower() for col in df.columns]
        for col_name in priority_cols:
            if col_name in df_cols_lower:
                return df.columns[df_cols_lower.index(col_name)]
        
        # If only one numeric column, use it
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        
        # Default to first numeric column
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        return None


def test_enhanced_discovery():
    """Test the enhanced data discovery system"""
    # This would be called with actual Pinecone client
    print("Enhanced Data Discovery System initialized")
    print("Supports comprehensive search for all daily variables including:")
    print("- Energy: WTI, OIL, CRUDE")
    print("- Metals: GOLD, SILVER, GLD")
    print("- Indices: SPY, QQQ, VIX")
    print("- Bonds: TIPS, TNX")
    print("- Currencies: DXY, EUR")
    print("- And any other variables with intelligent fallback")


if __name__ == "__main__":
    test_enhanced_discovery()