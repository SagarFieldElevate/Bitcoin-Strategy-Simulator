"""
Enhanced data extractor for intelligence-main index
Properly extracts time series data for macro variables
"""
import pandas as pd
import pinecone
import os
from typing import Dict, List, Optional

class EnhancedDataExtractor:
    def __init__(self):
        """Initialize connection to intelligence-main index"""
        pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.intelligence_index = pc.Index('intelligence-main')
        
        # Mapping of variables to their exact excel_name in the index
        self.variable_mapping = {
            'SPY': 'SPY Daily Close Price',
            'VIX': 'CBOE Volatility Index (VIX)',
            'GOLD': 'Gold Daily Close Price',
            'TIPS_10Y': '10-Year TIPS Yield (%)',
            'WTI': None  # Will search for oil data
        }
    
    def extract_macro_data(self, variables: List[str]) -> Dict[str, pd.Series]:
        """
        Extract time series data for multiple macro variables
        
        Args:
            variables: List of variable names (SPY, VIX, GOLD, TIPS_10Y, WTI)
            
        Returns:
            Dictionary mapping variable names to pandas Series with time series data
        """
        results = {}
        
        for var in variables:
            if var == 'BTC':
                continue  # BTC handled separately
                
            print(f"Extracting {var} data...")
            
            if var in self.variable_mapping:
                excel_name = self.variable_mapping[var]
                if excel_name:
                    time_series = self._extract_by_excel_name(excel_name)
                    if time_series is not None and len(time_series) > 0:
                        results[var] = time_series
                        print(f"Successfully extracted {len(time_series)} data points for {var}")
                    else:
                        print(f"No data found for {var}")
                else:
                    # Search for WTI/oil data
                    oil_data = self._search_oil_data()
                    if oil_data is not None:
                        results[var] = oil_data
                        print(f"Successfully extracted {len(oil_data)} data points for {var}")
                    else:
                        print(f"No oil data found for {var}")
            else:
                print(f"Unknown variable: {var}")
        
        return results
    
    def _extract_by_excel_name(self, excel_name: str) -> Optional[pd.Series]:
        """Extract time series data for a specific excel_name"""
        time_series_data = []
        
        # Use multiple query vectors to comprehensively search the index
        for i in range(15):  # Increased coverage
            try:
                query_vec = [0.1 if j % (i+2) == 0 else 0.0 for j in range(1536)]
                
                response = self.intelligence_index.query(
                    vector=query_vec,
                    top_k=1000,
                    include_metadata=True
                )
                
                for match in response.matches:
                    metadata = match.metadata or {}
                    
                    if metadata.get('excel_name', '') == excel_name:
                        raw_text = metadata.get('raw_text', '')
                        parsed_data = self._parse_raw_text(raw_text)
                        
                        if parsed_data:
                            time_series_data.append(parsed_data)
                            
            except Exception as e:
                continue
        
        if time_series_data:
            # Convert to DataFrame, remove duplicates, and sort
            df = pd.DataFrame(time_series_data)
            df = df.drop_duplicates(subset=['date'])
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
            return df['value']
        
        return None
    
    def _parse_raw_text(self, raw_text: str) -> Optional[Dict]:
        """Parse various raw_text formats to extract date and value"""
        try:
            if 'Date:' not in raw_text:
                return None
            
            # Extract date
            date_str = raw_text.split('Date: ')[1].split(' |')[0]
            date = pd.to_datetime(date_str)
            
            # Extract value based on format
            value = None
            
            if 'Close:' in raw_text:
                # Format: Close: 41.99
                value_str = raw_text.split('Close: ')[1].split(' |')[0]
                value = float(value_str)
                
            elif 'Price (USD):' in raw_text:
                # Format: SPY Close Price (USD): 598.42
                value_str = raw_text.split('Price (USD): ')[1].split(' |')[0]
                value = float(value_str)
                
            elif 'Yield (%):' in raw_text:
                # Format: 10-Year TIPS Yield (%): 2.45
                value_str = raw_text.split('Yield (%): ')[1].split(' |')[0]
                value = float(value_str)
                
            elif 'Index:' in raw_text:
                # Format: CBOE Volatility Index (VIX): 18.5
                if 'VIX' in raw_text:
                    value_str = raw_text.split('VIX): ')[1].split(' |')[0] if 'VIX): ' in raw_text else raw_text.split('Index: ')[1].split(' |')[0]
                else:
                    value_str = raw_text.split('Index: ')[1].split(' |')[0]
                value = float(value_str)
                
            elif 'Fear & Greed Index:' in raw_text:
                # Format: Fear & Greed Index: 72
                value_str = raw_text.split('Fear & Greed Index: ')[1].split(' |')[0]
                value = float(value_str)
            
            if value is not None:
                return {'date': date, 'value': value}
                
        except (ValueError, IndexError, AttributeError):
            pass
        
        return None
    
    def _search_oil_data(self) -> Optional[pd.Series]:
        """Search for WTI/oil data in the index"""
        oil_patterns = ['oil', 'wti', 'crude', 'petroleum', 'energy']
        
        # Search for oil-related excel_names
        for i in range(10):
            try:
                query_vec = [0.1 if j % (i+3) == 0 else 0.0 for j in range(1536)]
                
                response = self.intelligence_index.query(
                    vector=query_vec,
                    top_k=200,
                    include_metadata=True
                )
                
                for match in response.matches:
                    metadata = match.metadata or {}
                    excel_name = metadata.get('excel_name', '').lower()
                    
                    if any(pattern in excel_name for pattern in oil_patterns):
                        print(f"Found potential oil data: {metadata.get('excel_name', '')}")
                        return self._extract_by_excel_name(metadata.get('excel_name', ''))
                        
            except Exception as e:
                continue
        
        return None
    
    def test_all_extractions(self) -> Dict[str, int]:
        """Test data extraction for all macro variables"""
        test_results = {}
        
        variables = ['SPY', 'VIX', 'GOLD', 'TIPS_10Y', 'WTI']
        extracted_data = self.extract_macro_data(variables)
        
        for var in variables:
            if var in extracted_data:
                test_results[var] = len(extracted_data[var])
            else:
                test_results[var] = 0
        
        return test_results

def test_macro_data_extraction():
    """Test the enhanced data extraction system"""
    extractor = EnhancedDataExtractor()
    
    print("TESTING MACRO DATA EXTRACTION")
    print("=" * 40)
    
    results = extractor.test_all_extractions()
    
    print(f"\nExtraction Results:")
    total_available = 0
    for var, count in results.items():
        status = "✓ Available" if count > 100 else "✗ Insufficient" if count > 0 else "✗ Not found"
        print(f"  {var}: {count} data points - {status}")
        if count > 100:
            total_available += 1
    
    print(f"\nSummary: {total_available}/5 macro variables have sufficient data")
    
    if total_available >= 4:
        print("✓ Most macro variables available - multi-factor strategies can proceed")
    else:
        print("⚠ Limited macro data - recommend BTC-only strategies")
    
    return results

if __name__ == "__main__":
    test_macro_data_extraction()