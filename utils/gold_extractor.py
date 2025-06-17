"""
Direct gold data extraction from intelligence-main index
"""
import pandas as pd
import numpy as np

def extract_gold_data_direct(pinecone_client):
    """
    Extract complete gold time series using confirmed working method
    """
    try:
        intelligence_index = pinecone_client.pc.Index('intelligence-main')
        
        # Use the search patterns that we confirmed find gold data
        search_patterns = [
            [0.001 if i % 5 == 0 else 0.0 for i in range(1536)],
            [0.001 if i % 11 == 0 else 0.0 for i in range(1536)], 
            [0.001 if i % 13 == 0 else 0.0 for i in range(1536)], 
            [0.001 if i % 17 == 0 else 0.0 for i in range(1536)],
            [0.001 if i % 19 == 0 else 0.0 for i in range(1536)],
        ]
        
        all_gold_data = []
        
        for query_vector in search_patterns:
            response = intelligence_index.query(
                vector=query_vector,
                top_k=200,
                include_metadata=True
            )
            
            if hasattr(response, 'matches') and response.matches:
                for match in response.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata = match.metadata
                        excel_name = metadata.get('excel_name', '').lower()
                        raw_text = metadata.get('raw_text', '')
                        
                        if 'gold' in excel_name and 'Gold Close Price' in raw_text:
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
            
            print(f"Successfully extracted gold data: {len(series)} points from {series.index[0].date()} to {series.index[-1].date()}")
            print(f"Price range: ${series.min():.2f} to ${series.max():.2f}")
            
            return series
        else:
            print("No gold data found")
            return None
            
    except Exception as e:
        print(f"Error extracting gold data: {e}")
        return None

def test_gold_extraction(pinecone_client):
    """Test the gold extraction function"""
    gold_data = extract_gold_data_direct(pinecone_client)
    
    if gold_data is not None:
        print(f"\nGold data extraction successful!")
        print(f"Sample prices:")
        print(gold_data.head())
        print(f"\nRecent prices:")
        print(gold_data.tail())
        return True
    else:
        print("Gold data extraction failed")
        return False