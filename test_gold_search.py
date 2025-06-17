#!/usr/bin/env python3
"""
Direct test to find and extract gold data from intelligence-main
"""

from utils.pinecone_client import PineconeClient
import os
import pandas as pd
import json
from datetime import datetime

def find_gold_directly():
    """Find gold data using direct metadata search"""
    client = PineconeClient(os.getenv('PINECONE_API_KEY'))
    intelligence_index = client.pc.Index('intelligence-main')
    
    print("Searching for gold data with exhaustive patterns...")
    
    # Use the successful search patterns from our earlier discovery
    search_patterns = [
        [0.001 if i % 5 == 0 else 0.0 for i in range(1536)],
        [0.001 if i % 11 == 0 else 0.0 for i in range(1536)], 
        [0.001 if i % 13 == 0 else 0.0 for i in range(1536)], 
        [0.001 if i % 17 == 0 else 0.0 for i in range(1536)],
        [0.0] * 1536,
    ]
    
    gold_vectors = []
    
    for pattern_idx, query_vector in enumerate(search_patterns):
        response = intelligence_index.query(
            vector=query_vector,
            top_k=100,
            include_metadata=True
        )
        
        if hasattr(response, 'matches') and response.matches:
            for match in response.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    excel_name = metadata.get('excel_name', '')
                    raw_text = metadata.get('raw_text', '')
                    
                    # Look for gold in excel_name or raw_text
                    if 'gold' in excel_name.lower() or 'gold' in raw_text.lower():
                        gold_vectors.append({
                            'id': match.id,
                            'excel_name': excel_name,
                            'raw_text': raw_text,
                            'metadata': metadata,
                            'pattern': pattern_idx
                        })
    
    # Remove duplicates
    unique_gold = list({v['id']: v for v in gold_vectors}.values())
    
    if unique_gold:
        print(f"Found {len(unique_gold)} gold vectors")
        
        # Pick the first gold vector and extract time series
        gold_vector = unique_gold[0]
        print(f"Using vector: {gold_vector['excel_name']}")
        
        # Extract time series data
        time_series = extract_time_series(unique_gold)
        
        if time_series is not None and len(time_series) > 0:
            print(f"Successfully extracted {len(time_series)} gold price points")
            print(f"Date range: {time_series.index[0]} to {time_series.index[-1]}")
            print(f"Price range: ${time_series.min():.2f} to ${time_series.max():.2f}")
            return gold_vector, time_series
        else:
            print("Failed to extract time series from gold vectors")
    else:
        print("No gold vectors found")
    
    return None, None

def extract_time_series(gold_vectors):
    """Extract time series from gold vectors"""
    data_points = []
    
    for vector in gold_vectors:
        raw_text = vector['raw_text']
        
        # Parse the raw_text format: "Date: 2023-09-15 00:00:00 | Gold Close Price (USD): 1923.7"
        try:
            if '|' in raw_text and 'Gold Close Price' in raw_text:
                parts = raw_text.split('|')
                date_part = parts[0].strip()
                price_part = parts[1].strip()
                
                # Extract date
                date_str = date_part.replace('Date:', '').strip()
                date = pd.to_datetime(date_str)
                
                # Extract price
                price_str = price_part.split(':')[-1].strip()
                price = float(price_str)
                
                data_points.append({'date': date, 'price': price})
                
        except Exception as e:
            print(f"Error parsing vector {vector['id']}: {e}")
            continue
    
    if data_points:
        # Convert to pandas series
        df = pd.DataFrame(data_points)
        df = df.sort_values('date')
        series = pd.Series(df['price'].values, index=df['date'], name='GOLD')
        return series
    
    return None

if __name__ == "__main__":
    gold_vector, gold_data = find_gold_directly()
    
    if gold_data is not None:
        print("\nGold data extraction successful!")
        print(f"Sample prices: {gold_data.head()}")
    else:
        print("\nGold data extraction failed")