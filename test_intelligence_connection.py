"""
Test intelligence-main connection and MultiFactorDataFetcher functionality
"""

import os
from utils.multi_factor_data import MultiFactorDataFetcher
from utils.pinecone_client import PineconeClient

def test_intelligence_connection():
    """Test complete connection to intelligence-main and data fetching"""
    print("TESTING INTELLIGENCE-MAIN CONNECTION")
    print("=" * 50)
    
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ No PINECONE_API_KEY found")
        return
    
    print("1. Testing PineconeClient initialization...")
    pinecone_client = PineconeClient(api_key)
    if pinecone_client.connected:
        print("✅ PineconeClient connected to bitcoin-strategies")
    else:
        print("❌ PineconeClient failed to connect")
        return
    
    print("2. Testing MultiFactorDataFetcher initialization...")
    fetcher = MultiFactorDataFetcher(pinecone_client)
    
    print("3. Testing intelligence-main index access...")
    try:
        intelligence_index = pinecone_client.pc.Index("intelligence-main")
        print("✅ Successfully accessed intelligence-main index")
    except Exception as e:
        print(f"❌ Failed to access intelligence-main: {e}")
        return
    
    print("4. Testing variable search functionality...")
    test_variables = ["Bitcoin Daily Close Price", "QQQ Daily Close Price", "WTI Crude Oil Price (USD/Barrel)"]
    
    for variable in test_variables:
        print(f"\nTesting variable: {variable}")
        try:
            vector_info = fetcher.find_vector_for_variable(variable)
            if vector_info:
                print(f"✅ Found vector: {vector_info['name']}")
                print(f"   ID: {vector_info['id']}")
            else:
                print(f"❌ No vector found for {variable}")
        except Exception as e:
            print(f"❌ Error searching for {variable}: {e}")
    
    print("\n5. Testing multi-factor data fetching...")
    try:
        test_vars = ["Bitcoin Daily Close Price", "QQQ Daily Close Price"]
        data = fetcher.fetch_multi_factor_data(test_vars)
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched data: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        else:
            print("❌ No data returned from fetch_multi_factor_data")
    except Exception as e:
        print(f"❌ Error in multi-factor fetch: {e}")

if __name__ == "__main__":
    test_intelligence_connection()