"""
Test script to check Pinecone connection and list strategies
"""
import os
from utils.pinecone_client import PineconeClient

def test_pinecone_connection():
    # Get API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ No Pinecone API key found")
        return
    
    print(f"✅ Found Pinecone API key: {api_key[:10]}...")
    
    try:
        # Initialize client
        client = PineconeClient(api_key)
        print(f"✅ Pinecone client initialized, connected: {client.is_connected()}")
        
        # List strategies
        strategies = client.list_strategies()
        print(f"✅ Available strategies: {strategies}")
        
        # Test getting a specific strategy
        if strategies and len(strategies) > 1:
            strategy_name = strategies[1]  # Skip the default one
            strategy_code = client.get_strategy(strategy_name)
            print(f"✅ Retrieved strategy '{strategy_name}': {type(strategy_code)}")
        
    except Exception as e:
        print(f"❌ Error testing Pinecone: {str(e)}")

if __name__ == "__main__":
    test_pinecone_connection()