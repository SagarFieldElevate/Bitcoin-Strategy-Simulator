"""
Debug script to explore the bitcoin-strategies Pinecone index
"""
import os
from pinecone import Pinecone

def explore_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    
    # List all indexes
    indexes = pc.list_indexes()
    print("Available indexes:")
    for idx in indexes.indexes:
        print(f"  - {idx.name}: {idx.dimension} dimensions, {idx.metric} metric")
    
    # Explore both bitcoin-strategies and intelligence-main
    for index_name in ["bitcoin-strategies", "intelligence-main"]:
        print(f"\n=== Exploring {index_name} ===")
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"Stats: {stats}")
            
            # Query for trading strategies or bitcoin-related content
            if index_name == "intelligence-main":
                # Query for strategy-related content
                dummy_vector = [0.0] * 1536
                response = index.query(
                    vector=dummy_vector,
                    top_k=20,
                    include_metadata=True
                )
                
                print("Sample vectors from intelligence-main:")
                for i, match in enumerate(response.matches[:5]):
                    print(f"  {i+1}. ID: {match.id}")
                    if hasattr(match, 'metadata') and match.metadata:
                        print(f"     Metadata: {match.metadata}")
                    print()
            
            elif index_name == "bitcoin-strategies":
                # Try to get some vectors from bitcoin-strategies
                dummy_vector = [0.0] * 32
                response = index.query(
                    vector=dummy_vector,
                    top_k=10,
                    include_metadata=True
                )
                
                print("Sample vectors from bitcoin-strategies:")
                for i, match in enumerate(response.matches[:5]):
                    print(f"  {i+1}. ID: {match.id}")
                    if hasattr(match, 'metadata') and match.metadata:
                        print(f"     Metadata: {match.metadata}")
                    print()
                    
        except Exception as e:
            print(f"Error exploring {index_name}: {e}")

if __name__ == "__main__":
    explore_pinecone_index()