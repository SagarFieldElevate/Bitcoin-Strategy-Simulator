#!/usr/bin/env python3
"""
Simple test to verify the Bitcoin strategy simulation system is working
"""

import os
import sys
from utils.bitcoin_data import fetch_bitcoin_data
from utils.monte_carlo import MonteCarloSimulator
from utils.pinecone_client import PineconeClient

def test_basic_functionality():
    """Test core system functionality"""
    print("Testing Bitcoin Strategy Simulation System...")
    
    # Test 1: Bitcoin data fetching
    print("\n1. Testing Bitcoin data fetching...")
    try:
        btc_data = fetch_bitcoin_data()
        print(f"✓ Bitcoin data loaded: {len(btc_data)} days")
    except Exception as e:
        print(f"✗ Bitcoin data failed: {e}")
        return False
    
    # Test 2: Pinecone connection
    print("\n2. Testing Pinecone connection...")
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("✗ No Pinecone API key found")
            return False
        
        client = PineconeClient(api_key)
        strategies = client.list_strategies()
        print(f"✓ Pinecone connected: {len(strategies)} strategies")
    except Exception as e:
        print(f"✗ Pinecone failed: {e}")
        return False
    
    # Test 3: Monte Carlo simulator initialization
    print("\n3. Testing Monte Carlo simulator...")
    try:
        simulator = MonteCarloSimulator(btc_data, client)
        print("✓ Monte Carlo simulator initialized")
    except Exception as e:
        print(f"✗ Monte Carlo failed: {e}")
        return False
    
    # Test 4: Simple BTC-only simulation
    print("\n4. Testing BTC-only simulation...")
    try:
        results = simulator.run_simulation(
            n_simulations=10,
            simulation_days=30,
            selected_strategy=None,  # Default CEMD strategy
            market_condition=None,
            simulation_mode='btc_only'
        )
        
        median_cagr = results.get('median_cagr', -100)
        print(f"✓ BTC simulation completed: {median_cagr:.1f}% CAGR")
        
        if median_cagr == -100:
            print("⚠ Warning: Simulation returning -100% CAGR")
            return False
        
    except Exception as e:
        print(f"✗ BTC simulation failed: {e}")
        return False
    
    print("\n✓ All tests passed! System is working.")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)