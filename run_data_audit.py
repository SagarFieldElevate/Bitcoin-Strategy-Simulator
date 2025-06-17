#!/usr/bin/env python3
"""
Critical Data Integrity Audit for Hedge Fund Operations
Identifies strategies that require unavailable macro data
"""

from utils.pinecone_client import PineconeClient
from utils.simulation_router import SimulationRouter
from utils.multi_factor_data import MultiFactorDataFetcher
import os
from collections import Counter, defaultdict

def audit_strategy_data_requirements():
    """Audit all 254 strategies for data availability"""
    
    client = PineconeClient(os.getenv('PINECONE_API_KEY'))
    router = SimulationRouter()
    fetcher = MultiFactorDataFetcher(client)
    
    print("HEDGE FUND DATA INTEGRITY AUDIT")
    print("=" * 50)
    
    # Load strategies efficiently
    strategies = []
    dummy_vector = [0.0] * 32
    
    # Use multiple queries to get all strategies
    for i in range(10):
        offset_vector = [0.001 if j % (i+2) == 0 else 0.0 for j in range(32)]
        response = client.index.query(
            vector=offset_vector,
            top_k=50,
            include_metadata=True
        )
        
        if hasattr(response, 'matches'):
            for match in response.matches:
                if hasattr(match, 'metadata') and match.id not in [s.get('id') for s in strategies]:
                    strategies.append({
                        'id': match.id,
                        'name': match.metadata.get('description', 'Unknown')[:80],
                        'metadata': match.metadata
                    })
        
        if len(strategies) >= 254:
            break
    
    strategies = strategies[:254]  # Ensure exactly 254
    print(f"Loaded {len(strategies)} strategies for audit")
    
    # Track results
    all_variables = Counter()
    strategy_types = defaultdict(list)
    btc_only_strategies = []
    multi_factor_strategies = []
    problematic_strategies = []
    
    # Quick classification pass
    for i, strategy in enumerate(strategies):
        try:
            metadata = strategy['metadata']
            strategy_type = metadata.get('strategy_type', 'unknown')
            
            simulation_mode = router.select_simulation_mode(metadata)
            required_vars = router.get_required_variables(metadata, simulation_mode)
            
            strategy_info = {
                'name': strategy['name'],
                'type': strategy_type,
                'mode': simulation_mode,
                'variables': required_vars
            }
            
            all_variables.update(required_vars)
            strategy_types[strategy_type].append(strategy_info)
            
            if simulation_mode == 'btc_only':
                btc_only_strategies.append(strategy_info)
            else:
                multi_factor_strategies.append(strategy_info)
                
                # Flag strategies with non-BTC variables
                non_btc_vars = [v for v in required_vars if v != 'BTC']
                if non_btc_vars:
                    problematic_strategies.append({
                        'strategy': strategy_info,
                        'missing_data_vars': non_btc_vars
                    })
                    
        except Exception as e:
            print(f"Error processing strategy {i}: {e}")
    
    # Generate report
    print(f"\nSTRATEGY CLASSIFICATION:")
    print(f"BTC-only strategies: {len(btc_only_strategies)} (SAFE - authentic data available)")
    print(f"Multi-factor strategies: {len(multi_factor_strategies)}")
    print(f"Strategies requiring macro data: {len(problematic_strategies)}")
    
    print(f"\nMAC RO VARIABLE REQUIREMENTS:")
    for var, count in all_variables.most_common():
        if var != 'BTC':
            print(f"  {var}: {count} strategies require this variable")
    
    print(f"\nSTRATEGY TYPES BREAKDOWN:")
    for stype, strats in strategy_types.items():
        unique_vars = set()
        for s in strats:
            unique_vars.update(s['variables'])
        non_btc_vars = [v for v in unique_vars if v != 'BTC']
        if non_btc_vars:
            print(f"  {stype}: {len(strats)} strategies, requires {non_btc_vars}")
        else:
            print(f"  {stype}: {len(strats)} strategies, BTC-only (SAFE)")
    
    # Critical assessment
    critical_variables = [var for var, count in all_variables.most_common() if var != 'BTC']
    
    if critical_variables:
        print(f"\nCRITICAL HEDGE FUND RISK ASSESSMENT:")
        print(f"⚠️  {len(problematic_strategies)} strategies require macro data that may not be available")
        print(f"⚠️  Required variables: {critical_variables}")
        print(f"⚠️  These strategies CANNOT run without authentic macro data")
        print(f"⚠️  Running with synthetic data would compromise hedge fund integrity")
        
        print(f"\nRECOMMENDATION:")
        print(f"1. Verify authentic data availability for: {critical_variables}")
        print(f"2. Exclude strategies requiring unavailable data")
        print(f"3. Use only the {len(btc_only_strategies)} BTC-only strategies for safe operations")
    else:
        print(f"\n✅ ALL STRATEGIES USE ONLY BTC DATA - SAFE FOR HEDGE FUND OPERATIONS")
    
    return {
        'total_strategies': len(strategies),
        'btc_only_safe': len(btc_only_strategies),
        'multi_factor_risky': len(multi_factor_strategies),
        'critical_variables': critical_variables,
        'problematic_strategies': problematic_strategies
    }

if __name__ == "__main__":
    results = audit_strategy_data_requirements()
    
    print(f"\nAUDIT SUMMARY:")
    print(f"Safe for hedge fund: {results['btc_only_safe']} strategies")
    print(f"Require verification: {results['multi_factor_risky']} strategies")
    print(f"Critical data gaps: {len(results['critical_variables'])} variables")