"""
Data Integrity Validator for Hedge Fund Operations
Ensures no strategy uses synthetic or placeholder data
"""
import pandas as pd
from typing import List, Dict, Tuple, Optional

class DataIntegrityValidator:
    def __init__(self, pinecone_client, multi_factor_fetcher):
        self.pinecone_client = pinecone_client
        self.multi_factor_fetcher = multi_factor_fetcher
        self.validated_variables = {}  # Cache of validated variables
        self.min_data_points = 365  # Minimum 1 year of data required
        
    def validate_all_strategies(self) -> Dict:
        """
        Validate data availability for all 254 strategies
        Returns comprehensive audit results
        """
        print("HEDGE FUND DATA INTEGRITY AUDIT")
        print("=" * 50)
        
        # Load all strategies
        strategies = self._load_all_strategies()
        
        validation_results = {
            'total_strategies': len(strategies),
            'valid_strategies': [],
            'invalid_strategies': [],
            'variable_availability': {},
            'critical_issues': []
        }
        
        # Validate each strategy
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('name', f'Strategy_{i}')
            print(f"Validating {i+1}/{len(strategies)}: {strategy_name[:50]}...")
            
            is_valid, issues = self.validate_strategy(strategy)
            
            if is_valid:
                validation_results['valid_strategies'].append({
                    'name': strategy_name,
                    'metadata': strategy['metadata']
                })
            else:
                validation_results['invalid_strategies'].append({
                    'name': strategy_name,
                    'issues': issues,
                    'metadata': strategy['metadata']
                })
                validation_results['critical_issues'].extend(issues)
        
        # Generate summary report
        self._generate_audit_report(validation_results)
        
        return validation_results
    
    def validate_strategy(self, strategy: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single strategy for data availability
        Returns (is_valid, list_of_issues)
        """
        try:
            from .simulation_router import SimulationRouter
            router = SimulationRouter()
            
            metadata = strategy['metadata']
            simulation_mode = router.select_simulation_mode(metadata)
            required_variables = router.get_required_variables(metadata, simulation_mode)
            
            issues = []
            
            # Validate each required variable
            for variable in required_variables:
                is_available, variable_issues = self._validate_variable(variable)
                if not is_available:
                    issues.extend(variable_issues)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Strategy validation error: {str(e)}"]
    
    def _validate_variable(self, variable: str) -> Tuple[bool, List[str]]:
        """
        Validate data availability for a specific variable
        Returns (is_available, list_of_issues)
        """
        # Check cache first
        if variable in self.validated_variables:
            cached_result = self.validated_variables[variable]
            return cached_result['is_available'], cached_result['issues']
        
        issues = []
        
        try:
            if variable == 'BTC':
                # BTC is always available from yfinance
                data_points = self._count_btc_data_points()
                if data_points >= self.min_data_points:
                    self.validated_variables[variable] = {'is_available': True, 'issues': []}
                    return True, []
                else:
                    issue = f"BTC has insufficient data: {data_points} < {self.min_data_points} required"
                    issues.append(issue)
            
            elif variable.upper() in ['GOLD', 'GLD', 'XAU']:
                # Validate gold data from intelligence-main
                gold_data = self.multi_factor_fetcher.fetch_gold_data_direct()
                if gold_data is not None and len(gold_data) >= self.min_data_points:
                    self.validated_variables[variable] = {'is_available': True, 'issues': []}
                    return True, []
                else:
                    data_count = len(gold_data) if gold_data is not None else 0
                    issue = f"GOLD has insufficient data: {data_count} < {self.min_data_points} required"
                    issues.append(issue)
            
            else:
                # Validate other macro variables
                vector_info = self.multi_factor_fetcher.find_vector_for_variable(variable)
                if vector_info:
                    # Attempt to extract actual data
                    try:
                        time_series = self.multi_factor_fetcher.fetch_data_from_pinecone(vector_info)
                        if time_series is not None and len(time_series) >= self.min_data_points:
                            self.validated_variables[variable] = {'is_available': True, 'issues': []}
                            return True, []
                        else:
                            data_count = len(time_series) if time_series is not None else 0
                            issue = f"{variable} has insufficient data: {data_count} < {self.min_data_points} required"
                            issues.append(issue)
                    except Exception as e:
                        issue = f"{variable} data extraction failed: {str(e)}"
                        issues.append(issue)
                else:
                    issue = f"{variable} not found in intelligence-main index"
                    issues.append(issue)
        
        except Exception as e:
            issue = f"{variable} validation error: {str(e)}"
            issues.append(issue)
        
        # Cache negative result
        self.validated_variables[variable] = {'is_available': False, 'issues': issues}
        return False, issues
    
    def _count_btc_data_points(self) -> int:
        """Count available BTC data points"""
        try:
            from .bitcoin_data import fetch_bitcoin_data
            btc_data = fetch_bitcoin_data()
            return len(btc_data)
        except:
            return 0
    
    def _load_all_strategies(self) -> List[Dict]:
        """Load all 254 strategies from Pinecone"""
        strategies = []
        
        # Use multiple query patterns to ensure we get all strategies
        query_patterns = [
            [0.0] * 32,
            [0.001 if i % 2 == 0 else 0.0 for i in range(32)],
            [0.001 if i % 3 == 0 else 0.0 for i in range(32)],
            [0.001 if i % 5 == 0 else 0.0 for i in range(32)],
            [0.001 if i % 7 == 0 else 0.0 for i in range(32)],
        ]
        
        seen_ids = set()
        
        for pattern in query_patterns:
            response = self.pinecone_client.index.query(
                vector=pattern,
                top_k=100,
                include_metadata=True
            )
            
            if hasattr(response, 'matches'):
                for match in response.matches:
                    if match.id not in seen_ids and hasattr(match, 'metadata'):
                        strategies.append({
                            'id': match.id,
                            'name': match.metadata.get('description', 'Unknown Strategy'),
                            'metadata': match.metadata
                        })
                        seen_ids.add(match.id)
            
            if len(strategies) >= 254:
                break
        
        return strategies[:254]  # Ensure exactly 254
    
    def _generate_audit_report(self, results: Dict):
        """Generate comprehensive audit report"""
        total = results['total_strategies']
        valid = len(results['valid_strategies'])
        invalid = len(results['invalid_strategies'])
        
        print(f"\nDATA INTEGRITY AUDIT RESULTS")
        print(f"=" * 40)
        print(f"Total strategies audited: {total}")
        print(f"Strategies with complete authentic data: {valid}")
        print(f"Strategies requiring unavailable data: {invalid}")
        print(f"Data integrity percentage: {(valid/total)*100:.1f}%")
        
        if invalid > 0:
            print(f"\nCRITICAL ISSUES FOUND:")
            print(f"⚠️  {invalid} strategies cannot run without synthetic data")
            print(f"These strategies must be EXCLUDED from hedge fund operations")
            
            # Show most common missing variables
            missing_vars = {}
            for issue in results['critical_issues']:
                for word in issue.split():
                    if word.isupper() and len(word) > 2:
                        missing_vars[word] = missing_vars.get(word, 0) + 1
            
            if missing_vars:
                print(f"\nMost commonly missing variables:")
                for var, count in sorted(missing_vars.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {var}: {count} strategies affected")
        else:
            print(f"\n✅ ALL STRATEGIES HAVE COMPLETE AUTHENTIC DATA")
            print(f"Safe to proceed with hedge fund operations")

def run_comprehensive_audit(pinecone_client, multi_factor_fetcher):
    """Run complete data integrity audit"""
    validator = DataIntegrityValidator(pinecone_client, multi_factor_fetcher)
    return validator.validate_all_strategies()