"""
Smart simulation engine router that selects appropriate simulation mode
based on strategy type and dependencies
"""
import json
import os
from openai import OpenAI

class SimulationRouter:
    def __init__(self):
        """Initialize the simulation router with OpenAI client"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def select_simulation_mode(self, strategy_metadata):
        """
        Determine which simulation mode to use based on strategy metadata
        
        Returns:
            str: 'btc_only' or 'multi_factor'
        """
        strategy_type = strategy_metadata.get('strategy_type', '').lower()
        description = strategy_metadata.get('description', '')
        
        # Rule-based routing for clear cases
        if strategy_type in ['technical', 'ml']:
            return 'btc_only'
        elif strategy_type in ['correlation', 'multiway_combination']:
            return 'multi_factor'
        elif strategy_type == 'hybrid':
            # Use LLM to detect dependencies
            return self._detect_dependencies_with_llm(description)
        else:
            # Default fallback - analyze description
            return self._detect_dependencies_with_llm(description)
    
    def _detect_dependencies_with_llm(self, description):
        """
        Use LLM to detect if strategy depends on external macro variables
        """
        prompt = f"""Analyze this trading strategy description and determine if it depends on external variables beyond Bitcoin price.

Strategy description: "{description}"

Look for mentions of:
- Commodity prices (WTI oil, gold, silver, etc.)
- Economic indicators (CPI, inflation, GDP, unemployment, etc.)
- Interest rates and bonds (TIPS, Treasury yields, Fed rates, etc.)
- Currency indices (DXY, currency pairs, etc.)
- Other asset classes (stocks, ETFs, real estate, etc.)
- Cryptocurrency data (ETH volume, ETH price, other crypto assets, CoinGecko data)
- Market data sources (Yahoo Finance, CoinGecko, trading volume from other assets)
- Economic events or data releases

Respond with JSON in this exact format:
{{
  "has_external_dependencies": true/false,
  "detected_variables": ["list", "of", "variables"],
  "reasoning": "brief explanation"
}}

If the strategy only mentions Bitcoin price, technical indicators, or chart patterns, it has NO external dependencies."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing trading strategies to detect macro-economic dependencies. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            has_dependencies = result.get('has_external_dependencies', False)
            
            # Log the detection for debugging
            print(f"LLM Dependency Detection:")
            print(f"  Description: {description[:50]}...")
            print(f"  Has dependencies: {has_dependencies}")
            print(f"  Variables: {result.get('detected_variables', [])}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
            
            return 'multi_factor' if has_dependencies else 'btc_only'
            
        except Exception as e:
            print(f"Error in LLM dependency detection: {e}")
            # Conservative fallback - if we can't determine, use multi_factor
            return 'multi_factor'
    
    def get_required_variables(self, strategy_metadata, simulation_mode):
        """
        Get list of variables required for the simulation
        
        Returns:
            list: Variable names needed for simulation
        """
        if simulation_mode == 'btc_only':
            return ['BTC']
        
        # For multi_factor, determine which variables are needed
        description = strategy_metadata.get('description', '').lower()
        excel_names = strategy_metadata.get('excel_names', [])
        
        # Start with Bitcoin
        required_vars = ['BTC']
        
        # Common macro variables to check for
        macro_variables = {
            'wti': ['wti', 'oil', 'crude'],
            'tips_10y': ['tips', 'treasury', 'bond', 'yield'],
            'cpi': ['cpi', 'inflation', 'consumer price'],
            'dxy': ['dxy', 'dollar', 'currency'],
            'gold': ['gold', 'precious metal'],
            'vix': ['vix', 'volatility'],
            'spy': ['spy', 'stocks', 'equity', 's&p'],
            'eth': ['eth', 'ethereum', 'coingecko eth', 'eth volume', 'eth daily volume']
        }
        
        # Check description and excel_names for variable mentions
        text_to_check = f"{description} {' '.join(excel_names)}".lower()
        
        for var_code, keywords in macro_variables.items():
            if any(keyword in text_to_check for keyword in keywords):
                required_vars.append(var_code.upper())
        
        # Remove duplicates and ensure BTC is first
        unique_vars = ['BTC'] + [var for var in required_vars[1:] if var != 'BTC']
        
        return unique_vars