"""
Strategy processing utilities using OpenAI to generate structured conditions
"""
import json
import os
from openai import OpenAI
import streamlit as st
from functools import lru_cache
import hashlib

class StrategyProcessor:
    def __init__(self):
        """Initialize OpenAI client"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # In-memory cache for strategy conditions
        self._conditions_cache = {}
    
    def _get_strategy_hash(self, strategy_metadata):
        """Generate a hash for strategy metadata to use as cache key"""
        # Create a deterministic hash from strategy metadata
        key_parts = [
            strategy_metadata.get('description', ''),
            str(strategy_metadata.get('excel_names', [])),
            str(strategy_metadata.get('avg_holding_days', 3)),
            strategy_metadata.get('strategy_type', 'unknown')
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @st.cache_data(ttl=3600*24, show_spinner=False)  # Cache for 24 hours
    def _generate_conditions_cached(self, strategy_hash, description, excel_names, avg_holding_days, strategy_type):
        """Cached version of condition generation - uses Streamlit caching"""
        # This is a static method to work with Streamlit's caching
        return self._generate_conditions_internal({
            'description': description,
            'excel_names': excel_names,
            'avg_holding_days': avg_holding_days,
            'strategy_type': strategy_type
        })
    
    def generate_conditions(self, strategy_metadata):
        """
        Generate structured conditions from strategy metadata using OpenAI with caching
        """
        # Check in-memory cache first
        strategy_hash = self._get_strategy_hash(strategy_metadata)
        if strategy_hash in self._conditions_cache:
            print(f"Using cached conditions for strategy")
            return self._conditions_cache[strategy_hash]
        
        # Try Streamlit cache
        try:
            conditions = self._generate_conditions_cached(
                strategy_hash,
                strategy_metadata.get('description', ''),
                tuple(strategy_metadata.get('excel_names', [])),  # Convert to tuple for hashability
                strategy_metadata.get('avg_holding_days', 3),
                strategy_metadata.get('strategy_type', 'unknown')
            )
            self._conditions_cache[strategy_hash] = conditions
            return conditions
        except Exception as e:
            # If caching fails, generate directly
            print(f"Cache failed, generating conditions directly: {e}")
            conditions = self._generate_conditions_internal(strategy_metadata)
            self._conditions_cache[strategy_hash] = conditions
            return conditions
    
    def _generate_conditions_internal(self, strategy_metadata):
        """Internal method that actually generates conditions using OpenAI"""
        prompt = f"""You are a trading strategy compiler. Given the metadata of a Bitcoin strategy, extract the entry and exit conditions, asset dependencies, and turn it into a structured JSON schema called `conditions`.

Example format:
{{
  "entry": {{ "type": "...", "rule": "...", "delay_days": ... }},
  "exit": {{ "type": "...", "days": ... }},
  "position": {{ "side": "...", "size": "..." }},
  "depends_on": [...]
}}

Metadata:
Description: "{strategy_metadata.get('description', '')}"
Excel Series: {strategy_metadata.get('excel_names', [])}
Average Holding Days: {strategy_metadata.get('avg_holding_days', 3)}
Strategy Type: {strategy_metadata.get('strategy_type', 'unknown')}

Parse the description to identify:
1. Entry condition (what triggers the trade)
2. Delay period (if mentioned in description)
3. Exit condition (usually holding period based)
4. Position side (BUY = long, SELL = short)
5. Dependencies (assets from excel_names)

Generate the `conditions` JSON:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an expert at parsing trading strategy descriptions and generating structured JSON conditions. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            conditions_json = response.choices[0].message.content
            if conditions_json:
                return json.loads(conditions_json)
            else:
                raise ValueError("Empty response from OpenAI")
            
        except Exception as e:
            print(f"Error generating conditions: {e}")
            # Fallback structure
            return {
                "entry": {
                    "type": "signal",
                    "rule": "parsed from description",
                    "delay_days": strategy_metadata.get('avg_holding_days', 3)
                },
                "exit": {
                    "type": "holding_period",
                    "days": strategy_metadata.get('avg_holding_days', 3)
                },
                "position": {
                    "side": "long" if "BUY" in strategy_metadata.get('description', '') else "short",
                    "size": "100%"
                },
                "depends_on": strategy_metadata.get('excel_names', [])
            }
    
    def process_all_strategies(self, pinecone_client, max_strategies=None):
        """
        Process all strategies in Pinecone to generate conditions
        """
        if not pinecone_client or not pinecone_client.is_connected():
            return []
        
        try:
            processed_strategies = []
            
            # Get all strategies with pagination
            dummy_vector = [0.0] * 32
            
            # First query to get total count and initial batch
            response = pinecone_client.index.query(
                vector=dummy_vector,
                top_k=100,  # Maximum per query
                include_metadata=True
            )
            
            batch_count = 1
            if hasattr(response, 'matches'):
                for match in response.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        strategy = self._process_single_strategy(match)
                        if strategy:
                            processed_strategies.append(strategy)
                
                print(f"Processed batch {batch_count}: {len(response.matches)} strategies")
                
                # Continue with additional queries if we have more strategies
                # Since Pinecone doesn't have perfect pagination, we'll use different dummy vectors
                # to get different subsets of strategies
                for offset in range(100, 254, 100):
                    if max_strategies and len(processed_strategies) >= max_strategies:
                        break
                        
                    # Use slightly different vector to get different results
                    offset_vector = [float(i % 2) * 0.001 for i in range(32)]
                    response = pinecone_client.index.query(
                        vector=offset_vector,
                        top_k=100,
                        include_metadata=True
                    )
                    
                    batch_count += 1
                    new_strategies = 0
                    existing_ids = {s['id'] for s in processed_strategies}
                    
                    if hasattr(response, 'matches'):
                        for match in response.matches:
                            if hasattr(match, 'metadata') and match.metadata:
                                strategy_id = match.metadata.get('strategy_id', match.id)
                                if strategy_id not in existing_ids:
                                    strategy = self._process_single_strategy(match)
                                    if strategy:
                                        processed_strategies.append(strategy)
                                        existing_ids.add(strategy_id)
                                        new_strategies += 1
                    
                    print(f"Processed batch {batch_count}: {new_strategies} new strategies")
                    
                    if new_strategies == 0:  # No more new strategies found
                        break
            
            print(f"Total processed strategies: {len(processed_strategies)}")
            return processed_strategies
            
        except Exception as e:
            print(f"Error processing strategies: {e}")
            return []
    
    def _process_single_strategy(self, match):
        """Process a single strategy match from Pinecone"""
        try:
            metadata = match.metadata
            
            # Generate conditions if empty
            conditions_str = metadata.get('conditions', '{}')
            if not conditions_str or conditions_str == '{}':
                conditions = self.generate_conditions(metadata)
            else:
                # Parse existing conditions
                try:
                    conditions = json.loads(conditions_str)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse existing conditions: {e}")
                    conditions = self.generate_conditions(metadata)
            
            # Create processed strategy object
            strategy = {
                'id': metadata.get('strategy_id', match.id),
                'description': metadata.get('description', ''),
                'conditions': conditions,
                'performance': {
                    'total_return': metadata.get('total_return', 0),
                    'sharpe_ratio': metadata.get('sharpe_ratio', 0),
                    'max_drawdown': metadata.get('max_drawdown', 0),
                    'success_rate': metadata.get('success_rate', 0),
                    'total_trades': metadata.get('total_trades', 0),
                    'quality_score': metadata.get('quality_score', 0),
                    'risk_score': metadata.get('risk_score', 0)
                },
                'metadata': metadata
            }
            return strategy
            
        except Exception as e:
            print(f"Error processing single strategy: {e}")
            return None
    
    def execute_strategy_conditions(self, df, conditions):
        """
        Execute a strategy based on its structured conditions
        """
        import pandas as pd
        import numpy as np
        
        try:
            # Extract condition parameters
            entry_rule = conditions.get('entry', {}).get('rule', '')
            delay_days = conditions.get('entry', {}).get('delay_days', 0)
            exit_days = conditions.get('exit', {}).get('days', 3)
            position_side = conditions.get('position', {}).get('side', 'long')
            
            # Ensure numeric parameters are integers
            try:
                delay_days = int(delay_days)
            except (ValueError, TypeError):
                delay_days = 0
            
            try:
                exit_days = int(exit_days)
            except (ValueError, TypeError):
                exit_days = 3
            
            # Simple signal generation based on rule patterns
            signals = pd.Series(False, index=df.index)
            
            # Parse common rule patterns
            if 'increases' in entry_rule.lower() or '>' in entry_rule:
                # Look for price increases
                price_change = df['Close'].pct_change()
                signals = price_change > 0
            elif 'decreases' in entry_rule.lower() or '<' in entry_rule:
                # Look for price decreases  
                price_change = df['Close'].pct_change()
                signals = price_change < 0
            else:
                # Default: random signals based on volatility
                volatility = df['Close'].pct_change().rolling(20).std()
                signals = volatility > volatility.quantile(0.7)
            
            # Apply delay if specified
            if delay_days > 0:
                signals = signals.shift(delay_days).fillna(False)
            
            # Execute strategy with holding period
            pos, days, returns = 0, 0, []
            
            for i in range(len(df)):
                # Exit conditions
                if pos and days >= exit_days:
                    pos = 0
                    days = 0
                
                # Entry conditions
                if not pos and signals.iloc[i]:
                    pos = 1 if position_side == 'long' else -1
                    days = 0
                
                if pos:
                    days += 1
                
                # Calculate return
                ret = 0
                if pos and i > 0:
                    ret = pos * (df['Close'].iloc[i] / df['Close'].iloc[i-1] - 1)
                    ret = max(ret, -0.9999)  # Cap losses
                
                returns.append(ret)
            
            return pd.Series(returns, index=df.index)
            
        except Exception as e:
            print(f"Error executing strategy: {e}")
            # Return flat performance if strategy fails
            return pd.Series(0.0, index=df.index)