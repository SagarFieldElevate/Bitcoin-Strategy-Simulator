"""
Strategy processing utilities using OpenAI to generate structured conditions
"""
import json
import os
from openai import OpenAI

class StrategyProcessor:
    def __init__(self):
        """Initialize OpenAI client"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_conditions(self, strategy_metadata):
        """
        Generate structured conditions from strategy metadata using OpenAI
        """
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
            return json.loads(conditions_json)
            
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
    
    def process_all_strategies(self, pinecone_client):
        """
        Process all strategies in Pinecone to generate conditions
        """
        if not pinecone_client or not pinecone_client.is_connected():
            return []
        
        try:
            # Get all strategies from Pinecone
            dummy_vector = [0.0] * 32
            response = pinecone_client.index.query(
                vector=dummy_vector,
                top_k=100,  # Get more strategies
                include_metadata=True
            )
            
            processed_strategies = []
            
            if hasattr(response, 'matches'):
                for match in response.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata = match.metadata
                        
                        # Generate conditions if empty
                        if not metadata.get('conditions') or metadata.get('conditions') == '{}':
                            conditions = self.generate_conditions(metadata)
                            metadata['conditions'] = json.dumps(conditions)
                        else:
                            # Parse existing conditions
                            try:
                                conditions = json.loads(metadata.get('conditions', '{}'))
                            except:
                                conditions = self.generate_conditions(metadata)
                                metadata['conditions'] = json.dumps(conditions)
                        
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
                        processed_strategies.append(strategy)
            
            return processed_strategies
            
        except Exception as e:
            print(f"Error processing strategies: {e}")
            return []
    
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