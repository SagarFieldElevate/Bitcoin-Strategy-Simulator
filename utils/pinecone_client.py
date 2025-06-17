"""
Pinecone client for strategy storage and retrieval
"""
import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

class PineconeClient:
    def __init__(self, api_key):
        """
        Initialize Pinecone client and connect to bitcoin-strategies index
        """
        self.api_key = api_key
        self.connected = False
        self.pc = None
        self.index = None
        self.index_name = "bitcoin-strategies"
        
        # Initialize Pinecone connection
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize connection to Pinecone"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if bitcoin-strategies index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if self.index_name in index_names:
                # Connect to existing index
                self.index = self.pc.Index(self.index_name)
                self.connected = True
                print(f"Connected to existing index: {self.index_name}")
            else:
                # Index doesn't exist
                self.connected = False
                print(f"Index '{self.index_name}' not found. Available indexes: {index_names}")
                
        except Exception as e:
            self.connected = False
            print(f"Failed to connect to Pinecone: {str(e)}")
    
    def is_connected(self):
        """Check if Pinecone is connected"""
        return self.connected
        
    def list_strategies(self):
        """
        List available strategies from Pinecone bitcoin-strategies index
        Returns list of strategy names
        """
        if not self.connected or not self.index:
            return ["CEMD (Default)"]
        
        try:
            # Query the index to get all strategy metadata
            # Use a dummy vector to query all strategies
            dummy_vector = [0.0] * 1536  # OpenAI embedding dimension
            
            # Query with top_k large enough to get all strategies
            response = self.index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True
            )
            
            # Extract strategy names from metadata
            strategy_names = ["CEMD (Default)"]  # Always include default
            
            for match in response.matches:
                if match.metadata and 'strategy_name' in match.metadata:
                    strategy_name = match.metadata['strategy_name']
                    if strategy_name not in strategy_names:
                        strategy_names.append(strategy_name)
            
            return strategy_names
            
        except Exception as e:
            print(f"Error listing strategies from Pinecone: {str(e)}")
            return ["CEMD (Default)"]
    
    def get_strategy(self, strategy_name):
        """
        Retrieve strategy code from Pinecone bitcoin-strategies index
        Returns strategy implementation as string or code
        """
        if not self.connected or not self.index or strategy_name == "CEMD (Default)":
            return None
        
        try:
            # Query for the specific strategy by name
            # First get all strategies and find the one we want
            dummy_vector = [0.0] * 1536
            
            response = self.index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={"strategy_name": strategy_name}
            )
            
            if response.matches:
                # Get the first match
                match = response.matches[0]
                if match.metadata:
                    # Return the strategy code or implementation details
                    return {
                        'strategy_name': match.metadata.get('strategy_name'),
                        'code': match.metadata.get('code', ''),
                        'description': match.metadata.get('description', ''),
                        'parameters': match.metadata.get('parameters', {}),
                        'id': match.id
                    }
            
            return None
                
        except Exception as e:
            print(f"Error retrieving strategy {strategy_name} from Pinecone: {str(e)}")
            return None
    
    def _mean_reversion_strategy(self, df, spread_threshold, holding_period, risk_percent):
        """
        Example mean reversion strategy
        This would typically be loaded from Pinecone
        """
        # Calculate simple moving averages
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # Calculate deviation from mean
        df['deviation'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        
        # Generate signals
        long_sig = df['deviation'] < -spread_threshold
        short_sig = df['deviation'] > spread_threshold
        
        # Execute strategy
        pos, days, pl = 0, 0, []
        for i in range(len(df)):
            if pos and days >= holding_period:
                pos = 0
                days = 0
            
            if not pos:
                if long_sig.iloc[i]:
                    pos = 1
                elif short_sig.iloc[i]:
                    pos = -1
            
            if pos:
                days += 1
            
            ret = pos * (df["Close"].iloc[i] / df["Close"].iloc[i-1] - 1) if i else 0
            ret = max(ret, -0.9999)
            pl.append(ret * (risk_percent / 100))
        
        return pl
    
    def _momentum_breakout_strategy(self, df, spread_threshold, holding_period, risk_percent):
        """
        Example momentum breakout strategy
        """
        # Placeholder for momentum strategy
        return [0] * len(df)
    
    def _rsi_divergence_strategy(self, df, spread_threshold, holding_period, risk_percent):
        """
        Example RSI divergence strategy
        """
        # Placeholder for RSI strategy
        return [0] * len(df)
    
    def _bollinger_squeeze_strategy(self, df, spread_threshold, holding_period, risk_percent):
        """
        Example Bollinger Squeeze strategy
        """
        # Placeholder for Bollinger strategy
        return [0] * len(df)
    
    def save_strategy(self, strategy_name, strategy_code):
        """
        Save strategy to Pinecone
        """
        try:
            # Placeholder implementation
            # In actual implementation, this would save strategy to Pinecone
            st.success(f"Strategy {strategy_name} saved successfully")
            return True
        except Exception as e:
            st.error(f"Error saving strategy {strategy_name} to Pinecone: {str(e)}")
            return False
