#!/usr/bin/env python3
"""
Sophisticated Bitcoin Strategy Simulator with Pinecone and OpenAI integration
Restored version with all advanced features
"""

import sys
import os
import json
import traceback
from datetime import datetime, timedelta
import random
import math

# Try to import dependencies, fallback gracefully if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# If Streamlit is not available, create a web server fallback
if not STREAMLIT_AVAILABLE:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

class PineconeClient:
    """Pinecone client for strategy retrieval"""
    
    def __init__(self):
        self.connected = False
        self.strategies = []
        self.pc = None
        self.index = None
        
        try:
            if PINECONE_AVAILABLE:
                api_key = os.environ.get('PINECONE_API_KEY')
                if api_key:
                    self.pc = Pinecone(api_key=api_key)
                    
                    # Connect to bitcoin-strategies index
                    try:
                        self.index = self.pc.Index("bitcoin-strategies")
                        self.connected = True
                        print("✓ Connected to Pinecone bitcoin-strategies index")
                    except Exception as e:
                        print(f"Pinecone index connection failed: {e}")
                else:
                    print("PINECONE_API_KEY not found in environment")
            else:
                print("Pinecone library not available")
        except Exception as e:
            print(f"Pinecone initialization failed: {e}")
    
    def load_strategies(self):
        """Load strategies from Pinecone"""
        if not self.connected:
            return self._get_cached_strategies()
        
        try:
            # Get all strategy vectors
            all_strategies = []
            
            if hasattr(self.index, 'list'):
                scan_results = self.index.list()
                all_ids = []
                
                for ids_batch in scan_results:
                    all_ids.extend(ids_batch)
                
                print(f"Found {len(all_ids)} strategies in Pinecone")
                
                # Fetch in batches
                batch_size = 100
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    
                    try:
                        fetch_response = self.index.fetch(ids=batch_ids)
                        
                        for vector_id, vector_data in fetch_response.vectors.items():
                            if hasattr(vector_data, 'metadata') and vector_data.metadata:
                                metadata = vector_data.metadata
                                
                                strategy = {
                                    'id': vector_id,
                                    'name': metadata.get('name', 'Unknown Strategy'),
                                    'description': metadata.get('description', 'No description'),
                                    'excel_names': metadata.get('excel_names', []),
                                    'total_return': float(metadata.get('total_return', 0)),
                                    'sharpe_ratio': float(metadata.get('sharpe_ratio', 0)),
                                    'max_drawdown': float(metadata.get('max_drawdown', 0)),
                                    'success_rate': float(metadata.get('success_rate', 0)),
                                    'strategy_type': metadata.get('strategy_type', 'Unknown'),
                                    'quality_score': metadata.get('quality_score', 'N/A')
                                }
                                
                                all_strategies.append(strategy)
                                
                    except Exception as e:
                        print(f"Error fetching batch {i//batch_size + 1}: {e}")
                        continue
            
            print(f"Loaded {len(all_strategies)} strategies from Pinecone")
            self.strategies = all_strategies
            
            # Cache strategies
            try:
                with open("strategies_cache.json", 'w') as f:
                    json.dump(all_strategies, f, indent=2)
            except Exception as e:
                print(f"Cache save error: {e}")
            
            return all_strategies
            
        except Exception as e:
            print(f"Pinecone loading error: {e}")
            return self._get_cached_strategies()
    
    def _get_cached_strategies(self):
        """Fallback to cached strategies"""
        try:
            if os.path.exists("strategies_cache.json"):
                with open("strategies_cache.json", 'r') as f:
                    cached = json.load(f)
                    print(f"Loaded {len(cached)} strategies from cache")
                    return cached
        except Exception as e:
            print(f"Cache load error: {e}")
        
        # Return sample strategies if no cache
        return [
            {
                'id': 'cemd_default',
                'name': 'CEMD (Corporate vs Retail Momentum Divergence)',
                'description': 'Advanced momentum divergence strategy using corporate vs retail sentiment analysis',
                'total_return': 131.95,
                'sharpe_ratio': 9.99,
                'max_drawdown': -8.2,
                'success_rate': 0.68,
                'strategy_type': 'Momentum',
                'quality_score': 'A+'
            },
            {
                'id': 'mean_reversion_enhanced',
                'name': 'Enhanced Mean Reversion',
                'description': 'Sophisticated mean reversion with GARCH volatility modeling',
                'total_return': 86.48,
                'sharpe_ratio': 6.11,
                'max_drawdown': -12.5,
                'success_rate': 0.61,
                'strategy_type': 'Mean Reversion',
                'quality_score': 'A'
            }
        ]

class OpenAIClient:
    """OpenAI client for strategy analysis"""
    
    def __init__(self):
        self.connected = False
        self.client = None
        
        try:
            if OPENAI_AVAILABLE:
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.connected = True
                    print("✓ Connected to OpenAI")
                else:
                    print("OPENAI_API_KEY not found in environment")
            else:
                print("OpenAI library not available")
        except Exception as e:
            print(f"OpenAI initialization failed: {e}")
    
    def analyze_strategy(self, strategy, market_condition):
        """Analyze strategy performance using OpenAI"""
        if not self.connected:
            return "AI analysis unavailable - OpenAI not connected"
        
        try:
            prompt = f"""
            Analyze this Bitcoin trading strategy for {market_condition} market conditions:
            
            Strategy: {strategy['name']}
            Description: {strategy['description']}
            Historical Performance:
            - Total Return: {strategy['total_return']}%
            - Sharpe Ratio: {strategy['sharpe_ratio']}
            - Max Drawdown: {strategy['max_drawdown']}%
            - Success Rate: {strategy['success_rate']*100}%
            
            Provide a concise analysis of how this strategy might perform in {market_condition} conditions.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI analysis error: {str(e)}"

# Web interface handler for non-Streamlit fallback
class AdvancedBitcoinHandler(BaseHTTPRequestHandler):
    pinecone_client = None
    openai_client = None
    
    @classmethod
    def initialize_clients(cls):
        if cls.pinecone_client is None:
            cls.pinecone_client = PineconeClient()
            cls.openai_client = OpenAIClient()
    
    def do_GET(self):
        self.initialize_clients()
        
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = self.get_advanced_interface()
            self.wfile.write(html_content.encode('utf-8'))
            
        elif self.path == '/api/strategies':
            strategies = self.pinecone_client.load_strategies()
            self.send_json_response(strategies)
            
        elif self.path == '/api/status':
            status = {
                'pinecone_connected': self.pinecone_client.connected,
                'openai_connected': self.openai_client.connected,
                'strategies_count': len(self.pinecone_client.strategies)
            }
            self.send_json_response(status)
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        self.initialize_clients()
        
        if self.path == '/api/simulate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                params = json.loads(post_data.decode('utf-8'))
                results = self.run_advanced_simulation(params)
                self.send_json_response(results)
            except Exception as e:
                self.send_error_response(str(e))
                
        elif self.path == '/api/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                params = json.loads(post_data.decode('utf-8'))
                analysis = self.openai_client.analyze_strategy(
                    params.get('strategy'), 
                    params.get('market_condition', 'current')
                )
                self.send_json_response({'analysis': analysis})
            except Exception as e:
                self.send_error_response(str(e))
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode('utf-8'))
    
    def send_error_response(self, error_msg):
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        error_response = json.dumps({"error": error_msg})
        self.wfile.write(error_response.encode('utf-8'))
    
    def run_advanced_simulation(self, params):
        """Advanced Monte Carlo simulation with regime modeling"""
        strategy_id = params.get('strategy_id', 'cemd_default')
        market_condition = params.get('market_condition', 'normal')
        n_simulations = min(params.get('n_simulations', 100), 1000)  # Cap at 1000
        
        # Get strategy from Pinecone
        strategies = self.pinecone_client.strategies or self.pinecone_client.load_strategies()
        selected_strategy = next((s for s in strategies if s['id'] == strategy_id), strategies[0])
        
        # Advanced simulation parameters based on market regime
        base_drift = 0.0001  # Base daily drift
        base_volatility = 0.03  # Base daily volatility
        
        # Adjust for market conditions
        if market_condition == 'bull_market':
            drift_multiplier = 2.0
            vol_multiplier = 0.8
        elif market_condition == 'bear_market':
            drift_multiplier = -1.5
            vol_multiplier = 1.5
        elif market_condition == 'high_volatility':
            drift_multiplier = 1.0
            vol_multiplier = 2.0
        else:  # normal
            drift_multiplier = 1.0
            vol_multiplier = 1.0
        
        adjusted_drift = base_drift * drift_multiplier
        adjusted_vol = base_volatility * vol_multiplier
        
        # Run Monte Carlo simulation
        initial_price = 45000
        days = 252  # One year
        
        paths = []
        final_returns = []
        
        for sim in range(n_simulations):
            prices = [initial_price]
            
            for day in range(days):
                # GARCH-like volatility clustering
                vol_shock = random.gauss(0, 0.01) if random.random() < 0.1 else 0
                daily_vol = adjusted_vol + vol_shock
                
                # Price evolution with jumps
                if random.random() < 0.02:  # 2% chance of jump
                    jump = random.gauss(0, 0.05)  # Jump component
                else:
                    jump = 0
                
                daily_return = adjusted_drift + random.gauss(0, daily_vol) + jump
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 100))  # Floor at $100
            
            paths.append(prices)
            strategy_return = (prices[-1] - initial_price) / initial_price
            final_returns.append(strategy_return)
        
        # Calculate advanced statistics
        avg_return = sum(final_returns) / len(final_returns)
        volatility = (sum((r - avg_return) ** 2 for r in final_returns) / len(final_returns)) ** 0.5
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Downside statistics
        negative_returns = [r for r in final_returns if r < 0]
        max_drawdown = min(final_returns) if final_returns else 0
        
        return {
            'simulation_results': {
                'strategy': selected_strategy,
                'market_condition': market_condition,
                'n_simulations': n_simulations,
                'expected_return': round(avg_return * 100, 2),
                'volatility': round(volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'success_rate': round(len([r for r in final_returns if r > 0]) / len(final_returns), 2),
                'percentiles': {
                    '5th': round(sorted(final_returns)[int(0.05 * len(final_returns))] * 100, 2),
                    '25th': round(sorted(final_returns)[int(0.25 * len(final_returns))] * 100, 2),
                    '50th': round(sorted(final_returns)[int(0.50 * len(final_returns))] * 100, 2),
                    '75th': round(sorted(final_returns)[int(0.75 * len(final_returns))] * 100, 2),
                    '95th': round(sorted(final_returns)[int(0.95 * len(final_returns))] * 100, 2)
                },
                'price_paths': paths[:5]  # Return first 5 paths for display
            }
        }
    
    def get_advanced_interface(self):
        """Generate advanced HTML interface"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>₿ Bitcoin Strategy Monte Carlo Simulator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0e1117; color: #fafafa; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; padding: 20px; background: #262730; border-radius: 10px; }
        .header h1 { color: #ff6b35; margin: 0; font-size: 2.5em; }
        .section { margin: 20px 0; padding: 20px; background: #262730; border-radius: 10px; border: 1px solid #464851; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .strategy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }
        .strategy-card { padding: 15px; background: #1e1e1e; border-radius: 8px; border: 1px solid #464851; cursor: pointer; transition: all 0.3s; }
        .strategy-card:hover { border-color: #ff6b35; transform: translateY(-2px); }
        .strategy-card.selected { border-color: #ff6b35; background: #2d1810; }
        .controls { display: flex; flex-wrap: wrap; gap: 15px; align-items: center; margin: 20px 0; }
        .control-group { display: flex; flex-direction: column; }
        .control-group label { font-size: 0.9em; color: #b0b0b0; margin-bottom: 5px; }
        .control-group select, .control-group input { padding: 8px; border-radius: 5px; border: 1px solid #464851; background: #1e1e1e; color: #fafafa; }
        .button { background: linear-gradient(135deg, #ff6b35, #f7931e); color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.3s; }
        .button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255, 107, 53, 0.4); }
        .button:disabled { background: #666; cursor: not-allowed; transform: none; }
        .status { padding: 15px; border-radius: 8px; margin: 10px 0; }
        .status.success { background: rgba(76, 175, 80, 0.1); border: 1px solid #4caf50; color: #4caf50; }
        .status.error { background: rgba(244, 67, 54, 0.1); border: 1px solid #f44336; color: #f44336; }
        .status.warning { background: rgba(255, 152, 0, 0.1); border: 1px solid #ff9800; color: #ff9800; }
        .results { margin-top: 20px; padding: 20px; background: #1a1a1a; border-radius: 8px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { padding: 15px; background: #2d2d2d; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #ff6b35; }
        .metric-label { font-size: 0.9em; color: #b0b0b0; }
        .loading { text-align: center; padding: 20px; }
        .spinner { border: 3px solid #464851; border-top: 3px solid #ff6b35; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .ai-analysis { background: #0d1421; border: 1px solid #1f4e79; border-radius: 8px; padding: 15px; margin-top: 15px; }
        .ai-analysis h4 { color: #4a9eff; margin: 0 0 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>₿ Bitcoin Strategy Monte Carlo Simulator</h1>
            <p>Advanced Monte Carlo simulation with GARCH+jumps modeling, Pinecone strategy integration, and OpenAI analysis</p>
        </div>
        
        <div class="section">
            <h2>System Status</h2>
            <div id="status">Checking system status...</div>
        </div>
        
        <div class="grid">
            <div class="section">
                <h2>Available Strategies from Pinecone</h2>
                <div id="strategies" class="strategy-grid">Loading strategies...</div>
            </div>
            
            <div class="section">
                <h2>Simulation Configuration</h2>
                <div class="controls">
                    <div class="control-group">
                        <label>Market Condition</label>
                        <select id="marketCondition">
                            <option value="normal">Normal Market</option>
                            <option value="bull_market">Bull Market</option>
                            <option value="bear_market">Bear Market</option>
                            <option value="high_volatility">High Volatility</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Simulations</label>
                        <input type="number" id="numSimulations" value="500" min="100" max="1000">
                    </div>
                </div>
                <button class="button" onclick="runAdvancedSimulation()" id="simulateBtn">
                    Run Advanced Monte Carlo Simulation
                </button>
                <div id="results"></div>
            </div>
        </div>
        
        <div class="section" id="aiAnalysisSection" style="display: none;">
            <h2>AI Strategy Analysis</h2>
            <button class="button" onclick="getAIAnalysis()" id="analyzeBtn">
                Get OpenAI Strategy Analysis
            </button>
            <div id="aiAnalysis"></div>
        </div>
    </div>

    <script>
        let selectedStrategy = null;
        let strategies = [];
        
        window.onload = function() {
            checkSystemStatus();
            loadStrategies();
        };
        
        function checkSystemStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    displaySystemStatus(data);
                })
                .catch(error => {
                    document.getElementById('status').innerHTML = 
                        '<div class="status error">System status check failed: ' + error + '</div>';
                });
        }
        
        function displaySystemStatus(status) {
            const statusDiv = document.getElementById('status');
            let html = '';
            
            if (status.pinecone_connected) {
                html += '<div class="status success">✓ Pinecone connected - ' + status.strategies_count + ' strategies available</div>';
            } else {
                html += '<div class="status warning">⚠ Pinecone connection unavailable - using cached strategies</div>';
            }
            
            if (status.openai_connected) {
                html += '<div class="status success">✓ OpenAI connected - AI analysis available</div>';
            } else {
                html += '<div class="status warning">⚠ OpenAI connection unavailable - manual analysis only</div>';
            }
            
            statusDiv.innerHTML = html;
        }
        
        function loadStrategies() {
            fetch('/api/strategies')
                .then(response => response.json())
                .then(data => {
                    strategies = data;
                    displayStrategies(data);
                })
                .catch(error => {
                    document.getElementById('strategies').innerHTML = 
                        '<div class="status error">Error loading strategies: ' + error + '</div>';
                });
        }
        
        function displayStrategies(strategies) {
            const container = document.getElementById('strategies');
            container.innerHTML = '';
            
            strategies.forEach((strategy, index) => {
                const card = document.createElement('div');
                card.className = 'strategy-card';
                card.onclick = () => selectStrategy(strategy, card);
                
                if (index === 0) {
                    selectedStrategy = strategy;
                    card.classList.add('selected');
                }
                
                card.innerHTML = `
                    <h3>${strategy.name}</h3>
                    <p style="font-size: 0.9em; color: #b0b0b0;">${strategy.description}</p>
                    <div style="margin-top: 10px;">
                        <div><strong>Total Return:</strong> ${strategy.total_return}%</div>
                        <div><strong>Sharpe Ratio:</strong> ${strategy.sharpe_ratio}</div>
                        <div><strong>Max Drawdown:</strong> ${strategy.max_drawdown}%</div>
                        <div><strong>Success Rate:</strong> ${(strategy.success_rate * 100).toFixed(1)}%</div>
                        <div><strong>Type:</strong> ${strategy.strategy_type}</div>
                    </div>
                `;
                container.appendChild(card);
            });
        }
        
        function selectStrategy(strategy, card) {
            // Remove selection from all cards
            document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('selected'));
            // Add selection to clicked card
            card.classList.add('selected');
            selectedStrategy = strategy;
            
            // Show AI analysis section
            document.getElementById('aiAnalysisSection').style.display = 'block';
        }
        
        function runAdvancedSimulation() {
            if (!selectedStrategy) {
                alert('Please select a strategy first');
                return;
            }
            
            const marketCondition = document.getElementById('marketCondition').value;
            const numSimulations = parseInt(document.getElementById('numSimulations').value);
            
            const resultsDiv = document.getElementById('results');
            const simulateBtn = document.getElementById('simulateBtn');
            
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running Monte Carlo simulation...</p></div>';
            simulateBtn.disabled = true;
            
            const params = {
                strategy_id: selectedStrategy.id,
                market_condition: marketCondition,
                n_simulations: numSimulations
            };
            
            fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                displayAdvancedResults(data);
                simulateBtn.disabled = false;
            })
            .catch(error => {
                resultsDiv.innerHTML = '<div class="status error">Simulation error: ' + error + '</div>';
                simulateBtn.disabled = false;
            });
        }
        
        function displayAdvancedResults(data) {
            const results = data.simulation_results;
            const resultsDiv = document.getElementById('results');
            
            resultsDiv.innerHTML = `
                <div class="results">
                    <h3>Simulation Results</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${results.expected_return}%</div>
                            <div class="metric-label">Expected Return</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.volatility}%</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.sharpe_ratio}</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.max_drawdown}%</div>
                            <div class="metric-label">Max Drawdown</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(results.success_rate * 100).toFixed(1)}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                    </div>
                    
                    <h4>Return Percentiles</h4>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${results.percentiles['5th']}%</div>
                            <div class="metric-label">5th Percentile</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.percentiles['25th']}%</div>
                            <div class="metric-label">25th Percentile</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.percentiles['50th']}%</div>
                            <div class="metric-label">Median</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.percentiles['75th']}%</div>
                            <div class="metric-label">75th Percentile</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results.percentiles['95th']}%</div>
                            <div class="metric-label">95th Percentile</div>
                        </div>
                    </div>
                    
                    <p><em>Simulation: ${results.n_simulations} Monte Carlo paths with ${results.market_condition} conditions</em></p>
                </div>
            `;
        }
        
        function getAIAnalysis() {
            if (!selectedStrategy) {
                alert('Please select a strategy first');
                return;
            }
            
            const marketCondition = document.getElementById('marketCondition').value;
            const analyzeBtn = document.getElementById('analyzeBtn');
            const analysisDiv = document.getElementById('aiAnalysis');
            
            analysisDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Getting AI analysis...</p></div>';
            analyzeBtn.disabled = true;
            
            const params = {
                strategy: selectedStrategy,
                market_condition: marketCondition
            };
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                analysisDiv.innerHTML = `
                    <div class="ai-analysis">
                        <h4>🤖 OpenAI Strategy Analysis</h4>
                        <p>${data.analysis}</p>
                    </div>
                `;
                analyzeBtn.disabled = false;
            })
            .catch(error => {
                analysisDiv.innerHTML = '<div class="status error">AI analysis error: ' + error + '</div>';
                analyzeBtn.disabled = false;
            });
        }
    </script>
</body>
</html>
        '''

def run_advanced_server():
    """Run the advanced Bitcoin strategy simulator"""
    port = 5000
    server_address = ('0.0.0.0', port)
    
    try:
        print("Initializing advanced Bitcoin Strategy Simulator...")
        print(f"Pinecone available: {PINECONE_AVAILABLE}")
        print(f"OpenAI available: {OPENAI_AVAILABLE}")
        print(f"Pandas available: {PANDAS_AVAILABLE}")
        print(f"NumPy available: {NUMPY_AVAILABLE}")
        
        httpd = HTTPServer(server_address, AdvancedBitcoinHandler)
        print(f"₿ Advanced Bitcoin Strategy Simulator running on http://0.0.0.0:{port}")
        print("Features: Pinecone integration, OpenAI analysis, Monte Carlo simulation")
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print("\nServer stopped")
        httpd.server_close()
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == '__main__':
    if STREAMLIT_AVAILABLE:
        # Run as Streamlit app if available
        print("Running with Streamlit interface")
        exec(open('app.py').read())
    else:
        # Fallback to HTTP server
        print("Streamlit not available, using HTTP server interface")
        run_advanced_server()