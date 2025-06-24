import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time

from utils.bitcoin_data import fetch_bitcoin_data
from utils.monte_carlo import MonteCarloSimulator
from utils.pinecone_client import PineconeClient
from utils.visualization import create_fan_chart, create_terminal_price_histogram, create_cagr_distribution
from utils.market_conditions import MarketCondition, get_condition_description
from utils.simulation_router import SimulationRouter
from utils.multi_factor_data import MultiFactorDataFetcher

def simulate_portfolio_optimized(portfolio_inputs, n_paths=1000, n_days=365, strategy_lookup=None, 
                                market_condition_options=None, pinecone_client=None, bitcoin_data=None, progress_callback=None):
    """
    Optimized portfolio simulation that runs all strategies together
    """
    print("[Portfolio] Starting optimized portfolio simulation...")
    
    # Filter valid strategies
    valid_inputs = [p for p in portfolio_inputs if p["strategy_id"] != "Select a strategy..."]
    if not valid_inputs:
        raise ValueError("No valid strategies selected")
    
    # Normalize weights
    total_weight = sum(p["weight"] for p in valid_inputs)
    if abs(total_weight - 100.0) > 0.1:
        raise ValueError(f"Portfolio weights must sum to 100%, got {total_weight}")
    
    # Pre-process all strategies to get their requirements
    from utils.strategy_processor import StrategyProcessor
    processor = StrategyProcessor()
    router = SimulationRouter()
    
    strategy_configs = []
    all_required_vars = set(['BTC'])  # Always need BTC
    
    print("[Portfolio] Analyzing strategy requirements...")
    for p in valid_inputs:
        strategy_metadata = strategy_lookup.get(p["strategy_id"])
        if not strategy_metadata:
            continue
            
        # Generate strategy function
        strategy_func = processor.generate_conditions(strategy_metadata)
        
        # Determine requirements
        simulation_mode = router.select_simulation_mode(strategy_metadata)
        required_vars = router.get_required_variables(strategy_metadata, simulation_mode)
        
        if simulation_mode == 'multi_factor' and required_vars:
            all_required_vars.update(required_vars)
        
        strategy_configs.append({
            'input': p,
            'metadata': strategy_metadata,
            'function': strategy_func,
            'mode': simulation_mode,
            'variables': required_vars
        })
    
    print(f"[Portfolio] Total required variables: {all_required_vars}")
    
    # Initialize simulator
    simulator = MonteCarloSimulator(bitcoin_data, pinecone_client)
    
    # Generate base price paths once for all strategies
    print(f"[Portfolio] Generating {n_paths} base price paths...")
    
    # Convert condition name to MarketCondition enum
    market_condition = None
    for name, condition, _ in market_condition_options or []:
        if name == valid_inputs[0]["condition"]:  # Use first strategy's condition
            market_condition = condition
            break
    
    # Check if we need multi-factor simulation
    if len(all_required_vars) > 1:
        print("[Portfolio] Using multi-factor simulation...")
        # Fetch all required data once
        data_fetcher = MultiFactorDataFetcher(pinecone_client)
        historical_data = data_fetcher.fetch_multi_factor_data(list(all_required_vars))
        
        # Generate correlated paths for all variables
        correlation_matrix = data_fetcher.calculate_correlation_matrix(historical_data)
        simulated_paths = data_fetcher.simulate_multi_factor_series(
            historical_data, n_paths, n_days, correlation_matrix, 
            market_condition=market_condition
        )
        
        # Extract BTC paths
        if 'BTC' in simulated_paths:
            close_paths = np.array([simulated_paths['BTC'][i] for i in range(n_paths)])
        else:
            raise ValueError("BTC data missing from multi-factor simulation")
    else:
        print("[Portfolio] Using BTC-only simulation...")
        # Single factor (BTC only) simulation
        close_paths = simulator.generate_price_paths(n_paths, n_days, market_condition)
    
    # Pre-generate all OHLCV data at once
    print("[Portfolio] Synthesizing OHLCV data...")
    start_date = simulator.last_date + pd.Timedelta(days=1)
    all_ohlcv_data = simulator.synthesize_ohlcv_batch(close_paths, start_date)
    
    # Add multi-factor data to OHLCV if needed
    if len(all_required_vars) > 1:
        for i, ohlcv_df in enumerate(all_ohlcv_data):
            for var in all_required_vars:
                if var != 'BTC' and var in simulated_paths:
                    ohlcv_df[var] = simulated_paths[var][i]
    
    # Execute each strategy on the same paths and combine results
    print("[Portfolio] Executing strategies...")
    portfolio_returns = np.zeros((n_paths, n_days + 1))
    portfolio_returns[:, 0] = 1.0  # Start with $1
    
    for config_idx, config in enumerate(strategy_configs):
        weight = config['input']['weight'] / 100.0
        strategy_func = config['function']
        print(f"[Portfolio] Processing {config['metadata'].get('description', 'Unknown')[:50]}... (weight: {weight*100:.1f}%)")
        batch_size = min(100, n_paths)
        strategy_equity_curves = []
        for batch_start in range(0, n_paths, batch_size):
            batch_end = min(batch_start + batch_size, n_paths)
            batch_ohlcv = all_ohlcv_data[batch_start:batch_end]
            for ohlcv_df in batch_ohlcv:
                try:
                    returns = simulator.execute_strategy(ohlcv_df, strategy_func)
                    equity = simulator.calculate_equity_curve(returns)
                    strategy_equity_curves.append(equity.values)
                except Exception as e:
                    print(f"[Portfolio] Strategy execution failed: {e}")
                    strategy_equity_curves.append(np.ones(n_days + 1))
            # Update progress after each batch
            if progress_callback is not None:
                progress_callback((batch_end + config_idx * n_paths) / (n_paths * len(strategy_configs)))
        strategy_equity = np.array(strategy_equity_curves)
        weighted_contribution = strategy_equity * weight
        portfolio_returns += weighted_contribution
    
    print("[Portfolio] Portfolio simulation completed!")
    return portfolio_returns

# Page configuration
st.set_page_config(
    page_title="Bitcoin Strategy Simulator",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject futuristic CSS styling
st.markdown("""
<style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
        background-attachment: fixed;
    }
    
    /* Futuristic title styling */
    h1 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, #00D4FF 0%, #00FFF0 50%, #00D4FF 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-align: center;
        font-weight: 900;
        letter-spacing: 2px;
        margin-bottom: 0;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    /* Glowing headers */
    h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #00D4FF;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Futuristic metrics containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 240, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Neon buttons */
    .stButton > button {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(45deg, #00D4FF, #0099CC);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 1px;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6), 0 0 60px rgba(0, 212, 255, 0.4);
    }
    
    /* Primary button special styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #00FFF0, #00D4FF);
        box-shadow: 0 0 30px rgba(0, 255, 240, 0.5);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F2E 0%, #0E1117 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div > div {
        background-color: rgba(0, 212, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px;
        color: #FAFAFA;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #00FFF0);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Success/Error/Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
        font-family: 'Orbitron', monospace;
        letter-spacing: 1px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00D4FF, #0099CC);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Orbitron', monospace;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 8px;
    }
    
    /* DataFrame styling */
    .dataframe {
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Download button special */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #00FFF0, #00D4FF);
        border: 2px solid rgba(0, 255, 240, 0.5);
    }
    
    /* Animated gradient border for special containers */
    @keyframes gradient-border {
        0% { border-color: #00D4FF; }
        50% { border-color: #00FFF0; }
        100% { border-color: #00D4FF; }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1F2E;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00D4FF, #0099CC);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00FFF0, #00D4FF);
    }
    
    /* Constrain all selectboxes and number inputs in the main content area */
    section.main > div > div > div > div > div[data-testid="stVerticalBlock"] {
        max-width: 900px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    /* Constrain selectboxes and number inputs */
    div[data-testid="stSelectbox"], div[data-testid="stNumberInput"] {
        max-width: 700px !important;
        min-width: 120px !important;
    }
    /* Make columns in the config matrix tighter */
    div[data-testid="column"] {
        max-width: 700px !important;
    }
</style>
""", unsafe_allow_html=True)

# Futuristic title with subtitle
st.markdown("""
<h1 style='text-align: center; margin-bottom: 0;'>₿ BITCOIN STRATEGY SIMULATOR</h1>
<p style='text-align: center; font-family: Orbitron, monospace; color: #00D4FF; font-size: 18px; 
          letter-spacing: 3px; margin-top: 10px; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>
    ADVANCED MONTE CARLO ENGINE • GARCH+JUMPS • MULTI-FACTOR ANALYSIS
</p>
<div style='height: 2px; background: linear-gradient(90deg, transparent, #00D4FF, transparent); 
            margin: 20px 0 30px 0;'></div>
""", unsafe_allow_html=True)

# Initialize Pinecone client
@st.cache_resource
def init_pinecone():
    # Check multiple sources for the API key
    api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
    if not api_key:
        return None, "API key not found"
    try:
        client = PineconeClient(api_key)
        # Test connection by listing indexes
        client.pc.list_indexes()
        return client, "Connected"
    except Exception as e:
        return None, f"Connection failed: {str(e)}"

@st.cache_resource
def check_openai_connection():
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return False, "API key not found"
    try:
        if api_key.startswith("sk-"):
            return True, "Connected"
        else:
            return False, "Invalid API key format"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_strategies_from_pinecone(_pinecone_client):
    """Load all strategy IDs and their metadata from Pinecone."""
    if not _pinecone_client or not _pinecone_client.index:
        return pd.DataFrame()

    try:
        all_ids = []
        for page in _pinecone_client.index.list():
            all_ids.extend(page)

        if not all_ids:
            st.warning("No strategies found in Pinecone index.")
            return pd.DataFrame()

        batch_size = 100
        all_metadata = []
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            try:
                response = _pinecone_client.index.fetch(ids=batch_ids)
                if response and hasattr(response, 'vectors') and response.vectors:
                    for vec_id, vec_data in response.vectors.items():
                        if hasattr(vec_data, 'metadata'):
                            metadata = vec_data.metadata
                            metadata['id'] = vec_id
                            all_metadata.append(metadata)
                        else:
                            all_metadata.append({'id': vec_id, 'description': 'N/A', 'strategy_type': 'unknown'})
            except Exception as e:
                st.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        if not all_metadata:
            return pd.DataFrame()

        df = pd.DataFrame(all_metadata)
        
        if 'strategy_type' not in df.columns:
            df['strategy_type'] = 'unknown'
        df['strategy_type'] = df['strategy_type'].fillna('unknown')

        if 'description' not in df.columns:
            df['description'] = df['id'].apply(lambda x: x.replace('_', ' ').title())
        
        df['description'] = df['description'].astype(str)
        return df
    
    except Exception as e:
        st.error(f"An unexpected error occurred during strategy loading: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Initialize session state
if 'bitcoin_data' not in st.session_state:
    st.session_state.bitcoin_data = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'generated_strategy' not in st.session_state:
    st.session_state.generated_strategy = None
if 'current_strategy_name' not in st.session_state:
    st.session_state.current_strategy_name = None
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = 'baseline'  # Set default to 'baseline'
if 'required_variables' not in st.session_state:
    st.session_state.required_variables = None

# Initialize API connections
pinecone_client, pinecone_status = init_pinecone()
openai_connected, openai_status = check_openai_connection()

# Sidebar configuration
# API Connection Status with futuristic styling
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 240, 0.05)); 
            border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
    <h3 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0; text-align: center;
               text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>SYSTEM STATUS</h3>
</div>
""", unsafe_allow_html=True)

# Pinecone status with glow effect
pinecone_status_color = "#00FF00" if pinecone_client else "#FF0040"
pinecone_glow = "0 0 10px rgba(0, 255, 0, 0.8)" if pinecone_client else "0 0 10px rgba(255, 0, 64, 0.8)"
st.sidebar.markdown(f"""
<div style='display: flex; align-items: center; margin: 10px 0;'>
    <div style='width: 12px; height: 12px; border-radius: 50%; background: {pinecone_status_color}; 
                box-shadow: {pinecone_glow}; margin-right: 10px;'></div>
    <span style='font-family: Orbitron, monospace; font-size: 14px;'>
        <strong>PINECONE:</strong> {pinecone_status}
    </span>
</div>
""", unsafe_allow_html=True)

# OpenAI status with glow effect
openai_status_color = "#00FF00" if openai_connected else "#FF0040"
openai_glow = "0 0 10px rgba(0, 255, 0, 0.8)" if openai_connected else "0 0 10px rgba(255, 0, 64, 0.8)"
st.sidebar.markdown(f"""
<div style='display: flex; align-items: center; margin: 10px 0;'>
    <div style='width: 12px; height: 12px; border-radius: 50%; background: {openai_status_color}; 
                box-shadow: {openai_glow}; margin-right: 10px;'></div>
    <span style='font-family: Orbitron, monospace; font-size: 14px;'>
        <strong>OPENAI:</strong> {openai_status}
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='height: 1px; background: linear-gradient(90deg, transparent, #00D4FF, transparent); 
            margin: 20px 0;'></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 240, 0.05)); 
            border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
    <h3 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0; text-align: center;
               text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>SIMULATION PARAMETERS</h3>
</div>
""", unsafe_allow_html=True)

# Number of simulations slider
n_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=1000,
    max_value=20000,
    value=20000,
    step=1000,
    help="Higher numbers provide more accurate results but take longer to compute"
)

# Simulation period slider
simulation_days = st.sidebar.slider(
    "Simulation Period (Days)",
    min_value=30,
    max_value=365,
    value=365,
    step=5,
    help="Number of days to simulate forward"
)

# Market Condition Selection with Enhanced UI
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 240, 0.05)); 
            border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
    <h3 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0; text-align: center;
               text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>MARKET SCENARIO</h3>
</div>
""", unsafe_allow_html=True)

# Enhanced market condition options with emojis and better descriptions
market_condition_options = [
    ("🔄 Baseline (Historical Data)", None, "Use actual historical Bitcoin patterns"),
    ("🚀 High Vol Bull Market", MarketCondition.HIGH_VOL_UP, "Explosive growth with high swings"),
    ("💥 High Vol Crash", MarketCondition.HIGH_VOL_DOWN, "Panic selling, sharp drops"),
    ("🌊 High Vol Sideways", MarketCondition.HIGH_VOL_STABLE, "Volatile but directionless"),
    ("📈 Stable Bull Trend", MarketCondition.STABLE_VOL_UP, "Steady upward grind"),
    ("📉 Stable Bear Trend", MarketCondition.STABLE_VOL_DOWN, "Slow bleed downward"),
    ("➡️ Stable Sideways", MarketCondition.STABLE_VOL_STABLE, "Low volatility ranging")
]

# Custom radio button implementation for better UX
selected_condition_idx = st.sidebar.radio(
    "Select Market Scenario:",
    range(len(market_condition_options)),
    format_func=lambda idx: market_condition_options[idx][0],
    help="Choose a market environment to stress-test your strategy",
    key="market_condition_radio"
)

# Get selected condition details
selected_condition_name, selected_market_condition, short_desc = market_condition_options[selected_condition_idx]

# Display enhanced condition info
if selected_market_condition:
    # Get detailed description
    detailed_desc = get_condition_description(selected_market_condition)
    
    # Create info box with condition details
    st.sidebar.markdown(f"""
    <div style='background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2); 
                border-radius: 8px; padding: 12px; margin-top: 10px;'>
        <div style='font-family: Orbitron, monospace; color: #00FFF0; font-size: 14px; 
                   margin-bottom: 8px; font-weight: bold;'>{selected_condition_name}</div>
        <div style='font-family: Arial, sans-serif; color: #FAFAFA; font-size: 13px; 
                   line-height: 1.5;'>{detailed_desc}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2); 
                border-radius: 8px; padding: 12px; margin-top: 10px;'>
        <div style='font-family: Orbitron, monospace; color: #00FFF0; font-size: 14px; 
                   margin-bottom: 8px; font-weight: bold;'>Historical Baseline</div>
        <div style='font-family: Arial, sans-serif; color: #FAFAFA; font-size: 13px; 
                   line-height: 1.5;'>Using actual Bitcoin historical drift and volatility patterns 
                   without any scenario adjustments.</div>
    </div>
    """, unsafe_allow_html=True)

# --- Strategy selection ---
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 240, 0.05)); 
            border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
    <h3 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0; text-align: center;
               text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>STRATEGY SELECTION</h3>
</div>
""", unsafe_allow_html=True)

# Load strategies quickly from Pinecone
strategies_df = pd.DataFrame()
if pinecone_client:
    try:
        strategies_df = load_strategies_from_pinecone(pinecone_client)
        if not strategies_df.empty:
            st.sidebar.success(f"Loaded {len(strategies_df)} strategies")
        else:
            st.sidebar.warning("No strategies found in database")
    except Exception as e:
        st.sidebar.error(f"Error loading strategies: {str(e)}")
else:
    st.sidebar.warning("Pinecone connection required to load strategies")


# --- Strategy Filtering UI ---
filtered_strategies = []
strategy_options = ["Select a strategy..."]
strategy_lookup = {"Select a strategy...": None}

if not strategies_df.empty:
    # Dynamic Strategy Type Filter
    unique_types = sorted(strategies_df['strategy_type'].unique())
    
    # Header for filter section
    st.sidebar.markdown("""
    <div style='font-family: Orbitron, monospace; color: #00D4FF; font-size: 14px; 
               letter-spacing: 1px; margin-bottom: 10px;'>FILTER BY STRATEGY TYPE</div>
    """, unsafe_allow_html=True)
    
    # Create all options list
    all_options = ['Show all correlation strategies'] + [f'{t}' for t in unique_types]
    
    # Style the radio buttons to match the futuristic theme
    st.markdown("""
    <style>
    div[role="radiogroup"] > label {
        background: transparent !important;
        padding: 8px 0 !important;
    }
    div[role="radiogroup"] > label:hover {
        background: rgba(0, 212, 255, 0.05) !important;
    }
    div[role="radiogroup"] > label > div:first-child {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[role="radiogroup"] > label > div:first-child > div {
        width: 16px !important;
        height: 16px !important;
        border: 2px solid #00D4FF !important;
        background: transparent !important;
    }
    div[role="radiogroup"] > label > div:first-child > div[data-checked="true"] {
        background: #00D4FF !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
    }
    div[role="radiogroup"] > label > div:first-child > div[data-checked="true"]::after {
        content: '';
        display: block;
        width: 6px;
        height: 6px;
        background: #0E1117;
        border-radius: 50%;
        position: relative;
        top: 3px;
        left: 3px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's native radio button
    selected_option = st.sidebar.radio(
        "Strategy Type Filter",
        options=all_options,
        index=0,
        label_visibility="collapsed",
        help="Filter strategies by type"
    )
    
    # Simple separator
    st.sidebar.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # Apply type filter based on selected option
    if selected_option == 'Show all correlation strategies':
        filtered_df = strategies_df.copy()
    else:
        # Extract the strategy type from the formatted option
        filtered_df = strategies_df[strategies_df['strategy_type'] == selected_option].copy()

    # Search functionality
    search_term = st.sidebar.text_input(
        "Search within selected type",
        placeholder="Keywords (e.g., momentum, SPY)...",
        help="Search through filtered strategies by keywords in description"
    )

    if search_term:
        search_lower = search_term.lower()
        filtered_df = filtered_df[filtered_df['description'].astype(str).str.lower().str.contains(search_lower)]

    # Create strategy options from the filtered dataframe
    if not filtered_df.empty:
        # Sort by description for consistent ordering
        filtered_df = filtered_df.sort_values(by='description')
        filtered_strategies = filtered_df.to_dict('records')
        strategy_options = [s['description'] for s in filtered_strategies]
        strategy_lookup = {s['description']: s for s in filtered_strategies}
    else:
        st.sidebar.warning("No strategies match your criteria.")

selected_strategy_name = st.sidebar.selectbox(
    f"Choose from {len(filtered_strategies)} strategies",
    options=["Select a strategy..."] + strategy_options,
    help="Select a strategy to view details and run simulation"
)

# Strategy generation section
# Check if strategy has changed
if 'current_strategy_name' not in st.session_state or st.session_state.current_strategy_name != selected_strategy_name:
    st.session_state.generated_strategy = None
    st.session_state.current_strategy_name = selected_strategy_name
    st.session_state.simulation_mode = 'baseline'  # Reset to baseline on strategy change
    st.session_state.required_variables = None

strategy_generated = st.session_state.generated_strategy is not None
processed_strategy = st.session_state.generated_strategy
simulation_mode = st.session_state.simulation_mode
required_variables = st.session_state.required_variables

if selected_strategy_name != "Select a strategy...":
    # Show strategy selection status
    st.sidebar.success("✅ Strategy selected and ready for simulation")
    
    # Show detection preview if available
    if simulation_mode:
        st.sidebar.info(f"🔧 Mode: **{simulation_mode.replace('_', ' ').title()}**")
        if simulation_mode == 'multi_factor' and required_variables:
            st.sidebar.info(f"📊 Variables: {', '.join(required_variables)}")
else:
    st.sidebar.info("Please select a strategy to continue")

# Enhanced strategy details panel
if (selected_strategy_name != "Select a strategy..." and 
    selected_strategy_name in strategy_lookup and 
    strategy_lookup[selected_strategy_name] is not None):
    
    selected_strategy = strategy_lookup[selected_strategy_name]
    
    st.sidebar.subheader("Strategy Details")
    
    if selected_strategy is not None:
        strategy_type = selected_strategy.get('strategy_type', 'Unknown')
        st.sidebar.markdown(f"**Type:** `{strategy_type}`")
        
        description = selected_strategy.get('description', 'N/A')
        with st.sidebar.expander("📋 View Full Description", expanded=False):
            st.write(description)
        
        # Strategy dependencies
        dependencies = selected_strategy.get('dependencies', [])
        if dependencies:
            st.sidebar.markdown(f"**Dependencies:** {', '.join(dependencies)}")
        
        # Performance metrics in organized layout
        st.sidebar.subheader("Historical Performance")
        
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            # Get value, convert to numeric, handle missing, then format
            total_return = pd.to_numeric(selected_strategy.get('total_return', 0), errors='coerce')
            st.sidebar.metric("Total Return", f"{total_return:.1f}%" if pd.notna(total_return) else "N/A")

            sharpe = pd.to_numeric(selected_strategy.get('sharpe_ratio', 0), errors='coerce')
            st.sidebar.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
        
        with col_b:
            max_dd = pd.to_numeric(selected_strategy.get('max_drawdown', 0), errors='coerce')
            st.sidebar.metric("Max Drawdown", f"{abs(max_dd):.1f}%" if pd.notna(max_dd) else "N/A")
            
            success_rate = pd.to_numeric(selected_strategy.get('success_rate', 0), errors='coerce')
            st.sidebar.metric("Success Rate", f"{success_rate:.1%}" if pd.notna(success_rate) else "N/A")
        
        st.sidebar.subheader("Trading Details")
        avg_holding = selected_strategy.get('avg_holding_days', 'N/A')
        total_trades = selected_strategy.get('total_trades', 'N/A')
        
        st.sidebar.write(f"🕒 **Avg Holding:** {avg_holding} days")
        st.sidebar.write(f"📊 **Total Trades:** {total_trades}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Bitcoin Data")
    
    # Load Bitcoin data button
    if st.button("Load Bitcoin Data", type="primary"):
        with st.spinner("Fetching Bitcoin data..."):
            try:
                bitcoin_data = fetch_bitcoin_data()
                st.session_state.bitcoin_data = bitcoin_data
                st.success(f"Successfully loaded {len(bitcoin_data)} days of Bitcoin data")
            except Exception as e:
                st.error(f"Error loading Bitcoin data: {str(e)}")

    # Display Bitcoin data if available
    if st.session_state.bitcoin_data is not None:
        df = st.session_state.bitcoin_data
        
        # Show basic statistics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
        with col_b:
            daily_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
            st.metric("Daily Change", f"{daily_change:+.2f}%")
        with col_c:
            volatility = df['Close'].pct_change().std() * np.sqrt(365) * 100
            st.metric("Annual Volatility", f"{volatility:.1f}%")
        with col_d:
            st.metric("Data Points", f"{len(df):,}")
        
        # Futuristic price chart
        fig_price = go.Figure()
        
        # Add glow effect with multiple traces
        for width, opacity in [(8, 0.2), (4, 0.4), (2, 0.8)]:
            fig_price.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Bitcoin Price',
                line=dict(color='#00D4FF', width=width),
                opacity=opacity,
                showlegend=False
            ))
        
        # Main price line
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#00FFF0', width=2),
            showlegend=False
        ))
        
        # Futuristic layout
        fig_price.update_layout(
            title={
                'text': "BITCOIN PRICE MATRIX",
                'font': {'family': 'Orbitron, monospace', 'size': 20, 'color': '#00D4FF'}
            },
            xaxis_title="Temporal Coordinates",
            yaxis_title="Value Vector (USD)",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(14, 17, 23, 0.9)',
            font=dict(family='Orbitron, monospace', color='#FAFAFA'),
            xaxis=dict(
                gridcolor='rgba(0, 212, 255, 0.1)',
                zerolinecolor='rgba(0, 212, 255, 0.3)'
            ),
            yaxis=dict(
                gridcolor='rgba(0, 212, 255, 0.1)',
                zerolinecolor='rgba(0, 212, 255, 0.3)',
                tickformat='$,.0f'
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig_price, use_container_width=True, key="bitcoin_price_chart")

with col2:
    st.header("Quick Stats")
    if st.session_state.bitcoin_data is not None:
        df = st.session_state.bitcoin_data
        returns = df['Close'].pct_change().dropna()
        
        stats_data = {
            "Metric": ["Mean Daily Return", "Std Daily Return", "Skewness", "Kurtosis", "Max Drawdown"],
            "Value": [
                f"{returns.mean()*100:.4f}%",
                f"{returns.std()*100:.4f}%",
                f"{returns.skew():.4f}",
                f"{returns.kurtosis():.4f}",
                f"{((df['Close'] / df['Close'].cummax() - 1).min() * 100):.2f}%"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)

# Monte Carlo Simulation Section
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 240, 0.02)); 
            border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 15px; padding: 20px; margin: 30px 0;'>
    <h2 style='font-family: Orbitron, monospace; color: #00D4FF; text-align: center; margin-top: 0;
               text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);'>MONTE CARLO SIMULATION ENGINE</h2>
</div>
""", unsafe_allow_html=True)

if st.session_state.bitcoin_data is not None:
    # Simulation Mode Toggle
    simulation_type = st.radio(
        "Choose Simulation Mode:",
        options=["Single Strategy", "Portfolio Simulation"],
        index=0,
        horizontal=True,
        help="Single Strategy runs one strategy at a time. Portfolio Simulation allows you to combine multiple strategies with custom weights."
    )
    
    st.markdown("""
    <div style='height: 1px; background: linear-gradient(90deg, transparent, #00D4FF, transparent); 
                margin: 20px 0;'></div>
    """, unsafe_allow_html=True)
    
    if simulation_type == "Single Strategy":
        col_sim1, col_sim2 = st.columns([3, 1])
        with col_sim1:
            # Check if strategy has been selected
            if selected_strategy_name == "Select a strategy...":
                st.info("Please select a strategy to run simulation")
            else:
                if st.button("Run Simulation", type="primary"):
                    # Clear all previous simulation outputs
                    st.session_state.simulation_results = None
                    st.session_state.generated_strategy = None
                    st.session_state.current_strategy_name = None
                    st.session_state.simulation_mode = None
                    st.session_state.required_variables = None
                    # Enhanced progress tracking
                    progress_container = st.container()
                    with progress_container:
                        # Create multiple progress indicators
                        overall_progress = st.progress(0, text="Initializing simulation...")
                        status_placeholder = st.empty()
                        try:
                            # Step 1: Strategy Analysis
                            overall_progress.progress(10, text="Analyzing strategy...")
                            status_placeholder.info("🔍 Analyzing strategy and detecting dependencies")
                            selected_strategy_data = strategy_lookup[selected_strategy_name]
                            if not selected_strategy_data:
                                st.error("Could not find strategy data")
                            else:
                                # Step 2: Generate strategy code
                                overall_progress.progress(20, text="Generating strategy conditions...")
                                status_placeholder.info("⚙️ Converting strategy to executable conditions")
                                from utils.strategy_processor import StrategyProcessor
                                processor = StrategyProcessor()
                                processed_strategy = processor.generate_conditions(selected_strategy_data)
                                # Step 3: Smart routing
                                overall_progress.progress(30, text="Determining simulation mode...")
                                status_placeholder.info("🧠 Smart routing: selecting optimal simulation engine")
                                router = SimulationRouter()
                                simulation_mode = router.select_simulation_mode(selected_strategy_data)
                                required_variables = router.get_required_variables(selected_strategy_data, simulation_mode)
                                # Multi-factor strategy confirmation
                                if simulation_mode == 'multi_factor':
                                    macro_vars = [v for v in required_variables if v != 'BTC']
                                    if macro_vars:
                                        status_placeholder.info(f"📊 Multi-factor simulation with: {', '.join(macro_vars)}")
                                # Store in session state
                                st.session_state.generated_strategy = processed_strategy
                                st.session_state.current_strategy_name = selected_strategy_name
                                st.session_state.simulation_mode = simulation_mode
                                st.session_state.required_variables = required_variables
                                # Step 4: Data preparation
                                overall_progress.progress(40, text="Preparing simulation data...")
                                mode_display = simulation_mode.replace('_', ' ').title() if simulation_mode else "BTC Only"
                                vars_display = ', '.join(required_variables) if required_variables else "BTC"
                                status_placeholder.info(f"📊 Setting up {mode_display} simulation with {vars_display}")
                                # Step 5: Initialize simulator
                                overall_progress.progress(50, text="Initializing Monte Carlo engine...")
                                status_placeholder.info("🎲 Initializing Monte Carlo simulation engine")
                                simulator = MonteCarloSimulator(st.session_state.bitcoin_data, pinecone_client)
                                # Step 6: Run simulation with detailed progress
                                overall_progress.progress(60, text="Running Monte Carlo simulation...")
                                status_placeholder.info(f"🚀 Running {n_simulations} simulations over {simulation_days} days")
                                
                                # Simple progress callback function
                                def simple_progress_callback(progress):
                                    overall_progress.progress(60 + int(progress * 30), text=f"Running simulations... {int(progress * 100)}%")
                                
                                results = simulator.run_simulation(
                                    n_simulations=n_simulations,
                                    simulation_days=simulation_days,
                                    selected_strategy=processed_strategy,
                                    market_condition=selected_market_condition,
                                    progress_callback=simple_progress_callback,
                                    simulation_mode=simulation_mode or 'btc_only',
                                    required_variables=required_variables or ['BTC']
                                )
                                # Completion
                                overall_progress.progress(100, text="Simulation completed!")
                                st.session_state.simulation_results = results
                                
                                completion_mode = results.get('simulation_mode', simulation_mode)
                                completion_display = completion_mode.replace('_', ' ').title() if completion_mode else "BTC Only"
                                
                                status_placeholder.success(f"✅ {completion_display} simulation completed successfully!")
                                
                                # Show simulation summary
                                if results.get('simulation_mode') == 'multi_factor':
                                    used_vars = results.get('variables_used', required_variables)
                                    if used_vars:
                                        st.info(f"📊 Simulated variables: {', '.join(used_vars)}")
                                
                                # Brief results preview
                                median_cagr = results.get('median_cagr', 0)
                                worst_cagr = results.get('worst_decile_cagr', 0)
                                st.info(f"📈 Results preview: Median CAGR {median_cagr:.1f}%, Worst 10% CAGR {worst_cagr:.1f}%")
                        
                        except Exception as e:
                            st.error(f"Error running simulation: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        with col_sim2:
            if st.session_state.simulation_results is not None:
                results = st.session_state.simulation_results
                st.metric("Median CAGR", f"{results['median_cagr']:.2f}%")
                st.metric("Worst 10% CAGR", f"{results['worst_decile_cagr']:.2f}%")
                st.metric("Median Max DD", f"{results['median_max_drawdown']:.2f}%")

    elif simulation_type == "Portfolio Simulation":
        # Portfolio Simulation UI with Dynamic Table
        st.markdown(f"""
        <div style='background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2); 
                    border-radius: 10px; padding: 20px; margin-bottom: 20px; max-width: 900px; margin-left: auto; margin-right: auto;'>
            <h3 style='font-family: Orbitron, monospace; color: #00FFF0; margin-top: 0;'>
                📊 PORTFOLIO BUILDER
            </h3>
            <p style='color: #FAFAFA; margin-bottom: 10px;'>
                Configure multiple strategies with custom weights for your portfolio.
            </p>
            <div style='background: rgba(0, 212, 255, 0.02); border: 1px solid rgba(0, 212, 255, 0.1); 
                        border-radius: 8px; padding: 10px; margin-top: 10px;'>
                <span style='font-family: Orbitron, monospace; color: #00D4FF; font-size: 13px;'>
                    MARKET SCENARIO: 
                </span>
                <span style='font-family: Orbitron, monospace; color: #00FFF0; font-size: 13px; font-weight: bold;'>
                    {selected_condition_name}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Number of strategies input
        num_strategies = st.number_input(
            "Number of strategies in portfolio", 
            min_value=2, 
            max_value=10, 
            value=2,
            help="Choose how many strategies to include in your portfolio"
        )
        
        # Create market condition options for portfolio
        condition_options = [name for name, _, _ in market_condition_options]
        
        # Portfolio inputs table
        portfolio_inputs = []
        total_weight = 0
        
        st.markdown("""
        <div style='background: rgba(0, 212, 255, 0.02); border: 1px solid rgba(0, 212, 255, 0.1); 
                    border-radius: 10px; padding: 15px; margin: 20px 0; max-width: 900px; margin-left: auto; margin-right: auto;'>
            <h4 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0;'>
                STRATEGY CONFIGURATION MATRIX
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Table headers
        col_header1, col_header2 = st.columns([5, 1])
        with col_header1:
            st.markdown("**Strategy Selection**")
        with col_header2:
            st.markdown("**Weight (%)**")
        
        # Dynamic strategy input rows
        for i in range(num_strategies):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                strategy_id = st.selectbox(
                    f"Strategy {i+1}", 
                    strategy_options, 
                    key=f"portfolio_strategy_{i}",
                    label_visibility="collapsed"
                )
            
            with col2:
                default_weight = 100.0 / num_strategies
                weight = st.number_input(
                    f"Weight {i+1} (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=default_weight,
                    step=5.0,
                    key=f"portfolio_weight_{i}",
                    label_visibility="collapsed"
                )
                total_weight += weight
            
            # Store portfolio input with the selected market condition from sidebar
            portfolio_inputs.append({
                "strategy_id": strategy_id,
                "weight": weight,
                "condition": selected_condition_name  # Use the sidebar-selected condition
            })
        
        # Portfolio validation and summary
        st.markdown("---")
        
        # Display total weight with validation
        weight_color = "#00FF00" if abs(total_weight - 100) < 0.1 else "#FF0040"
        st.markdown(f"""
        <div style='text-align: center; margin: 20px 0; max-width: 900px; margin-left: auto; margin-right: auto;'>
            <span style='font-family: Orbitron, monospace; color: {weight_color}; 
                        font-size: 24px; font-weight: bold;'>
                Total Portfolio Weight: {total_weight:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation checks
        valid_strategies = [p for p in portfolio_inputs if p["strategy_id"] != "Select a strategy..."]
        unique_strategies = len(set(p["strategy_id"] for p in valid_strategies))
        
        # Weight validation check
        total_weight_check = sum(p["weight"] for p in portfolio_inputs)
        
        if len(valid_strategies) < 2:
            st.error("❌ Please select at least 2 different strategies for portfolio simulation.")
        elif unique_strategies != len(valid_strategies):
            st.warning("⚠️ Duplicate strategies detected. Each strategy should be unique in the portfolio.")
        elif abs(total_weight_check - 100.0) > 0.1:
            st.warning(f"⚠️ Portfolio weights must sum to 100%. Current total: {total_weight_check:.2f}%")
        else:
            st.success("✅ Ready to simulate portfolio - all validations passed!")
        
        # Portfolio summary
        if valid_strategies:
            st.markdown("""
            <div style='background: rgba(0, 212, 255, 0.02); border: 1px solid rgba(0, 212, 255, 0.1); 
                        border-radius: 10px; padding: 15px; margin: 20px 0; max-width: 900px; margin-left: auto; margin-right: auto;'>
                <h4 style='font-family: Orbitron, monospace; color: #00D4FF; margin-top: 0;'>
                    PORTFOLIO SUMMARY
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Portfolio configuration summary
            st.markdown("### Portfolio Configuration")
            # Tasteful display of the market scenario used
            st.markdown(f"""
            <div style='background: rgba(0, 212, 255, 0.04); border: 1px solid rgba(0, 212, 255, 0.12); border-radius: 8px; padding: 10px 15px; margin-bottom: 10px; max-width: 900px; margin-left: auto; margin-right: auto;'>
                <span style='font-family: Orbitron, monospace; color: #00D4FF; font-size: 13px;'>
                    MARKET SCENARIO:
                </span>
                <span style='font-family: Orbitron, monospace; color: #00FFF0; font-size: 13px; font-weight: bold;'>
                    {selected_condition_name}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            for i, strategy_input in enumerate(valid_strategies):
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px; margin: 5px 0; background: rgba(0, 212, 255, 0.05); 
                           border-radius: 5px; max-width: 900px; margin-left: auto; margin-right: auto;'>
                    <span style='font-family: Orbitron, monospace; color: #00FFF0;'>
                        {i+1}. {strategy_input['strategy_id']}
                    </span>
                    <span style='font-family: Orbitron, monospace; color: #FAFAFA;'>
                        {strategy_input['weight']:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        # Run portfolio simulation button
        portfolio_valid = (len(valid_strategies) >= 2 and 
                          unique_strategies == len(valid_strategies) and 
                          abs(total_weight_check - 100.0) <= 0.1)
        
        if st.button("🚀 Run Portfolio Simulation", 
                    type="primary", 
                    disabled=not portfolio_valid,
                    key="run_portfolio_sim"):
            if portfolio_valid:
                print("[Portfolio] Button clicked: Valid configuration. Starting simulation...")
                # Clear previous results
                st.session_state.simulation_results = None
                print("[Portfolio] Cleared previous results.")
                # Run portfolio simulation with progress tracking
                progress_container = st.container()
                with progress_container:
                    overall_progress = st.progress(0, text="Initializing portfolio simulation...")
                    status_placeholder = st.empty()
                    sim_counter_placeholder = st.empty()
                    try:
                        start_time = time.time()
                        print("[Portfolio] Using optimized portfolio simulation...")
                        status_placeholder.info("🔄 Running optimized portfolio simulation...")
                        overall_progress.progress(20, text="Analyzing portfolio strategies...")
                        # Live simulation counter
                        n_total = n_simulations
                        def live_progress_callback(progress):
                            completed = int(progress * n_total)
                            overall_progress.progress(min(int(progress * 100), 100), text=f"Running simulations... ({completed:,} / {n_total:,})")
                            sim_counter_placeholder.markdown(f"**Simulations completed:** {completed:,} / {n_total:,}")
                        portfolio_returns = simulate_portfolio_optimized(
                            portfolio_inputs=valid_strategies,
                            n_paths=n_simulations,
                            n_days=simulation_days,
                            strategy_lookup=strategy_lookup,
                            market_condition_options=market_condition_options,
                            pinecone_client=pinecone_client,
                            bitcoin_data=st.session_state.bitcoin_data,
                            progress_callback=live_progress_callback
                        )
                        print("[Portfolio] Simulation finished. Now computing metrics...")
                        overall_progress.progress(80, text="Computing portfolio metrics...")
                        status_placeholder.info("📊 Computing portfolio performance metrics...")
                        sim_counter_placeholder.empty()
                        elapsed_time = time.time() - start_time
                        print(f"[Portfolio] Simulation completed in {elapsed_time:.2f} seconds")
                        print(f"[Portfolio] Metrics: CAGR={np.median(portfolio_returns[:, -1]):.2f}%, DD={np.median(np.min(portfolio_returns, axis=1)):.2f}%, WinRate={np.sum(portfolio_returns[:, -1] > 0) / len(portfolio_returns):.1%}")
                        overall_progress.progress(100, text="Portfolio simulation completed!")
                        status_placeholder.success(f"✅ Portfolio simulation completed in {elapsed_time:.1f} seconds!")
                        st.session_state.simulation_results = {
                            'returns': portfolio_returns,
                            'cagr': np.median(portfolio_returns[:, -1]),
                            'cagr_values': portfolio_returns[:, -1],
                            'max_drawdown': np.median(np.min(portfolio_returns, axis=1)),
                            'drawdown_values': np.min(portfolio_returns, axis=1),
                            'win_rate': np.sum(portfolio_returns[:, -1] > 0) / len(portfolio_returns),
                            'configuration': valid_strategies,
                            'n_simulations': n_simulations,
                            'simulation_days': simulation_days
                        }
                        print("[Portfolio] Results stored in session state.")
                        st.rerun()
                    except Exception as e:
                        print(f"[Portfolio] Exception occurred: {e}")
                        st.error(f"Error running portfolio simulation: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                print("[Portfolio] Button clicked: Invalid configuration. Not running simulation.")
                st.error("Please fix portfolio configuration issues before running simulation.")

# Portfolio Results Display - Only show for portfolio simulations
if ('simulation_results' in st.session_state and st.session_state.simulation_results is not None and 
    simulation_type == "Portfolio Simulation"):
    results = st.session_state.simulation_results
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 240, 0.02)); 
                border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 15px; padding: 20px; margin: 30px 0;'>
        <h2 style='font-family: Orbitron, monospace; color: #00D4FF; text-align: center; margin-top: 0;
                   text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);'>PORTFOLIO SIMULATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)

    # Tabbed results layout
    tab1, tab2, tab3, tab4 = st.tabs(["Price Paths", "CAGR Distribution", "Terminal Prices", "Detailed Metrics"])

    with tab1:
        st.header("Monte Carlo Price Path Fan Chart")
        st.subheader("Monte Carlo Bitcoin Price Paths - Fan Chart")
        # Check if this is portfolio or single strategy results
        if 'returns' in results:
            # Portfolio results - use returns (equity curves)
            portfolio_simulation_days = results['returns'].shape[1] - 1
            fan_chart = create_fan_chart(results['returns'], portfolio_simulation_days)
        else:
            # Single strategy results - use close_paths
            fan_chart = create_fan_chart(results['close_paths'], simulation_days)
        st.plotly_chart(fan_chart, use_container_width=False, width=900, key="portfolio_results_fan_chart")

    with tab2:
        st.header("CAGR Distribution")
        st.subheader("Distribution of Annualized Returns (CAGR)")
        cagr_hist = create_cagr_distribution(results['cagr_values'])
        st.plotly_chart(cagr_hist, use_container_width=False, width=900, key="portfolio_results_cagr_chart")
        
        # CAGR statistics
        col_cagr1, col_cagr2, col_cagr3, col_cagr4 = st.columns(4)
        with col_cagr1:
            st.metric("Mean CAGR", f"{np.mean(results['cagr_values']):.2f}%")
        with col_cagr2:
            st.metric("Std CAGR", f"{np.std(results['cagr_values']):.2f}%")
        with col_cagr3:
            st.metric("95th Percentile", f"{np.percentile(results['cagr_values'], 95):.2f}%")
        with col_cagr4:
            st.metric("5th Percentile", f"{np.percentile(results['cagr_values'], 5):.2f}%")

    with tab3:
        st.header("Terminal Price Distribution")
        st.subheader("Distribution of Terminal Prices")
        # Check if this is portfolio or single strategy results
        if 'returns' in results:
            # Portfolio results - use returns (equity curves)
            terminal_hist = create_terminal_price_histogram(results['returns'])
            terminal_prices = results['returns'][:, -1]
        else:
            # Single strategy results - use close_paths
            terminal_hist = create_terminal_price_histogram(results['close_paths'])
            terminal_prices = results['close_paths'][:, -1]
        st.plotly_chart(terminal_hist, use_container_width=False, width=900, key="portfolio_results_terminal_chart")
        
        # Terminal price statistics
        col_term1, col_term2, col_term3, col_term4 = st.columns(4)
        with col_term1:
            st.metric("Mean Terminal", f"${np.mean(terminal_prices):,.0f}")
        with col_term2:
            st.metric("Median Terminal", f"${np.median(terminal_prices):,.0f}")
        with col_term3:
            st.metric("95th Percentile", f"${np.percentile(terminal_prices, 95):,.0f}")
        with col_term4:
            st.metric("5th Percentile", f"${np.percentile(terminal_prices, 5):,.0f}")

    with tab4:
        st.header("Detailed Metrics")
        # Compute detailed metrics
        # Calculate simulation_days from data or use sidebar value
        if 'returns' in results:
            # Portfolio results - calculate from returns shape
            result_simulation_days = results['returns'].shape[1] - 1
        else:
            # Single strategy results - use stored value or sidebar value
            result_simulation_days = results.get('simulation_days', simulation_days)
        
        simulation_period = f"{result_simulation_days} days"
        cagr_values = results.get('cagr_values')
        drawdown_values = results.get('drawdown_values')
        if cagr_values is not None and drawdown_values is not None:
            median_cagr = np.median(cagr_values)
            mean_cagr = np.mean(cagr_values)
            std_cagr = np.std(cagr_values)
            worst_decile_cagr = np.percentile(cagr_values, 10)
            best_decile_cagr = np.percentile(cagr_values, 90)
            median_max_drawdown = np.median(drawdown_values)
            mean_max_drawdown = np.mean(drawdown_values)
            profitable_simulations = np.sum(np.array(cagr_values) > 0)
            win_rate = results.get('win_rate', 0)
            metrics_table = pd.DataFrame([
                ["Simulation Period", simulation_period],
                ["Median CAGR", f"{median_cagr:.2f}%"],
                ["Mean CAGR", f"{mean_cagr:.2f}%"],
                ["Standard Deviation CAGR", f"{std_cagr:.2f}%"],
                ["Worst Decile CAGR", f"{worst_decile_cagr:.2f}%"],
                ["Best Decile CAGR", f"{best_decile_cagr:.2f}%"],
                ["Median Max Drawdown", f"{median_max_drawdown:.2f}%"],
                ["Mean Max Drawdown", f"{mean_max_drawdown:.2f}%"],
                ["Profitable Simulations", f"{profitable_simulations:,}"],
                ["Win Rate", f"{win_rate:.1f}%"]
            ], columns=["Metric", "Value"])
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        else:
            st.info("No detailed metrics available for this simulation.")

# Results visualization for single strategy simulations - Only show for single strategy
if ('simulation_results' in st.session_state and st.session_state.simulation_results is not None and 
    simulation_type == "Single Strategy"):
    # Add header for single strategy results
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 240, 0.02)); 
                border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 15px; padding: 20px; margin: 30px 0;'>
        <h2 style='font-family: Orbitron, monospace; color: #00D4FF; text-align: center; margin-top: 0;
                   text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);'>SINGLE STRATEGY SIMULATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)

    # Header with export functionality
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.header("Simulation Results")
    
    with col_header2:
        # Export functionality
        if st.button("📊 Export Results", help="Download simulation results as CSV"):
            results = st.session_state.simulation_results
            
            # Create comprehensive results DataFrame
            export_data = {
                'Simulation_Number': range(1, len(results['cagr_values']) + 1),
                'CAGR_Percent': results['cagr_values'],
                'Max_Drawdown_Percent': results['drawdown_values'],
                'Final_Price': [path[-1] for path in results['close_paths']],
                'Strategy': st.session_state.current_strategy_name or 'Unknown',
                'Simulation_Mode': results.get('simulation_mode', 'btc_only'),
                'Market_Condition': selected_market_condition.value if selected_market_condition else 'None',
                'Simulation_Days': simulation_days,
                'Variables_Used': ', '.join(results.get('variables_used', ['BTC']))
            }
            
            export_df = pd.DataFrame(export_data)
            
            # Add summary statistics at the end
            summary_row = {
                'Simulation_Number': 'SUMMARY',
                'CAGR_Percent': f"Median: {results['median_cagr']:.2f}%",
                'Max_Drawdown_Percent': f"Median: {results['median_max_drawdown']:.2f}%",
                'Final_Price': f"Range: ${min(export_data['Final_Price']):,.0f} - ${max(export_data['Final_Price']):,.0f}",
                'Strategy': st.session_state.current_strategy_name or 'Unknown',
                'Simulation_Mode': results.get('simulation_mode', 'btc_only'),
                'Market_Condition': selected_market_condition.value if selected_market_condition else 'None',
                'Simulation_Days': simulation_days,
                'Variables_Used': ', '.join(results.get('variables_used', ['BTC']))
            }
            
            # Add summary as last row
            export_df = pd.concat([export_df, pd.DataFrame([summary_row])], ignore_index=True)
            
            # Convert to CSV
            csv_data = export_df.to_csv(index=False)
            
            # Create filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name_clean = (st.session_state.current_strategy_name or "strategy").replace(" ", "_")[:30]
            filename = f"bitcoin_simulation_{strategy_name_clean}_{timestamp}.csv"
            
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Download detailed simulation results"
            )
    
    results = st.session_state.simulation_results
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Price Paths", "CAGR Distribution", "Terminal Prices", "Detailed Metrics"])
    
    with tab1:
        st.subheader("Monte Carlo Price Path Fan Chart")
        # Check if this is portfolio or single strategy results
        if 'returns' in results:
            # Portfolio results - use returns (equity curves)
            portfolio_simulation_days = results['returns'].shape[1] - 1
            fan_chart = create_fan_chart(results['returns'], portfolio_simulation_days)
        else:
            # Single strategy results - use close_paths
            fan_chart = create_fan_chart(results['close_paths'], simulation_days)
        st.plotly_chart(fan_chart, use_container_width=False, width=900, key="single_strategy_fan_chart")
    
    with tab2:
        st.subheader("CAGR Distribution")
        cagr_hist = create_cagr_distribution(results['cagr_values'])
        st.plotly_chart(cagr_hist, use_container_width=False, width=900, key="single_strategy_cagr_chart")
        
        # CAGR statistics
        col_cagr1, col_cagr2, col_cagr3, col_cagr4 = st.columns(4)
        with col_cagr1:
            st.metric("Mean CAGR", f"{np.mean(results['cagr_values']):.2f}%")
        with col_cagr2:
            st.metric("Std CAGR", f"{np.std(results['cagr_values']):.2f}%")
        with col_cagr3:
            st.metric("95th Percentile", f"{np.percentile(results['cagr_values'], 95):.2f}%")
        with col_cagr4:
            st.metric("5th Percentile", f"{np.percentile(results['cagr_values'], 5):.2f}%")
    
    with tab3:
        st.subheader("Terminal Price Distribution")
        # Check if this is portfolio or single strategy results
        if 'returns' in results:
            # Portfolio results - use returns (equity curves)
            terminal_hist = create_terminal_price_histogram(results['returns'])
            terminal_prices = results['returns'][:, -1]
        else:
            # Single strategy results - use close_paths
            terminal_hist = create_terminal_price_histogram(results['close_paths'])
            terminal_prices = results['close_paths'][:, -1]
        st.plotly_chart(terminal_hist, use_container_width=False, width=900, key="single_strategy_terminal_chart")
        
        # Terminal price statistics
        col_term1, col_term2, col_term3, col_term4 = st.columns(4)
        with col_term1:
            st.metric("Mean Terminal", f"${np.mean(terminal_prices):,.0f}")
        with col_term2:
            st.metric("Median Terminal", f"${np.median(terminal_prices):,.0f}")
        with col_term3:
            st.metric("95th Percentile", f"${np.percentile(terminal_prices, 95):,.0f}")
        with col_term4:
            st.metric("5th Percentile", f"${np.percentile(terminal_prices, 5):,.0f}")
    
    with tab4:
        st.subheader("Detailed Performance Metrics")
        
        # Create detailed metrics table
        metrics_data = {
            "Metric": [
                "Total Simulations",
                "Simulation Period",
                "Median CAGR",
                "Mean CAGR",
                "Standard Deviation CAGR",
                "Worst Decile CAGR",
                "Best Decile CAGR",
                "Median Max Drawdown",
                "Mean Max Drawdown",
                "Profitable Simulations",
                "Win Rate"
            ],
            "Value": [
                f"{n_simulations:,}",
                f"{simulation_days} days",
                f"{results['median_cagr']:.2f}%",
                f"{np.mean(results['cagr_values']):.2f}%",
                f"{np.std(results['cagr_values']):.2f}%",
                f"{results['worst_decile_cagr']:.2f}%",
                f"{np.percentile(results['cagr_values'], 90):.2f}%",
                f"{results['median_max_drawdown']:.2f}%",
                f"{np.mean(results['drawdown_values']):.2f}%",
                f"{len([x for x in results['cagr_values'] if x > 0]):,}",
                f"{len([x for x in results['cagr_values'] if x > 0]) / len(results['cagr_values']) * 100:.1f}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

else:
    st.info("Please load Bitcoin data first to run Monte Carlo simulations.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Monte Carlo simulation with GARCH+jumps modeling")