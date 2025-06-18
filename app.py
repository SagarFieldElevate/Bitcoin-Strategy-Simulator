import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

from utils.bitcoin_data import fetch_bitcoin_data
from utils.monte_carlo import MonteCarloSimulator
from utils.pinecone_client import PineconeClient
from utils.visualization import create_fan_chart, create_terminal_price_histogram, create_cagr_distribution
from utils.market_conditions import MarketCondition, get_condition_description
from utils.simulation_router import SimulationRouter


# Page configuration
st.set_page_config(
    page_title="Bitcoin Strategy Simulator",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("₿ Bitcoin Strategy Monte Carlo Simulator")
st.markdown("Advanced Monte Carlo simulation with GARCH+jumps modeling for Bitcoin trading strategies")

# Initialize Pinecone client
@st.cache_resource
def init_pinecone():
    # Check multiple sources for the API key
    api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
    if not api_key:
        return None, "API key not found"
    try:
        client = PineconeClient(api_key)
        # Test connection
        strategies = client.list_strategies()
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
        # Simple test to verify API key format
        if api_key.startswith("sk-"):
            return True, "Connected"
        else:
            return False, "Invalid API key format"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# Load strategies with efficient file-based caching
@st.cache_data(ttl=3600)  # Cache in memory for 1 hour
def load_strategies_cached():
    """Load strategies from file cache or Pinecone if cache is stale"""
    import json
    import os
    from datetime import datetime
    
    cache_file = "strategies_cache.json"
    cache_max_age = 86400  # 24 hours in seconds
    
    # Check if cache file exists and is recent
    if os.path.exists(cache_file):
        try:
            file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
            if file_age < cache_max_age:
                with open(cache_file, 'r') as f:
                    strategies = json.load(f)
                    print(f"Loaded {len(strategies)} strategies from cache (age: {file_age/3600:.1f}h)")
                    return strategies
        except Exception as e:
            print(f"Cache read error: {e}")
    
    # Cache miss or stale - load fresh from Pinecone
    return load_strategies_from_pinecone()

def load_strategies_from_pinecone():
    """Load fresh strategies from Pinecone and update cache"""
    import json
    from regime_correlations import is_daily_only_strategy
    
    pinecone_client, status = init_pinecone()
    if not pinecone_client:
        print(f"Pinecone unavailable: {status}")
        return []
    
    try:
        all_strategies = []
        
        # Get all vector IDs
        if hasattr(pinecone_client.index, 'list'):
            scan_results = pinecone_client.index.list()
            all_ids = []
            
            for ids_batch in scan_results:
                all_ids.extend(ids_batch)
            
            print(f"Found {len(all_ids)} total vectors in index")
            
            # Fetch in batches
            batch_size = 100
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                
                try:
                    if hasattr(pinecone_client.index, 'fetch'):
                        fetch_response = pinecone_client.index.fetch(ids=batch_ids)
                        
                        for vector_id, vector_data in fetch_response.vectors.items():
                            if hasattr(vector_data, 'metadata') and vector_data.metadata:
                                strategy = {
                                    'id': vector_id,
                                    'name': vector_data.metadata.get('name', 'Unknown Strategy'),
                                    'description': vector_data.metadata.get('description', 'No description'),
                                    'excel_names': vector_data.metadata.get('excel_names', [])
                                }
                                
                                # Include all strategies regardless of frequency
                                all_strategies.append(strategy)
                                
                except Exception as e:
                    print(f"Error fetching batch {i//batch_size + 1}: {e}")
                    continue
        
        print(f"Loaded {len(all_strategies)} strategies from Pinecone")
        
        # Save to cache file
        try:
            with open("strategies_cache.json", 'w') as f:
                json.dump(all_strategies, f, indent=2)
            print("Strategies cached to file")
        except Exception as e:
            print(f"Cache save error: {e}")
        
        return all_strategies
        
    except Exception as e:
        print(f"Pinecone loading error: {e}")
        return []

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
    st.session_state.simulation_mode = None
if 'required_variables' not in st.session_state:
    st.session_state.required_variables = None

# Initialize API connections
pinecone_client, pinecone_status = init_pinecone()
openai_connected, openai_status = check_openai_connection()

# Sidebar configuration
# API Connection Status
st.sidebar.header("API Connection Status")

# Pinecone status
pinecone_color = "🟢" if pinecone_client else "🔴"
st.sidebar.markdown(f"{pinecone_color} **Pinecone:** {pinecone_status}")

# OpenAI status  
openai_color = "🟢" if openai_connected else "🔴"
st.sidebar.markdown(f"{openai_color} **OpenAI:** {openai_status}")

st.sidebar.markdown("---")

st.sidebar.header("Simulation Parameters")

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

# Market Condition Selection
st.sidebar.header("Market Scenario")
market_condition_options = [
    ("Baseline (Historical)", None),
    ("High Volatility Bull", MarketCondition.HIGH_VOL_UP),
    ("High Volatility Bear (Crash)", MarketCondition.HIGH_VOL_DOWN),
    ("High Volatility Sideways", MarketCondition.HIGH_VOL_STABLE),
    ("Stable Volatility Bull", MarketCondition.STABLE_VOL_UP),
    ("Stable Volatility Bear", MarketCondition.STABLE_VOL_DOWN),
    ("Stable Volatility Sideways", MarketCondition.STABLE_VOL_STABLE)
]

selected_condition_name = st.sidebar.selectbox(
    "Market Condition",
    options=[name for name, _ in market_condition_options],
    help="Select market scenario to simulate"
)

# Get the corresponding MarketCondition enum
selected_market_condition = next(
    (condition for name, condition in market_condition_options if name == selected_condition_name),
    None
)

# Show description of selected market condition
if selected_market_condition:
    st.sidebar.info(get_condition_description(selected_market_condition))
    
    # Display market condition effects
    st.sidebar.subheader("Market Scenario Effects")
    if selected_market_condition == MarketCondition.HIGH_VOL_UP:
        st.sidebar.write("• Drift: +50% bias")
        st.sidebar.write("• Volatility: +75%")
    elif selected_market_condition == MarketCondition.HIGH_VOL_DOWN:
        st.sidebar.write("• Drift: -150% (crash)")
        st.sidebar.write("• Volatility: +75%")
    elif selected_market_condition == MarketCondition.HIGH_VOL_STABLE:
        st.sidebar.write("• Drift: No bias")
        st.sidebar.write("• Volatility: +50%")
    elif selected_market_condition == MarketCondition.STABLE_VOL_UP:
        st.sidebar.write("• Drift: +25% bias")
        st.sidebar.write("• Volatility: -50%")
    elif selected_market_condition == MarketCondition.STABLE_VOL_DOWN:
        st.sidebar.write("• Drift: -125% bias")
        st.sidebar.write("• Volatility: -50%")
    elif selected_market_condition == MarketCondition.STABLE_VOL_STABLE:
        st.sidebar.write("• Drift: No bias")
        st.sidebar.write("• Volatility: -50%")
else:
    st.sidebar.write("Using historical Bitcoin drift and volatility patterns")



# Strategy selection
st.sidebar.header("Strategy Selection")

# Load strategies with caching
strategies = []
try:
    strategies = load_strategies_cached()
    if strategies:
        st.sidebar.success(f"Loaded {len(strategies)} strategies")
    else:
        st.sidebar.warning("No strategies found in database")
except Exception as e:
    st.sidebar.error(f"Error loading strategies: {str(e)}")

# Strategy search and filtering
st.sidebar.subheader("Strategy Selection")

# Strategy type filter for user convenience
total_count = len(strategies) if strategies else 0
data_filter = st.sidebar.radio(
    "Strategy Type Filter:",
    [f"All strategies ({total_count})", "BTC-only strategies", "Multi-factor strategies"],
    index=0,
    help="Filter strategies by data requirements"
)

# Search functionality
search_term = st.sidebar.text_input(
    "Search strategies", 
    placeholder="Keywords (SPY, momentum, etc.)",
    help="Search through all strategies by keywords"
)

# Apply data integrity filter
filtered_strategies = strategies if strategies else []

if data_filter.startswith("BTC-only") and strategies:
    # Filter for BTC-only strategies (excel_names contains only BTC variables)
    btc_strategies = []
    for strategy in strategies:
        excel_names = strategy.get('excel_names', [])
        
        # Only classify as BTC-only if ALL variables contain "BTC" or "Bitcoin"
        is_btc_only = all('btc' in name.lower() or 'bitcoin' in name.lower() for name in excel_names)
        
        if is_btc_only:
            btc_strategies.append(strategy)
    
    filtered_strategies = btc_strategies
    st.sidebar.info(f"📊 {len(filtered_strategies)} BTC-only strategies")

elif data_filter.startswith("Multi-factor") and strategies:
    # Filter for multi-factor strategies (excel_names contains any non-BTC variables)
    multi_strategies = []
    for strategy in strategies:
        excel_names = strategy.get('excel_names', [])
        
        # Classify as multi-factor if ANY variable is not BTC/Bitcoin
        has_non_btc = any('btc' not in name.lower() and 'bitcoin' not in name.lower() for name in excel_names)
        
        if has_non_btc:
            multi_strategies.append(strategy)
    
    filtered_strategies = multi_strategies
    st.sidebar.info(f"📈 {len(filtered_strategies)} multi-factor strategies")

# Apply search filter
if search_term and filtered_strategies:
    search_filtered = []
    search_lower = search_term.lower()
    for s in filtered_strategies:
        desc_match = search_lower in s['description'].lower()
        type_match = search_lower in s.get('metadata', {}).get('strategy_type', '').lower()
        
        if desc_match or type_match:
            search_filtered.append(s)
    
    filtered_strategies = search_filtered
    
    if filtered_strategies:
        st.sidebar.info(f"Found {len(filtered_strategies)} matches")
    else:
        st.sidebar.warning("No matches found")

# Strategy type filter
if filtered_strategies:
    strategy_types = list(set([s.get('metadata', {}).get('strategy_type', 'unknown') for s in filtered_strategies]))
    strategy_types.sort()
    
    selected_type = st.sidebar.selectbox(
        "Filter by type",
        ["All Types"] + strategy_types
    )
    
    # Apply type filter
    if selected_type != "All Types":
        filtered_strategies = [s for s in filtered_strategies if s.get('metadata', {}).get('strategy_type', 'unknown') == selected_type]

# Create strategy options
strategy_options = ["Select a strategy..."]
strategy_lookup = {"Select a strategy...": None}

if filtered_strategies:
    for strategy in filtered_strategies:
        display_name = strategy['description']
        strategy_options.append(display_name)
        strategy_lookup[display_name] = strategy

selected_strategy_name = st.sidebar.selectbox(
    f"Choose from {len(filtered_strategies)} strategies" if filtered_strategies else "No strategies available",
    options=strategy_options,
    help="Select a strategy to view details and run simulation"
)

# Strategy generation section
# Check if strategy has changed
if st.session_state.current_strategy_name != selected_strategy_name:
    st.session_state.generated_strategy = None
    st.session_state.current_strategy_name = selected_strategy_name
    st.session_state.simulation_mode = None
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
    metadata = selected_strategy.get('metadata', {})
    
    st.sidebar.subheader("Strategy Details")
    
    # Strategy type and description
    strategy_type = metadata.get('strategy_type', 'Unknown')
    st.sidebar.markdown(f"**Type:** `{strategy_type}`")
    
    # Truncated description with expansion option
    description = selected_strategy.get('description', 'N/A')
    if len(description) > 100:
        with st.sidebar.expander("📋 View Full Description"):
            st.write(description)
        st.sidebar.write(f"**Brief:** {description[:100]}...")
    else:
        st.sidebar.write(f"**Description:** {description}")
    
    # Strategy dependencies
    dependencies = metadata.get('dependencies', [])
    if dependencies:
        st.sidebar.markdown(f"**Dependencies:** {', '.join(dependencies)}")
    
    # Performance metrics in organized layout
    st.sidebar.subheader("Historical Performance")
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        total_return = metadata.get('total_return', 0)
        st.sidebar.metric("Total Return", f"{total_return:.1f}%", 
                         delta=None if total_return == 0 else ("Profitable" if total_return > 0 else "Loss"))
        
        sharpe = metadata.get('sharpe_ratio', 0)
        st.sidebar.metric("Sharpe Ratio", f"{sharpe:.2f}",
                         delta=None if sharpe == 0 else ("Good" if sharpe > 1 else "Poor" if sharpe < 0.5 else "Fair"))
    
    with col_b:
        max_dd = metadata.get('max_drawdown', 0)
        st.sidebar.metric("Max Drawdown", f"{abs(max_dd):.1f}%",
                         delta=None if max_dd == 0 else ("Low Risk" if abs(max_dd) < 10 else "High Risk"))
        
        success_rate = metadata.get('success_rate', 0)
        st.sidebar.metric("Success Rate", f"{success_rate:.1%}",
                         delta=None if success_rate == 0 else ("High" if success_rate > 0.6 else "Low"))
    
    # Additional strategy metrics
    st.sidebar.subheader("Trading Details")
    
    avg_holding = metadata.get('avg_holding_days', 'N/A')
    total_trades = metadata.get('total_trades', 'N/A')
    quality_score = metadata.get('quality_score', 'N/A')
    
    st.sidebar.write(f"🕒 **Avg Holding:** {avg_holding} days")
    st.sidebar.write(f"📊 **Total Trades:** {total_trades}")
    st.sidebar.write(f"⭐ **Quality Score:** {quality_score}/100")
    
    # Risk assessment
    if total_return != 0 and max_dd != 0:
        risk_level = "Low" if abs(max_dd) < 10 else "Medium" if abs(max_dd) < 20 else "High"
        risk_color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
        st.sidebar.write(f"{risk_color} **Risk Level:** {risk_level}")
    
    # Simulation readiness indicator
    st.sidebar.subheader("Simulation Ready")
    if simulation_mode:
        mode_display = simulation_mode.replace('_', ' ').title()
        st.sidebar.success(f"✅ {mode_display} Mode")
        if simulation_mode == 'multi_factor' and required_variables:
            st.sidebar.info(f"📊 Variables: {', '.join(required_variables)}")
    else:
        st.sidebar.info("🔄 Ready for analysis")

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
        
        # Price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#FF6B35', width=2)
        ))
        fig_price.update_layout(
            title="Bitcoin Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_price, use_container_width=True)

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
st.header("Monte Carlo Simulation")

if st.session_state.bitcoin_data is not None:
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
                            
                            # Enhanced progress callback with step tracking
                            def enhanced_progress_callback(progress):
                                # Map simulation progress to remaining 40% (60-100%)
                                adjusted_progress = 60 + (progress * 40)
                                sim_number = int(progress * n_simulations)
                                overall_progress.progress(adjusted_progress / 100, 
                                                        text=f"Simulation {sim_number}/{n_simulations} ({progress*100:.1f}%)")
                                
                                if progress < 0.5:
                                    status_placeholder.info(f"🔄 Generating price paths: {sim_number}/{n_simulations}")
                                else:
                                    status_placeholder.info(f"📈 Executing strategy logic: {sim_number}/{n_simulations}")
                            
                            # Data validation before simulation
                            if simulation_mode == 'multi_factor' and required_variables:
                                status_placeholder.info("🔍 Validating required data availability...")
                                missing_data = []
                                
                                # Test data availability for each required variable
                                from utils.multi_factor_data import MultiFactorDataFetcher
                                data_fetcher = MultiFactorDataFetcher(pinecone_client)
                                
                                for var in required_variables:
                                    if var != 'BTC':  # BTC data already validated
                                        try:
                                            # Quick test fetch to validate data availability
                                            if var.upper() in ['WTI', 'CRUDE', 'OIL', 'CL']:
                                                test_data = data_fetcher.fetch_wti_data_direct()
                                                if test_data is None or len(test_data) < 10:
                                                    missing_data.append(f'{var} (Oil prices)')
                                            elif var.upper() in ['GOLD', 'GLD', 'XAU']:
                                                test_data = data_fetcher.fetch_gold_data_direct()
                                                if test_data is None or len(test_data) < 10:
                                                    missing_data.append(f'{var} (Gold prices)')
                                            else:
                                                # For other variables, use the enhanced LLM-based search
                                                vector_info = data_fetcher.find_vector_for_variable(var)
                                                if not vector_info or vector_info.get('confidence', 0) < 0.3:
                                                    missing_data.append(f'{var} (Economic data)')
                                                else:
                                                    # Test actual data extraction
                                                    test_data = data_fetcher.fetch_data_from_pinecone(vector_info)
                                                    if test_data is None or len(test_data) < 10:
                                                        missing_data.append(f'{var} (Data extraction failed)')
                                        except Exception as e:
                                            print(f"Data validation failed for {var}: {e}")
                                            missing_data.append(f'{var} (Data access error)')
                                
                                if missing_data:
                                    overall_progress.progress(100, text="Data validation failed")
                                    status_placeholder.empty()
                                    
                                    st.error("❌ **Required Data Not Available**")
                                    st.error(f"Cannot run simulation - missing data for: {', '.join(missing_data)}")
                                    st.warning("**Solutions:**")
                                    st.warning("• Select a BTC-only strategy instead")
                                    st.warning("• Choose a different multi-factor strategy")
                                    st.warning("• Contact support if data should be available")
                                    
                                    # Clear progress indicators
                                    progress_container.empty()
                                    st.stop()
                            
                            results = simulator.run_simulation(
                                n_simulations=n_simulations,
                                simulation_days=simulation_days,
                                selected_strategy=processed_strategy,
                                market_condition=selected_market_condition,
                                progress_callback=enhanced_progress_callback,
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

# Results visualization
if st.session_state.simulation_results is not None:
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
        fan_chart = create_fan_chart(results['close_paths'], simulation_days)
        st.plotly_chart(fan_chart, use_container_width=True)
    
    with tab2:
        st.subheader("CAGR Distribution")
        cagr_hist = create_cagr_distribution(results['cagr_values'])
        st.plotly_chart(cagr_hist, use_container_width=True)
        
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
        terminal_hist = create_terminal_price_histogram(results['close_paths'])
        st.plotly_chart(terminal_hist, use_container_width=True)
        
        # Terminal price statistics
        terminal_prices = results['close_paths'][:, -1]
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
