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

# Load strategies from Pinecone (simple and fast)
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_strategies_fast(_pinecone_client):
    """Load strategy descriptions from Pinecone quickly"""
    if not _pinecone_client:
        return []
    
    try:
        # Get strategies from Pinecone
        dummy_vector = [0.0] * 32
        response = _pinecone_client.index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True
        )
        
        strategies = []
        if hasattr(response, 'matches'):
            for match in response.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    description = match.metadata.get('description', '')
                    if description:
                        strategies.append({
                            'id': match.metadata.get('strategy_id', match.id),
                            'description': description,
                            'metadata': match.metadata
                        })
        
        return strategies
    except Exception as e:
        print(f"Error loading strategies: {e}")
        return []

# Initialize session state
if 'bitcoin_data' not in st.session_state:
    st.session_state.bitcoin_data = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

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

# Load strategies quickly from Pinecone
strategies = []
if pinecone_client:
    try:
        strategies = load_strategies_fast(pinecone_client)
        if strategies:
            st.sidebar.success(f"Loaded {len(strategies)} strategies")
        else:
            st.sidebar.warning("No strategies found in database")
    except Exception as e:
        st.sidebar.error(f"Error loading strategies: {str(e)}")
else:
    st.sidebar.warning("Pinecone connection required to load strategies")

# Strategy selection dropdown
strategy_options = ["CEMD (Default)"]
strategy_lookup = {"CEMD (Default)": None}

if strategies:
    for strategy in strategies:
        # Use just the description as the display name
        display_name = strategy['description']
        strategy_options.append(display_name)
        strategy_lookup[display_name] = strategy

selected_strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    options=strategy_options,
    help="Choose a strategy to simulate"
)

selected_strategy = strategy_lookup[selected_strategy_name]

# Display strategy details if selected
if selected_strategy:
    st.sidebar.subheader("Strategy Details")
    st.sidebar.write(f"**Description:** {selected_strategy['description']}")
    
    # Performance metrics from metadata
    metadata = selected_strategy['metadata']
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        st.sidebar.metric("Total Return", f"{metadata.get('total_return', 0):.1f}%")
        st.sidebar.metric("Sharpe Ratio", f"{metadata.get('sharpe_ratio', 0):.2f}")
    with col_b:
        st.sidebar.metric("Max Drawdown", f"{metadata.get('max_drawdown', 0):.1f}%")
        st.sidebar.metric("Success Rate", f"{metadata.get('success_rate', 0):.1%}")
    
    # Show additional strategy info
    st.sidebar.write(f"**Holding Period:** {metadata.get('avg_holding_days', 'N/A')} days")
    st.sidebar.write(f"**Total Trades:** {metadata.get('total_trades', 'N/A')}")
    st.sidebar.write(f"**Quality Score:** {metadata.get('quality_score', 'N/A')}/100")

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
        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {n_simulations:,} simulations over {simulation_days} days..."):
                try:
                    # Initialize Monte Carlo simulator
                    simulator = MonteCarloSimulator(st.session_state.bitcoin_data)
                    
                    # Run simulation with market condition
                    progress_bar = st.progress(0)
                    
                    results = simulator.run_simulation(
                        n_simulations=n_simulations,
                        simulation_days=simulation_days,
                        selected_strategy=selected_strategy,
                        market_condition=selected_market_condition,
                        progress_callback=lambda p: progress_bar.progress(p)
                    )
                    
                    st.session_state.simulation_results = results
                    st.success("Simulation completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
    
    with col_sim2:
        if st.session_state.simulation_results is not None:
            results = st.session_state.simulation_results
            st.metric("Median CAGR", f"{results['median_cagr']:.2f}%")
            st.metric("Worst 10% CAGR", f"{results['worst_decile_cagr']:.2f}%")
            st.metric("Median Max DD", f"{results['median_max_drawdown']:.2f}%")

# Results visualization
if st.session_state.simulation_results is not None:
    st.header("Simulation Results")
    
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
