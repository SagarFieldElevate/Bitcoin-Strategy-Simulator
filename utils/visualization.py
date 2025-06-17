"""
Visualization utilities using Plotly for interactive charts
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_fan_chart(close_paths, simulation_days):
    """
    Create fan chart showing price path percentiles
    """
    days_axis = np.arange(simulation_days + 1)
    percentiles = np.percentile(close_paths, [5, 25, 50, 75, 95], axis=0)
    mean_path = close_paths.mean(axis=0)
    
    fig = go.Figure()
    
    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[4],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='95th percentile'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[0],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(128,128,128,0.2)',
        name='5-95% band',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[3],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='75th percentile'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[1],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(128,128,128,0.4)',
        name='25-75% band',
        showlegend=True
    ))
    
    # Add median and mean lines
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[2],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Median path'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=mean_path,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Mean path'
    ))
    
    fig.update_layout(
        title="Monte Carlo Bitcoin Price Paths - Fan Chart",
        xaxis_title="Days Ahead",
        yaxis_title="Price (USD)",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_cagr_distribution(cagr_values):
    """
    Create histogram of CAGR distribution
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=cagr_values,
        nbinsx=50,
        marker=dict(
            color='steelblue',
            line=dict(color='black', width=1)
        ),
        opacity=0.7,
        name='CAGR Distribution'
    ))
    
    # Add vertical lines for key percentiles
    median_cagr = np.median(cagr_values)
    p10_cagr = np.percentile(cagr_values, 10)
    p90_cagr = np.percentile(cagr_values, 90)
    
    fig.add_vline(x=median_cagr, line_dash="dash", line_color="red", 
                  annotation_text=f"Median: {median_cagr:.1f}%")
    fig.add_vline(x=p10_cagr, line_dash="dot", line_color="orange",
                  annotation_text=f"10th %: {p10_cagr:.1f}%")
    fig.add_vline(x=p90_cagr, line_dash="dot", line_color="green",
                  annotation_text=f"90th %: {p90_cagr:.1f}%")
    
    fig.update_layout(
        title="Distribution of Annualized Returns (CAGR)",
        xaxis_title="CAGR (%)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig

def create_terminal_price_histogram(close_paths):
    """
    Create histogram of terminal (final) prices
    """
    terminal_prices = close_paths[:, -1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=terminal_prices,
        nbinsx=50,
        marker=dict(
            color='lightcoral',
            line=dict(color='black', width=1)
        ),
        opacity=0.7,
        name='Terminal Price Distribution'
    ))
    
    # Add vertical lines for key percentiles
    median_price = np.median(terminal_prices)
    p10_price = np.percentile(terminal_prices, 10)
    p90_price = np.percentile(terminal_prices, 90)
    
    fig.add_vline(x=median_price, line_dash="dash", line_color="red",
                  annotation_text=f"Median: ${median_price:,.0f}")
    fig.add_vline(x=p10_price, line_dash="dot", line_color="orange",
                  annotation_text=f"10th %: ${p10_price:,.0f}")
    fig.add_vline(x=p90_price, line_dash="dot", line_color="green",
                  annotation_text=f"90th %: ${p90_price:,.0f}")
    
    fig.update_layout(
        title="Distribution of Terminal Prices",
        xaxis_title="Price (USD)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig

def create_sample_paths_chart(close_paths, n_sample=50):
    """
    Create spaghetti plot of sample price paths
    """
    fig = go.Figure()
    
    # Sample random paths
    sample_indices = np.random.choice(close_paths.shape[0], size=min(n_sample, close_paths.shape[0]), replace=False)
    days_axis = np.arange(close_paths.shape[1])
    
    for i, idx in enumerate(sample_indices):
        fig.add_trace(go.Scatter(
            x=days_axis,
            y=close_paths[idx],
            mode='lines',
            line=dict(width=0.8),
            opacity=0.3,
            showlegend=False,
            hovertemplate='Day: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Sample of {len(sample_indices)} Individual Monte Carlo Paths",
        xaxis_title="Days Ahead",
        yaxis_title="Price (USD)",
        height=400
    )
    
    return fig

def create_drawdown_distribution(drawdown_values):
    """
    Create histogram of maximum drawdown distribution
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=drawdown_values,
        nbinsx=50,
        marker=dict(
            color='darkred',
            line=dict(color='black', width=1)
        ),
        opacity=0.7,
        name='Max Drawdown Distribution'
    ))
    
    # Add vertical lines for key percentiles
    median_dd = np.median(drawdown_values)
    p10_dd = np.percentile(drawdown_values, 10)
    p90_dd = np.percentile(drawdown_values, 90)
    
    fig.add_vline(x=median_dd, line_dash="dash", line_color="red",
                  annotation_text=f"Median: {median_dd:.1f}%")
    fig.add_vline(x=p10_dd, line_dash="dot", line_color="orange",
                  annotation_text=f"10th %: {p10_dd:.1f}%")
    fig.add_vline(x=p90_dd, line_dash="dot", line_color="green",
                  annotation_text=f"90th %: {p90_dd:.1f}%")
    
    fig.update_layout(
        title="Distribution of Maximum Drawdowns",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig
