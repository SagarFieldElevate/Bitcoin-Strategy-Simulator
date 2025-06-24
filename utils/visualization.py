"""
Visualization utilities using Plotly for interactive charts
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_fan_chart(close_paths, simulation_days):
    """
    Create futuristic fan chart showing price path percentiles
    """
    days_axis = np.arange(simulation_days + 1)
    percentiles = np.percentile(close_paths, [5, 25, 50, 75, 95], axis=0)
    mean_path = close_paths.mean(axis=0)
    
    fig = go.Figure()
    
    # Add percentile bands with futuristic gradients
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
        fillcolor='rgba(0, 212, 255, 0.1)',
        name='5-95% Confidence Zone',
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
        fillcolor='rgba(0, 255, 240, 0.2)',
        name='25-75% Core Zone',
        showlegend=True
    ))
    
    # Add median and mean lines with glow effect
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=percentiles[2],
        mode='lines',
        line=dict(color='#FF00FF', width=3, dash='dash'),
        name='Median Trajectory'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis,
        y=mean_path,
        mode='lines',
        line=dict(color='#00D4FF', width=3),
        name='Mean Trajectory'
    ))
    
    # Futuristic dark theme layout
    fig.update_layout(
        title={
            'text': "QUANTUM PRICE PROJECTION MATRIX",
            'font': {'family': 'Orbitron, monospace', 'size': 24, 'color': '#00D4FF'}
        },
        xaxis_title="Temporal Horizon (Days)",
        yaxis_title="Price Vector (USD)",
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.9)',
        font=dict(family='Orbitron, monospace', color='#FAFAFA'),
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.1)',
            zerolinecolor='rgba(0, 212, 255, 0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.1)',
            zerolinecolor='rgba(0, 212, 255, 0.3)'
        ),
        legend=dict(
            bgcolor='rgba(0, 212, 255, 0.05)',
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        )
    )
    
    return fig

def create_cagr_distribution(cagr_values):
    """
    Create futuristic histogram of CAGR distribution
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=cagr_values,
        nbinsx=50,
        marker=dict(
            color='#00D4FF',
            line=dict(color='#00FFF0', width=1),
            pattern=dict(shape="")
        ),
        opacity=0.8,
        name='Return Distribution'
    ))
    
    # Add vertical lines for key percentiles with glow effect
    median_cagr = np.median(cagr_values)
    p10_cagr = np.percentile(cagr_values, 10)
    p90_cagr = np.percentile(cagr_values, 90)
    
    fig.add_vline(x=median_cagr, line_dash="dash", line_color="#FF00FF", line_width=3,
                  annotation_text=f"MEDIAN: {median_cagr:.1f}%",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#FF00FF'))
    fig.add_vline(x=p10_cagr, line_dash="dot", line_color="#FF0040", line_width=2,
                  annotation_text=f"RISK: {p10_cagr:.1f}%",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#FF0040'))
    fig.add_vline(x=p90_cagr, line_dash="dot", line_color="#00FF00", line_width=2,
                  annotation_text=f"OPPORTUNITY: {p90_cagr:.1f}%",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#00FF00'))
    
    # Futuristic dark theme layout
    fig.update_layout(
        title={
            'text': "RETURN PROBABILITY MATRIX",
            'font': {'family': 'Orbitron, monospace', 'size': 24, 'color': '#00D4FF'}
        },
        xaxis_title="Annualized Return Vector (%)",
        yaxis_title="Probability Density",
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
            zerolinecolor='rgba(0, 212, 255, 0.3)'
        ),
        bargap=0.1
    )
    
    return fig

def create_terminal_price_histogram(close_paths):
    """
    Create futuristic histogram of terminal (final) prices
    """
    terminal_prices = close_paths[:, -1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=terminal_prices,
        nbinsx=50,
        marker=dict(
            color='#00FFF0',
            line=dict(color='#00D4FF', width=1),
            pattern=dict(shape="")
        ),
        opacity=0.8,
        name='Terminal Price Distribution'
    ))
    
    # Add vertical lines for key percentiles with glow effect
    median_price = np.median(terminal_prices)
    p10_price = np.percentile(terminal_prices, 10)
    p90_price = np.percentile(terminal_prices, 90)
    
    fig.add_vline(x=median_price, line_dash="dash", line_color="#FF00FF", line_width=3,
                  annotation_text=f"MEDIAN: ${median_price:,.0f}",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#FF00FF'))
    fig.add_vline(x=p10_price, line_dash="dot", line_color="#FF0040", line_width=2,
                  annotation_text=f"DOWNSIDE: ${p10_price:,.0f}",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#FF0040'))
    fig.add_vline(x=p90_price, line_dash="dot", line_color="#00FF00", line_width=2,
                  annotation_text=f"UPSIDE: ${p90_price:,.0f}",
                  annotation_font=dict(family='Orbitron, monospace', size=12, color='#00FF00'))
    
    # Futuristic dark theme layout
    fig.update_layout(
        title={
            'text': "TERMINAL VALUE PROJECTION",
            'font': {'family': 'Orbitron, monospace', 'size': 24, 'color': '#00D4FF'}
        },
        xaxis_title="Terminal Price Vector (USD)",
        yaxis_title="Probability Density",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.9)',
        font=dict(family='Orbitron, monospace', color='#FAFAFA'),
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.1)',
            zerolinecolor='rgba(0, 212, 255, 0.3)',
            tickformat='$,.0f'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.1)',
            zerolinecolor='rgba(0, 212, 255, 0.3)'
        ),
        bargap=0.1
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
