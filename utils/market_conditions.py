"""
Market condition scenarios for Monte Carlo simulation
"""
from enum import Enum
import numpy as np

class MarketCondition(Enum):
    HIGH_VOL_UP = "High Volatility Bull"
    HIGH_VOL_DOWN = "High Volatility Bear (Crash)"
    HIGH_VOL_STABLE = "High Volatility Sideways"
    STABLE_VOL_UP = "Stable Volatility Bull"
    STABLE_VOL_DOWN = "Stable Volatility Bear"
    STABLE_VOL_STABLE = "Stable Volatility Sideways"

def adjust_mu_sigma_for_condition(mu, sigma, condition: MarketCondition):
    """
    Adjust drift (μ) and volatility (σ) based on market condition
    
    Args:
        mu: Base drift from GARCH calibration
        sigma: Base volatility from GARCH calibration  
        condition: MarketCondition enum
        
    Returns:
        tuple: (adjusted_mu, adjusted_sigma)
    """
    # Debug logging - original values
    print(f"[MARKET DEBUG] Base μ = {mu:.6f} (daily), Drift/year = {mu*365:.2%}")
    print(f"[MARKET DEBUG] Base σ = {sigma:.4f} (daily), Vol/year = {sigma*np.sqrt(365):.2%}")
    print(f"[MARKET DEBUG] Condition: {condition.value}")
    
    # Apply adjustments based on market condition
    if condition == MarketCondition.HIGH_VOL_UP:
        adjusted_mu, adjusted_sigma = mu * 1.5, sigma * 1.75
    elif condition == MarketCondition.HIGH_VOL_DOWN:
        # Force strong negative drift for bear crash scenarios
        base_negative_mu = -abs(mu) * 1.5
        # Add floor to ensure meaningful negative drift
        adjusted_mu = min(base_negative_mu, -0.001)  # At least -36.5% annually
        adjusted_sigma = sigma * 1.75
    elif condition == MarketCondition.HIGH_VOL_STABLE:
        adjusted_mu, adjusted_sigma = 0.0, sigma * 1.5
    elif condition == MarketCondition.STABLE_VOL_UP:
        adjusted_mu, adjusted_sigma = mu * 1.25, sigma * 0.5
    elif condition == MarketCondition.STABLE_VOL_DOWN:
        # Force meaningful negative drift for stable bear
        base_negative_mu = -abs(mu) * 1.25
        adjusted_mu = min(base_negative_mu, -0.0005)  # At least -18% annually
        adjusted_sigma = sigma * 0.5
    elif condition == MarketCondition.STABLE_VOL_STABLE:
        adjusted_mu, adjusted_sigma = 0.0, sigma * 0.5
    else:
        adjusted_mu, adjusted_sigma = mu, sigma  # fallback if no match
    
    # Debug logging - adjusted values
    print(f"[Sim Regime: {condition.value}] Adjusted μ = {adjusted_mu:.6f}, σ = {adjusted_sigma:.4f}")
    print(f"[Sim Regime: {condition.value}] Adjusted Drift/year = {adjusted_mu*365:.2%}, Vol/year = {adjusted_sigma*np.sqrt(365):.2%}")
    
    return adjusted_mu, adjusted_sigma

def get_condition_description(condition: MarketCondition):
    """Get detailed description of market condition"""
    descriptions = {
        MarketCondition.HIGH_VOL_UP: "Strong bull market with high volatility - often near tops or during rebounds",
        MarketCondition.HIGH_VOL_DOWN: "Panic/crash environment - high volatility with sharp negative drift", 
        MarketCondition.HIGH_VOL_STABLE: "Sideways chop with high noise - often seen before breakouts",
        MarketCondition.STABLE_VOL_UP: "Healthy uptrend - grinding up with low volatility",
        MarketCondition.STABLE_VOL_DOWN: "Slow bleed - common in early bear markets or long drawdowns",
        MarketCondition.STABLE_VOL_STABLE: "Boring sideways market - useful for mean reversion testing"
    }
    return descriptions.get(condition, "Unknown market condition")