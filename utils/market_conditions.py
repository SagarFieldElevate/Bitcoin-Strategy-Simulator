"""
Market condition scenarios for Monte Carlo simulation
"""
from enum import Enum

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
    if condition == MarketCondition.HIGH_VOL_UP:
        return mu * 1.5, sigma * 1.75
    elif condition == MarketCondition.HIGH_VOL_DOWN:
        return -abs(mu) * 1.5, sigma * 1.75
    elif condition == MarketCondition.HIGH_VOL_STABLE:
        return 0.0, sigma * 1.5
    elif condition == MarketCondition.STABLE_VOL_UP:
        return mu * 1.25, sigma * 0.5
    elif condition == MarketCondition.STABLE_VOL_DOWN:
        return -abs(mu) * 1.25, sigma * 0.5
    elif condition == MarketCondition.STABLE_VOL_STABLE:
        return 0.0, sigma * 0.5
    else:
        return mu, sigma  # fallback if no match

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