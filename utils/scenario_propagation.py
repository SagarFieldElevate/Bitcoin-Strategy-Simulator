"""
Scenario propagation system using regime-specific correlation matrices.
Determines market scenarios for each variable based on correlation relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.market_conditions import MarketCondition
from regime_correlations import REGIME_CORRELATIONS, get_regime_correlation

class ScenarioPropagator:
    def __init__(self):
        """Initialize the scenario propagation system."""
        self.supported_assets = ["BTC", "SPY", "GOLD", "TIPS"]
        self.market_conditions = [
            MarketCondition.HIGH_VOL_UP,
            MarketCondition.HIGH_VOL_DOWN,
            MarketCondition.HIGH_VOL_STABLE,
            MarketCondition.STABLE_VOL_UP,
            MarketCondition.STABLE_VOL_DOWN,
            MarketCondition.STABLE_VOL_STABLE
        ]
    
    def propagate_scenarios(self, primary_asset: str, primary_scenario: MarketCondition, 
                          required_assets: List[str], method: str = "correlation_weighted") -> Dict[str, MarketCondition]:
        """
        Propagate market scenarios from a primary asset to other required assets.
        
        Args:
            primary_asset: The asset with the known scenario (e.g., "BTC")
            primary_scenario: The scenario for the primary asset
            required_assets: List of assets that need scenario assignment
            method: Propagation method ("correlation_weighted" or "deterministic")
            
        Returns:
            Dictionary mapping asset names to their assigned scenarios
        """
        scenarios = {primary_asset: primary_scenario}
        
        # Get the regime name for correlation lookup
        regime_name = primary_scenario.name
        
        for asset in required_assets:
            if asset == primary_asset:
                continue
                
            if asset not in self.supported_assets:
                # For unsupported assets, use the same scenario as primary
                scenarios[asset] = primary_scenario
                continue
            
            if method == "correlation_weighted":
                scenarios[asset] = self._correlation_weighted_assignment(
                    primary_asset, primary_scenario, asset, regime_name
                )
            elif method == "deterministic":
                scenarios[asset] = self._deterministic_assignment(
                    primary_asset, primary_scenario, asset, regime_name
                )
            else:
                raise ValueError(f"Unknown propagation method: {method}")
        
        return scenarios
    
    def _correlation_weighted_assignment(self, primary_asset: str, primary_scenario: MarketCondition, 
                                       target_asset: str, regime_name: str) -> MarketCondition:
        """
        Assign scenario based on correlation strength and direction.
        """
        try:
            correlation = get_regime_correlation(regime_name, primary_asset, target_asset)
        except (ValueError, KeyError):
            # Fallback to same scenario if correlation not available
            return primary_scenario
        
        # Strong positive correlation (>0.5) - same scenario type
        if correlation > 0.5:
            return primary_scenario
        
        # Strong negative correlation (<-0.3) - opposite scenario
        elif correlation < -0.3:
            return self._get_opposite_scenario(primary_scenario)
        
        # Weak correlation - use neutral/stable scenario
        else:
            return self._get_neutral_scenario(primary_scenario)
    
    def _deterministic_assignment(self, primary_asset: str, primary_scenario: MarketCondition,
                                target_asset: str, regime_name: str) -> MarketCondition:
        """
        Deterministic scenario assignment based on asset class relationships.
        """
        # Define asset class relationships
        risk_assets = ["BTC", "SPY"]
        safe_havens = ["GOLD", "TIPS"]
        
        primary_is_risk = primary_asset in risk_assets
        target_is_risk = target_asset in risk_assets
        
        # Same asset class - similar behavior
        if primary_is_risk == target_is_risk:
            return primary_scenario
        
        # Different asset classes - opposite behavior in extreme scenarios
        if primary_scenario in [MarketCondition.HIGH_VOL_DOWN, MarketCondition.HIGH_VOL_UP]:
            return self._get_opposite_scenario(primary_scenario)
        
        # For stable scenarios, use moderate version
        return self._get_neutral_scenario(primary_scenario)
    
    def _get_opposite_scenario(self, scenario: MarketCondition) -> MarketCondition:
        """Get the opposite scenario (bull <-> bear, keeping volatility level)."""
        opposite_map = {
            MarketCondition.HIGH_VOL_UP: MarketCondition.HIGH_VOL_DOWN,
            MarketCondition.HIGH_VOL_DOWN: MarketCondition.HIGH_VOL_UP,
            MarketCondition.STABLE_VOL_UP: MarketCondition.STABLE_VOL_DOWN,
            MarketCondition.STABLE_VOL_DOWN: MarketCondition.STABLE_VOL_UP,
            MarketCondition.HIGH_VOL_STABLE: MarketCondition.HIGH_VOL_STABLE,  # Stable stays stable
            MarketCondition.STABLE_VOL_STABLE: MarketCondition.STABLE_VOL_STABLE
        }
        return opposite_map.get(scenario, scenario)
    
    def _get_neutral_scenario(self, scenario: MarketCondition) -> MarketCondition:
        """Get a neutral/stable scenario with similar volatility level."""
        if "HIGH_VOL" in scenario.name:
            return MarketCondition.HIGH_VOL_STABLE
        else:
            return MarketCondition.STABLE_VOL_STABLE
    
    def generate_scenario_matrix(self, primary_asset: str, required_assets: List[str]) -> Dict[str, Dict[str, MarketCondition]]:
        """
        Generate a complete scenario matrix showing all possible propagations.
        
        Returns:
            Dictionary mapping primary scenarios to asset scenario assignments
        """
        scenario_matrix = {}
        
        for primary_scenario in self.market_conditions:
            scenarios = self.propagate_scenarios(
                primary_asset, primary_scenario, required_assets, "correlation_weighted"
            )
            scenario_matrix[primary_scenario.name] = scenarios
        
        return scenario_matrix
    
    def explain_scenario_logic(self, primary_asset: str, primary_scenario: MarketCondition,
                              target_asset: str, assigned_scenario: MarketCondition) -> str:
        """
        Provide explanation for why a particular scenario was assigned.
        """
        if primary_asset == target_asset:
            return f"{target_asset} is the primary asset with user-selected scenario"
        
        regime_name = primary_scenario.name
        
        try:
            correlation = get_regime_correlation(regime_name, primary_asset, target_asset)
            
            if correlation > 0.5:
                return f"{target_asset} has strong positive correlation ({correlation:.1f}) with {primary_asset} in {regime_name} regime"
            elif correlation < -0.3:
                return f"{target_asset} has negative correlation ({correlation:.1f}) with {primary_asset}, assigned opposite scenario"
            else:
                return f"{target_asset} has weak correlation ({correlation:.1f}) with {primary_asset}, assigned neutral scenario"
                
        except (ValueError, KeyError):
            return f"{target_asset} correlation data unavailable, using same scenario as {primary_asset}"

def test_scenario_propagation():
    """Test the scenario propagation system."""
    propagator = ScenarioPropagator()
    
    print("SCENARIO PROPAGATION TEST")
    print("=" * 40)
    
    # Test case: BTC crash scenario
    primary_asset = "BTC"
    primary_scenario = MarketCondition.HIGH_VOL_DOWN
    required_assets = ["BTC", "SPY", "GOLD", "TIPS"]
    
    print(f"Primary: {primary_asset} = {primary_scenario.value}")
    print()
    
    # Test correlation-weighted method
    scenarios = propagator.propagate_scenarios(
        primary_asset, primary_scenario, required_assets, "correlation_weighted"
    )
    
    print("Correlation-Weighted Assignments:")
    for asset, scenario in scenarios.items():
        explanation = propagator.explain_scenario_logic(
            primary_asset, primary_scenario, asset, scenario
        )
        print(f"  {asset}: {scenario.value}")
        print(f"    → {explanation}")
    
    print()
    
    # Test deterministic method
    scenarios_det = propagator.propagate_scenarios(
        primary_asset, primary_scenario, required_assets, "deterministic"
    )
    
    print("Deterministic Assignments:")
    for asset, scenario in scenarios_det.items():
        print(f"  {asset}: {scenario.value}")

if __name__ == "__main__":
    test_scenario_propagation()