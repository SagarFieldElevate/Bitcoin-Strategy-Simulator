"""
Monte Carlo simulation with GARCH+jumps modeling
"""
import numpy as np
import pandas as pd
from arch import arch_model
import streamlit as st
from .market_conditions import MarketCondition, adjust_mu_sigma_for_condition

class MonteCarloSimulator:
    def __init__(self, bitcoin_data, pinecone_client=None):
        """
        Initialize Monte Carlo simulator with Bitcoin data
        """
        self.df_hist = bitcoin_data.copy()
        self.pinecone_client = pinecone_client
        
        # Extract price and volume data
        self.prices = self.df_hist["Close"].astype(float).values
        self.vol_hist = self.df_hist["Volume"].replace(0, np.nan).dropna().astype(float).values
        self.last_date = self.df_hist.index[-1]
        
        # Calibrate model parameters
        self._calibrate_parameters()
    
    def _calibrate_parameters(self):
        """
        Calibrate drift, jump parameters, and GARCH parameters
        """
        # Calculate daily returns
        daily_ret = np.diff(np.log(self.prices))
        self.mu_daily = daily_ret.mean()
        
        # Debug logging for GARCH calibration
        print(f"[GARCH DEBUG] Bitcoin data points: {len(self.prices)}")
        print(f"[GARCH DEBUG] Daily returns calculated: {len(daily_ret)} observations")
        print(f"[GARCH DEBUG] Raw daily μ = {self.mu_daily:.6f} ({self.mu_daily*365:.2%} annually)")
        print(f"[GARCH DEBUG] Daily volatility = {daily_ret.std():.4f} ({daily_ret.std()*np.sqrt(365):.2%} annually)")
        
        # Jump calibration
        sigma_day = daily_ret.std()
        jump_mask = np.abs(daily_ret) > 6 * sigma_day
        self.jump_lambda = jump_mask.mean()
        self.jump_mu = daily_ret[jump_mask].mean() if jump_mask.any() else 0.0
        self.jump_sigma = daily_ret[jump_mask].std() if jump_mask.any() else 0.0
        
        print(f"[GARCH DEBUG] Jump frequency (lambda): {self.jump_lambda:.4f}")
        print(f"[GARCH DEBUG] Jump size (mu): {self.jump_mu:.4f}")
        print(f"[GARCH DEBUG] Jump volatility (sigma): {self.jump_sigma:.4f}")
        
        # Fit GARCH(1,1) model
        try:
            garch_mod = arch_model(daily_ret * 100, p=1, q=1, mean='zero')
            garch_fit = garch_mod.fit(disp="off")
            self.omega = garch_fit.params["omega"]
            self.alpha = garch_fit.params["alpha[1]"]
            self.beta = garch_fit.params["beta[1]"]
        except Exception as e:
            # Fallback to simple parameters if GARCH fitting fails
            st.warning(f"GARCH fitting failed, using fallback parameters: {str(e)}")
            self.omega = 0.1
            self.alpha = 0.1
            self.beta = 0.8
        
        # Store initial variance
        self.initial_variance = daily_ret.var() * 10000
    
    def generate_price_paths(self, n_simulations, simulation_days, market_condition=None, seed=42):
        """
        Generate Monte Carlo price paths using GARCH+jumps model with market condition adjustments
        """
        MIN_PRICE = 1e-6
        rng = np.random.default_rng(seed)
        
        # Adjust drift and volatility for market condition
        mu_adjusted = self.mu_daily
        sigma_multiplier = 1.0
        
        if market_condition:
            # Calculate base volatility from GARCH parameters
            base_vol = np.sqrt(self.initial_variance) / 100
            print(f"[PATH DEBUG] Calling adjust_mu_sigma_for_condition with mu={self.mu_daily:.6f}, sigma={base_vol:.4f}")
            mu_adjusted, sigma_adjusted = adjust_mu_sigma_for_condition(
                self.mu_daily, base_vol, market_condition
            )
            sigma_multiplier = sigma_adjusted / base_vol
            print(f"[PATH DEBUG] Market condition applied: mu_adjusted={mu_adjusted:.6f}, sigma_multiplier={sigma_multiplier:.4f}")
        else:
            print(f"[PATH DEBUG] No market condition applied, using base mu={self.mu_daily:.6f}")
        
        close_paths = np.empty((n_simulations, simulation_days + 1))
        close_paths[:, 0] = self.prices[-1]  # Start from current price
        
        for path in range(n_simulations):
            sigma2 = self.initial_variance
            closes = [self.prices[-1]]
            
            for t in range(1, simulation_days + 1):
                # Protect against zero or negative prices
                prev = max(closes[-1], MIN_PRICE)
                prev2 = max(closes[-2], MIN_PRICE) if t > 1 else prev
                
                # Calculate previous return for GARCH
                last_ret = np.log(prev / prev2) * 100 if t > 1 else 0
                
                # Update variance using GARCH(1,1)
                sigma2 = self.omega + self.alpha * last_ret**2 + self.beta * sigma2
                sigma_t = np.sqrt(max(sigma2, 1e-12)) / 100 * sigma_multiplier
                
                # Generate random components
                Z = rng.standard_normal()
                J = rng.poisson(self.jump_lambda) * rng.normal(self.jump_mu, self.jump_sigma)
                
                # Calculate return with market condition adjustment
                ret = mu_adjusted - 0.5 * sigma_t**2 + sigma_t * Z + J
                
                # Debug logging for first few price calculations
                if path == 0 and t <= 3:
                    print(f"[PATH DEBUG] Day {t}: mu_adj={mu_adjusted:.6f}, sigma_t={sigma_t:.4f}, Z={Z:.4f}, J={J:.4f}")
                    print(f"[PATH DEBUG] Day {t}: ret={ret:.6f}, prev_price={prev:.2f}")
                
                # Calculate new price
                price = prev * np.exp(ret)
                closes.append(max(price, MIN_PRICE))
                
                if path == 0 and t <= 3:
                    print(f"[PATH DEBUG] Day {t}: new_price={price:.2f}, change={((price/prev)-1)*100:.2f}%")
            
            close_paths[path] = closes
        
        return close_paths
    
    def synthesize_ohlcv(self, close_path, start_date):
        """
        Create synthetic OHLCV data from close prices
        """
        rng = np.random.default_rng(42)
        
        dates = pd.date_range(start=start_date, periods=len(close_path), freq="D")
        closes = pd.Series(close_path, index=dates, name="Close")
        opens = closes.shift(1).fillna(closes.iloc[0])
        
        # Calculate realistic high/low based on returns
        rpct = closes.pct_change().abs().fillna(0)
        high = closes * (1 + 0.5 * rpct)
        low = closes * (1 - 0.5 * rpct)
        
        # Generate volume using historical distribution
        vol = pd.Series(rng.choice(self.vol_hist, size=len(dates)), index=dates)
        
        return pd.DataFrame({
            "Open": opens,
            "High": high, 
            "Low": low,
            "Close": closes,
            "Volume": vol
        })
    
    def vwap(self, df, window):
        """Calculate Volume Weighted Average Price"""
        pv = df["Close"] * df["Volume"]
        return pv.rolling(window).sum() / df["Volume"].rolling(window).sum()
    
    def cemd_strategy(self, df, spread_threshold=2.0, holding_period=5, risk_percent=100.0):
        """
        Default CEMD (Corporate vs Retail Momentum Divergence) strategy
        """
        df = df.copy()
        df["VW"] = self.vwap(df, 20)
        df["rng"] = (df["High"] - df["Low"]) / df["Close"] * 100
        
        # Calculate momentum and pressure
        mom = (df["Close"] - df["Open"]) / df["Open"] * 100
        press = df["Volume"] * mom
        inst = press.rolling(10).mean()
        retail = press.rolling(30).mean()
        div = (inst - retail) / (retail.abs() + 1e-4) * 100
        rng20 = df["rng"].rolling(20).mean()
        
        # Generate signals
        long_sig = (div > spread_threshold) & (df["Close"] > df["VW"]) & (df["rng"] < rng20)
        short_sig = (div < -spread_threshold) & (df["Close"] < df["VW"]) & (df["rng"] < rng20)
        
        # Execute strategy
        pos, days, pl = 0, 0, []
        for i in range(len(df)):
            # Exit conditions
            if pos and (days >= holding_period or df.index[i].weekday() == 4):
                pos = 0
                days = 0
            
            # Entry conditions
            if not pos:
                if long_sig.iloc[i]:
                    pos = 1
                elif short_sig.iloc[i]:
                    pos = -1
            
            if pos:
                days += 1
            
            # Calculate return
            ret = pos * (df["Close"].iloc[i] / df["Close"].iloc[i-1] - 1) if i else 0
            ret = max(ret, -0.9999)  # Cap at -99.99%
            pl.append(ret * (risk_percent / 100))
        
        return pd.Series(pl, index=df.index)
    
    def execute_strategy(self, df, strategy=None):
        """
        Execute strategy - either default CEMD or processed strategy from Pinecone
        """
        if strategy is None or strategy == "CEMD (Default)":
            return self.cemd_strategy(df)
        else:
            try:
                # Import strategy processor here to avoid circular imports
                from .strategy_processor import StrategyProcessor
                processor = StrategyProcessor()
                
                # Execute strategy using its conditions
                if isinstance(strategy, dict):
                    if 'conditions' in strategy:
                        # Strategy object with conditions
                        return processor.execute_strategy_conditions(df, strategy['conditions'])
                    else:
                        # Direct conditions JSON from OpenAI
                        return processor.execute_strategy_conditions(df, strategy)
                else:
                    # Fallback to default strategy
                    return self.cemd_strategy(df)
                    
            except Exception as e:
                print(f"Failed to execute strategy, using default CEMD: {str(e)}")
                return self.cemd_strategy(df)
    
    def calculate_equity_curve(self, returns):
        """Calculate equity curve from returns"""
        return (1 + returns).cumprod()
    
    def run_simulation(self, n_simulations, simulation_days, selected_strategy=None, 
                      market_condition=None, progress_callback=None, simulation_mode='btc_only', 
                      required_variables=None):
        """
        Run complete Monte Carlo simulation with market condition scenarios and multi-factor support
        """
        if simulation_mode == 'multi_factor' and required_variables:
            return self.run_multi_factor_simulation(
                n_simulations, simulation_days, selected_strategy, 
                market_condition, progress_callback, required_variables
            )
        else:
            return self.run_btc_only_simulation(
                n_simulations, simulation_days, selected_strategy, 
                market_condition, progress_callback
            )
    
    def run_btc_only_simulation(self, n_simulations, simulation_days, selected_strategy=None, 
                               market_condition=None, progress_callback=None):
        """
        Run BTC-only Monte Carlo simulation (original logic)
        """
        # Generate price paths with market condition adjustments
        close_paths = self.generate_price_paths(n_simulations, simulation_days, market_condition)
        
        # Calculate strategy performance for each path
        cagr_values = []
        drawdown_values = []
        
        for i, path in enumerate(close_paths):
            if progress_callback:
                progress_callback(i / len(close_paths))
            
            # Create OHLCV data
            ohlcv_df = self.synthesize_ohlcv(path, self.last_date + pd.Timedelta(days=1))
            
            # Execute strategy
            returns = self.execute_strategy(ohlcv_df, selected_strategy)
            
            # Calculate equity curve
            equity = self.calculate_equity_curve(returns)
            
            if len(equity) == 0 or equity.iloc[-1] <= 0 or pd.isna(equity.iloc[-1]):
                cagr_values.append(-100.0)
                drawdown_values.append(-100.0)
            else:
                # Calculate CAGR
                final_equity = equity.iloc[-1]
                cagr = (final_equity ** (365 / len(equity)) - 1) * 100
                cagr_values.append(cagr)
                
                # Calculate maximum drawdown
                max_dd = (equity / equity.cummax() - 1).min() * 100
                drawdown_values.append(max_dd)
        
        if progress_callback:
            progress_callback(1.0)
        
        # Calculate summary statistics
        results = {
            'simulation_mode': 'btc_only',
            'close_paths': close_paths,
            'cagr_values': cagr_values,
            'drawdown_values': drawdown_values,
            'median_cagr': np.median(cagr_values),
            'worst_decile_cagr': np.percentile(cagr_values, 10),
            'median_max_drawdown': np.median(drawdown_values)
        }
        
        return results
    
    def run_multi_factor_simulation(self, n_simulations, simulation_days, selected_strategy=None, 
                                   market_condition=None, progress_callback=None, required_variables=None):
        """
        Run multi-factor Monte Carlo simulation with dynamic variable detection and regime-specific correlations
        """
        from .multi_factor_data import MultiFactorDataFetcher
        from .scenario_propagation import ScenarioPropagator
        from regime_correlations import REGIME_CORRELATIONS, VARIABLE_NAME_MAPPING
        from correlation_utils import build_matrix_from_dict
        
        # Detect variables from strategy metadata if not explicitly provided
        if required_variables is None and selected_strategy is not None:
            required_variables = self._detect_strategy_variables(selected_strategy)
        
        print(f"Running multi-factor simulation with variables: {required_variables}")
        
        # Map variable names to short codes for correlation matrix lookup
        variable_codes = self._map_variables_to_codes(required_variables)
        print(f"Mapped to correlation codes: {variable_codes}")
        
        # Fetch historical multi-factor data
        data_fetcher = MultiFactorDataFetcher(self.pinecone_client)
        historical_data = data_fetcher.fetch_multi_factor_data(required_variables)
        available_variables = historical_data.columns.tolist()
        available_codes = self._map_variables_to_codes(available_variables)
        
        print(f"Successfully fetched data for {len(available_variables)} variables: {available_variables}")
        
        # Generate regime-specific scenarios for each available variable
        if market_condition is not None:
            propagator = ScenarioPropagator()
            primary_asset = "BTC"  # Use BTC as primary driver
            
            # Propagate scenarios based on correlations using short codes
            regime_scenarios = propagator.propagate_scenarios(
                primary_asset, market_condition, available_codes, "correlation_weighted"
            )
            
            print(f"Regime scenario assignments:")
            for var, scenario in regime_scenarios.items():
                print(f"  {var}: {scenario.value}")
            
            # Build filtered correlation matrix for available variables only
            regime_name = market_condition.name
            
            print(f"\n=== SIMULATION DEBUG INFO ===")
            print(f"Selected Market Condition: {market_condition.value}")
            print(f"Variables being simulated: {available_variables}")
            print(f"Mapped to correlation codes: {available_codes}")
            
            if regime_name in REGIME_CORRELATIONS and len(available_codes) > 1:
                correlation_matrix = build_matrix_from_dict(
                    REGIME_CORRELATIONS[regime_name], available_codes
                )
                print(f"\nUsing {regime_name} regime correlation matrix:")
                import pandas as pd
                corr_df = pd.DataFrame(correlation_matrix, index=available_codes, columns=available_codes)
                print(corr_df.round(3))
                print()
            else:
                # Fallback to historical correlation
                correlation_matrix = data_fetcher.calculate_correlation_matrix(historical_data)
                print(f"\nUsing historical correlation matrix (regime data unavailable):")
                import pandas as pd
                corr_df = pd.DataFrame(correlation_matrix, index=available_variables, columns=available_variables)
                print(corr_df.round(3))
                print()
                regime_scenarios = None  # No regime scenarios if no regime correlation
        else:
            # No market condition specified, use base parameters
            regime_scenarios = None
            correlation_matrix = data_fetcher.calculate_correlation_matrix(historical_data)
            print("Using historical correlation matrix and base parameters")
        
        # Generate correlated paths with regime-specific scenarios
        simulated_paths = data_fetcher.simulate_multi_factor_series(
            historical_data, n_simulations, simulation_days, 
            correlation_matrix, market_condition=None, regime_scenarios=regime_scenarios
        )
        
        # Calculate strategy performance for each simulation
        cagr_values = []
        drawdown_values = []
        
        for i in range(n_simulations):
            if progress_callback:
                progress_callback(i / n_simulations)
            
            # Create multi-factor DataFrame for this simulation
            sim_data = {}
            for var in required_variables:
                if var in simulated_paths:
                    sim_data[var] = simulated_paths[var][i]
            
            # Create date index
            start_date = self.last_date + pd.Timedelta(days=1)
            date_index = pd.date_range(start=start_date, periods=simulation_days, freq='D')
            
            multi_factor_df = pd.DataFrame(sim_data, index=date_index)
            
            # For strategy execution, we still need OHLCV format for BTC
            if 'BTC' in sim_data:
                btc_path = sim_data['BTC']
                ohlcv_df = self.synthesize_ohlcv(btc_path, start_date)
                
                # Add other variables as additional columns
                for var in required_variables:
                    if var != 'BTC' and var in sim_data:
                        ohlcv_df[var] = sim_data[var]
            else:
                # Fallback to BTC-only if BTC not in required variables
                print("Warning: BTC not in required variables, using fallback")
                continue
            
            # Execute strategy with multi-factor data
            returns = self.execute_strategy(ohlcv_df, selected_strategy)
            
            # Calculate equity curve
            equity = self.calculate_equity_curve(returns)
            
            if len(equity) == 0 or equity.iloc[-1] <= 0 or pd.isna(equity.iloc[-1]):
                cagr_values.append(-100.0)
                drawdown_values.append(-100.0)
            else:
                # Calculate CAGR
                final_equity = equity.iloc[-1]
                cagr = (final_equity ** (365 / len(equity)) - 1) * 100
                cagr_values.append(cagr)
                
                # Calculate maximum drawdown
                max_dd = (equity / equity.cummax() - 1).min() * 100
                drawdown_values.append(max_dd)
        
        if progress_callback:
            progress_callback(1.0)
        
        # Calculate summary statistics
        cagr_values = np.array(cagr_values)
        drawdown_values = np.array(drawdown_values)
        
        results = {
            'simulation_mode': 'multi_factor',
            'variables_used': required_variables,
            'close_paths': simulated_paths.get('BTC', []),  # Return BTC paths for visualization
            'all_paths': simulated_paths,  # Include all variable paths
            'cagr_values': cagr_values,
            'drawdown_values': drawdown_values,
            'median_cagr': np.median(cagr_values),
            'worst_decile_cagr': np.percentile(cagr_values, 10),
            'median_max_drawdown': np.median(drawdown_values)
        }
        
        return results
