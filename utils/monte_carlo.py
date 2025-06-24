"""
Monte Carlo simulation with GARCH+jumps modeling - Optimized Version
"""
import numpy as np
import pandas as pd
from arch import arch_model
import streamlit as st
from .market_conditions import MarketCondition, adjust_mu_sigma_for_condition
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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
        
        # Cache for strategy conditions and multi-factor data
        self._strategy_cache = {}
        self._multi_factor_cache = {}
        
        # Pre-compute common values
        self._precompute_common_values()
    
    def _precompute_common_values(self):
        """Pre-compute commonly used values for optimization"""
        # Pre-compute log returns for faster GARCH updates
        self.log_returns = np.diff(np.log(self.prices))
        
        # Pre-compute volume statistics
        self.vol_mean = self.vol_hist.mean()
        self.vol_std = self.vol_hist.std()
        
        # Pre-compute price statistics
        self.price_mean = self.prices.mean()
        self.price_std = self.prices.std()
    
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
    
    def generate_price_paths_vectorized(self, n_simulations, simulation_days, market_condition=None, seed=42):
        """
        Optimized vectorized price path generation using NumPy broadcasting
        """
        MIN_PRICE = 1e-6
        rng = np.random.default_rng(seed)
        
        # Adjust drift and volatility for market condition
        mu_adjusted = self.mu_daily
        sigma_multiplier = 1.0
        
        if market_condition:
            base_vol = np.sqrt(self.initial_variance) / 100
            mu_adjusted, sigma_adjusted = adjust_mu_sigma_for_condition(self.mu_daily, base_vol, market_condition)
            sigma_multiplier = sigma_adjusted / base_vol if base_vol > 0 else 1.0
        
        # Initialize arrays for all paths at once
        close_paths = np.zeros((n_simulations, simulation_days + 1))
        close_paths[:, 0] = self.prices[-1]
        
        # Pre-generate all random numbers for efficiency
        Z_all = rng.standard_normal((n_simulations, simulation_days))
        J_poisson = rng.poisson(self.jump_lambda, (n_simulations, simulation_days))
        J_normal = rng.normal(self.jump_mu, self.jump_sigma, (n_simulations, simulation_days))
        J_all = J_poisson * J_normal
        
        # Vectorized GARCH variance evolution
        sigma2 = np.full(n_simulations, self.initial_variance)
        
        for t in range(simulation_days):
            # Update variance (simplified for vectorization)
            if t > 0:
                prev_ret = np.log(close_paths[:, t] / close_paths[:, t-1]) * 100
                sigma2 = self.omega + self.alpha * prev_ret**2 + self.beta * sigma2
            
            sigma_t = np.sqrt(np.maximum(sigma2, 1e-12)) / 100 * sigma_multiplier
            
            # Calculate returns for all paths at once
            ret = mu_adjusted - 0.5 * sigma_t**2 + sigma_t * Z_all[:, t] + J_all[:, t]
            
            # Update prices (vectorized)
            close_paths[:, t + 1] = close_paths[:, t] * np.exp(ret)
            close_paths[:, t + 1] = np.maximum(close_paths[:, t + 1], MIN_PRICE)

        return close_paths
    
    # Use the optimized version as the default
    generate_price_paths = generate_price_paths_vectorized
    
    def synthesize_ohlcv_batch_optimized(self, close_paths, start_date):
        """
        Optimized batch OHLCV synthesis using NumPy operations
        """
        n_sims, n_days = close_paths.shape
        dates = pd.date_range(start=start_date, periods=n_days, freq="D")
        
        # Pre-generate random volumes for all simulations at once
        rng = np.random.default_rng(42)
        vol_indices = rng.integers(0, len(self.vol_hist), size=(n_sims, n_days))
        vol_matrix = self.vol_hist[vol_indices]
        
        # Pre-calculate returns for all paths
        returns = np.zeros_like(close_paths)
        returns[:, 1:] = close_paths[:, 1:] / close_paths[:, :-1] - 1
        returns_abs = np.abs(returns)
        
        # Create OHLCV data for all simulations
        ohlcv_data = []
        
        # Process in chunks for memory efficiency
        chunk_size = min(1000, n_sims)
        
        for chunk_start in range(0, n_sims, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_sims)
            chunk_indices = range(chunk_start, chunk_end)
            
            for i in chunk_indices:
                close_path = close_paths[i]
                
                # Vectorized OHLC calculation
                closes = pd.Series(close_path, index=dates, name="Close")
                opens = closes.shift(1).fillna(closes.iloc[0])
                
                # Efficient high/low calculation
                high_multiplier = 1 + 0.5 * returns_abs[i]
                low_multiplier = 1 - 0.5 * returns_abs[i]
                
                high = closes * high_multiplier
                low = closes * low_multiplier
                
                ohlcv_data.append(pd.DataFrame({
                    "Open": opens,
                    "High": high, 
                    "Low": low,
                    "Close": closes,
                    "Volume": vol_matrix[i]
                }))
        
        return ohlcv_data
    
    # Use optimized version as default
    synthesize_ohlcv_batch = synthesize_ohlcv_batch_optimized
    
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
    

    
    def execute_strategy_cached(self, df, strategy):
        """Execute strategy with caching for repeated calls"""
        # Create a hash of the strategy for caching
        strategy_hash = str(strategy) if isinstance(strategy, dict) else str(type(strategy))
        df_hash = str(len(df)) + str(df.index[0]) + str(df.index[-1])
        cache_key = f"{strategy_hash}_{df_hash}"
        
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key].copy()
        
        # Execute strategy
        result = self.execute_strategy(df, strategy)
        
        # Cache result (limit cache size)
        if len(self._strategy_cache) > 1000:
            # Remove oldest entries
            self._strategy_cache = dict(list(self._strategy_cache.items())[-500:])
        
        self._strategy_cache[cache_key] = result.copy()
        return result
    
    def execute_strategy(self, df, strategy):
        """
        Execute a strategy from Pinecone. A valid strategy must be provided.
        """
        if strategy is None:
            raise ValueError("No strategy provided. A valid strategy must be selected to run simulation.")
        
        # Strategy must be a dictionary containing conditions
        if not isinstance(strategy, dict):
            raise ValueError(f"Invalid strategy format. Expected a dictionary, but got {type(strategy)}. Please select a valid strategy.")
        
        try:
            # Import strategy processor here to avoid circular imports
            from .strategy_processor import StrategyProcessor
            processor = StrategyProcessor()
            
            # The strategy object can come in two forms from Pinecone/frontend
            conditions = strategy.get('conditions', strategy)
            return processor.execute_strategy_conditions(df, conditions)
                    
        except Exception as e:
            # Add context to the error and re-raise it.
            # This will stop the simulation and display the error in the Streamlit app.
            error_message = f"Execution failed for strategy '{strategy.get('id', 'Unknown')}'. Reason: {str(e)}"
            print(f"[STRATEGY ERROR] {error_message}")
            raise RuntimeError(error_message) from e
    
    def calculate_equity_curve(self, returns):
        """Calculate equity curve from returns"""
        return (1 + returns).cumprod()
    
    def run_simulation(self, n_simulations, simulation_days, selected_strategy, 
                      market_condition=None, progress_callback=None, simulation_mode='btc_only', 
                      required_variables=None):
        """
        Run complete Monte Carlo simulation with market condition scenarios and multi-factor support.
        A valid strategy must be provided.
        """
        if selected_strategy is None:
            raise ValueError("No strategy provided. A valid strategy must be selected to run simulation.")
            
        if simulation_mode == 'multi_factor' and required_variables:
            return self.run_multi_factor_simulation(
                n_simulations, simulation_days, selected_strategy, 
                market_condition, progress_callback, required_variables
            )
        else:
            return self.run_btc_only_simulation_optimized(
                n_simulations, simulation_days, selected_strategy, 
                market_condition, progress_callback
            )
    
    def run_btc_only_simulation_optimized(self, n_simulations, simulation_days, selected_strategy, 
                               market_condition=None, progress_callback=None):
        """
        Optimized BTC-only Monte Carlo simulation with parallel processing
        """
        import time
        start_time = time.time()
        
        # Generate price paths with market condition adjustments
        print(f"Generating {n_simulations} price paths...")
        close_paths = self.generate_price_paths_vectorized(n_simulations, simulation_days, market_condition)
        
        # Pre-generate all OHLCV data at once for better performance
        print(f"Synthesizing OHLCV data for all paths...")
        start_date = self.last_date + pd.Timedelta(days=1)
        
        # Batch synthesize OHLCV data
        all_ohlcv_data = self.synthesize_ohlcv_batch_optimized(close_paths, start_date)
        
        # Validate that a strategy is provided
        if selected_strategy is None:
            raise ValueError("No strategy provided. A valid strategy must be selected to run simulation.")
        
        # Process strategy with parallel processing
        print("Processing strategy with parallel execution...")
        
        # Determine optimal number of workers
        n_workers = min(multiprocessing.cpu_count(), 8)
        batch_size = max(10, n_simulations // (n_workers * 10))
        
        cagr_values = []
        drawdown_values = []
        
        def process_batch(batch_data):
            """Process a batch of simulations"""
            batch_results = {'cagr': [], 'drawdown': []}
            
            for ohlcv_df in batch_data:
                try:
                    returns = self.execute_strategy_cached(ohlcv_df, selected_strategy)
                    equity = self.calculate_equity_curve(returns)
                    
                    if len(equity) == 0 or equity.iloc[-1] <= 0 or pd.isna(equity.iloc[-1]):
                        batch_results['cagr'].append(-100.0)
                        batch_results['drawdown'].append(-100.0)
                    else:
                        final_equity = equity.iloc[-1]
                        cagr = (final_equity ** (365 / len(equity)) - 1) * 100
                        batch_results['cagr'].append(cagr)
                        
                        max_dd = (equity / equity.cummax() - 1).min() * 100
                        batch_results['drawdown'].append(max_dd)
                except Exception as e:
                    print(f"Strategy execution failed: {e}")
                    batch_results['cagr'].append(-100.0)
                    batch_results['drawdown'].append(-100.0)
            
            return batch_results
        
        # Process in batches using ThreadPoolExecutor (GIL-friendly for I/O bound operations)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for batch_start in range(0, n_simulations, batch_size):
                batch_end = min(batch_start + batch_size, n_simulations)
                batch_ohlcv = all_ohlcv_data[batch_start:batch_end]
                
                future = executor.submit(process_batch, batch_ohlcv)
                futures.append((future, batch_start, batch_end))
            
            # Collect results
            for future, batch_start, batch_end in futures:
                batch_results = future.result()
                cagr_values.extend(batch_results['cagr'])
                drawdown_values.extend(batch_results['drawdown'])
                
                # Update progress
                if progress_callback:
                    progress_callback(batch_end / n_simulations)
        
        if progress_callback:
            progress_callback(1.0)
        
        # Calculate summary statistics
        cagr_array = np.array(cagr_values)
        drawdown_array = np.array(drawdown_values)
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Average time per simulation: {(end_time - start_time) / n_simulations * 1000:.2f} ms")
        
        results = {
            'simulation_mode': 'btc_only',
            'close_paths': close_paths,
            'cagr_values': cagr_array,
            'drawdown_values': drawdown_array,
            'median_cagr': np.median(cagr_array),
            'worst_decile_cagr': np.percentile(cagr_array, 10),
            'median_max_drawdown': np.median(drawdown_array)
        }
        
        return results
    
    # Use optimized version as default
    run_btc_only_simulation = run_btc_only_simulation_optimized
    
    def _detect_strategy_variables(self, strategy):
        """
        Detect variables used by a strategy from its excel_names field
        
        Args:
            strategy: Strategy dictionary with excel_names field
            
        Returns:
            list: Variable names used by the strategy
        """
        if not strategy or 'excel_names' not in strategy:
            return ['BTC']  # Default to Bitcoin-only
        
        variables = strategy['excel_names']
        if not variables:
            return ['BTC']
        
        # Always ensure BTC is included if not already present
        if 'Bitcoin Daily Close Price' not in variables:
            variables = ['Bitcoin Daily Close Price'] + list(variables)
        
        return variables
    
    def _map_variables_to_codes(self, variables):
        """
        Map variable names to short codes for correlation matrix lookup
        
        Args:
            variables: List of variable names
            
        Returns:
            list: Corresponding short codes
        """
        from regime_correlations import VARIABLE_NAME_MAPPING
        
        codes = []
        for var in variables:
            if var in VARIABLE_NAME_MAPPING:
                codes.append(VARIABLE_NAME_MAPPING[var])
            elif var == 'BTC':
                codes.append('BTC')
            else:
                # Default mapping for unmapped variables
                codes.append(var.replace(' ', '_').upper())
        
        return codes
    
    def run_multi_factor_simulation(self, n_simulations, simulation_days, selected_strategy, 
                                   market_condition=None, progress_callback=None, required_variables=None):
        """
        Run multi-factor Monte Carlo simulation with dynamic variable detection and regime-specific correlations
        """
        import pandas as pd
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
        
        # Update progress for data fetching phase
        if progress_callback:
            progress_callback(0.1)  # 10% for data fetching start
        
        # Check cache for multi-factor data
        cache_key = f"{','.join(sorted(required_variables))}_{market_condition}"
        
        if cache_key in self._multi_factor_cache:
            print("Using cached multi-factor data")
            historical_data = self._multi_factor_cache[cache_key]['historical_data']
            available_variables = self._multi_factor_cache[cache_key]['available_variables']
            available_codes = self._multi_factor_cache[cache_key]['available_codes']
        else:
            # Fetch historical multi-factor data
            data_fetcher = MultiFactorDataFetcher(self.pinecone_client)
            historical_data = data_fetcher.fetch_multi_factor_data(required_variables)
            available_variables = historical_data.columns.tolist()
            available_codes = self._map_variables_to_codes(available_variables)
            
            # Cache the data
            self._multi_factor_cache[cache_key] = {
                'historical_data': historical_data,
                'available_variables': available_variables,
                'available_codes': available_codes
            }
        
        print(f"Successfully fetched data for {len(available_variables)} variables: {available_variables}")
        
        # Update progress after data fetching
        if progress_callback:
            progress_callback(0.2)  # 20% after data fetch complete
        
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
                data_fetcher = MultiFactorDataFetcher(self.pinecone_client)
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
            data_fetcher = MultiFactorDataFetcher(self.pinecone_client)
            correlation_matrix = data_fetcher.calculate_correlation_matrix(historical_data)
            print("Using historical correlation matrix and base parameters")
        
        # Generate correlated paths with regime-specific scenarios
        data_fetcher = MultiFactorDataFetcher(self.pinecone_client)
        simulated_paths = data_fetcher.simulate_multi_factor_series(
            historical_data, n_simulations, simulation_days, 
            correlation_matrix, market_condition=None, regime_scenarios=regime_scenarios
        )
        
        # Calculate strategy performance for each simulation
        cagr_values = []
        drawdown_values = []
        
        # Process in parallel batches for efficiency
        batch_size = min(100, n_simulations)
        
        for batch_start in range(0, n_simulations, batch_size):
            batch_end = min(batch_start + batch_size, n_simulations)
            
            for i in range(batch_start, batch_end):
                if progress_callback:
                    progress_callback(0.2 + 0.8 * i / n_simulations)
            
            # Create multi-factor DataFrame for this simulation
            sim_data = {}
            for var in required_variables:
                if var in simulated_paths:
                    sim_data[var] = simulated_paths[var][i]
            
            # Create date index
            start_date = self.last_date + pd.Timedelta(days=1)
            date_index = pd.date_range(start=start_date, periods=simulation_days + 1, freq='D')
            
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