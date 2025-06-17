"""
Monte Carlo simulation with GARCH+jumps modeling
"""
import numpy as np
import pandas as pd
from arch import arch_model
import streamlit as st
from .market_conditions import MarketCondition, adjust_mu_sigma_for_condition

class MonteCarloSimulator:
    def __init__(self, bitcoin_data):
        """
        Initialize Monte Carlo simulator with Bitcoin data
        """
        self.df_hist = bitcoin_data.copy()
        
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
        
        # Jump calibration
        sigma_day = daily_ret.std()
        jump_mask = np.abs(daily_ret) > 6 * sigma_day
        self.jump_lambda = jump_mask.mean()
        self.jump_mu = daily_ret[jump_mask].mean() if jump_mask.any() else 0.0
        self.jump_sigma = daily_ret[jump_mask].std() if jump_mask.any() else 0.0
        
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
            mu_adjusted, sigma_adjusted = adjust_mu_sigma_for_condition(
                self.mu_daily, base_vol, market_condition
            )
            sigma_multiplier = sigma_adjusted / base_vol
        
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
                
                # Calculate new price
                price = prev * np.exp(ret)
                closes.append(max(price, MIN_PRICE))
            
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
                if isinstance(strategy, dict) and 'conditions' in strategy:
                    return processor.execute_strategy_conditions(df, strategy['conditions'])
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
                      market_condition=None, progress_callback=None):
        """
        Run complete Monte Carlo simulation with market condition scenarios
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
            'close_paths': close_paths,
            'cagr_values': cagr_values,
            'drawdown_values': drawdown_values,
            'median_cagr': np.median(cagr_values),
            'worst_decile_cagr': np.percentile(cagr_values, 10),
            'median_max_drawdown': np.median(drawdown_values)
        }
        
        return results
