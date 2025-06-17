"""
Database utilities for storing simulation results and strategy data
"""
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import streamlit as st

Base = declarative_base()

class SimulationRun(Base):
    __tablename__ = 'simulation_runs'
    
    id = Column(Integer, primary_key=True)
    run_date = Column(DateTime, default=datetime.utcnow)
    strategy_name = Column(String(100))
    n_simulations = Column(Integer)
    simulation_days = Column(Integer)
    spread_threshold = Column(Float)
    holding_period = Column(Integer)
    risk_percent = Column(Float)
    median_cagr = Column(Float)
    worst_decile_cagr = Column(Float)
    median_max_drawdown = Column(Float)
    mean_cagr = Column(Float)
    std_cagr = Column(Float)
    profitable_sims = Column(Integer)
    win_rate = Column(Float)
    terminal_price_mean = Column(Float)
    terminal_price_median = Column(Float)

class StrategyPerformance(Base):
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(100))
    run_date = Column(DateTime, default=datetime.utcnow)
    simulation_run_id = Column(Integer)
    cagr_values = Column(Text)  # JSON array
    drawdown_values = Column(Text)  # JSON array
    terminal_prices = Column(Text)  # JSON array

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            return True
        except Exception as e:
            st.error(f"Error creating database tables: {str(e)}")
            return False
    
    def save_simulation_results(self, results, parameters):
        """Save simulation results to database"""
        try:
            session = self.SessionLocal()
            
            # Calculate additional metrics
            cagr_values = results['cagr_values']
            terminal_prices = results['close_paths'][:, -1]
            profitable_sims = len([x for x in cagr_values if x > 0])
            win_rate = profitable_sims / len(cagr_values) * 100
            
            # Create simulation run record
            sim_run = SimulationRun(
                strategy_name=parameters.get('strategy_name', 'Unknown'),
                n_simulations=parameters.get('n_simulations', 0),
                simulation_days=parameters.get('simulation_days', 0),
                spread_threshold=parameters.get('spread_threshold', 0),
                holding_period=parameters.get('holding_period', 0),
                risk_percent=parameters.get('risk_percent', 0),
                median_cagr=results['median_cagr'],
                worst_decile_cagr=results['worst_decile_cagr'],
                median_max_drawdown=results['median_max_drawdown'],
                mean_cagr=float(np.mean(cagr_values)),
                std_cagr=float(np.std(cagr_values)),
                profitable_sims=profitable_sims,
                win_rate=win_rate,
                terminal_price_mean=float(np.mean(terminal_prices)),
                terminal_price_median=float(np.median(terminal_prices))
            )
            
            session.add(sim_run)
            session.commit()
            
            # Create strategy performance record
            strategy_perf = StrategyPerformance(
                strategy_name=parameters.get('strategy_name', 'Unknown'),
                simulation_run_id=sim_run.id,
                cagr_values=json.dumps([float(x) for x in cagr_values]),
                drawdown_values=json.dumps([float(x) for x in results['drawdown_values']]),
                terminal_prices=json.dumps([float(x) for x in terminal_prices])
            )
            
            session.add(strategy_perf)
            session.commit()
            session.close()
            
            return sim_run.id
            
        except Exception as e:
            st.error(f"Error saving simulation results: {str(e)}")
            return None
    
    def get_simulation_history(self, limit=50):
        """Get recent simulation runs"""
        try:
            session = self.SessionLocal()
            runs = session.query(SimulationRun).order_by(SimulationRun.run_date.desc()).limit(limit).all()
            session.close()
            
            if not runs:
                return pd.DataFrame()
            
            data = []
            for run in runs:
                data.append({
                    'ID': run.id,
                    'Date': run.run_date,
                    'Strategy': run.strategy_name,
                    'Simulations': run.n_simulations,
                    'Days': run.simulation_days,
                    'Median CAGR': f"{run.median_cagr:.2f}%",
                    'Worst 10% CAGR': f"{run.worst_decile_cagr:.2f}%",
                    'Max DD': f"{run.median_max_drawdown:.2f}%",
                    'Win Rate': f"{run.win_rate:.1f}%"
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            st.error(f"Error retrieving simulation history: {str(e)}")
            return pd.DataFrame()
    
    def get_strategy_comparison(self):
        """Get strategy performance comparison"""
        try:
            session = self.SessionLocal()
            
            # Query for strategy comparison
            query = text("""
                SELECT 
                    strategy_name,
                    COUNT(*) as runs_count,
                    AVG(median_cagr) as avg_median_cagr,
                    AVG(worst_decile_cagr) as avg_worst_decile_cagr,
                    AVG(median_max_drawdown) as avg_max_drawdown,
                    AVG(win_rate) as avg_win_rate,
                    MAX(run_date) as last_run
                FROM simulation_runs 
                GROUP BY strategy_name 
                ORDER BY avg_median_cagr DESC
            """)
            
            result = session.execute(query)
            data = []
            
            for row in result:
                data.append({
                    'Strategy': row.strategy_name,
                    'Runs': row.runs_count,
                    'Avg Median CAGR': f"{row.avg_median_cagr:.2f}%",
                    'Avg Worst 10%': f"{row.avg_worst_decile_cagr:.2f}%",
                    'Avg Max DD': f"{row.avg_max_drawdown:.2f}%",
                    'Avg Win Rate': f"{row.avg_win_rate:.1f}%",
                    'Last Run': row.last_run.strftime('%Y-%m-%d %H:%M') if row.last_run else 'N/A'
                })
            
            session.close()
            return pd.DataFrame(data)
            
        except Exception as e:
            st.error(f"Error retrieving strategy comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_detailed_results(self, run_id):
        """Get detailed results for a specific simulation run"""
        try:
            session = self.SessionLocal()
            
            # Get simulation run details
            sim_run = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()
            if not sim_run:
                return None
            
            # Get strategy performance data
            strategy_perf = session.query(StrategyPerformance).filter(
                StrategyPerformance.simulation_run_id == run_id
            ).first()
            
            session.close()
            
            if strategy_perf:
                return {
                    'simulation_run': sim_run,
                    'cagr_values': json.loads(strategy_perf.cagr_values),
                    'drawdown_values': json.loads(strategy_perf.drawdown_values),
                    'terminal_prices': json.loads(strategy_perf.terminal_prices)
                }
            
            return {'simulation_run': sim_run}
            
        except Exception as e:
            st.error(f"Error retrieving detailed results: {str(e)}")
            return None
    
    def delete_simulation_run(self, run_id):
        """Delete a simulation run and its associated data"""
        try:
            session = self.SessionLocal()
            
            # Delete strategy performance data first
            session.query(StrategyPerformance).filter(
                StrategyPerformance.simulation_run_id == run_id
            ).delete()
            
            # Delete simulation run
            session.query(SimulationRun).filter(SimulationRun.id == run_id).delete()
            
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            st.error(f"Error deleting simulation run: {str(e)}")
            return False

@st.cache_resource
def get_database_manager():
    """Get database manager instance"""
    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        return db_manager
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        return None