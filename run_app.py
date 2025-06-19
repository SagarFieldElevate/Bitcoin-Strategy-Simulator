#!/usr/bin/env python3
"""
Direct runner for the sophisticated Bitcoin Strategy Simulator
Bypasses package installation issues by using system Python
"""

import sys
import os
import subprocess

# Add current directory to Python path
sys.path.insert(0, '/home/runner/workspace')

def check_dependencies():
    """Check if required dependencies are available"""
    required = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'yfinance', 'openai', 'pinecone', 'arch'
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep}")
    
    return missing

def install_missing(packages):
    """Install missing packages using pip"""
    if not packages:
        return True
        
    try:
        # Try pip install with user flag
        cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully installed: {', '.join(packages)}")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Installation error: {e}")
        return False

def run_streamlit():
    """Run the original Streamlit app"""
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/runner/workspace'
        
        # Run streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', 
               '--server.port', '5000', '--server.address', '0.0.0.0']
        
        subprocess.run(cmd, env=env)
        
    except Exception as e:
        print(f"Streamlit run error: {e}")

if __name__ == '__main__':
    print("Bitcoin Strategy Simulator - Dependency Check")
    
    missing = check_dependencies()
    
    if missing:
        print(f"\nAttempting to install missing packages: {missing}")
        if install_missing(missing):
            print("Dependencies installed successfully")
        else:
            print("Some dependencies failed to install, proceeding anyway...")
    
    print("\nStarting Streamlit app...")
    run_streamlit()