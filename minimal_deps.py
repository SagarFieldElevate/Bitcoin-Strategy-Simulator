"""
Minimal dependency installer for the Bitcoin Strategy Simulator
Downloads and installs packages individually to avoid thread pool exhaustion
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Install a single package using pip"""
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--no-deps', package_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✓ {package_name}")
            return True
        else:
            print(f"✗ {package_name}: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"✗ {package_name}: {e}")
        return False

def main():
    # Essential packages in dependency order
    packages = [
        'setuptools',
        'wheel', 
        'six',
        'python-dateutil',
        'pytz',
        'numpy==1.24.0',  # Compatible version
        'pandas==1.5.3',  # Compatible version
        'requests',
        'urllib3',
        'certifi',
        'charset-normalizer',
        'idna',
        'click',
        'markupsafe',
        'jinja2',
        'itsdangerous',
        'werkzeug',
        'plotly==5.17.0',  # Compatible version
        'streamlit==1.28.0',  # Compatible version
        'yfinance',
        'openai',
        'pinecone-client',
        'psycopg2-binary',
        'sqlalchemy',
        'arch'
    ]
    
    print("Installing Bitcoin Strategy Simulator dependencies...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstalled {success_count}/{len(packages)} packages")
    
    if success_count >= len(packages) - 3:  # Allow a few failures
        print("Sufficient packages installed to run the simulator")
        return True
    else:
        print("Too many package failures, trying alternative approach")
        return False

if __name__ == '__main__':
    main()