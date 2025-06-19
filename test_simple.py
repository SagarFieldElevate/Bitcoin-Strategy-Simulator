#!/usr/bin/env python3
"""
Simple test to verify basic Python functionality
"""

try:
    import sys
    print(f"Python version: {sys.version}")
    
    # Test basic imports
    import json
    print("✓ JSON import successful")
    
    import os
    print("✓ OS import successful")
    
    # Test if we can import numpy directly
    try:
        import numpy as np
        print(f"✓ NumPy import successful: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
    
    # Test pandas
    try:
        import pandas as pd
        print(f"✓ Pandas import successful: {pd.__version__}")
    except Exception as e:
        print(f"✗ Pandas import failed: {e}")
        
    print("Basic test completed")
    
except Exception as e:
    print(f"Critical error: {e}")