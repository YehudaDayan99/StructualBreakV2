#!/usr/bin/env python3
"""Debug script to test imports step by step."""

print("=== DEBUGGING IMPORTS ===")
print(f"Current working directory: {__import__('pathlib').Path().resolve()}")
print(f"Python executable: {__import__('sys').executable}")

print("\n1. Testing basic imports...")
try:
    import sys
    print("✓ sys imported")
except Exception as e:
    print(f"❌ sys import failed: {e}")

try:
    import pathlib
    print("✓ pathlib imported")
except Exception as e:
    print(f"❌ pathlib import failed: {e}")

print("\n2. Testing numpy import...")
try:
    import numpy as np
    print(f"✓ numpy imported, version: {np.__version__}")
except Exception as e:
    print(f"❌ numpy import failed: {e}")

print("\n3. Testing pandas import...")
try:
    import pandas as pd
    print(f"✓ pandas imported, version: {pd.__version__}")
except Exception as e:
    print(f"❌ pandas import failed: {e}")

print("\n4. Testing matplotlib import...")
try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except Exception as e:
    print(f"❌ matplotlib import failed: {e}")

print("\n5. Testing sys.path manipulation...")
try:
    repo_parent = pathlib.Path().resolve().parent.parent
    print(f"Repo parent path: {repo_parent}")
    print(f"StructualBreakV2 exists: {(repo_parent / 'StructualBreakV2' / '__init__.py').exists()}")
    
    if str(repo_parent) not in sys.path:
        sys.path.insert(0, str(repo_parent))
        print(f"✓ Added to sys.path: {repo_parent}")
    else:
        print(f"✓ Already in sys.path: {repo_parent}")
        
    print(f"Current sys.path (first 3 entries):")
    for i, path in enumerate(sys.path[:3]):
        print(f"  {i}: {path}")
        
except Exception as e:
    print(f"❌ sys.path manipulation failed: {e}")

print("\n6. Testing StructualBreakV2 import...")
try:
    from StructualBreakV2 import compute_predictors_for_values, run_batch, Wavelet21Method
    print("✓ StructualBreakV2 imports successful")
    
    # Test instantiation
    wavelet21 = Wavelet21Method()
    print("✓ Wavelet21Method instantiation successful")
    
except Exception as e:
    print(f"❌ StructualBreakV2 import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")
