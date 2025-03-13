#!/usr/bin/env python3
"""
Test script to verify possible module import paths.
"""

import sys
import os
import importlib

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Testing module paths from: {project_root}")
print(f"Python path: {sys.path}")

# Algorithm modules to test
algorithms = ["momentum_algorithm", "breakout_algorithm", "mean_reversion_algorithm"]

# Different import paths to try
path_patterns = [
    "algorithms.{algo}",
    "src.algorithms.{algo}"
]

# Test importing each module with each path pattern
for algo in algorithms:
    print(f"\nTesting import for {algo}:")
    for pattern in path_patterns:
        module_path = pattern.format(algo=algo)
        try:
            module = importlib.import_module(module_path)
            print(f"✓ Successfully imported {module_path}")
            
            # Check for expected classes
            module_dir = dir(module)
            algorithm_classes = [name for name in module_dir if name.endswith('Algorithm')]
            print(f"  Found classes: {algorithm_classes}")
            print(f"  Module file: {module.__file__}")
        except Exception as e:
            print(f"✗ Failed to import {module_path}: {e}")

print("\nChecking for tick data functions:")
try:
    from src.base_bot import BaseBot
    print("✓ Successfully imported BaseBot from src.base_bot")
except Exception as e:
    print(f"✗ Failed to import BaseBot: {e}")

try:
    import asyncpg
    print("✓ Successfully imported asyncpg")
except Exception as e:
    print(f"✗ Failed to import asyncpg: {e}")