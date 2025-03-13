#!/usr/bin/env python3
"""
Test script to verify that algorithm imports are working correctly.
"""

import sys
import os
import importlib

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Testing algorithm imports from: {project_root}")
print(f"Python path: {sys.path}")

# Try to import all algorithm modules
algorithm_modules = [
    'algorithms.momentum_algorithm',
    'algorithms.breakout_algorithm',
    'algorithms.mean_reversion_algorithm',
    'algorithms.minute_momentum_algorithm',
    'algorithms.price_pattern_algorithm',
    'algorithms.support_resistance_algorithm',
    'algorithms.volatility_breakout_algorithm',
    'algorithms.volume_surge_algorithm'
]

print("\nTesting algorithm module imports:")
for module_name in algorithm_modules:
    try:
        module = importlib.import_module(module_name)
        module_classes = [name for name in dir(module) if name.endswith('Algorithm') and not name.startswith('__')]
        print(f"✓ Successfully imported {module_name}")
        print(f"  Found classes: {module_classes}")
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")

print("\nChecking YAML configurations:")
# Get all YAML files and check their algorithm_module entries
import glob
import yaml

yaml_files = glob.glob(os.path.join(project_root, 'src/bots/*.yaml'))
for yaml_file in sorted(yaml_files)[:5]:  # Check just the first 5 files
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            algorithm_module = config.get('algorithm_module', 'Not specified')
            print(f"YAML file: {os.path.basename(yaml_file)}")
            print(f"  algorithm_module: {algorithm_module}")
    except Exception as e:
        print(f"Error processing {yaml_file}: {e}")