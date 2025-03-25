#!/usr/bin/env python3
"""
Check all bot configurations to see which ones are having issues.
"""

import os
import yaml
import sys
import importlib

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_bots():
    # Directory containing bot YAML files
    bots_dir = os.path.join(project_root, 'src', 'bots')
    
    # Check if the directory exists
    if not os.path.exists(bots_dir):
        print(f"Error: Bot directory {bots_dir} does not exist")
        return
    
    # Get all YAML files
    yaml_files = [f for f in os.listdir(bots_dir) if f.endswith('.yaml') or f.endswith('.yml')]
    yaml_files.sort()
    
    print(f"Found {len(yaml_files)} bot configuration files")
    
    # Track algorithm module usage
    algo_counts = {}
    problem_bots = []
    
    # Check each YAML file
    for yaml_file in yaml_files:
        file_path = os.path.join(bots_dir, yaml_file)
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            bot_id = config.get('bot_id')
            ticker = config.get('ticker')
            algo_module = config.get('algorithm_module')
            
            if algo_module:
                # Count algorithm usage
                algo_counts[algo_module] = algo_counts.get(algo_module, 0) + 1
                
                # Skip actual imports - just check if the algorithm module exists
                algo_name = algo_module.split('.')[-1]
                
                # Map of algorithm file names to expected class names
                algo_class_map = {
                    "momentum_algorithm": "MomentumAlgorithm",
                    "breakout_algorithm": "BreakoutAlgorithm",
                    "mean_reversion_algorithm": "Mean_reversionAlgorithm",
                    "minute_momentum_algorithm": "Minute_momentumAlgorithm",
                    "price_pattern_algorithm": "Price_patternAlgorithm",
                    "support_resistance_algorithm": "Support_resistanceAlgorithm",
                    "volatility_breakout_algorithm": "Volatility_breakoutAlgorithm",
                    "volume_surge_algorithm": "Volume_surgeAlgorithm"
                }
                
                if algo_name in algo_class_map:
                    class_name = algo_class_map[algo_name]
                    print(f"✓ Bot {bot_id} ({yaml_file}): {algo_module} -> {class_name}")
                else:
                    problem_bots.append((bot_id, yaml_file, f"Unknown algorithm module: {algo_module}"))
                    print(f"✗ Bot {bot_id} ({yaml_file}): Unknown algorithm module: {algo_module}")
            else:
                problem_bots.append((bot_id, yaml_file, "No algorithm_module specified"))
                print(f"✗ Bot {bot_id} ({yaml_file}): No algorithm_module specified")
                
        except Exception as e:
            problem_bots.append((None, yaml_file, f"Error loading YAML: {e}"))
            print(f"✗ Error loading {yaml_file}: {e}")
    
    # Print algorithm usage statistics
    print("\nAlgorithm Usage:")
    for algo, count in sorted(algo_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{algo}: {count} bots")
    
    # Print summary of problem bots
    if problem_bots:
        print(f"\nFound {len(problem_bots)} problematic bot configurations:")
        for bot_id, yaml_file, error in problem_bots:
            print(f"Bot {bot_id} ({yaml_file}): {error}")
    else:
        print("\nAll bot configurations appear valid!")

if __name__ == "__main__":
    check_bots()