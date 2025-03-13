#!/usr/bin/env python3
"""
Fix incorrect algorithm paths in YAML files.
"""

import os
import yaml
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))

def fix_yaml_files():
    # Directory containing bot YAML files
    bots_dir = os.path.join(project_root, 'src', 'bots')
    
    # Check if the directory exists
    if not os.path.exists(bots_dir):
        print(f"Error: Bot directory {bots_dir} does not exist")
        return
    
    # Get all YAML files with volitility in the name
    yaml_files = [f for f in os.listdir(bots_dir) if 
                  (f.endswith('.yaml') or f.endswith('.yml')) and
                  "volitility" in f]
    yaml_files.sort()
    
    print(f"Found {len(yaml_files)} YAML files with 'volitility' in the name")
    
    # Track which files were fixed
    fixed_files = []
    
    # Fix each YAML file
    for yaml_file in yaml_files:
        file_path = os.path.join(bots_dir, yaml_file)
        try:
            # Read the YAML file
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Fix the algorithm module path
            fixed_content = content.replace(
                'algorithms.volitility_breakout_algorithm',
                'algorithms.volatility_breakout_algorithm'
            )
            
            # Write the fixed content back
            with open(file_path, 'w') as f:
                f.write(fixed_content)
                
            fixed_files.append(yaml_file)
            print(f"✓ Fixed {yaml_file}")
                
        except Exception as e:
            print(f"✗ Error fixing {yaml_file}: {e}")
    
    # Print summary
    if fixed_files:
        print(f"\nSuccessfully fixed {len(fixed_files)} YAML files")
    else:
        print("\nNo files were fixed")
        
if __name__ == "__main__":
    fix_yaml_files()